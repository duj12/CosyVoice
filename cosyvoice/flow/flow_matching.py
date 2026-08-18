# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from cosyvoice.flow.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed
import onnxruntime
import threading
import queue


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.lock = threading.Lock()

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = min(flow_cache.shape[2], mu.size(2))
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :cache_size, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :cache_size, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        trt_context, trt_stream, trt_engine = None, None, None
        if getattr(self.estimator, '__class__').__name__ == 'TrtContextWrapper':
            with self.lock:
                [trt_context, trt_stream], trt_engine = self.estimator.acquire_estimator()
        else:
            from contextlib import nullcontext
            trt_stream = nullcontext
        with trt_stream:   #TensorRT并发推理时，必须把多步采样放到特定cuda stream里面，不然不同进程间Trt推理会相互影响
            # Do not use concat, it may cause memory format changed and trt infer with wrong results!
            x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
            mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
            mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
            t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
            spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
            cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)

            max_retry_times = 10
            for step in range(1, len(t_span)):
                # Classifier-Free Guidance inference introduced in VoiceBox
                mask_in[:] = mask
                mu_in[0] = mu
                t_in[:] = t.unsqueeze(0)
                spks_in[0] = spks
                cond_in[0] = cond

                retries = 1
                while retries < max_retry_times:
                    x_in[:] = x
                    dphi_dt = self.forward_estimator(
                        x_in, mask_in,  mu_in, t_in, spks_in, cond_in, streaming,
                        trt_context=trt_context, trt_stream=trt_stream, trt_engine=trt_engine
                    )
                    # 第一步采样时遇到NaN输出，更新X
                    if step == 1 and torch.isnan(dphi_dt).any():
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"NaN detected in decoder step {step}, try {retries} time.")
                        logger.info(f"x:{x}\n mu:{mu}\n, dphi_dt:{dphi_dt}")
                        x = torch.randn_like(x) / retries   # temperature = 1/retyies
                        logger.info(f"reset input noise as x: {x} \n")
                        retries += 1
                    else:
                        break

                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
                x = x + dt * dphi_dt
                t = t + dt
                sol.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

        if getattr(self.estimator, '__class__').__name__ == 'TrtContextWrapper':
            with self.lock:
                self.estimator.release_estimator(trt_context, trt_stream)

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False, **kwargs):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        elif isinstance(self.estimator, onnxruntime.InferenceSession):
            ort_inputs = {
                'x': x.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                't': t.cpu().numpy(),
                'spks': spks.cpu().numpy(),
                'cond': cond.cpu().numpy()
            }
            output_onnx = self.estimator.run(None, ort_inputs)[0]
            return torch.from_numpy(output_onnx).to(x.device)

        elif getattr(self.estimator, '__class__').__name__ == 'TrtContextWrapper':
            # 从上层多步采样的地方获取context, stream传进来，上层也需要用特定的stream
            trt_context = kwargs.get('trt_context')
            trt_stream = kwargs.get('trt_stream')
            trt_engine = kwargs.get('trt_engine')
            with trt_stream:
                trt_context.set_input_shape('x', (2, 80, x.size(2)))
                trt_context.set_input_shape('mask', (2, 1, x.size(2)))
                trt_context.set_input_shape('mu', (2, 80, x.size(2)))
                trt_context.set_input_shape('t', (2,))
                trt_context.set_input_shape('spks', (2, 80))
                trt_context.set_input_shape('cond', (2, 80, x.size(2)))
                output = torch.zeros_like(x)
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             output.contiguous().data_ptr()]
                for i, j in enumerate(data_ptrs):
                    trt_context.set_tensor_address(
                        trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert trt_context.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()

                return output

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        set_all_random_seed(0)
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None


class StreamConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None
