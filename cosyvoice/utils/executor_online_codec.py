# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#                    Jing Du
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


def get_codec_and_spkemb(batch_dict, codec_model, spkemb_model,
                         codec_type="s3tokenizer", sample_rate=24000):
    wave = batch_dict['speech'].to(codec_model.device)
    wave_len = batch_dict['speech_len'].to(codec_model.device)
    with torch.no_grad():
        if codec_type == 'facodec':
            enc_out, prosody_latent = codec_model.encode(wave.unsqueeze(1))
            vq_post_emb, vq_id, spk_embs = codec_model.quantize(
                enc_out, prosody_latent)
            codec_speaker_embs = spk_embs  # B D
            speech_code = vq_id  # B T 6
            speech_code_len = wave_len // codec_model.fa_en.hop_length
        else:
            mels = []
            import s3tokenizer
            for i in range(wave.size(0)):
                audio = wave[i, :wave_len[i]]
                # whisper speech code use 16k sample_rate
                if sample_rate != 16000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        sample_rate, 16000).to(audio.device)
                    audio = resampler(audio)
                mels.append(s3tokenizer.log_mel_spectrogram(audio))
            mels, mels_lens = s3tokenizer.padding(mels)
            mels = mels.to(codec_model.device)
            mels_lens = mels_lens.to(codec_model.device)
            speech_code, speech_code_len = codec_model.quantize(mels, mels_lens)

        speech_code = speech_code.clone()
        speech_code_len = speech_code_len.clone()

        # the speaker_embed_model use 24k wave tensor input, if not 24k, resample is needed
        speaker_embs = spkemb_model(wave.unsqueeze(1), wave_len)  # B D

    batch_dict['speech_token'] = speech_code
    batch_dict['speech_token_len'] = speech_code_len
    batch_dict['embedding'] = speaker_embs

    return batch_dict


class Executor:

    def __init__(self, gan: bool = False):
        self.gan = gan
        self.step = 0
        self.epoch = 0
        self.configs = {}
        self.data_idx = 0  # for multi data list, when resume training, we should start from the recent data index
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader,
                       writer, info_dict, scaler, group_join, codec_model=None, spkemb_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict = get_codec_and_spkemb(
                        batch_dict, codec_model, spkemb_model,
                        self.configs['codec_type'], self.configs['sample_rate'])

                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step ) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False,
                            codec_model=codec_model, spkemb_model=spkemb_model)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        # dist.barrier()
        # self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True,
        #         codec_model=codec_model, spkemb_model=spkemb_model)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join, codec_model=None, spkemb_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False,
                            codec_model=codec_model, spkemb_model=spkemb_model)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        # dist.barrier()
        # self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True,
        #         codec_model=codec_model, spkemb_model=spkemb_model)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True, codec_model=None, spkemb_model=None):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            info_dict["data_idx"] = self.data_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'

            if self.gan is False:   # we only need speech token and speaker emb when train llm and flow
                batch_dict = get_codec_and_spkemb(
                    batch_dict, codec_model, spkemb_model,
                    self.configs['codec_type'], self.configs['sample_rate'])

            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step)
        save_model(model, model_name, info_dict)
