# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#                    Jing Du
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

import logging
import os
import torch
import json
import re
import datetime
import yaml

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR
from cosyvoice.dataset.dataset_kaldidata import Dataset as KaldiDataset
from cosyvoice.dataset.dataset_jsondata import Dataset as JsonDataset
import s3tokenizer
from cosyvoice.speaker.speaker_encoder import SpeakerEmbedding


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        model._set_static_graph()
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict, only_yaml=False):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if not only_yaml:
        if info_dict["train_engine"] == "torch_ddp":
            if rank == 0:
                torch.save(model.module.state_dict(), save_model_path)
        else:
            with torch.no_grad():
                model.save_checkpoint(save_dir=model_dir,
                                      tag=model_name,
                                      client_state=info_dict)
        if rank == 0:
            logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(
                rank, save_model_path))

    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to yaml {}'.format(rank, info_path))


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, batch, scaler, info_dict):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step, lr, rank, ' '.join(['{}_{}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step)

def init_kaldi_dataset(args, configs, gan, train_data_indexes):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']

    train_data = [configs['train_data'][i] for i in train_data_indexes]

    train_dataset = KaldiDataset(train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = KaldiDataset(configs['cv_data'], data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader

def init_json_dataset(args, configs, gan, train_data_indexes):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']

    train_data = [configs['train_data'][i] for i in train_data_indexes]

    train_dataset = JsonDataset(train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = JsonDataset(configs['cv_data'], data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=0,
                                prefetch_factor=None)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader

def get_latest_ckpt(ckpt_dir, regex="epoch_*.pt"):
    import glob
    f_list = glob.glob(os.path.join(ckpt_dir, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(f_list) != 0:
        x = f_list[-1]
        epoch = x.split("epoch_")[1].split("_")[0]
        y = f"{ckpt_dir}/epoch_{epoch}_whole.pt"
        if os.path.exists(y):
            x = y
        return x
    else:
        return "failed to find latest_checkpoint_path:" \
               + os.path.join(ckpt_dir, regex)

def get_resume_params(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        info_dict = load_hyperpyyaml(file)
    return info_dict

def freeze(model):
    for _, param in model.named_parameters():
        param.requires_grad = False
    return model

def init_codec_and_embed_model(configs, rank=0):
    if configs['codec_type'] == 's3tokenizer':
        codec_model = s3tokenizer.load_model(
            'speech_tokenizer_v1_25hz', configs['s3tokenizer_ckpt'])
        logging.info(f"loaded codec model ckpt {configs['s3tokenizer_ckpt']}")
    elif configs['codec_type'] == 's3tokenizer-v2':
        codec_model = s3tokenizer.load_model(
            'speech_tokenizer_v2_25hz', configs['s3tokenizer_ckpt'])
    codec_model = codec_model.cuda(rank)
    codec_model = freeze(codec_model)

    use_freeze_spkemb = configs.get('use_freeze_spkemb', True)
    if use_freeze_spkemb:  # 使用离线的说话人向量提取模型，不和主干网络一起训练
        spkemb_model = SpeakerEmbedding(
            ckpt_path=configs['speaker_encoder_ckpt']).cuda(rank)
        spkemb_model = freeze(spkemb_model)
    else:
        spkemb_model = None

    return codec_model, spkemb_model

def get_codec_and_spkemb(batch_dict, codec_model, spkemb_model,
                         configs):
    wave = batch_dict['speech'].to(codec_model.device)
    wave_len = batch_dict['speech_len'].to(codec_model.device)
    with torch.no_grad():
        if configs['codec_type'] == 'facodec':
            enc_out, prosody_latent = codec_model.encode(wave.unsqueeze(1))
            vq_post_emb, vq_id, spk_embs = codec_model.quantize(
                enc_out, prosody_latent)
            codec_speaker_embs = spk_embs  # B D
            speech_code = vq_id  # B T 6
            speech_code_len = wave_len // codec_model.fa_en.hop_length
        elif configs['codec_type'].startswith('s3tokenizer'):
            mels = []
            import s3tokenizer
            for i in range(wave.size(0)):
                audio = wave[i, :wave_len[i]]
                # whisper speech code use 16k sample_rate
                if configs['sample_rate'] != 16000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        configs['sample_rate'], 16000).to(audio.device)
                    audio = resampler(audio)
                mels.append(s3tokenizer.log_mel_spectrogram(audio))
            mels, mels_lens = s3tokenizer.padding(mels)
            mels = mels.to(codec_model.device)
            mels_lens = mels_lens.to(codec_model.device)
            speech_code, speech_code_len = codec_model.quantize(mels, mels_lens)

        speech_code = speech_code.clone()
        speech_code_len = speech_code_len.clone()
    batch_dict['speech_token'] = speech_code
    batch_dict['speech_token_len'] = speech_code_len

    if spkemb_model is not None:
        spk_audio_crop = configs.get('spk_audio_crop', 0)
        if spk_audio_crop:
            wave_len = wave_len.to('cpu')
            crop_length = spk_audio_crop * configs['sample_rate']
            extracted_waves = []
            spk_wave_len = []
            for b, true_length in enumerate(wave_len):
                if true_length < crop_length:  # 需要拼接至crop_length
                    repeat_times = (crop_length + wave_len[b] - 1) // wave_len[b]
                    extracted_wave = torch.cat([wave[b][:true_length]] * repeat_times)
                    extracted_wave = extracted_wave[:crop_length]
                    spk_wave_len.append(crop_length)
                else:
                    random_length = torch.randint(crop_length, true_length + 1, (1,)).item()
                    start_idx = torch.randint(0, true_length-random_length+1, (1,)).item()
                    extracted_wave = wave[b, start_idx:start_idx + random_length]
                    spk_wave_len.append(random_length)

                extracted_waves.append(extracted_wave)

            from torch.nn.utils.rnn import pad_sequence
            spk_wave = pad_sequence(extracted_waves, batch_first=True, padding_value=0)
            spk_wave = spk_wave.to(codec_model.device)
            spk_wave_len = torch.tensor(spk_wave_len).to(codec_model.device)
        else:
            spk_wave = wave
            spk_wave_len = wave_len

        with torch.no_grad():
            # the speaker_embed_model use 24k wave tensor input, if not 24k, resample is needed
            speaker_embs = spkemb_model(spk_wave.unsqueeze(1), spk_wave_len)  # B D
        batch_dict['embedding'] = speaker_embs

    return batch_dict
