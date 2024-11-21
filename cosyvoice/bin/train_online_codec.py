# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               Jing Du  (thuduj12@163.com)
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

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor_online_codec import Executor
from cosyvoice.dataset.dataset_kaldidata import Dataset
from torch.utils.data import DataLoader

from cosyvoice.utils.train_utils import (
    init_distributed,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)

import s3tokenizer

import sys
# local codec model and code
sys.path.append(".")
from facodec.facodecInfer import FACodecInfer
# local spk_emb model
sys.path.append("/data/megastore/Projects/DuJing/code/vits_new")
from vits.model.models import SpeakerEmbedding

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=False,
                        help='train data dir, you can assign it in .yaml')
    parser.add_argument('--cv_data', required=False, help='cv data dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def init_dataset_and_dataloader(args, configs, gan, train_data_indexes):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']

    train_data = [configs['train_data'][i] for i in train_data_indexes]

    train_dataset = Dataset(train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = Dataset(configs['cv_data'], data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

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
    if configs['codec_type'] == 'facodec':
        codec_model = FACodecInfer().cuda(rank)
    else:
        codec_model = s3tokenizer.S3Tokenizer('speech_tokenizer_v1_25hz')
        codec_model.init_from_onnx(
            "../../../pretrained_models/CosyVoice-300M-25Hz/speech_tokenizer_v1.onnx")
        codec_model = codec_model.cuda(rank)

    spkemb_model = SpeakerEmbedding(
        ckpt_path="/data/megastore/Projects/DuJing/code/vits_new/egs/art_codec/speaker_encoder/speaker_encoder.pt").cuda(rank)

    codec_model = freeze(codec_model)
    spkemb_model = freeze(spkemb_model)
    return codec_model, spkemb_model

@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    model = configs[args.model]
    start_epoch = 0
    resume_info = None
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            if os.path.isdir(args.checkpoint):  # the ckpt path is a dir, we will use the most recent ckpt file
                ckpt_path = get_latest_ckpt(args.checkpoint)
                args.checkpoint = ckpt_path
                yaml_path = ckpt_path.replace(".pt", ".yaml")
                if os.path.exists(yaml_path):
                    resume_info = get_resume_params(yaml_path)
                    logging.info(resume_info)

            saved_state_dict = torch.load(args.checkpoint, map_location='cpu')
            new_state_dict = {}
            if gan:
                if "generator.m_source.l_linear.weight" in saved_state_dict:
                    # checkpoint include generator and discriminator
                    dest_model = model
                else:   # checkpoint only include generator
                    dest_model = model.generator
                    logging.warning('discriminator is not pretrained!')
            else:
                dest_model = model

            for k, v in dest_model.state_dict().items():
                if k not in saved_state_dict:
                    logging.warning(
                        f"{k} is not saved in the checkpoint {args.checkpoint}")
                    new_state_dict[k] = v
                elif v.size() != saved_state_dict[k].size():
                    logging.warning(
                        f"**{k} size is not same in the checkpoint:"
                        f" cur size={v.size()}, "
                        f" saved size={saved_state_dict[k].size()}")
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = saved_state_dict[k]

            dest_model.load_state_dict(new_state_dict, strict=False)
            logging.info(f'Loaded checkpoint {args.checkpoint}')
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)
    rank = int(os.environ["LOCAL_RANK"])

    if not gan:
        codec_model, spkemb_model = init_codec_and_embed_model(configs, rank)

    else:
        codec_model = None
        spkemb_model = None

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    save_model(model, 'init', info_dict)

    # Get executor
    executor = Executor(gan=gan)
    if resume_info:
        executor.step = resume_info["step"]
        start_epoch = resume_info['epoch']
        optimizer.param_groups[0]['lr'] = resume_info["lr"]
        scheduler.set_step(resume_info["step"])
        if scheduler_d is not None:
            scheduler_d.set_step(resume_info["step"])

    executor.configs = configs
    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Start training loop
    for epoch in range(start_epoch, info_dict['max_epoch']):
        executor.epoch = epoch

        for data_indexes in configs['train_data_indexes']:
            # Get dataset & dataloader
            train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
                init_dataset_and_dataloader(args, configs, gan, data_indexes)

            train_dataset.set_epoch(epoch)
            dist.barrier()
            group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
            if gan is True:
                executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                            writer, info_dict, scaler, group_join, codec_model, spkemb_model)
            else:
                executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join, codec_model, spkemb_model)

        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
