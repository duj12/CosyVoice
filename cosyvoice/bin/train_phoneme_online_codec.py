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
logging.getLogger('numba').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed
import cosyvoice.loralib as lora
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record
from cosyvoice.utils.executor_online_codec import Executor
from cosyvoice.utils.ema import EMA as ExponentialMovingAverage
from cosyvoice.utils.train_utils import (
    init_distributed,  init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config,
    get_latest_ckpt, get_resume_params,
    init_json_dataset, init_codec_and_embed_model
)


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

    # add additional LoRA parameters
    use_lora = configs.get("use_lora", False)
    if use_lora:
        model = lora.replace_specific_layer_4lora(model, configs)
        lora.mark_only_lora_as_trainable(model)
    lora.getModelSize_lora(model)

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
            if 'generator' in saved_state_dict:
                logging.info('use pretrained generator.')
                saved_state_dict = saved_state_dict['generator']

            new_state_dict = {}
            if gan and 'generator.m_source.l_linear.weight' not in saved_state_dict and 'generator.input_embedding.weight' not in saved_state_dict:
                # 模型参数只保存了generator
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

    codec_model, spkemb_model = init_codec_and_embed_model(configs, rank)

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
        if 'data_idx' in resume_info:
            executor.data_idx = resume_info['data_idx'] + 1

    executor.configs = configs
    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if scaler is not None:
        logging.info(f"use cuda amp training, dtype: {configs.get('dtype', 'fp32')}.")
    else:
        logging.info(f"use cuda.matmul.allow_tf32, dtype: tf32.")
        torch.backends.cuda.matmul.allow_tf32 = True  # speedup large matrix multiply
        torch.backends.cudnn.allow_tf32 = True  # speedup conv layers

    # ema_decay = configs.get('ema_decay', 0.9)
    # ema = ExponentialMovingAverage(model, decay=ema_decay)
    # ema.to(f'cuda:{rank}')

    # Start training loop
    for epoch in range(start_epoch, info_dict['max_epoch']):
        executor.epoch = epoch

        for data_i, data_indexes in enumerate(configs['train_data_indexes']):
            if executor.data_idx >= len(configs['train_data_indexes']) - 1:
                executor.data_idx = 0
            if data_i != executor.data_idx:
                logging.warning(f"Jumped train data {data_indexes}")
                continue

            # Get dataset & dataloader
            logging.info(f"Loading train data index {data_indexes}")
            train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
                init_json_dataset(args, configs, gan, data_indexes)

            train_dataset.set_epoch(epoch)
            dist.barrier()
            group_join = None
            # group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
            if gan is True:
                executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                            writer, info_dict, scaler, group_join, codec_model, spkemb_model)
            else:
                executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join, codec_model, spkemb_model)

            executor.data_idx = data_i + 1
            if data_i >= len(configs['train_data_indexes']) - 1:
                executor.data_idx = 0

            del train_dataset
            del train_data_loader

            # dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
