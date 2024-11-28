# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import random
import os
import math
import logging
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # force datalist even
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[:self.world_size]
            data = data[self.rank::self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[:self.num_workers]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, utt2wav, utt2text, utt2spk, shuffle=True, partition=True, tts_text=None):
        self.lists = lists
        self.utt2wav = utt2wav
        self.utt2text = utt2text
        self.utt2spk = utt2spk
        self.tts_text = tts_text  # a list, each prompt will generate all texts in the list
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            utt = self.lists[index]
            sample = {}
            sample['utt'] = utt
            sample['wav'] = self.utt2wav[utt]
            if utt in self.utt2text:
                sample['text'] = self.utt2text[utt]
            if utt in self.utt2spk:
                sample['spk'] = self.utt2spk[utt]
            else:
                sample['spk'] = utt

            sample.update(sampler_info)

            if self.tts_text is not None:
                if isinstance(self.tts_text, dict):
                    for text in self.tts_text[utt]:
                        new_sample = sample.copy()
                        new_sample.update({'tts_text': text})
                        yield new_sample

                elif isinstance(self.tts_text, list):
                    for text in self.tts_text:
                        new_sample = sample.copy()
                        new_sample.update({'tts_text': text})
                        yield new_sample
            else:
                yield sample


def Dataset(data_dir,
            data_pipeline,
            mode='train',
            gan=False,
            shuffle=True,
            partition=True,
            tts_file=None):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert mode in ['train', 'inference']

    utt2wav = {}
    utt2text = {}
    utt2spk = {}

    def add_one_data(data_dir):
        logging.info(f"Loading data: {data_dir}")
        assert os.path.exists(f"{data_dir}/wav.scp") # \
               # and os.path.exists(f"{data_dir}/text")
               # and os.path.exists(f"{data_dir}/utt2spk")

        with open(f"{data_dir}/wav.scp", 'r', encoding='utf-8') as f_scp:
            for line in f_scp:
                line = line.strip().split(maxsplit=1)
                if len(line) != 2:
                    continue
                utt, wav = line[0], line[1]
                utt2wav[utt] = wav

        if os.path.exists(f"{data_dir}/text"):
            with open(f"{data_dir}/text", 'r', encoding='utf-8') as f_text:
                for line in f_text:
                    line = line.strip().split(maxsplit=1)
                    if len(line) != 2:
                        continue
                    utt, text = line[0], line[1]
                    utt2text[utt] = text

        if os.path.exists(f"{data_dir}/utt2spk"):
            with open(f"{data_dir}/utt2spk", 'r', encoding='utf-8') as f_spk:
                for line in f_spk:
                    line = line.strip().split(maxsplit=1)
                    if len(line) != 2:
                        continue
                    utt, spk = line[0], line[1]
                    utt2spk[utt] = spk

        logging.info(f"Current utts: {len(utt2wav.keys())}")

    if isinstance(data_dir, list):
        for sub_data in data_dir:
            add_one_data(sub_data)
    else:
        add_one_data(data_dir)

    valid_utt_list = list(utt2wav.keys())
    if len(utt2text) != 0 and mode=='train':
        valid_utt_list = list(set(utt2wav.keys()) & set(utt2text.keys()))
    logging.info(f"Total utts: {len(valid_utt_list)}")

    tts_text = None
    if mode=="inference" and os.path.exists(tts_file):
        # valid_utt_list.sort()
        with open(tts_file, 'r', encoding='utf-8') as f_ttstext:
            if tts_file.endswith('.txt'):
                tts_text = []
                for line in f_ttstext:
                    line = line.strip()
                    tts_text.append(line)
            elif tts_file.endswith('.json'):
                import json
                tts_text=json.load(f_ttstext)
            logging.info(f"read {len(tts_text)} lines from {tts_file}")

    dataset = DataList(valid_utt_list, utt2wav, utt2text, utt2spk,
                       shuffle=shuffle, partition=partition, tts_text=tts_text)

    if gan is True:
        # map partial arg to padding func in gan mode
        data_pipeline[-1] = partial(data_pipeline[-1], gan=gan)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset
