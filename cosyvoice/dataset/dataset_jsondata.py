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
import logging
import json
import math
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

    def __init__(self, lists, utt2wav, utt2text, utt2pho, utt2spk, utt2dur=None,
                 shuffle=True, partition=True, tts_text=None, eval=False, need_text=True):
        self.lists = lists
        self.utt2wav = utt2wav
        self.utt2text = utt2text
        self.utt2pho = utt2pho
        self.utt2spk = utt2spk
        self.utt2dur = utt2dur
        self.tts_text = tts_text  # a list, each prompt will generate all texts in the list
        self.need_text = need_text
        if not eval:
            self.sampler = DistributedSampler(shuffle, partition)
        else:
            self.sampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        if self.sampler is not None:
            sampler_info = self.sampler.update()
            indexes = self.sampler.sample(self.lists)
        else:
            sampler_info = {}
            indexes = range(len(self.lists))

        for index in indexes:
            utt = self.lists[index]
            sample = {}
            sample['utt'] = utt
            sample['wav'] = self.utt2wav[utt]
            if self.need_text:
                if utt not in self.utt2text:
                    logging.warning(f'key {utt} not in self.utt2text. jumped.')
                    continue
                else:
                    sample['text'] = self.utt2text[utt]
                if utt not in self.utt2pho:
                    logging.warning(f'key {utt} not in self.utt2pho. jumped.')
                    continue
                else:
                    sample['pho'] = self.utt2pho[utt]
            if utt in self.utt2spk:
                sample['spk'] = self.utt2spk[utt]
            else:
                sample['spk'] = utt

            if self.utt2dur is not None and utt in self.utt2dur:
                sample['mfa_duration'] = self.utt2dur[utt]

            sample.update(sampler_info)

            if self.tts_text is not None:
                for text in self.tts_text:
                    new_sample = sample.copy()
                    new_sample.update({'tts_text': text})
                    yield new_sample
            else:
                yield sample


def Dataset(json_file,
            data_pipeline,
            mode='train',
            gan=False,
            shuffle=True,
            partition=True,
            tts_file=None,
            eval=False,
            rich_sample_short_utt=0,
            need_text=True
    ):
    """ Construct dataset from arguments

        json_file is like :
        {
          "ID_0|ID_0": [
            7697.139833333331,
            [
              [
                "USF01_YJ_0001",
                "4.130041666666667",
                {
                  "text": ["F14", "ER14", "S14", "T14", "#1", "AH14", "VV14", "#1", "AO14", "L14", "L14", "EH14", "T14", "S14", "#1", "HH14", "EH14", "D14", "#1", "D14", "AW14", "N14", "S14", "T14", "EH14", "R14", "Z14", "#1", "T14", "UW14", "#1", "T14", "AO14", "K14", "#1", "AH14", "B14", "AW14", "T14", "#1", "AY14", "P14", "AE14", "D14", "#1", "OW14", "EH14", "S14", "#4", "<p>"],
                  "mfa_duration": [0.18, 0.13, 0.08000000000000002, 0.04999999999999999, 0.04999999999999999, 0.050000000000000044, 0.19999999999999996, 0.19000000000000006, 0.17000000000000004, 0.07999999999999985, 0.06000000000000005, 0.030000000000000027, 0.07000000000000006, 0.05999999999999983, 0.050000000000000044, 0.040000000000000036, 0.1100000000000001, 0.05999999999999983, 0.09000000000000008, 0.050000000000000044, 0.10999999999999988, 0.06000000000000005, 0.05999999999999983, 0.050000000000000266, 0.040000000000000036, 0.08999999999999986, 0.10000000000000009, 0.06999999999999984, 0.040000000000000036, 0.040000000000000036, 0.1299999999999999, 0.08000000000000007, 0.1499999999999999, 0.10000000000000009, 0.10999999999999988, 0.03000000000000025, 0.16999999999999993, 0.16000000000000014, 0.23999999999999977, 0.5000416666666672],
                  "head_end_sil": [0.0, 0.5000416666666672]
                }
              ],
                ...
          ...
        }
        speaker
           speaker's total duration
           [
             [
               utt1
               duration
               {
                 "text": [  phoneme sequence list ]
                 ...
                }
              ]
              ...
            ]


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
    utt2pho = {}
    utt2spk = {}
    utt2text = {}
    utt2dur = {}
    valid_utt_list = []

    def add_one_data(json_file):
        logging.info(f"Loading data: {json_file}, short data sample times {rich_sample_short_utt}")
        duplicated_utt = 0
        if isinstance(json_file, list):
            json_file, language, repeat_time = json_file
        with open(json_file, 'r', encoding='utf8') as fin:
            dataset_info = json.load(fin)
        data_dir = os.path.dirname(json_file)
        data_name = os.path.basename(data_dir)
        wave_dir = os.path.join(data_dir, "Formatted")
        kaldi_data_dir = os.path.join(data_dir, data_name)

        for speaker, info in dataset_info.items():
            speaker_folder = speaker.split('|')[0]
            sid = speaker.split('|')[1]
            total_dur, file_list = info
            for fname, dur, sequence in file_list:
                pho = sequence['text']
                wav_path = os.path.join(wave_dir, speaker_folder, '{}.wav'.format(fname))
                utt = f"{data_name}-{fname}"
                speaker = f"{data_name}-{sid}"
                if utt in utt2wav:
                    # 同一个数据集里面有同名的音频，他们文本可能不同，处理数据时text文件只保留了其中一个的文本，直接删掉这些重名数据好了
                    # logging.warning(f"{utt} is duplicated. remove it from train data.")
                    # del utt2wav[utt], utt2pho[utt], utt2spk[utt]
                    duplicated_utt += 1
                    continue

                utt2wav[utt] = wav_path
                utt2pho[utt] = pho
                utt2spk[utt] = speaker
                if 'mfa_duration' in sequence:
                    mfa_duration = sequence['mfa_duration']
                    duration = []
                    mfa_idx = 0
                    for idx, p in enumerate(pho):
                        if p.startswith('#'):   # 韵律音素，插入时长为0
                            duration.append(0)
                            continue
                        duration.append(float(mfa_duration[mfa_idx]))
                        mfa_idx += 1
                    # assert len(pho) == len(duration)
                    utt2dur[utt] = duration

                valid_utt_list.append(utt)
                if rich_sample_short_utt>0 and len(pho) < 20:  # 对音素序列长度低于20的音频富采样
                    valid_utt_list.extend([utt]*rich_sample_short_utt)

        text_path = os.path.join(kaldi_data_dir, "text_punc")
        if not os.path.exists(text_path):
            text_path = os.path.join(kaldi_data_dir, 'text')
            assert(os.path.exists(text_path)), f"text path of {data_name} not exist! you shou check it."

        with open(text_path, 'r', encoding='utf-8') as ftext:
            for line in ftext:
                line = line.strip().split(maxsplit=1)
                if len(line)!=2:
                    continue
                utt, text = line[0], line[1]
                utt = f"{data_name}-{utt}"
                if utt not in utt2pho:
                    continue
                utt2text[utt] = text

        del dataset_info
        logging.info(f"Current utts: {len(utt2wav.keys())}. total samples: {len(valid_utt_list)}, duplicated:{duplicated_utt}")

    if isinstance(json_file, list):
        for sub_data in json_file:
            add_one_data(sub_data)
    else:
        add_one_data(json_file)

    # if need_text:
    #     valid_utt_list = list(set(utt2wav.keys()) & set(utt2text.keys()))
    logging.info(f"Total utts: {len(utt2wav.keys())}. Actual total samples {len(valid_utt_list)}")

    tts_text = None
    if mode=="inference" and os.path.exists(tts_file):
        tts_text = []
        with open(tts_file, 'r', encoding='utf-8') as f_ttstext:
            for line in f_ttstext:
                line = line.strip()
                tts_text.append(line)
            logging.info(f"read {len(tts_text)} lines from {tts_file}")

    dataset = DataList(valid_utt_list, utt2wav, utt2text, utt2pho, utt2spk,
                       utt2dur, shuffle=shuffle, partition=partition,
                       tts_text=tts_text, eval=eval, need_text=need_text)

    if gan is True:
        # map partial arg to padding func in gan mode
        data_pipeline[-1] = partial(data_pipeline[-1], gan=gan)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset
