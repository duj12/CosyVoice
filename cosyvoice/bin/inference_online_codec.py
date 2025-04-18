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
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import os
import time
import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.train_utils import (
    init_codec_and_embed_model, get_codec_and_spkemb)
from cosyvoice.dataset.dataset_kaldidata import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data', required=True, help='prompt data file, in kaldi format')
    parser.add_argument('--tts_text', required=True, help='tts input file, each line is a sentence to generate')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode',
                        default='zero_shot',
                        choices=['sft', 'zero_shot'],
                        help='inference mode')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    args = parser.parse_args()
    logging.info(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init cosyvoice models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16=False, sr=configs['sample_rate'])
    model.load(args.llm_model, args.flow_model, args.hifigan_model)

    codec_model, spkemb_model = init_codec_and_embed_model(configs, 0)
    data_pipeline = configs['infer_data_pipeline'] if 'infer_data_pipeline' in configs else configs['data_pipeline']
    test_dataset = Dataset(args.prompt_data, data_pipeline=data_pipeline, mode='inference', shuffle=False, partition=False,
                           tts_file=args.tts_text)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    args.result_dir = os.path.abspath(args.result_dir)
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        time_start = time.perf_counter()
        for wav_idx, batch in tqdm(enumerate(test_data_loader)):
            batch = get_codec_and_spkemb(batch, codec_model, spkemb_model, configs)

            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"
            tts_text = batch["tts_text"]
            print(utts, tts_text)

            tts_text_token = batch["tts_text_token"].to(device)
            tts_text_token_len = batch["tts_text_token_len"].to(device)
            utt_embedding = batch["embedding"].to(device)
            spk_embedding = batch["embedding"].to(device)

            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)
            speech_feat = batch["speech_feat"].to(device)
            speech_feat_len = batch["speech_feat_len"].to(device)

            if args.mode == 'sft':
                # 实验发现如果flow不输入speech token和speech feat会更稳定，
                # 主要靠embedding和tts_predict_speech_token去重建语音。
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                               'llm_embedding': spk_embedding, 'flow_embedding': spk_embedding,
                               # 'flow_prompt_speech_token': speech_token,
                               # 'flow_prompt_speech_token_len': speech_token_len,
                               # 'prompt_speech_feat': speech_feat,
                               # 'prompt_speech_feat_len': speech_feat_len,
                }

            else:
                text_token = batch["text_token"].to(device)
                text_token_len = batch["text_token_len"].to(device)
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                               'prompt_text': text_token, 'prompt_text_len': text_token_len,
                               'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                               'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                               'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                               'llm_embedding': utt_embedding, 'flow_embedding': utt_embedding}
            tts_speeches = []
            for model_output in model.tts(**model_input):
                tts_speeches.append(model_output['tts_speech'])

            time_end = time.perf_counter()
            time_tts = time_end - time_start

            tts_speeches = torch.concat(tts_speeches, dim=1)
            time_stamp = time.strftime('%m%d%H%M', time.localtime())
            tts_key = f'{wav_idx+1}_{utts[0]}_{tts_text[0][:10]}_{time_stamp}'
            tts_fn = os.path.join(args.result_dir, f'{tts_key}.wav')
            torchaudio.save(tts_fn, tts_speeches, sample_rate=24000)

            time_audio = tts_speeches.size(-1) / configs['sample_rate']
            rtf = time_tts / time_audio
            logging.info(f'RTF {rtf}, {tts_fn}')
            f.write(f'{tts_key}\t{tts_fn}\t{rtf}\n')
            f.flush()

            time_start = time.perf_counter()   # 开始时间更新

    f.close()
    logging.info(f'Result wav.scp saved in {fn}')


if __name__ == '__main__':
    main()
