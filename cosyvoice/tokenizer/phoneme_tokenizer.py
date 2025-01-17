# Copyright (c) 2024  Jing Du  (thuduj12@163.com)
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

import json
import sys
from cosyvoice.tokenizer.preprocess import extract_mandarin_only, extract_non_mandarin
from cosyvoice.tokenizer.phoneme_frontend import get_frontend_result

class PhonemeTokenizer:
    def __init__(self,
                 phoneme_dict="cosyvoice/tokenizer/assets/hnttsa_phoneme2id.json",
                 mode='train', use_pause_label=True
    ):
        with open(phoneme_dict, 'r', encoding='utf-8') as fin:
            self.phoneme2id = json.load(fin)
        self.mode = mode
        self.cn_frontend_model = None
        self.en_frontend_model = None
        self.use_pause_label = use_pause_label
        if mode == 'inference':  # hard code frontend model, import from local path
            TTS_root = "/data/megastore/Projects/DuJing/code/TTS"
            sys.path.append(TTS_root)
            from tts.frontend.init_text_frontend import init_text_frontend
            if self.cn_frontend_model is None:
                self.cn_frontend_model = init_text_frontend('hntts')
            if self.en_frontend_model is None:
                self.en_frontend_model = init_text_frontend('entts')

    def encode(self, phoneme_list):
        if self.mode=='train':
            return self._parse_pho_tone_lang_prsd(phoneme_list)
        else:  # call frontmodel to get phoneme sequence
            text = phoneme_list
            # detect the language, support cn and en
            language = self._detect_language(text)
            if language == 0: # chinese
                result = get_frontend_result(text, self.cn_frontend_model)
            else:  # english
                result = get_frontend_result(text, self.en_frontend_model)

            pho, tone, lang = result['pho'], result['tone'], result['lang']
            return self._extract_prosody(pho, tone, lang)


    def _ispunc_mark(self, phoneme):
        lista = set([".", "。", ",", "，", "?", "？", "!", "！", ":", "：",
                 ";", "；", "、", "·", "…", "—", "-", "|", "~", "'",
                 '/', "\"", "“", "”", "(", "（", ")", "）"])
        return phoneme in lista

    def _islabel_mark(self, phoneme):
        lista = set(['<k>', '<p>', '<g>', '<t>', '<s>'])
        return phoneme in lista

    def _isprosody_mark(self, phoneme):
        lista = set(['#1', '#2', '#3', '#4', '$1', '$2', '$3', '$4'])
        return phoneme in lista

    def _parse_pho_tone_lang_prsd(self, phonemes):
        pho_ids, tone_ids, lang_ids, prsd_ids = list(), list(), list(), list()
        for i, phoneme in enumerate(phonemes):
            # prosody
            if self._isprosody_mark(phoneme):
                prsd_id = int(phoneme[-1])  # 1,2,3,4
                if len(prsd_ids) != 0:
                    prsd_ids[-1] = prsd_id

                # the prosody is not add into the sequence
                continue

            if not self.use_pause_label and self._islabel_mark(phoneme):
                continue

            # normal phoneme, and punctuation or human labeled pause in audio
            else:
                if phoneme[-2:].isdigit():
                    pho = phoneme[:-2]
                    tone_id = int(phoneme[-2:])
                elif phoneme[-1].isdigit():
                    pho = phoneme[:-1]
                    tone_id = int(phoneme[-1])
                else:
                    pho = phoneme
                    tone_id = 0

            pho_id = self.phoneme2id[pho]
            pho_ids.append(pho_id)
            tone_ids.append(tone_id)
            # check eng
            if tone_id == 14:
                lang_ids.append(1)
            else:
                lang_ids.append(0)
            prsd_ids.append(0)

        return pho_ids, tone_ids, lang_ids, prsd_ids

    def _detect_language(self, text, zh2en_ratio=1.0):
        '''
        :param text:
        :param zh2en_ratio:
        :return: 0: chinese, 1: english
        '''
        chinese = extract_mandarin_only(text)
        non_chinese = extract_non_mandarin(text)
        len_zh = len(chinese)   # chinese chars
        len_en = len(non_chinese.split(' '))  # english words
        if len_zh / len_en > zh2en_ratio:
            return 0
        else:
            return 1

    def _extract_prosody(self, phonemes, tones, langs):
        pho_ids, tone_ids, lang_ids, prsd_ids = list(), list(), list(), list()
        for i, pho in enumerate(phonemes):
            # prosody
            if self._isprosody_mark(pho):
                prsd_id = int(pho[-1])  # 1,2,3,4
                if len(prsd_ids) != 0:
                    prsd_ids[-1] = prsd_id

                # the prosody is not add into the sequence
                continue

            if not self.use_pause_label and self._islabel_mark(pho):
                continue

            # normal phoneme, and punctuation or human labeled pause in audio
            else:
                pho_id = self.phoneme2id[pho]
                pho_ids.append(pho_id)
                tone_ids.append(tones[i])
                lang_ids.append(langs[i])
                prsd_ids.append(0)

        return pho_ids, tone_ids, lang_ids, prsd_ids

def get_tokenizer(phoneme_dict="cosyvoice/tokenizer/assets/hnttsa_phoneme2id.json",
                  mode='train', use_pause_label=True):
    return PhonemeTokenizer(phoneme_dict, mode=mode, use_pause_label=use_pause_label)
