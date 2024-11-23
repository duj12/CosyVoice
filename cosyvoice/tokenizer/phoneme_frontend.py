# Copyright (c) 2024   Jing Du  (thuduj12@163.com)
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

import re
import copy
import os
import json

ENG_LAUGHTER_DIC = {'haa15': 'haa', 'hee15': 'hee', 'hnn15': 'hnn', 'hii15': 'hii', 'hyy15': 'hyy', 'hoo15': 'hoo'}
CUR_DIR = os.path.dirname(__file__)

laugh2pho = {v: k for k, v in ENG_LAUGHTER_DIC.items()}
with open('cosyvoice/tokenizer/assets/tanci.json', 'r', encoding='utf-8') as fin:
    interjection2pho = json.load(fin)

def get_token2phoneme_list(result, token_length):
    """
    由前端结果中的音素序列，以及音素对应到原始文本中的token位置，得到每个词对应的音素的列表
    :param result:
    :return:
    """
    phonemes_list = []
    cur_adx = 0
    one_phos = []
    pho_nums = 0  # token对应pho的数量
    token2phonums = []

    for apho, adx in zip(result['phonemes'], result['pho2token_ids']):
        if adx == cur_adx:
            one_phos.append(apho)
            pho_nums += 1    # token对应音素数量+1
        elif adx == cur_adx + 1:
            cur_adx += 1
            phonemes_list.append(one_phos.copy())
            token2phonums.append(pho_nums)

            # 下一个token对应的音素和数量
            one_phos = [apho]
            pho_nums = 1

        else:
            pass

        if adx == token_length - 1:  # 最后一个
            phonemes_list.append(one_phos.copy())
            token2phonums.append(pho_nums)
            one_phos = []
            pho_nums = 0

    return phonemes_list, token2phonums

def remove_word_boundary(result):
    words = result['words']
    symbols = result['symbols']
    prosody_with_er = result['prosody']  # 儿字单独作为一个占位符的韵律

    # 去掉分词结果, 处理儿化音
    chars = list()
    pinyins = list()
    prosody = list()
    prsd_idx = 0

    for i, (word, symbol) in enumerate(zip(words, symbols)):
        word = word.strip()
        symbol = symbol.split(' ')
        for j, s in enumerate(symbol):
            if s.startswith('@eng@'):  # 英文
                len_s = len(s.replace('@eng@', ''))
                chars.append(word[j:j + len_s])
                pinyins.append(s)
                prosody.append(prosody_with_er[prsd_idx])
            elif s.startswith('@punc@'):
                chars.append(word)
                pinyins.append(s)
                prosody.append(prosody_with_er[prsd_idx])
            elif word == '@#@':  # 注音的单词
                chars.append(word)
                pinyins.append(s)
                prosody.append(prosody_with_er[prsd_idx])
            else:  # 中文
                # 处理儿化，token需要把儿字并到前一个字
                if s == '@er@' and len(chars) != 0:
                    chars[-1] += word[j]
                    # 拼音占位符@er@不需要了
                    # er对应的韵律也不要了，直接儿化前面那个字的韵律，改成儿字对应韵律
                    prosody[-1] = prosody_with_er[prsd_idx]

                else:
                    chars.append(word[j])  # 中文, 不是儿化，正常处理
                    pinyins.append(s)
                    prosody.append(prosody_with_er[prsd_idx])

            prsd_idx += 1

    return chars, pinyins, prosody

def remove_kpsgt_tag(text):
    """
    去除<k>|<p>|<s>|<g>|<t>
    返回（去除后的文本，<k>|<p>|<s>|<g>|<t>在原始文本的索引）
    """
    pattern = re.compile(r'<k>|<p>|<s>|<g>|<t>')
    input_text = re.sub(pattern, '', text)

    text_wo_space = re.sub('\s', '', text)

    matches = [(match.group(), match.start(), match.end())
               for match in re.finditer(pattern, text_wo_space)]

    return input_text, matches

def restore_prosody_tag(chars, prosody_tag, prosody_kpsgt_idx):
    # 将<k><p><t><s><g>插回【韵律】中
    cnt = 0
    new_prosody = []
    assert len(chars) == len(prosody_tag)
    for char, prosody in zip(chars, prosody_tag):
        # 加入原本的韵律
        if prosody in ['#0', '#1', '#2', '#3', '#4', ]:
            prosody = prosody[1:]
        new_prosody.append(prosody)

        cnt += len(char)

        # 备注：先判断是否是插入的<p>， 对应韵律先加入
        while prosody_kpsgt_idx and cnt == prosody_kpsgt_idx[0][1]:
            if prosody_kpsgt_idx[0][0] in ['<k>', '<p>']:
                new_prosody.append('2')
            elif prosody_kpsgt_idx[0][0] in ['<t>']:
                if len(new_prosody) > 0:
                    new_prosody.append(new_prosody[-1])
                else:
                    new_prosody.append('1')
            else:
                new_prosody.append('1')
            cnt += len(prosody_kpsgt_idx[0][0])
            prosody_kpsgt_idx.pop(0)

    return new_prosody

def insert_kpsgt_into_tokens(chars, pinyins, phoneme_list, kpsgt_idx):
    cnt = 0
    new_chars = []
    new_pinyins = []
    new_phoneme_list = []
    for i, (char, pinyin, phones) in enumerate(
            zip(chars, pinyins, phoneme_list)):
        # 先加入当前的
        new_chars.append(char)
        new_pinyins.append(pinyin)
        new_phoneme_list.append(phones)

        cnt += len(char)
        while kpsgt_idx and cnt == kpsgt_idx[0][1]:
            new_chars.append(kpsgt_idx[0][0])
            new_pinyins.append(kpsgt_idx[0][0])
            new_phoneme_list.append([kpsgt_idx[0][0]])  # 这里要加入列表
            cnt += len(kpsgt_idx[0][0])
            kpsgt_idx.pop(0)

    return new_chars, new_pinyins, new_phoneme_list

def insert_kpsgt_into_phonemes(pho, tone, lang, phoneme_list, token_prsd):
    # 有人工标注的<kpsgt>等，把这个标记也插入到音素序列中, 对于<kpsgt>的韵律，也需要插入到序列中
    new_phonemes = []
    new_tones = []
    new_languages = []

    p_idx = 0
    for t_idx, word_phoneme in enumerate(phoneme_list):
        for p in word_phoneme:
            while (pho[p_idx].startswith('#')):  # 原来音素序列中有的韵律
                new_phonemes.append(pho[p_idx])
                new_tones.append(tone[p_idx])
                new_languages.append(lang[p_idx])
                p_idx += 1
            if not p.startswith('<'):
                assert p == pho[p_idx]
                new_phonemes.append(pho[p_idx])
                new_tones.append(tone[p_idx])
                new_languages.append(lang[p_idx])
                p_idx += 1
            else:  # 遇到<kptsg>, 除了将自身加入，还需要将韵律也加入
                new_phonemes.append(p)
                new_tones.append(0)
                new_languages.append(0)
                # 当前韵律不为0
                prsd = token_prsd[t_idx]
                if int(prsd) != 0:
                    new_phonemes.append(f"#{prsd}")
                    new_tones.append(0)
                    new_languages.append(0)

    return new_phonemes, new_tones, new_languages

def get_frontend_result(text, text_frontend_model):
    input_text, kpsgt_idx = remove_kpsgt_tag(text)
    prosody_kpsgt_idx = copy.deepcopy(kpsgt_idx)
    # 送入前端的文本，需要把<>这些特殊符号去掉
    result = text_frontend_model.get_frontend_outputs(input_text)
    kpsgt_num = len(kpsgt_idx)

    # 去掉分词结果，把文本拆分成token，和对应的拼音以及韵律
    chars, pinyins, prosody = remove_word_boundary(result)
    assert(len(chars) == len(pinyins) == len(prosody)), \
        f"Frontend result {result}. " \
        f"len(chars):{len(chars)}, len(phos):{len(pinyins)}, len(prosody):{len(prosody)}"
    # 得到每个token对应的phoneme及其数量
    phoneme_list, token2phonum = get_token2phoneme_list(result, len(chars))

    # 替換笑声
    for j, (char, pho) in enumerate(zip(chars, pinyins)):
        if char.lower() in laugh2pho:
            pinyins[j] = laugh2pho[char.lower()]

    # 替换叹词
    for j, (char, pho) in enumerate(zip(chars, pinyins)):
        if char.lower() in interjection2pho:
            pinyins[j] = interjection2pho[char.lower()]

    if kpsgt_num > 0:
        # 将<k><p><t><s><g>插回【韵律】中, 需要用到原始不带kptsg的chars
        prosody = restore_prosody_tag(chars, prosody, prosody_kpsgt_idx)  # 插入kptsg的韵律同时去掉了韵律前面的#号
        prosody = [int(p) for p in prosody]

        # 将<k><p><t><s><g>插回【拼音】中
        chars, pinyins, phoneme_list = insert_kpsgt_into_tokens(chars, pinyins, phoneme_list, kpsgt_idx)

        assert(len(prosody_kpsgt_idx)==len(kpsgt_idx)==0), f"{len(prosody_kpsgt_idx)}!=0, {len(kpsgt_idx)}!=0, {result}"
    else:
        prosody = [int(p[1:]) for p in prosody]  # 去掉韵律前面的#号,转成int

    # 将<p>等token的phoneme数量置为0
    new_token2phonum = [len(p) if not p[0].startswith("<") else 0 for p in phoneme_list]
    result['ori_text'] = text                   # 原本的文本，没有去掉<>
    result['text_token'] = chars                # 划分后的每个字
    result['pinyins'] = pinyins                 # 每个字对应的拼音
    result['phoneme_list'] = phoneme_list       # 每个字对应的音素
    result['token_prsd'] = prosody              # 每个字对应的韵律
    result['token2phonum'] = new_token2phonum   # 每个字对应的音素数量

    token2phonums_w_prsd = []
    for i, n in enumerate(new_token2phonum):
        cur_pho_prsd = int(prosody[i])
        token2phonums_w_prsd.append(n)  # 正常的token，直接把之前计算的音素数量添加进去
        if cur_pho_prsd != 0:
            token2phonums_w_prsd.append(0)  # 韵律token, phonum使用0占位符表示
    result['token2phonum_w_prsd'] = token2phonums_w_prsd

    pho = result['phonemes']
    tone = result['tones']
    lang = result['language_ids']
    if kpsgt_num > 0:  # 有人工标注的<kpsgt>等，把这个标记也插入到音素序列中
        new_pho, new_tone, new_lang = insert_kpsgt_into_phonemes(pho, tone, lang, phoneme_list, prosody)
        result['pho'] = new_pho
        result['tone'] = new_tone
        result['lang'] = new_lang
    else:
        result['pho'] = pho
        result['tone'] = tone
        result['lang'] = lang

    total_pho_num = 0
    for n in token2phonums_w_prsd:
        if n==0:
            total_pho_num += 1  # 是韵律或者是<>
        else:
            total_pho_num += n
    true_pho_num = len(result['pho'])
    assert total_pho_num == true_pho_num, f"Frontend result {result}. "

    return result
