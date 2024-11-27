import os
import re
import json
import sys
TTS_root = "/data/megastore/Projects/DuJing/code/TTS"
sys.path.append(TTS_root)
from tts.utils.text_splitter import split_text

def dumps_json_pretty(data, indent=2, depth=2):
    assert depth > 0
    space = ' ' * indent
    s = json.dumps(data, ensure_ascii=False, indent=indent)
    lines = s.splitlines()
    N = len(lines)

    # determine which lines to be shortened
    def is_over_depth_line(i):
        return i in range(N) and lines[i].startswith(space * (depth + 1))

    def is_open_bracket_line(i):
        return not is_over_depth_line(i) and is_over_depth_line(i + 1)

    def is_close_bracket_line(i):
        return not is_over_depth_line(i) and is_over_depth_line(i - 1)

    def shorten_line(line_index):
        if not is_open_bracket_line(line_index):
            return lines[line_index]
        # shorten over-depth lines
        start = line_index
        end = start
        while not is_close_bracket_line(end):
            end += 1
        has_trailing_comma = lines[end][-1] == ','
        _lines = [lines[start][-1], *lines[start + 1:end],
                  lines[end].replace(',', '')]
        d = json.dumps(json.loads(' '.join(_lines)), ensure_ascii=False)
        return lines[line_index][:-1] + d + (',' if has_trailing_comma else '')

    s = '\n'.join([
        shorten_line(i)
        for i in range(N) if
        not is_over_depth_line(i) and not is_close_bracket_line(i)
    ])

    return s

def write_json_pretty(to_path, data, intent=4):
    with open(to_path, 'w', encoding='utf-8') as writer:
        if intent:
            # json.dump(data, writer, ensure_ascii=False, indent=4)
            writer.write(dumps_json_pretty(data, indent=4))
        else:
            json.dump(data, writer, ensure_ascii=False)


def process(base_path, sub_dirs, txt_path, language):
    temp_list = []
    tts_text = {}
    for dir_name in sub_dirs:
        data_name = os.path.basename(os.path.dirname(base_path))
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            # 查找第一个 wav 文件
            wav_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
            if wav_files:
                wav_file = wav_files[0]
                utt = os.path.splitext(wav_file)[0]
                utt = f"{data_name}_{dir_name}"
                wav_path = os.path.join(dir_path, wav_file)
                temp_list.append((utt,wav_path))

    with open(txt_path) as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            utt_i = i % len(temp_list)
            (utt, path) = temp_list[utt_i]
            if not utt in tts_text:
                tts_text[utt] = []
            texts = split_text(line, language=language)
            for text in texts:
                tts_text[utt].append(text)

    return temp_list, tts_text

if __name__ == "__main__":
    inputs = [
        ("/data/megastore/SHARE/TTS/VoiceClone/NTTS/Formatted_wavnorm", ["ID_7","ID_16","ID_18"], "zhongcao_script_20231206.txt", 'chinese'),
        ("/data/megastore/SHARE/TTS/VoiceClone/NTTS/Formatted_wavnorm", ["ID_19","ID_20","ID_24","ID_25","ID_28"], "hntts_分享_script_20240918.txt", 'chinese'),
        ("/data/megastore/SHARE/TTS/VoiceClone/NTTS/Formatted_wavnorm", ["ID_22", "ID_23"], "hntts_文旅_script_20240827.txt", 'chinese'),
        ("/data/megastore/SHARE/TTS/VoiceClone/ENTTS/Formatted_wavnorm", ["ID_2","ID_3"], "entts_talkshow_script_20240506.txt", 'english'),
        ("/data/megastore/SHARE/TTS/VoiceClone/ENTTS/Formatted_wavnorm", ["ID_7"], "entts_teach_script_20240708.txt", 'english')
    ]

    wav_list = []
    tts_text = {}
    for input in inputs:
        base_path, sub_dirs, text_file, language = input
        temp_list, temp_text = process(base_path, sub_dirs, text_file, language)
        wav_list += temp_list
        tts_text.update(temp_text)

    scp_file = "wav.scp"
    txt_file = "tts_text.json"

    with open(scp_file, 'w') as fout:
        for (utt, path) in wav_list:
            fout.write(f"{utt}\t{path}\n")
    write_json_pretty(txt_file, tts_text)




