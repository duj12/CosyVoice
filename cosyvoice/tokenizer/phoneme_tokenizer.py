import json

class PhonemeTokenizer:
    def __init__(self, phoneme_dict="assets/hnttsa_phoneme2id.json"):
        with open(phoneme_dict, 'r', encoding='utf-8') as fin:
            self.phoneme2id = json.load(fin)

    def encode(self, phoneme_list):
        return self._parse_pho_tone_lang_prsd(phoneme_list)

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


def get_tokenizer(phoneme_dict="assets/hnttsa_phoneme2id.json"):
    return PhonemeTokenizer(phoneme_dict)
