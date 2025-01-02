# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#                    Jing Du (thuduj12@163.com)
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
import os.path

import logging
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from librosa.filters import mel as librosa_mel_fn

torchaudio.set_audio_backend('soundfile')


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        # sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))
        if not os.path.exists(sample['wav']):
            logging.warning(f"wav path {sample['wav']} does not exist, this utt is jumped.")
            continue
        sample['speech'], sample['sample_rate'] = torchaudio.load(sample['wav'])    # load from wav_path
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)

        if mode == 'inference':  # do not filter any utt when we inference
            yield sample

        else:
            # sample['wav'] is torch.Tensor, we have 100 frames every second
            num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
            if num_frames < min_length:
                logging.warning(f"{sample['wav']} less than {min_length} frame(1s=100f)")
                continue
            if num_frames > max_length:
                logging.warning(f"{sample['wav']} more than {max_length} frame(1s=100f)")
                continue

            if 'text_token' in sample:
                if len(sample['text_token']) < token_min_length:
                    logging.warning(f"utt: {sample['utt']}, text: {sample['text']} less than {token_min_length}")
                    continue
                if len(sample['text_token']) > token_max_length:
                    logging.warning(f"utt: {sample['utt']}, text: {sample['text']} more than {token_max_length}")
                    continue

                if num_frames != 0:
                    if len(sample['text_token']) / num_frames < min_output_input_ratio:
                        logging.warning(f"utt: {sample['utt']}, text to audio frame ratio less than {min_output_input_ratio}")
                        continue
                    if len(sample['text_token']) / num_frames > max_output_input_ratio:
                        logging.warning(f"utt: {sample['utt']}, text to audio frame ratio more than {max_output_input_ratio}")
                        continue
            yield sample

def transcribe(data, get_transcriber, mode='inference'):
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample

        if 'text' not in sample:
            transcriber = get_transcriber()
            transribe_sr = 16000
            speech = sample['speech']
            if sample['sample_rate'] != transribe_sr:
                speech = torchaudio.transforms.Resample(
                    orig_freq=sample['sample_rate'], new_freq=transribe_sr)(speech)
            input = speech[0]
            sample['text'] = transcriber.transcribe(speech_or_path=input)
            logging.info(f"prompt text: {sample['text']}")

        yield sample


def resample(data, resample_rate=24000, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                logging.warning(f"audio sample_rate {sample['sample_rate']} less than {min_sample_rate}, the clip is droped")
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            if mode == 'train':
                waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
        sample['speech'] = waveform
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample['speech_feat'] = mat
        yield sample


def compute_f0(data, pitch_extractor, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        waveform = sample['speech']
        mat = pitch_extractor(waveform).transpose(1, 2)
        mat = F.interpolate(mat, size=sample['speech_feat'].shape[0], mode='linear')
        sample['pitch_feat'] = mat[0, 0]
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special='all', mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample

def tokenize_phoneme(data, get_tokenizer, mode='train'):
    """ Decode text to phoneme
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer(mode=mode)

    for sample in data:
        if mode == 'train':
            assert 'text' in sample
        if 'text' in sample:
            try:
                pho_ids, tone_ids, lang_ids, prsd_ids = tokenizer.encode(sample['text'])
            except Exception as e:
                logging.error(e)
                continue

            if not (len(prsd_ids)==len(tone_ids)==len(lang_ids)==len(prsd_ids)):
                logging.warning(f"{sample['utt']}, {sample['wav']} phoneme error.")
                continue

            sample['text_token'] = pho_ids
            sample['text_tone'] = tone_ids
            sample['text_lang'] = lang_ids
            sample['text_prsd'] = prsd_ids

        if mode == 'inference':
            pho_ids1, tone_ids1, lang_ids1, prsd_ids1 = tokenizer.encode(sample['tts_text'])
            sample['tts_text_token'] = pho_ids1
            sample['tts_text_tone'] = tone_ids1
            sample['tts_text_lang'] = lang_ids1
            sample['tts_text_prsd'] = prsd_ids1
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0), reverse=True)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0), reverse=True)
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', gan=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech'].size(0) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        speech = pad_sequence(speech, batch_first=True, padding_value=0)
        # speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        # speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        # speech_token = pad_sequence(speech_token,
        #                             batch_first=True,
        #                             padding_value=0)
        # utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        # spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)

        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            # "speech_token": speech_token,
            # "speech_token_len": speech_token_len,
            # "speech_feat": speech_feat,
            # "speech_feat_len": speech_feat_len,
            # "utt_embedding": utt_embedding,
            # "spk_embedding": spk_embedding,
        }
        if 'speech_feat' in sample[0]:
            speech_feat = [sample[i]['speech_feat'] for i in order]
            speech_feat_len = torch.tensor([i.size(0) for i in speech_feat],
                                           dtype=torch.int32)
            speech_feat = pad_sequence(speech_feat,
                                       batch_first=True,
                                       padding_value=0)
            batch.update({
                "speech_feat": speech_feat,
                "speech_feat_len": speech_feat_len,
            })

        if 'text_token' in sample[0]:
            text = [sample[i]['text'] for i in order]
            text_token = [torch.tensor(sample[i]['text_token']) for i in order]
            text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
            text_token = pad_sequence(text_token, batch_first=True, padding_value=0)

            batch.update({
                "text": text,
                "text_token": text_token,
                "text_token_len": text_token_len,
            })


        if 'text_tone' in sample[0]:
            text_tone = [torch.tensor(sample[i]['text_tone']) for i in order]
            text_tone = pad_sequence(text_tone, batch_first=True, padding_value=0)
            text_lang = [torch.tensor(sample[i]['text_lang']) for i in order]
            text_lang = pad_sequence(text_lang, batch_first=True, padding_value=0)
            text_prsd = [torch.tensor(sample[i]['text_prsd']) for i in order]
            text_prsd = pad_sequence(text_prsd, batch_first=True, padding_value=0)

            batch['text_token'] = torch.cat([text_token.unsqueeze(-1),
                                             text_tone.unsqueeze(-1),
                                             text_lang.unsqueeze(-1),
                                             text_prsd.unsqueeze(-1)], dim=-1)

        if gan is True:
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        # else:
        #     # only gan train needs speech, delete it to save memory
        #     del batch["speech"]
        #     del batch["speech_len"]
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})

            if 'tts_text_tone' in sample[0]:
                text_tone = [torch.tensor(sample[i]['tts_text_tone']) for i in
                             order]
                text_tone = pad_sequence(text_tone, batch_first=True,
                                         padding_value=0)
                text_lang = [torch.tensor(sample[i]['tts_text_lang']) for i in
                             order]
                text_lang = pad_sequence(text_lang, batch_first=True,
                                         padding_value=0)
                text_prsd = [torch.tensor(sample[i]['tts_text_prsd']) for i in
                             order]
                text_prsd = pad_sequence(text_prsd, batch_first=True,
                                         padding_value=0)

                batch['tts_text_token'] = torch.cat([tts_text_token.unsqueeze(-1),
                                                 text_tone.unsqueeze(-1),
                                                 text_lang.unsqueeze(-1),
                                                 text_prsd.unsqueeze(-1)],
                                                dim=-1)
        # if use_spk_embedding is True:
        #     batch["embedding"] = batch["spk_embedding"]
        # else:
        #     batch["embedding"] = batch["utt_embedding"]
        yield batch
