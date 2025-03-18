import argparse
import logging
import os
import json
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from cosyvoice.speaker.speaker_encoder import SpeakerEmbedding


def get_args():
    parser = argparse.ArgumentParser(
        description='Extract speaker vectors offline.')
    parser.add_argument('--spkemb_ckpt',
                        default="../../../pretrained_models/speaker_encoder_v2.pt",
                        help='checkpoint model')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data_json', type=list, default=[
        ["/data/megastore/SHARE/TTS/VoiceClone/NTTS/NTTS.ID36+37.10min.json", 0, 1],
        ["/data/megastore/SHARE/TTS/VoiceClone3/MengNiu/MengNiu.vits.json", 0, 1],
        ["/data/megastore/SHARE/TTS/VoiceClone3/ENTTS_VC1/ENTTS_VC1.ID12.json", 1, 1],
        ["/data/megastore/SHARE/TTS/VoiceClone3/ENTTS_VC3/ENTTS_VC3.ID88.json", 1, 1]],
                        help='the json file path')
    parser.add_argument('--save_path', type=str,
                        default="../../../pretrained_models/fewshot_spk_vec.pt",
                        help='the vector save path')
    parser.add_argument('--data_scp', type=str, default="",
                        help='the wav.scp path')

    args = parser.parse_args()
    return args


def get_wav_path(json_file_list):
    spk2wavpath = {}
    for json_file in json_file_list:
        logging.info(f"Loading data: {json_file}")
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
            if speaker_folder not in spk2wavpath:
                spk2wavpath[speaker_folder] = []
            total_dur, file_list = info
            for fname, dur, sequence in file_list:
                pho = sequence['text']
                wav_path = os.path.join(wave_dir, speaker_folder,
                                        '{}.wav'.format(fname))
                spk2wavpath[speaker_folder].append(wav_path)

    return spk2wavpath


def get_spkemb(wav_path_list, spkemb_model):
    wav_list = []
    wav_len_list = []
    for wav_path in wav_path_list:
        wav, fs = torchaudio.load(wav_path)
        target_sample_rate = spkemb_model.sampling_rate
        if fs != target_sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=fs,
                                   new_freq=target_sample_rate)
            wav = resampler(wav)
            fs = target_sample_rate
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        wav_list.append(wav)
        wav_len_list.append(len(wav))

    spk_wave = pad_sequence(wav_list, batch_first=True, padding_value=0.0)
    spk_wave = spk_wave.cuda()
    spk_wave_len = torch.tensor(wav_len_list).cuda()

    with torch.no_grad():
        spkemb = spkemb_model(spk_wave.unsqueeze(1), spk_wave_len)

    return spkemb


def batch_iterator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


if __name__ == "__main__":
    args = get_args()
    spkemb_model = SpeakerEmbedding(ckpt_path=args.spkemb_ckpt).cuda()
    spk2wavpath = get_wav_path(args.data_json)

    speaker_vectors = {}
    speaker_utts = {}
    for spk in spk2wavpath:
        wav_path_list = spk2wavpath[spk]
        for batch in batch_iterator(wav_path_list, args.batch_size):
            batch_emb = get_spkemb(batch, spkemb_model)
            if spk not in speaker_vectors:
                speaker_vectors[spk] = torch.sum(batch_emb, dim=0)
            else:
                speaker_vectors[spk] += torch.sum(batch_emb, dim=0)
            if spk not in speaker_utts:
                speaker_utts[spk] = len(batch)
            else:
                speaker_utts[spk] += len(batch)

    avg_spk_vec = {}
    for spk in speaker_vectors:
        avg_spk_vec[spk] = speaker_vectors[spk] / speaker_utts[spk]

    torch.save(avg_spk_vec, args.save_path)
