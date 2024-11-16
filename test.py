import torchaudio
import os
import itertools
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

# export PYTHONPATH=third_party/Matcha-TTS
current_path = os.environ.get('PYTHONPATH', '')
Matcha_path = 'third_party/Matcha-TTS'
if Matcha_path not in current_path:
    os.environ['PYTHONPATH'] = Matcha_path + os.pathsep + current_path


sensevoice = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="/data/megastore/Projects/DuJing/code/SenseVoice/model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)


# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# # sft usage
# print(cosyvoice.list_avaliable_spks())
# output = cosyvoice.inference_sft('那有哪些美剧是不太适合学英语的呢？我来给大家举几个例子吧。第一个《破产姐妹》，我不知道为什么总有人推荐这一部，我先声明一下，我真的很喜欢很喜欢破产姐妹，它真的很下饭。我大学有一段时间就是天天去食堂打包吃的，然后回到宿舍，我就边看边吃，甚至听到她那个片头曲，我就会很有食欲，但是我真的真的没有办法用它来学英语。一个是语速太快了；第二全是开车的台词，你说是生活中、考试试中哪儿会用到？所以我觉得破产姐妹下饭必备，学英语还是算了。', '中文女')
# torchaudio.save('sft.wav', output['tts_speech'], 24000)
# output = cosyvoice.inference_sft('The problems of efficiency today are less drastic but more chronic, they can also prolong the evils that they were intended to solve and took the electronic medical record. It seemed to be the answer to the problem of doctors handwriting, and it had the benefit of providing much better data for treatments. In practice, it has meant much more electronic paperwork and physicians are now complaining that they have less rather than more time to see patients individually. The obsession with efficiency can actually make us less efficient.', '英文男')
# torchaudio.save('sft-en.wav', output['tts_speech'], 24000)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')
# zero_shot usage
ref_audios = [
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_3_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_5_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_6_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_8_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_11_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_14_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_18_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_20_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/CN_21_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/EN_4_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/EN_15_30s.wav',
    # '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/test/EN_22_30s.wav',

    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/jim_10s.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/caikangyong_10s.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/ASR_lingli_10s.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/ASR_lola_10s.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/jyr/ASR_jiangyueren_10s.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/dubbing.wav',
    '/data/megastore/Projects/DuJing/code/TTS/tts/ref_audios/cute2.wav',
]

texts = [
    ('zh','那有哪些美剧是不太适合学英语的呢？我来给大家举几个例子吧。第一个《破产姐妹》，我不知道为什么总有人推荐这一部，我先声明一下，我真的很喜欢很喜欢破产姐妹，它真的很下饭。我大学有一段时间就是天天去食堂打包吃的，然后回到宿舍，我就边看边吃，甚至听到她那个片头曲，我就会很有食欲，但是我真的真的没有办法用它来学英语。一个是语速太快了；第二全是开车的台词，你说是生活中、考试试中哪儿会用到？所以我觉得破产姐妹下饭必备，学英语还是算了。'),
    ('en','The problems of efficiency today are less drastic but more chronic, they can also prolong the evils that they were intended to solve and took the electronic medical record. It seemed to be the answer to the problem of doctors handwriting, and it had the benefit of providing much better data for treatments. In practice, it has meant much more electronic paperwork and physicians are now complaining that they have less rather than more time to see patients individually. The obsession with efficiency can actually make us less efficient.'),
]

for ref_audio in ref_audios:
    print(f'Cloning the reference audio: {ref_audio}')
    id = os.path.basename(ref_audio)
    id = os.path.splitext(id)[0]
    prompt_speech_16k = load_wav(ref_audio, 16000)
    print(prompt_speech_16k.size())
    if prompt_speech_16k.size(1) > 160000:
        prompt_speech_16k = prompt_speech_16k[:, :160000]

    # en
    res = sensevoice.generate(
        input=prompt_speech_16k[0],
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    prompt_text = rich_transcription_postprocess(res[0]["text"])
    print("prompt_text:", prompt_text)

    for (lang,long_text) in texts:
        if lang == 'zh':
            text_list = long_text.split("。")
        elif lang == 'en':
            text_list = long_text.split(".")
        else:
            raise NotImplementedError
        total_audio = []
        for text in text_list:
            text = text.strip()
            if len(text)<1:
                continue
            if lang == 'zh':
                #text = "<|zh|>" + text
                text+="。"
            elif lang == 'en':
                #text = "<|en|>" + text
                text+="."

            for i,j in enumerate(cosyvoice.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_speech_16k,
                    stream=False)):

                torchaudio.save(f'{id}_{text[0:10]}_{i}.wav', j['tts_speech'], 24000)



# # cross_lingual usage
# prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
# output = cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k)
# torchaudio.save('cross_lingual.wav', output['tts_speech'], 24000)
#
# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# # instruct usage
# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.')
# torchaudio.save('instruct.wav', output['tts_speech'], 24000)