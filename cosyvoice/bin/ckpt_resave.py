import librosa
import logging
import torch
import torchaudio
import s3tokenizer
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils.rnn import pad_sequence
import cosyvoice.loralib as lora
from cosyvoice.speaker.speaker_encoder import SpeakerEmbedding

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

ref_audios0 = [
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC1/EN_Altman_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC1/EN_Musk_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC1/EN_Zuckerberg_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_DQQHJJ_F_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_GSGJJ_F_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_QSFX_M_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_F_13_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_F_19_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_F_20_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_M_1_reference.wav',

    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_M_7_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_M_8_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WGFG_M_13_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_WRFX_F_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_ZGLJS_F_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_ZGLJS_M_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_ZGLJS_M_2_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC2/EN_ZSZYJJ_M_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_CWQHJJ_F_2_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_CWQHJJ_M_1_reference.wav',

    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_GSGJJ_M_1_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_F_7_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_F_8_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_F_14_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_F_16_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_F_21_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_M_9_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_M_14_reference.wav',
    '/data/megastore/SHARE/TTS/ref_audios/EN_VC3/EN_WGFG_M_19_reference.wav',
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_CWQHJJ_M_2_reference.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_CWQHJJ_M_3_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_CWQHJJ_M_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_CWQHJJ_M_5_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_DLDJS_M_1_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_QSFX_F_1_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/ID35.wav",
    # "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_QSFX_F_2_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_QSFX_F_3_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_QSFX_F_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_QSFX_M_2_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_F_10_reference.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_F_12_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_F_18_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_F_23_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_F_9_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_M_10_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_M_11_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_M_15_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_M_16_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_WGFG_M_17_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_F_2_reference.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_F_3_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_F_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_F_5_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_M_3_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_M_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_M_5_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZGLJS_M_6_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_F_1_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_F_2_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_F_3_reference.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_F_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_F_5_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_2_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_3_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_4_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_5_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_6_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_7_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_8_reference.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC4/EN_ZSZYJJ_M_9_reference.wav",
    # 70个
]
ref_audios1 = [
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_DLDJS_F_1.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_DLDJS_F_2.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_DLDJS_F_3.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_QSFX_F_5.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_QSFX_F_6.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_QSFX_M_3.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZGLJS_F_6.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZGLJS_F_7.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZGLJS_F_8.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_6.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_7.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_8.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_9.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_10.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC5/EN_ZSZYJJ_F_11.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_CWQHJJ_M_6.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_CWQHJJ_M_7.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_DLDJS_F_4.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_HLFX_F_1.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_7.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_8.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_9.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_10.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_11.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_12.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_13.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_F_14.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_M_4.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_M_5.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_M_6.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_M_7.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_QSFX_M_8.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_WGFG_M_18.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_WGFG_M_20.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_WRFX_F_2.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_WRFX_F_3.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZGLJS_F_9.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZGLJS_M_7.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZGLJS_M_8.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_12.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_13.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_14.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_15.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_16.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_17.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_18.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_19.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_20.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_F_21.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC6/EN_ZSZYJJ_M_10.wav",
    # 120个
]
ref_audios2 = [
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_CWQHJJ_F_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_CWQHJJ_F_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_DLDJS_F_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_GLDJS_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_10_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_11_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_12_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_13_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_14_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_16_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_2_JYJ.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_3_JYJ.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_F_5_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_1_JYJ.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_4_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_5_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_HLFX_M_6_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_11_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_12_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_1_JYJ.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_5_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_7_WKD.wav",
    # "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_F_8_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/ID146.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_QSFX_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_WGFG_M_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_WGFG_M_4_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_WRFX_F_11_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_WRFX_F_6_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_WRFX_F_8_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZGLJS_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZSZYJJ_F_1_VC.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZSZYJJ_M_1_VC.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZSZYJJ_M_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZSZYJJ_M_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZSZYJJ_M_4_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC7/EN_ZYZYJJ_M_5_XP.wav",
    # 160个
]
ref_audios3 = [
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_DQQHJJ_F_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_GSGJJ_F_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_GSGJJ_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_18_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_19_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_23_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_24_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_25_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_26_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_27_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_28_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_29_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_30_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_31_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_32_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_33_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_34_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_35_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_36_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_37_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_F_38_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_12_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_13_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_14_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_15_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_16_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_18_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_19_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_7_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_HLFX_M_8_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_F_14_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_F_15_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_F_16_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_F_18_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_F_19_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_M_3_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_M_5_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_QSFX_M_6_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_WRFX_M_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC8/EN_ZBFG_M_1_XP.wav",
    # 200个
]
ref_audios4 = [
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_CWQHJJ_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_CWQHJJ_M_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_GSGJJ_F_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_20_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_21_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_22_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_39_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_40_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_41_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_42_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_43_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_45_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_46_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_48_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_49_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_50_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_51_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_52_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_53_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_54_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_56_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_57_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_58_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_59_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_60_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_61_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_62_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_63_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_F_64_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_17_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_20_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_21_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_22_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_23_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_24_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_25_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_26_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_27_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_28_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_30_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_31_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_32_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_33_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_34_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_35_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_37_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_38_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_39_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_3_XST.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_40_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_42_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_43_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_44_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_46_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_HLFX_M_6_XST.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_QSFX_F_20_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_QSFX_F_21_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_QSFX_M_7_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_WGFG_M_1_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_WRFX_M_3_XP.wav",

    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_ZBFG_M_2_XP.wav",
    "/data/megastore/SHARE/TTS/ref_audios/EN_VC9/EN_ZBFG_M_3_XP.wav",
    # 262个
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/Scarlett_ID262_HER_00027.wav', 262),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/Scarlett_ID263_MS_00008.wav', 263),
]
ref_audios5 = [
    # ('/data/megastore/SHARE/TTS/ref_audios/CN_VC/CN_VC_ID_0_ref.wav', 1000),
    # ('/data/megastore/SHARE/TTS/ref_audios/CN_VC/财联社傅老师_参考音频_final.wav', 1001),
    # ('/data/megastore/SHARE/TTS/ref_audios/CN_VC/蒙牛_参考音频.wav', 1002),
    ('/data/megastore/SHARE/TTS/ref_audios/zeroshot/沈晓明zeroshot参考音频_10s.WAV', 1006),
]

ref_audios_zs = ref_audios0+ref_audios1+ref_audios2+ref_audios3+ref_audios4+ref_audios5

fs_ref_audios0 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/Scarlett_ID262_HER_00027.wav', 262),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/Scarlett_ID263_MS_00008.wav', 263),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ENVCM_ID1_Musk_reference.wav', 1),
]
fs_ref_audios1 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCF_ID3_FX_reference.wav', 3),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCF_ID4_FX_reference.wav', 4),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCF_ID7_FX_reference.wav', 7),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCM_ID5_FX_reference.wav', 5),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCF_ID18_LS_reference.wav', 18),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCM_ID10_FX_reference.wav', 10),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCM_ID12_ZB_reference.wav', 12),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc1/ENVCM_ID15_LS_reference.wav', 15),
]
fs_ref_audios2 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCF_ID40_FX_reference.wav', 40),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCF_ID39_ZB_reference.wav', 39),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCF_ID80_YJ_reference.wav', 80),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID19_FX_reference.wav', 19),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID26_ZB_reference.wav', 26),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID28_FX_reference.wav', 28),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID38_FX_reference.wav', 38),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID46_FX_reference.wav', 46),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID48_FX_reference.wav', 48),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc2/ENVCM_ID62_YJ_reference.wav', 62),
]
fs_ref_audios3 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID51_LS_reference.wav', 51),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID52_LS_reference.wav', 52),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID59_YJ1_reference.wav', 59),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID78_LS_reference.wav', 78),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID79_FX_reference.wav', 79),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID84_YJ1_reference.wav', 84),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID88_FX_reference.wav', 88),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID92_FX_reference.wav', 92),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID96_FX_reference.wav', 96),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID104_FX_reference.wav', 104),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID106_LS_reference.wav', 106),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID133_FX_reference.wav', 133),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCF_ID134_FX_reference.wav', 134),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID32_YJ_reference.wav', 32),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID64_YJ_reference.wav', 64),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID68_YJ_reference.wav', 68),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID85_YJ_reference.wav', 85),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID100_ZB_reference.wav', 100),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID102_FX_reference.wav', 102),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc3/ENVCM_ID108_LS_reference.wav', 108),

]
fs_ref_audios4 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID143_FX_reference.wav', 143),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID145_FX_reference.wav', 145),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID152_YJ_reference.wav', 152),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID161_FX_reference.wav', 161),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID166_FX_reference.wav', 166),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCF_ID168_FX_reference.wav', 168),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCM_ID136_FX_reference.wav', 136),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCM_ID148_FX_reference.wav', 148),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCM_ID149_YJ_reference.wav', 149),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc4/ENVCM_ID153_LS_reference.wav', 153),
]
fs_ref_audios5 = [
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc5/ENVCF_ID177_FX_reference.wav', 177),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc5/ENVCF_ID190_FX_reference.wav', 190),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc5/ENVCF_ID203_FX_reference.wav', 203),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc5/ENVCM_ID189_FX_reference.wav', 189),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/ttsvc5/ENVCM_ID199_ZB_reference.wav', 199),
]
fs_ref_audios6 = [
    #('/data/megastore/SHARE/TTS/ref_audios/CN_VC/蒙牛_参考音频.wav', 1003),
    #('/data/megastore/SHARE/TTS/ref_audios/fewshot/CNVCF_MNDZ_0126.wav', 1003),
    #('/data/megastore/SHARE/TTS/ref_audios/fewshot/CNVCF_MNDZ_0137.wav', 1003),
    # ('/data/megastore/SHARE/TTS/ref_audios/fewshot/CNVCF_DWXQ_0024.wav', 1004),
    # ('/data/megastore/SHARE/TTS/ref_audios/fewshot/CNVCM_GDTY_0022.wav', 1005),
    ('/data/megastore/SHARE/TTS/ref_audios/fewshot/CNVCF_MFLJ_0060.wav', 1007),
]

fs_ref_audios7 = [
    ("/data/megastore/SHARE/TTS/ref_audios/zeroshot/CNF15_GS_0004.wav", 1000, 'ID_37'),
    ("/data/megastore/SHARE/TTS/ref_audios/zeroshot/CNM15_YJ_0006.wav", 1001, 'ID_36'),
    ("/data/megastore/SHARE/TTS/ref_audios/zeroshot/CNVCF_MNDZ_0005.wav", 1002, 'mengniu_0'),
    ("/data/megastore/SHARE/TTS/ref_audios/zeroshot/ENVCF_ID88_FX_reference.wav", 0, 'ENVCF_ID88_FX'),
    ("/data/megastore/SHARE/TTS/ref_audios/zeroshot/ENVCM_ID12_ZB_reference.wav", 1, 'ENVCM_ID12_ZB'),

]



def init_model(config, ckpt_llm, ckpt_flow, ckpt_se, ckpt_codec,
               qwen_root="/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/acoustics/qwen/CosyVoice-BlankEN"):
    logger.info(f"load llm config from {config}")
    with open(config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': qwen_root,
        })
    llm = configs['llm']

    use_lora = configs.get("llm_use_lora", False)
    configs['lora_r'] = configs.get('llm_lora_r', 16)
    configs['lora_alpha'] = configs.get('llm_lora_alpha', 16)
    if use_lora:
        llm = lora.replace_specific_layer_4lora(llm, configs)
        lora.getModelSize_lora(llm)
        # lora.mark_only_infer_base_model(llm)  # 初始化后默认只推理基座模型参数

    logger.info(f"load llm from {ckpt_llm}")
    checkpoint = torch.load(ckpt_llm, map_location='cuda')
    llm.load_state_dict(checkpoint, strict=False)
    llm.eval()

    flow = configs['flow']
    use_lora = configs.get("flow_use_lora", False)
    configs['lora_r'] = configs.get('flow_lora_r', 16)
    configs['lora_alpha'] = configs.get('flow_lora_alpha', 16)
    if use_lora:
        flow = lora.replace_specific_layer_4lora(flow, configs)
        lora.getModelSize_lora(flow)
        # lora.mark_only_infer_base_model(flow)  # 初始化后默认只推理基座模型参数

    logger.info(f"load flow from {ckpt_flow}")
    checkpoint = torch.load(ckpt_flow, map_location='cuda')
    flow.load_state_dict(checkpoint, strict=False)
    flow.eval()

    mel_fn = configs['compute_fbank']
    spkemb_model = SpeakerEmbedding(ckpt_path=ckpt_se).cuda()
    spkemb_model.eval()

    codec_model = s3tokenizer.load_model("speech_tokenizer_v2_25hz", ckpt_codec).cuda()
    codec_model.eval()

    return configs, llm, flow, spkemb_model, codec_model, mel_fn


def wav2token(waves_padded, codec_model, sr_in=24000, wave_lengths=None):
    '''
        waves_padded: B T
    '''
    if not wave_lengths:
        wave_lengths = torch.LongTensor(1).to(waves_padded.device)
        wave_lengths[0] = waves_padded.size(-1)
    mels = []
    batch_size = waves_padded.size(0)
    for i in range(batch_size):
        audio = waves_padded[i, :wave_lengths[i]]
        # whisper speech code use 16k sample_rate
        if sr_in != 16000:
            resampler = torchaudio.transforms.Resample(
                sr_in, 16000).to(audio.device)
            audio = resampler(audio)
        mels.append(s3tokenizer.log_mel_spectrogram(audio))
    mels, mels_lens = s3tokenizer.padding(mels)
    mels = mels.to(waves_padded.device)
    mels_lens = mels_lens.to(waves_padded.device)
    speech_code, speech_code_len = codec_model.quantize(
        mels, mels_lens)
    return speech_code, speech_code_len


def wav2melfeat(waves_padded,compute_fbank_func, sr_in=24000):
    '''
        waves_padded: B T
    '''
    prompt_speech_feat = compute_fbank_func(
        [{
            'sample_rate': sr_in,
            'speech': waves_padded,
            'utt': None
        }]
    )
    prompt_speech_feat = list(prompt_speech_feat)[0]["speech_feat"].to(
        waves_padded.device)
    if prompt_speech_feat.ndim <= 2:
        prompt_speech_feat = prompt_speech_feat.unsqueeze(0)
    return prompt_speech_feat
# 注意下面这个方法得到的向量和工程代码不匹配，会出现问题。
def get_spkemb(wav_path, spkemb_model):
    logger.info(f"extract speaker vector of {wav_path}")
    wav, fs = torchaudio.load(wav_path)
    target_sample_rate = spkemb_model.sampling_rate
    if fs != target_sample_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=fs,
                               new_freq=target_sample_rate)
        wav = resampler(wav)
        fs = target_sample_rate

    spk_wave = wav.cuda()
    spk_wave_len = torch.tensor([wav.size(1)]).cuda()

    with torch.no_grad():
        spkemb = spkemb_model(spk_wave.unsqueeze(1), spk_wave_len)

    return spkemb

def get_spkemb2(wav_path, spkemb_model):
    logger.info(f"extract speaker vector of {wav_path}")
    sample_rate = spkemb_model.sampling_rate
    ref_wave = librosa.load(wav_path, sr=sample_rate)[0]
    wave = torch.from_numpy(ref_wave).cuda()

    # 将wave切分成片段，然后分别提取音色特征之后再平均
    total_length = len(wave)
    chunk_length = 60 * sample_rate

    if total_length < chunk_length:
        repeat_times = (chunk_length + total_length - 1) // total_length
        extracted_wave = torch.cat([wave] * repeat_times)
        wave = extracted_wave[:chunk_length]

    avg_vec = []
    for start in range(0, total_length, chunk_length):
        last = total_length - start
        # last_chunk is a little bit more than chunk. 只要少于1.5chunk，就直接一次提取完了
        if last >= chunk_length and last < chunk_length * 1.5:
            x_chunk = wave[start:]
            start = total_length  # to break the loop
        else:
            x_chunk = wave[start: start + chunk_length]
        x_chunk = x_chunk.unsqueeze(0)  # 1,T
        x_length = torch.LongTensor([x_chunk.size(-1), ]).to(x_chunk.device)
        vec = spkemb_model(x_chunk, x_length)
        avg_vec.append(vec)

        if start == total_length:
            break

    avg_vec = torch.stack(avg_vec)
    g = torch.mean(avg_vec, dim=0)

    return g

def check_base_lora_param(base_ckpt, lora_ckpt):
    base_dict = torch.load(base_ckpt, map_location='cpu')
    lora_dict = torch.load(lora_ckpt, map_location='cpu')
    print(f"Check base ckpt {base_ckpt} and lora ckpt {lora_ckpt}")
    for k in base_dict.keys():
        base_param = base_dict[k]
        lora_param = lora_dict[k]

        different = torch.sum(base_param) - torch.sum(lora_param)
        if different!=0:
            print(f"key:{k}, diff: {different}")

if __name__ == '__main__':
    config_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/vc_config_v2.yaml"
    llm_base_model_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/LLM/llm_v2.pt"
    flow_base_model_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/Flow/flow_v2.pt"
    llm_model_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/exp/llm_pho_31w_tts_lora_all/epoch_52_step_370000.pt"
    llm_model_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/exp/llm_pho_31w_tts_lora_all1/epoch_53_step_370000.pt"
    flow_model_path = "/data/megastore/Projects/DuJing/code/CosyVoice/examples/tts_vc/cosyvoice2/exp/flow_15w_tts_lora/epoch_8_step_20000.pt"

    check_base_lora_param(llm_base_model_path, llm_model_path)
    check_base_lora_param(flow_base_model_path, flow_model_path)

    se_model_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/SpeakerEncoder/speaker_encoder_v2.pt"
    codec_model_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/s3tokenizer/"
    offline_vec_path = "/data/megastore/SHARE/TTS/pretrained_models/fewshot_spk_vec.pt"

    llm_save_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/LLM/llm_v2_lora_0.pt"
    flow_save_path = "/data/megastore/Projects/DuJing/code/lam_tts/tts/checkpoints/LAM-VC/Flow/flow_v2_lora_0.pt"
    version = "v2.0.250319";    ref_audios = fs_ref_audios7;  only_lora = True

    print(f"model_path:\n {llm_model_path}\n {flow_model_path} \n {se_model_path} \n {codec_model_path} \n {offline_vec_path}")

    fewshot_vecs = torch.load(offline_vec_path, map_location='cpu')
    configs, llm, flow, spkemb_model, codec_model, mel_fn = init_model(config_path, llm_model_path, flow_model_path, se_model_path, codec_model_path)

    speaker_infos = {}

    use_offline_spkemb = configs.get("use_offline_spkemb", False)

    for ref_path in ref_audios:
        if isinstance(ref_path, tuple):
            ref_path, tts_id, spk_name = ref_path
            logger.info(f"ID {tts_id}, speaker name {spk_name}, ref_audio {ref_path}")

        sr = configs['sample_rate']
        ref_wave = librosa.load(ref_path, sr=sr)[0]
        wave = torch.from_numpy(ref_wave).to('cuda')

        # 选取最后10s作为ref_audio, 用做flow模块prompt, 添加0.5s静音
        flow_audio = wave[-10 * sr:].unsqueeze(0)
        silence = (0 * wave[-int(0.5 * sr):]).unsqueeze(0)
        flow_audio = torch.cat([flow_audio, silence], dim=1)

        flow_prompt_token, prompt_len = wav2token(flow_audio, codec_model)
        flow_prompt_feat = wav2melfeat(flow_audio, mel_fn)

        if use_offline_spkemb:
            logger.info(f"use the offline vector of speaker {spk_name}")
            spk_vec = fewshot_vecs[spk_name].unsqueeze(0)  # B D
        else:
            # spk_vec0 = get_spkemb(ref_path, spkemb_model)  # B D
            spk_vec = get_spkemb2(ref_path, spkemb_model)

        cur_info = {
            'llm_prompt_text': None,
            'llm_prompt_token': None,
            'flow_prompt_feat': flow_prompt_feat,
            'flow_prompt_token': flow_prompt_token,
            'speaker_embedding': spk_vec,
            'llm_speaker_embedding': spk_vec,
        }
        speaker_infos[tts_id] = cur_info

    flow_state_dict = flow.state_dict()
    llm_state_dict = llm.state_dict()

    if only_lora:
        flow_keys = list(flow_state_dict.keys())
        for k in flow_keys:
            if 'lora' not in k:
                del flow_state_dict[k]
        llm_keys = list(llm_state_dict.keys())
        for k in llm_keys:
            if 'lora' not in k:
                del llm_state_dict[k]
    logger.info(f"Save llm with speaker infos into {llm_save_path}")
    torch.save({
        'version': version,
        'state_dict': llm_state_dict,
        'speaker_infos': speaker_infos,
    }, llm_save_path)
    logger.info(f"Save flow into {flow_save_path}")
    torch.save({
        'version': version,
        'state_dict': flow_state_dict,
    }, flow_save_path)
