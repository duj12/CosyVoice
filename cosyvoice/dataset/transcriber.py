

class Transcriber:
    def __init__(self, model="iic/SenseVoiceSmall", device="cuda:0"):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=False,
            remote_code="model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 15000},
            device=device)


    def transcribe(self, speech_or_path):
        from funasr.utils.postprocess_utils import \
            rich_transcription_postprocess
        res = self.model.generate(
            input=speech_or_path,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text

def get_transcriber():
    return Transcriber()
