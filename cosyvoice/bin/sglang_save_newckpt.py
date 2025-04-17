from transformers import Qwen2ForCausalLM, Qwen2Config
import torch
import json
from hyperpyyaml import load_hyperpyyaml


class Qwen2ForCausalLM_lamtts(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 克隆的大概这么写：
        llm_input_size = 896
        llm_output_size = 896
        speech_token_size = 6561
        self.llm_decoder = torch.nn.Linear(llm_output_size,
                                           speech_token_size + 3)
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3,
                                                   llm_input_size)

        # 下面是我之前训TTS的时候的网络层名称
        # CUR_DIR = os.path.dirname(__file__)
        # sys.path.append(os.path.join(CUR_DIR, "acoustics"))
        # from acoustics.valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
        # from acoustics.valle.models.valle import FeatureMapping
        # decoder_dim = 896
        # ar_logits_out_dim = 6561 + 2
        # self.num_quantizers = 1
        # self.ar_audio_embedding_group = torch.nn.ModuleList([TokenEmbedding(decoder_dim, ar_logits_out_dim + 1) for i in range(self.num_quantizers)])
        # self.ar_group_decoder_group = torch.nn.ModuleList([FeatureMapping(decoder_dim, ar_logits_out_dim) for idx in range(self.num_quantizers)])

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs

if __name__ == "__main__":
    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    vc_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/llm_v20.pt"
    vc_config_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/vc_config_v2.yaml"

    true_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/llm_v2.pt"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/VC2_forsglang"

    config_dict = json.load(open(f"{pretrain_path}/config.json"))
    config = Qwen2Config(**config_dict)
    custom_model = Qwen2ForCausalLM_lamtts(config)

    state_dict = torch.load(vc_model_path)
    print(state_dict.keys())
    with open(vc_config_path, 'r') as f:
        vc_configs = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': pretrain_path,
            'qwen_sglang_config': None,
            'flow': None,  # llm加载时不需要加载flow和hift模块
            'hift': None,
        })

    VC_model = vc_configs['llm']
    VC_model.load_state_dict(state_dict, strict=False)
    # 将克隆模型中qwen的embed_tokens模块剥离出来， 然后将llm去掉节省显存。
    VC_model.qwen_token_embed.load_state_dict(
        VC_model.llm.model.model.embed_tokens.state_dict())
    # VC_model.llm = None  先保留llm模块，方便fp32推理
    new_vc_state_dict = VC_model.state_dict()
    torch.save(new_vc_state_dict, true_model_path)

    # 把cosy的结构里套娃的实际的qwen的参数摘出来
    new_state_dict = {
        k.replace('llm.model.', '') if 'llm.model.' in k else k: v 
        for k, v in state_dict.items()
    }
    custom_model = custom_model.to(torch.bfloat16)
    custom_model.load_state_dict(new_state_dict, strict=False)
    # 生成model.safetensors而非pytorch_model.bin
    custom_model.save_pretrained(save_root, safe_serialization=True)