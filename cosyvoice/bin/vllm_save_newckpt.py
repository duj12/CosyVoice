import torch
import os
from hyperpyyaml import load_hyperpyyaml

def export_cosyvoice2_vllm(model, model_path, device):
    if os.path.exists(model_path):
        return
    pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
    vocab_size = model.speech_embedding.num_embeddings
    feature_size = model.speech_embedding.embedding_dim
    pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to

    dtype = torch.bfloat16
    # lm_head
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=pad_vocab_size, bias=True)
    with torch.no_grad():
        new_lm_head.weight[:vocab_size] = model.llm_decoder.weight
        new_lm_head.bias[:vocab_size] = model.llm_decoder.bias
        new_lm_head.weight[vocab_size:] = 0
        new_lm_head.bias[vocab_size:] = 0
    model.llm.model.lm_head = new_lm_head
    new_codec_embed = torch.nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
    # embed_tokens
    embed_tokens = model.llm.model.model.embed_tokens
    with torch.no_grad():
        new_codec_embed.weight[:vocab_size] = model.speech_embedding.weight
        new_codec_embed.weight[vocab_size:] = 0
    model.llm.model.set_input_embeddings(new_codec_embed)
    model.llm.model.to(device)
    model.llm.model.to(dtype)
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = pad_vocab_size
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = True
    model.llm.model.config.max_position_embeddings = 1024  # 把最长输入序列从32768缩小为1024，节省推理显存
    model.llm.model.save_pretrained(model_path)
    os.system('sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g {}/config.json'.format(os.path.abspath(model_path)))
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.set_input_embeddings(embed_tokens)  # 恢复原来的文本embed


if __name__ == "__main__":
    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    vc_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/llm_v2.pt"
    vc_config_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/vc_config_v2.yaml"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/vllm"

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

    export_cosyvoice2_vllm(VC_model, save_root, 'cpu')