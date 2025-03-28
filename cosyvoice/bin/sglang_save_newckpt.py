from transformers import Qwen2ForCausalLM, AutoConfig
import torch
import sys, os



if __name__ == "__main__":
    ### qwen的config位置、实际训好的模型 和 保存路径 NOTE 自己改下！
    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    true_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/llm_v2.pt"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/VC2_forsglang"

    ### 这个只是为了存文件，forward定义修改在sglang的模型文件里
    class Qwen2ForCausalLM_lamtts(Qwen2ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            # 克隆的大概这么写：
            llm_input_size = 896
            llm_output_size = 896
            speech_token_size = 6561
            self.llm_decoder = torch.nn.Linear(llm_output_size, speech_token_size + 3)
            self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
            
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

    ### 实例化自定义模型并加载实际训好的模型
    config = AutoConfig.from_pretrained(pretrain_path)
    custom_model = Qwen2ForCausalLM_lamtts(config)

    state_dict = torch.load(true_model_path) # ["model"]
    print(state_dict.keys())

    new_state_dict = {
        k.replace('llm.model.', '') if 'llm.model.' in k else k: v 
        for k, v in state_dict.items()
    } # 把cosy的结构里套娃的实际的qwen的参数摘出来

    # custom_model = custom_model.to(torch.bfloat16)

    custom_model.load_state_dict(new_state_dict, strict=False)

    ### 保存
    custom_model.save_pretrained(save_root, safe_serialization=True) # 生成model.safetensors而非pytorch_model.bin
    '''
        注意：
        保存出的config里
        architectures对应的是EntryClass，sglang会根据这个对应的EntryClass来找到model下面对应的文件
        model_type只能是qwen2，好像自定义的不行
    '''