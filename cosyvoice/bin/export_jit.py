from __future__ import print_function
from hyperpyyaml import load_hyperpyyaml
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import torch
from cosyvoice.utils.file_utils import logging



def get_optimized_script(model, preserved_attrs=[]):
    script = torch.jit.script(model)
    if preserved_attrs != []:
        script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
    else:
        script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    return script


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch._C._jit_set_fusion_strategy([('STATIC', 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    vc_config_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/vc_config_v2.yaml"
    vc_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/Flow/flow_v2.pt"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/Flow"
    print(f"model path: {vc_model_path}")
    state_dict = torch.load(vc_model_path)
    print(state_dict.keys())
    with open(vc_config_path, 'r') as f:
        vc_configs = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': pretrain_path,
            'qwen_sglang_config': None,
            'llm': None,
            'hift': None,
        })

    VC_model = vc_configs['flow'].eval()
    VC_model.load_state_dict(state_dict, strict=False)

    # export flow encoder
    flow_encoder = VC_model.encoder
    script = get_optimized_script(flow_encoder)
    script.save('{}/flow.encoder.fp32.zip'.format(save_root))
    script = get_optimized_script(flow_encoder.half())
    script.save('{}/flow.encoder.fp16.zip'.format(save_root))
    script = get_optimized_script(flow_encoder.to(torch.bfloat16))
    script.save('{}/flow.encoder.bf16.zip'.format(save_root))
    logging.info('successfully export flow_encoder')


if __name__ == '__main__':
    main()
