from __future__ import print_function
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import sys
import onnxruntime
import random
import torch
import tensorrt as trt
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import logging


def get_dummy_input(batch_size, seq_len, out_channels, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    return x, mask, mu, t, spks, cond


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    vc_config_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/vc_config_v2.yaml"
    vc_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/Flow/flow_v2.pt"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/Flow"

    state_dict = torch.load(vc_model_path)
    print(state_dict.keys())
    with open(vc_config_path, 'r') as f:
        vc_configs = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': pretrain_path,
            'qwen_sglang_config': None,
            'llm': None,
            'hift': None,
        })

    model = vc_configs['flow'].cuda()
    model.load_state_dict(state_dict, strict=False)

    # 1. export flow decoder estimator
    estimator = model.decoder.estimator
    estimator.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len = 2, 256
    out_channels = estimator.out_channels
    onnx_model_path = '{}/flow.decoder.estimator.fp32.onnx'.format(save_root)
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, seq_len, out_channels, device)
    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        onnx_model_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'estimator_out': {2: 'seq_len'},
        }
    )

    # 2. test computation consistency
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
    estimator_onnx = onnxruntime.InferenceSession(onnx_model_path,
                                                  sess_options=option, providers=providers)

    # for _ in tqdm(range(10)):
    #     x, mask, mu, t, spks, cond = get_dummy_input(batch_size, random.randint(16, 512), out_channels, device)
    #     output_pytorch = estimator(x, mask, mu, t, spks, cond)
    #     ort_inputs = {
    #         'x': x.cpu().numpy(),
    #         'mask': mask.cpu().numpy(),
    #         'mu': mu.cpu().numpy(),
    #         't': t.cpu().numpy(),
    #         'spks': spks.cpu().numpy(),
    #         'cond': cond.cpu().numpy()
    #     }
    #     output_onnx = estimator_onnx.run(None, ort_inputs)[0]
    #     torch.testing.assert_allclose(output_pytorch, torch.from_numpy(output_onnx).to(device), rtol=1e-2, atol=1e-4)
    logging.info('successfully export estimator')

    # convert onnx into tensorrt
    trt_model_path = f"{save_root}/flow.decoder.estimator.fp32.trt"
    convert_onnx_to_trt(trt_model_path, get_trt_kwargs(), onnx_model_path, False)
    trt_model_path = f"{save_root}/flow.decoder.estimator.fp16.trt"
    convert_onnx_to_trt(trt_model_path, get_trt_kwargs(), onnx_model_path, True)

def get_trt_kwargs():
    min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
    opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
    max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
    input_names = ["x", "mask", "mu", "cond"]
    return {'min_shape': min_shape, 'opt_shape': opt_shape,
            'max_shape': max_shape, 'input_names': input_names}

def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):

    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


if __name__ == "__main__":
    main()
