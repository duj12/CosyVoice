import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from typing import Dict
from torch.nn.utils import weight_norm, remove_weight_norm
from .layers import LoRALayer, Linear, Embedding, Conv1d, ConvTranspose1d, CausalConv1d
from cosyvoice.flow.decoder import CausalConv1d as Cosy_CausalConv1d

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def mark_only_infer_base_model(model):
    logger.info(f"Mark the LoRA module, only infer Base model.")
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.infer_base_model = True

def mark_infer_base_and_lora(model):
    logger.info(f"Mark the LoRA module, infer Base+LoRA model.")
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.infer_base_model = False


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or "weight_cache" in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k or "weight_cache" in k }
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k or "weight_cache" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def getModelSize_lora(model, lora_bias="none"):
    param_size_dict = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALayer):
            continue

        parent_name = name.split(".")[0]
        if parent_name not in param_size_dict:
            param_size_dict[parent_name] = 0
        
        if hasattr(module, 'lora_A'):
            param = module.lora_A
            param_size_dict[parent_name] += param.nelement() * param.element_size() / 1024 / 1024
            param = module.lora_B
            param_size_dict[parent_name] += param.nelement() * param.element_size() / 1024 / 1024
        if lora_bias == "lora_only":
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    param = module.bias
                    param_size_dict[parent_name] += param.nelement() * param.element_size() / 1024 / 1024

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    all_size = 0
    for key, param_size in param_size_dict.items():
        logger.info(f"lora节点{key}大小为：{param_size:.3f}MB")
        all_size += param_size
    logger.info(f"lora模型总大小为：{all_size:.3f}MB")
    return

def replace_specific_layer_4lora(model, hps):
    lora_r_new = hps["lora_r"]
    lora_alpha = hps["lora_alpha"]
    lora_dropout = 0.01
    lora_init_weights = hps.get("lora_init_weights", "normal")

    lora_skip_modules = hps.get("lora_skip_modules", ['llm'])

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        need_lora = True
        for skip_module in lora_skip_modules:
            if name.startswith(skip_module):
                logger.info(f"LoRA skip name {name}")
                need_lora = False
                break
        if not need_lora:
            continue

        # Check if the module is an instance of the specified layers
        if isinstance(module, torch.nn.Linear):
            out_features, in_features = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            localnet = Linear(in_features, out_features, bias=False if module.bias is None else True,
                r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False,
                lora_init_weights=lora_init_weights)
            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)

        elif isinstance(module, torch.nn.Embedding):
            num_embeddings, embedding_dim = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            localnet = Embedding(num_embeddings, embedding_dim,
                r=lora_r_new, lora_alpha=lora_alpha, merge_weights=False,
                lora_init_weights=lora_init_weights)
            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.Conv1d):
            device = module.weight.device
            dtype = module.weight.dtype

            if isinstance(module, Cosy_CausalConv1d):
                localnet = CausalConv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    bias=False if module.bias is None else True,
                    r=lora_r_new, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, merge_weights=False,
                    lora_init_weights=lora_init_weights
                )
            else:
                localnet = Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    bias=False if module.bias is None else True,
                    r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False,
                    lora_init_weights=lora_init_weights
                )
            # 判断是否有weight_norm 会修改weight！
            if getattr(module, "weight_g", None) is not None:
                localnet = weight_norm(localnet)

            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            device = module.weight.device
            dtype = module.weight.dtype

            localnet = ConvTranspose1d(
                in_channels=module.in_channels, 
                out_channels=module.out_channels, 
                kernel_size=module.kernel_size[0], 
                stride=module.stride,
                padding=module.padding, 
                output_padding=module.output_padding, 
                dilation=module.dilation, 
                groups=module.groups, 
                device=module.weight.device,
                dtype=module.weight.dtype,
                bias=False if module.bias is None else True,
                r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False,
                lora_init_weights=lora_init_weights)
            # NOTE 判断是否有weight_norm 会修改weight！
            if getattr(module, "weight_g", None) is not None:
                localnet = weight_norm(localnet)

            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.Conv2d):
            # print(f"{name}, {module} is not supported now.")
            pass

    return model

def set_layer_from_name(net, name, target_layer):
    tokens = name.strip().split('.')
    layer = net
    for t in tokens[:-1]:
        if not t.isnumeric():
            layer = getattr(layer, t)
        else:
            layer = layer[int(t)]
    setattr(layer, tokens[-1], target_layer)