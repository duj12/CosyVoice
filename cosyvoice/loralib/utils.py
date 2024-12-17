#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from typing import Dict

from torch.nn.utils import weight_norm, remove_weight_norm
from .layers import LoRALayer, Linear, Embedding, Conv1d, ConvTranspose1d

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

def getModelSize_lora(model, hps):
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
        if hps.lora_bias == "lora_only":
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    param = module.bias
                    param_size_dict[parent_name] += param.nelement() * param.element_size() / 1024 / 1024

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        # buffer_sum += buffer.nelement()

    # all_size = (param_size + buffer_size) / 1024 / 1024
    all_size = 0
    for key, param_size in param_size_dict.items():
        logger.info(f"lora节点{key}大小为：{param_size:.3f}MB")
        all_size += param_size
    logger.info(f"lora模型总大小为：{all_size:.3f}MB")
    if "enc_q" in param_size_dict:
        logger.info(f"lora模型去除enc_q总大小为：{all_size-param_size_dict['enc_q']:.3f}MB")
    return

def adjust_r(name, hps, lora_r):
    lora_max = hps.lora_max
    lora_min = hps.lora_min

    father_type = name.split(".")[0]
    lora_lambda = getattr(hps, "lora_lambda_{}".format(father_type), 1.0)

    if father_type in ["flow", "dec", "dur", "enc_p", "enc_q",
                       "speaker_encoder", "style_encoder", "gst"]:
        lora_r = int(lora_r * lora_lambda)

    return lora_r

def replace_specific_layer_4lora(model, hps):
    lora_r = hps.lora_r #16
    lora_alpha = hps.lora_alpha #32
    lora_dropout = hps.lora_dropout #0.01
    lora_max = hps.lora_max #8
    lora_mid_scale = hps.lora_mid_scale #16
    lora_min = hps.lora_min #2
    lora_init_weights = hps.lora_init_weights

    unique_dim = 4

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, torch.nn.Linear):
            out_features, in_features = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype

            lora_init_weights_local = "normal" if out_features<=unique_dim or in_features<=unique_dim else lora_init_weights
            lora_r_new = adjust_r(name, hps, lora_r)
            # print("{} {} r:{} in_feat:{} out_feat:{}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", lora_r_new, in_features, out_features), device)
            localnet = Linear(in_features, out_features, bias=False if module.bias is None else True, 
                r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=hps.merge_weights, 
                lora_init_weights=lora_init_weights_local)
            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.Embedding):
            num_embeddings, embedding_dim = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype

            lora_init_weights_local = "normal"
            lora_r_new = adjust_r(name, hps, lora_r)
            # print("{} {} r:{} num_emb:{} emb_dim:{}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", lora_r_new, num_embeddings, embedding_dim), device)
            localnet = Embedding(num_embeddings, embedding_dim, 
                r=lora_r_new, lora_alpha=lora_alpha, merge_weights=hps.merge_weights, 
                lora_init_weights=lora_init_weights_local)
            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.Conv1d):
            # device = "cuda:0" 
            device = module.weight.device ### NOTE 很恶心
            dtype = module.weight.dtype
            lora_r_new = min(lora_max, lora_r, max(((module.in_channels + module.out_channels)//2)//lora_mid_scale, lora_min))
            lora_r_new = adjust_r(name, hps, lora_r_new)

            lora_init_weights_local = lora_init_weights
            if "noconvk" in hps.lora_init_weights:
                lora_init_weights_local = "normal" if module.kernel_size[0]>1 else lora_init_weights
            if getattr(module, "weight_g", None) is not None and "nownorm" in hps.lora_init_weights:
                lora_init_weights_local = "normal"
            # print("{} {} r:{} in_c:{} out_c:{} ker:{}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", lora_r_new, module.in_channels, module.out_channels, module.kernel_size[0]), device)
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
                r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=hps.merge_weights, 
                lora_init_weights=lora_init_weights_local)
            # 判断是否有weight_norm 会修改weight！
            if getattr(module, "weight_g", None) is not None:
                localnet = weight_norm(localnet)

            localnet.to(dtype)
            localnet.to(device)
            set_layer_from_name(model, name, localnet)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            device = module.weight.device
            dtype = module.weight.dtype
            lora_r_new = min(lora_max, lora_r, max(((module.in_channels + module.out_channels)//2)//lora_mid_scale, lora_min))
            lora_r_new = adjust_r(name, hps, lora_r_new)

            lora_init_weights_local = lora_init_weights
            if "noconvk" in hps.lora_init_weights:
                lora_init_weights_local = "normal" if module.kernel_size[0]>1 else lora_init_weights
            if getattr(module, "weight_g", None) is not None and "nownorm" in hps.lora_init_weights:
                lora_init_weights_local = "normal"
            # print("{} {} r:{} in_c:{} out_c:{} trans ker:{}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", lora_r_new, module.in_channels, module.out_channels, module.kernel_size[0]), device)
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
                r=lora_r_new, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=hps.merge_weights, 
                lora_init_weights=lora_init_weights_local)
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

def init_pissa_weight(model, hps):
    lora_r = hps.lora_r #16
    lora_alpha = hps.lora_alpha #32
    lora_dropout = hps.lora_dropout #0.01
    lora_max = hps.lora_max #8
    lora_mid_scale = hps.lora_mid_scale #16
    lora_min = hps.lora_min #2
    lora_init_weights = hps.lora_init_weights

    if "pissa" not in lora_init_weights:
        return model

    unique_dim = 4

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, torch.nn.Linear):
            out_features, in_features = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype

            lora_init_weights_local = "normal" if out_features<=unique_dim or in_features<=unique_dim else lora_init_weights
            lora_r_new = adjust_r(name, hps, lora_r)
            # print("init {} {} {}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", device))
            module.init_parameters()
        elif isinstance(module, torch.nn.Embedding):
            num_embeddings, embedding_dim = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype

            lora_init_weights_local = "normal"
            lora_r_new = adjust_r(name, hps, lora_r)
            # print("init {} {} {}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", device))
            module.init_parameters()
        elif isinstance(module, torch.nn.Conv1d):
            device = module.weight.device
            dtype = module.weight.dtype
            lora_r_new = min(lora_max, lora_r, max(((module.in_channels + module.out_channels)//2)//lora_mid_scale, lora_min))
            lora_r_new = adjust_r(name, hps, lora_r_new)

            lora_init_weights_local = lora_init_weights
            if "noconvk" in hps.lora_init_weights:
                lora_init_weights_local = "normal" if module.kernel_size[0]>1 else lora_init_weights
            if getattr(module, "weight_g", None) is not None and "nownorm" in hps.lora_init_weights:
                lora_init_weights_local = "normal"
            # print("init {} {} {}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", device))
            if getattr(module, "weight_g", None) is not None and "rmwnorm" in lora_init_weights:
                module = remove_weight_norm(module)
            module.init_parameters()
        elif isinstance(module, torch.nn.ConvTranspose1d):
            device = module.weight.device
            dtype = module.weight.dtype
            lora_r_new = min(lora_max, lora_r, max(((module.in_channels + module.out_channels)//2)//lora_mid_scale, lora_min))
            lora_r_new = adjust_r(name, hps, lora_r_new)

            lora_init_weights_local = lora_init_weights
            if "noconvk" in hps.lora_init_weights:
                lora_init_weights_local = "normal" if module.kernel_size[0]>1 else lora_init_weights
            if getattr(module, "weight_g", None) is not None and "nownorm" in hps.lora_init_weights:
                lora_init_weights_local = "normal"
            # print("init {} {} {}".format(name, lora_init_weights_local.upper() if lora_init_weights_local!='normal' else "", device))
            if getattr(module, "weight_g", None) is not None and "rmwnorm" in lora_init_weights:
                module = remove_weight_norm(module)
            module.init_parameters()
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