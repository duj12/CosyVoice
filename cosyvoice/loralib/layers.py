#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

from torch.version import __version__ as __torch_version

torch_version_list = __torch_version.split('.')
torch_version = '.'.join(torch_version_list[0:2])

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):  
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.infer_base_model = False   # 为True时，只对基座模型进行推理


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        lora_init_weights: str = "normal",
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r if "noscale" not in lora_init_weights else 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        # init
        nn.Embedding.reset_parameters(self)
        self.lora_init_weights = lora_init_weights
        if "pissa" not in lora_init_weights:
            self.init_parameters()

    def init_parameters(self):
        if hasattr(self, 'lora_A'):
            if self.lora_init_weights=="normal":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.zeros_(self.lora_B)
                nn.init.normal_(self.lora_A)
            elif "pissa" in self.lora_init_weights:
                if self.lora_init_weights[:5]=="pissa":
                    V, S, Uh = torch.linalg.svd(self.weight.data, full_matrices=False)
                    Vr = V[:, : self.r]
                    Sr = S[: self.r]
                    Sr /= self.scaling
                    Uhr = Uh[: self.r]
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    # print("debug embedding:", self.weight.data.size(), self.r, int(self.lora_init_weights.split("_niter_")[-1]))
                    Vr, Sr, Ur = torch.svd_lowrank(self.weight.data, self.r, niter=int(self.lora_init_weights.split("_niter_")[-1]))
                    Sr /= self.scaling
                    Uhr = Ur.t()
                else:
                    assert(0)
                
                lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
                lora_B = Vr @ torch.diag(torch.sqrt(Sr))
                self.lora_A.data = lora_A
                self.lora_B.data = lora_B
                dtype = self.weight.dtype
                weight = self.weight.data - self.scaling * lora_B @ lora_A
                weight = weight.to(dtype)
                self.weight.data = weight

                if self.lora_init_weights[:5]=="pissa":
                    del V, S, Uh, Vr, Sr, Uhr
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    del Vr, Sr, Ur, Uhr
                else:
                    assert(0)
            else:
                assert(0)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged and not self.infer_base_model:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        lora_init_weights: str = "normal",
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        assert(r>0)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r if "noscale" not in lora_init_weights else 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        # init 
        nn.Linear.reset_parameters(self)
        self.lora_init_weights = lora_init_weights
        if "pissa" not in lora_init_weights:
            self.init_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def init_parameters(self):
        if hasattr(self, 'lora_A'):
            if self.lora_init_weights=="normal":
                # initialize B the same way as the default for nn.Linear and A to zero
                # this is different than what is described in the paper but should not affect performance
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif "pissa" in self.lora_init_weights:
                if self.lora_init_weights[:5]=="pissa":
                    V, S, Uh = torch.linalg.svd(self.weight.data, full_matrices=False)
                    Vr = V[:, : self.r]
                    Sr = S[: self.r]
                    Sr /= self.scaling
                    Uhr = Uh[: self.r]
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    # print("debug linear:", self.weight.data.size(), self.r, int(self.lora_init_weights.split("_niter_")[-1]))
                    Vr, Sr, Ur = torch.svd_lowrank(self.weight.data, self.r, niter=int(self.lora_init_weights.split("_niter_")[-1]))
                    Sr /= self.scaling
                    Uhr = Ur.t()
                else:
                    assert(0)
                
                lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
                lora_B = Vr @ torch.diag(torch.sqrt(Sr))
                self.lora_A.data = lora_A
                self.lora_B.data = lora_B
                dtype = self.weight.dtype
                if "cachewnorm" in self.lora_init_weights:
                    self.weight_cache = nn.Parameter((self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling)
                else:
                    weight = self.weight.data - self.scaling * lora_B @ lora_A
                    weight = weight.to(dtype)
                    self.weight.data = weight

                if self.lora_init_weights[:5]=="pissa":
                    del V, S, Uh, Vr, Sr, Uhr
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    del Vr, Sr, Ur, Uhr
            else:
                assert(0)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged and not self.infer_base_model:
            if "cachewnorm" in self.lora_init_weights:
                result = F.linear(x, T(self.weight), bias=self.bias)
                result -= F.linear(x, T(self.weight_cache), bias=self.bias)
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            else:
                result = F.linear(x, T(self.weight), bias=self.bias)            
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(
        self, 
        in_channels=0, 
        out_channels=0, 
        kernel_size=0, 
        r=0, 
        lora_alpha=1, 
        lora_dropout=0., 
        merge_weights=True, 
        lora_init_weights="normal",
        **kwargs
    ):
        # print("?super(ConvLoRA, self).__init__()开始")
        # super(ConvLoRA, self).__init__()
        # nn.Module.__init__(self)
        # print("?super(ConvLoRA, self).__init__()完毕")
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int), kernel_size
        # Actual trainable parameters
        if r > 0:
            self.r = self.r * kernel_size if "fixconvk" not in lora_init_weights else self.r
            if "pissa" in lora_init_weights:
                weight = self.weight.data
                n,c,h = weight.size(0), weight.size(1), weight.size(2)
                w = weight.size(3) if len(weight.size())>=4 else 1
                self.r = min(n, self.r, c*h*w)
            lora_init_weights = "normal" if self.r==1 else lora_init_weights

            # if "fullconv" in lora_init_weights:
            #     self.lora_A = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            #     self.lora_B = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            # else:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, in_channels*kernel_size**(self.weight.dim()-2))))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels//self.groups, self.r)))

            self.scaling = self.lora_alpha / self.r if "noscale" not in lora_init_weights else 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        # init 
        self.reset_parameters()
        self.lora_init_weights = lora_init_weights
        if "pissa" not in lora_init_weights:
            self.init_parameters()

        self.merged = False

    def init_parameters(self):
        if hasattr(self, 'lora_A'):
            if self.lora_init_weights=="normal":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif "pissa" in self.lora_init_weights:
                weight = self.weight.data
                n,c,h = weight.size(0), weight.size(1), weight.size(2)
                w = weight.size(3) if len(weight.size())>=4 else -1
                weight = weight.view(n, -1)
                if self.lora_init_weights[:5]=="pissa":
                    V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
                    Vr = V[:, : self.r]
                    Sr = S[: self.r]
                    Sr /= self.scaling
                    Uhr = Uh[: self.r]
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    Vr, Sr, Ur = torch.svd_lowrank(weight, self.r, niter=int(self.lora_init_weights.split("_niter_")[-1]))
                    Sr /= self.scaling
                    Uhr = Ur.t()
                else:
                    assert(0)
                
                lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
                lora_B = Vr @ torch.diag(torch.sqrt(Sr))
                # if self.lora_A.data.size() != lora_A.size() or self.lora_B.data.size() != lora_B.size():
                #     print("self.weight.data.size(), weight.size()", self.weight.data.size(), weight.size())
                #     print("V.size(), S.size(), Uh.size()", V.size(), S.size(), Uh.size())
                #     print("self.lora_A={},self.lora_B={},lora_A={},lora_B={}".format(self.lora_A.data.size(), self.lora_B.data.size(), lora_A.size(), lora_B.size()))
                #     print("self.r,c,h,w", self.r,c,h,w)
                #     assert(0)
                # else:
                #     print("lora_A={},lora_B={}".format(lora_A.size(), lora_B.size()))
                    
                self.lora_A.data = lora_A#.view(self.r,c,h,w) if w>=1 else lora_A.view(self.r,c,h)
                self.lora_B.data = lora_B#[:,:,None,None]
                dtype = self.weight.dtype
                if "cachewnorm" in self.lora_init_weights:
                    self.weight_cache = nn.Parameter((self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling)
                else:
                    weight = self.weight.data - (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                    weight = weight.to(dtype)
                    self.weight.data = weight

                if self.lora_init_weights[:5]=="pissa":
                    del V, S, Uh, Vr, Sr, Uhr
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    del Vr, Sr, Ur, Uhr
                else:
                    assert(0)
            else:
                assert(0)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.weight = self.weight.to(self.lora_B.device)
                    self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.weight = self.weight.to(self.lora_B.device)
                    self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged and not self.infer_base_model:
            if "cachewnorm" in self.lora_init_weights:
                return self._conv_forward(
                    x, 
                    self.weight - self.weight_cache + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                    self.bias
                )
            else:
                return self._conv_forward(
                    x, 
                    self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                    self.bias
                )
        return self._conv_forward(x, self.weight, self.bias)

# Can Extend to other ones like this
class Conv1d(ConvLoRA, nn.Conv1d):
    def __init__(self, 
        *args, 
        r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, lora_init_weights="normal", 
        **kwargs
    ):
        # super(Conv1d, self).__init__(*args, **kwargs)
        # print("nn.Conv1d.__init__开始")
        nn.Conv1d.__init__(self, *args, **kwargs)
        # print("nn.Conv1d.__init__完毕")
        # print("ConvLoRA.__init__开始")
        ConvLoRA.__init__(self, 
            *args, 
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, lora_init_weights=lora_init_weights,
            **kwargs
        )
        # print("ConvLoRA.__init__完毕")

class ConvTransposeLoRA(nn.Module, LoRALayer):
    def __init__(
        self, 
        in_channels=0, 
        out_channels=0, 
        kernel_size=0, 
        r=0, 
        lora_alpha=1, 
        lora_dropout=0., 
        merge_weights=True, 
        lora_init_weights="normal",
        **kwargs
    ):
        # print("?super(ConvLoRA, self).__init__()开始")
        # super(ConvLoRA, self).__init__()
        # nn.Module.__init__(self)
        # print("?super(ConvLoRA, self).__init__()完毕")
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int), kernel_size
        # Actual trainable parameters
        if r > 0:
            self.r = self.r * kernel_size if "fixconvk" not in lora_init_weights else self.r
            if "pissa" in lora_init_weights:
                weight = self.weight.data
                n,c,h = weight.size(0), weight.size(1), weight.size(2)
                w = weight.size(3) if len(weight.size())>=4 else 1
                self.r = min(n, self.r, c*h*w)
            lora_init_weights = "normal" if self.r==1 else lora_init_weights

            self.lora_A = nn.Parameter(
                self.weight.new_zeros((self.r, out_channels//self.groups*kernel_size**(self.weight.dim()-2)))
            )
            self.lora_B = nn.Parameter(
              self.weight.new_zeros((in_channels, self.r))
            )
            self.scaling = self.lora_alpha / self.r if "noscale" not in lora_init_weights else 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()
        self.lora_init_weights = lora_init_weights
        if "pissa" not in lora_init_weights:
            self.init_parameters()

        self.merged = False

    def init_parameters(self):
        if hasattr(self, 'lora_A'):
            if self.lora_init_weights=="normal":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif "pissa" in self.lora_init_weights:
                weight = self.weight.data
                n,c,h = weight.size(0), weight.size(1), weight.size(2)
                w = weight.size(3) if len(weight.size())>=4 else -1
                weight = weight.view(n, -1)
                if self.lora_init_weights[:5]=="pissa":
                    V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
                    Vr = V[:, : self.r]
                    Sr = S[: self.r]
                    Sr /= self.scaling
                    Uhr = Uh[: self.r]
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    # print("debug conv:", self.weight.data.size(), self.r, int(self.lora_init_weights.split("_niter_")[-1]))
                    Vr, Sr, Ur = torch.svd_lowrank(weight, self.r, niter=int(self.lora_init_weights.split("_niter_")[-1]))
                    Sr /= self.scaling
                    Uhr = Ur.t()
                else:
                    assert(0)
                
                lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
                lora_B = Vr @ torch.diag(torch.sqrt(Sr))
                # if self.lora_A.data.size() != lora_A.size() or self.lora_B.data.size() != lora_B.size():
                #     print("self.weight.data.size(), weight.size()", self.weight.data.size(), weight.size())
                #     print("V.size(), S.size(), Uh.size()", V.size(), S.size(), Uh.size())
                #     print("self.lora_A={},self.lora_B={},lora_A={},lora_B={}".format(self.lora_A.data.size(), self.lora_B.data.size(), lora_A.size(), lora_B.size()))
                #     print("self.r,c,h,w", self.r,c,h,w)
                #     assert(0)
                # else:
                #     print("lora_A={},lora_B={}".format(lora_A.size(), lora_B.size()))

                self.lora_A.data = lora_A#.view(self.r,c,h,w) if w>=1 else lora_A.view(self.r,c,h)
                self.lora_B.data = lora_B#[:,:,None,None]
                dtype = self.weight.dtype
                if "cachewnorm" in self.lora_init_weights:
                    self.weight_cache = nn.Parameter((self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling)
                else:
                    weight = self.weight.data - (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                    weight = weight.to(dtype)
                    self.weight.data = weight

                if self.lora_init_weights[:5]=="pissa":
                    del V, S, Uh, Vr, Sr, Uhr
                elif len(self.lora_init_weights.split("_niter_")) == 2:
                    del Vr, Sr, Ur, Uhr
                else:
                    assert(0)
            else:
                assert(0)

    def train(self, mode=True):
        super(ConvTransposeLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.weight = self.weight.to(self.lora_B.device)
                    self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.weight = self.weight.to(self.lora_B.device)
                    self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x, output_size=None):
        if float(torch_version) >= 1.12:
            output_padding = self._output_padding(
                    input=x,
                    output_size=output_size,
                    stride=self.stride,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                    num_spatial_dims=self.num_spatial_dims, ### NOTE 非常坑！torch 1.12这里多了个参数！！！！
                    dilation=self.dilation
                )
        else:
            output_padding = self._output_padding(
                input=x,
                output_size=output_size,
                stride=self.stride,
                padding=self.padding,
                kernel_size=self.kernel_size,
                # num_spatial_dims=self.num_spatial_dims,  ### NOTE 非常坑！torch 1.12这里多了个参数！！！！
                dilation=self.dilation
            )
        
        if self.r > 0 and not self.merged and not self.infer_base_model:

            if self.num_spatial_dims==1:
                if "cachewnorm" in self.lora_init_weights:
                    return F.conv_transpose1d(
                            x, 
                            self.weight - self.weight_cache + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, 
                            self.bias, 
                            self.stride, 
                            self.padding,
                            output_padding, 
                            self.groups, 
                            self.dilation
                        )
                else:
                    return F.conv_transpose1d(
                            x, 
                            self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, 
                            self.bias, 
                            self.stride, 
                            self.padding,
                            output_padding, 
                            self.groups, 
                            self.dilation
                        )
            else:
                assert(0), "还没写好"
        else:
            return F.conv_transpose1d(
                    x, 
                    self.weight, 
                    self.bias, 
                    self.stride, 
                    self.padding,
                    output_padding, 
                    self.groups, 
                    self.dilation
                )

class ConvTranspose1d(ConvTransposeLoRA, nn.ConvTranspose1d):
    def __init__(self, 
        *args, 
        r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, lora_init_weights="normal", 
        **kwargs
    ):
        self.num_spatial_dims = 1
        # super(ConvTranspose1d, self).__init__(*args, **kwargs)
        # print("nn.ConvTranspose1d.__init__开始")
        nn.ConvTranspose1d.__init__(self, *args, **kwargs)
        # print("nn.ConvTranspose1d.__init__完毕")
        # print("ConvLoRA.__init__开始")
        ConvTransposeLoRA.__init__(self, 
            *args, 
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, lora_init_weights=lora_init_weights,
            **kwargs
        )
        # print("ConvLoRA.__init__完毕")