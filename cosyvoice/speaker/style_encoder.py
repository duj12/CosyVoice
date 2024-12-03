# Copyright (c) 2024  Jing Du  (thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import cosyvoice.speaker.modules as modules


class StyleEncoder_v2(nn.Module):
    ''' Mel StyleEncoder, extract style embedding from Mels.
    refer to Meta-StyleSpeech: https://arxiv.org/pdf/2106.03153.pdf
    unlike GST's ref encoder, we use self attention to extract style
    Multi-layer transformer.
    '''

    def __init__(self, in_dim, style_hidden=128, style_vector_dim=128,
                 style_kernel_size=5, style_head=4, dropout=0.1,
                 num_layers=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout
        self.num_layers = num_layers

        self.spectral_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()

        self.in_fc = modules.LinearNorm(self.in_dim, self.hidden_dim)

        for _ in range(num_layers):
            spectral = nn.Sequential(
                modules.LinearNorm(self.hidden_dim, self.hidden_dim),
                modules.Mish(),
                nn.Dropout(self.dropout),
                modules.LinearNorm(self.hidden_dim, self.hidden_dim),
                modules.Mish(),
                nn.Dropout(self.dropout)
            )
            self.spectral_layers.append(spectral)

            temporal = nn.Sequential(
                modules.Conv1dGLU(self.hidden_dim, self.hidden_dim,
                                  self.kernel_size, self.dropout),
                modules.Conv1dGLU(self.hidden_dim, self.hidden_dim,
                                  self.kernel_size, self.dropout),
            )
            self.temporal_layers.append(temporal)

            attn = modules.MultiHeadAttention(
                self.n_head, self.hidden_dim, self.hidden_dim // self.n_head,
                self.hidden_dim // self.n_head, self.dropout)
            self.attn_layers.append(attn)

        self.fc = modules.LinearNorm(self.hidden_dim, self.out_dim)


    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        """
        :param x:  B, T, D
        :param mask:  [[1,1,1,1,1,...0,0,0],...]
        :return: style vector B, D
        """
        max_len = x.shape[1]
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        else:
            slf_attn_mask = None

        x = self.in_fc(x)
        for i in range(self.num_layers):
            # spectral
            x = self.spectral_layers[i](x)
            # temporal
            x = x.transpose(1, 2)
            x = self.temporal_layers[i](x)
            x = x.transpose(1, 2)
            # self-attention
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0)
            x, _ = self.attn_layers[i](x, mask=slf_attn_mask)

        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w


class GlobalStyleTokens_v2(nn.Module):
    """
    refer to GST https://arxiv.org/abs/1803.09017
    https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
    Style embedding -> Attention to Tokens -> Final Style Embedding
    Multi-Layer, Multi-Head Attention.

    TODO: Add regularization to minimize the tokens' similarity.
    """
    def __init__(self, token_num=128, embed_dim=128, num_heads=4,
                 num_layers=1):

        super().__init__()
        self.num_layers = num_layers
        self.embeds = nn.ParameterList()
        self.attentions = nn.ModuleList()

        d_q = embed_dim
        d_k = embed_dim // num_heads  # we assign smaller vectors in dimension

        for _ in range(num_layers):
            embed = nn.Parameter(
                torch.FloatTensor(token_num, embed_dim // num_heads))
            nn.init.normal_(embed, mean=0, std=0.5)
            self.embeds.append(embed)

            attention = modules.MultiHeadAttention_GivenK(
                query_dim=d_q, key_dim=d_k,
                num_units=embed_dim, num_heads=num_heads)
            self.attentions.append(attention)

    def forward(self, inputs, return_score=False):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E]
        # [N, token_num, E // num_heads]

        for i in range(self.num_layers):
            keys = F.tanh(self.embeds[i]).unsqueeze(0).expand(N, -1, -1)
            query = self.attentions[i](query, keys, return_score)

        return query


class StyleFuser(nn.Module):
    def __init__(self, feat_dim, style_dim, fuse_type="Add"):
        super().__init__()
        self.fuse_type = fuse_type
        if fuse_type == "Add":
            self.style_fuser = nn.Linear(style_dim, feat_dim, bias=False)
        elif fuse_type == "Concat":
            self.style_fuser = nn.Linear(
                style_dim+feat_dim, feat_dim, bias=False)
        elif fuse_type == "AdaLN":
            self.style_fuser = modules.StyleAdaptiveLayerNorm(
                feat_dim, style_dim)
        else:
            raise NotImplementedError(f"fuse type {fuse_type} not support yet.")

    def forward(self, input_feat, style_vector):
        """
        :param input_feat: B, T, D1
        :param style_vector: B, 1, D2
        :return:  B, T, D1
        """
        T = input_feat.size(1)
        if self.fuse_type == "Add":
            style_vector = self.style_fuser(style_vector).repeat(1, T, 1)
            output_feat = input_feat + style_vector
        elif self.fuse_type == "Concat":
            # fuse gst into prior, by concat and affine.
            style_vector = style_vector.repeat(1, T, 1)
            x_with_gst = torch.concat((input_feat, style_vector), dim=2)
            output_feat = self.style_fuser(x_with_gst)
        elif self.fuse_type == "AdaLN":
            output_feat = self.style_fuser(input_feat, style_vector)

        return output_feat