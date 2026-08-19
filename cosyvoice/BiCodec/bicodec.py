# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#               2025 Jing Du
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
from pathlib import Path
from typing import Dict, Any
from safetensors.torch import load_file
from omegaconf import OmegaConf, DictConfig

from cosyvoice.BiCodec.encoder_decoder.feat_encoder import Encoder
from cosyvoice.BiCodec.encoder_decoder.feat_decoder import Decoder
from cosyvoice.BiCodec.encoder_decoder.wave_generator import WaveGenerator
from cosyvoice.BiCodec.vq.factorized_vector_quantize import FactorizedVectorQuantize

def load_config(config_path: Path) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)
    return config


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        **kwargs
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.prenet = prenet
        self.postnet = postnet

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.
        
        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])

        model = cls(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            prenet=prenet,
            postnet=postnet,
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model. 暂时不用，训练时使用

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.

        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]   # wav2vec2.0提取特征
        embedding = batch["embedding"]

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        conditions = embedding
        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "audios": batch["wav"].unsqueeze(1),
        }

    def tokenize(self, feat):
        """
        Tokenizes the input audio into semantic and global tokens.
        Args:
            feat: input feature
        Returns:
            tuple: Semantic tokens and global tokens.
        """
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        return semantic_tokens

    def detokenize(self, semantic_tokens, speaker_embedding):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            speaker_embedding (tensor): Global speaker vector.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        x = self.prenet(z_q, speaker_embedding)
        x = x + speaker_embedding.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


class BiCodecDecoder(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
            self,
            quantizer: nn.Module,
            prenet: nn.Module,
            decoder: nn.Module,
            **kwargs
    ) -> None:
        """
        Initializes the BiCodec model with the required components.
        Args:
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            prenet (nn.Module): Prenet network.
        """
        super().__init__()
        self.quantizer = quantizer
        self.prenet = prenet
        self.decoder = decoder

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodecDecoder":
        """
        Loads the model from a checkpoint.
        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.
        Returns:
            BiCodec: The initialized BiCodec decoder model.
        """
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        decoder = WaveGenerator(**config["decoder"])

        model = cls(
            quantizer=quantizer,
            prenet=prenet,
            decoder=decoder,
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                              strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        return model

    def detokenize(self, semantic_tokens, speaker_embedding):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            speaker_embedding (tensor): Global speaker vector.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        x = self.prenet(z_q, speaker_embedding)
        x = x + speaker_embedding.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)