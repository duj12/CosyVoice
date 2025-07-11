from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import cosyvoice.speaker.commons as commons
from cosyvoice.hifigan.discriminator import \
    feature_loss, generator_loss, discriminator_loss
from cosyvoice.utils.losses import tpr_loss, mel_loss, kl_loss


class HiFiGan(nn.Module):
    def __init__(self, generator, discriminator, mel_spec_transform,
                 multi_mel_spectral_recon_loss_weight=45, feat_match_loss_weight=2.0,
                 tpr_loss_weight=1.0, tpr_loss_tau=0.04):
        super(HiFiGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.mel_spec_transform = mel_spec_transform
        self.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.tpr_loss_weight = tpr_loss_weight
        self.tpr_loss_tau = tpr_loss_tau

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch['turn'] == 'generator':
            return self.forward_generator(batch, device)
        else:
            return self.forward_discriminator(batch, device)

    def forward_generator(self, batch, device):
        real_speech = batch['speech'].to(device)
        pitch_feat, mel_feat = None, None
        if 'pitch_feat' in batch:
            pitch_feat = batch['pitch_feat'].to(device)
        if 'speech_feat' in batch:
            mel_feat = batch['speech_feat'].to(device)
        # 1. calculate generator outputs
        generated_speech, generated_f0 = self.generator(batch, device)

        loss_mel_recon = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_kl = torch.tensor(0.0, dtype=torch.float32, device=device)
        if isinstance(generated_f0, tuple):
            if self.generator.__class__.__name__.startswith("BigVGAN"):
                generated_mel, generated_f0 = generated_f0
                loss_mel_recon = F.mse_loss(generated_mel, mel_feat, reduction="mean")
            elif self.generator.__class__.__name__.startswith("VitsDecoder"):
                (ids_slice, x_mask, y_mask, z, z_p, m_p, logs_p, m_q, logs_q) = generated_f0
                real_speech = commons.slice_segments(
                    real_speech.unsqueeze(1), ids_slice * self.generator.hop_length,
                    self.generator.segment_size).squeeze(1)
                generated_speech = generated_speech.squeeze(1)

                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)
        # 3. calculate generator losses, feature loss, mel loss, tpr losses [Optional]
        loss_gen, _ = generator_loss(y_d_gs)
        loss_fm = feature_loss(fmap_rs, fmap_gs)
        loss_mel = mel_loss(real_speech, generated_speech, self.mel_spec_transform)
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_f0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        if generated_f0 is not None and pitch_feat is not None:
            loss_f0 = F.l1_loss(generated_f0, pitch_feat)
        loss = loss_gen + self.feat_match_loss_weight * loss_fm + \
            self.multi_mel_spectral_recon_loss_weight * loss_mel + \
            self.tpr_loss_weight * loss_tpr + loss_f0 + loss_mel_recon + loss_kl
        return {'loss': loss, 'loss_gen': loss_gen, 'loss_fm': loss_fm,
                'loss_mel': loss_mel, 'loss_tpr': loss_tpr, 'loss_f0': loss_f0,
                "loss_mel_recon": loss_mel_recon, "loss_kl": loss_kl, }

    def forward_discriminator(self, batch, device):
        real_speech = batch['speech'].to(device)
        # 1. calculate generator outputs
        with torch.no_grad():
            generated_speech, generated_f0 = self.generator(batch, device)

        if self.generator.__class__.__name__.startswith("VitsDecoder"):
            (ids_slice, x_mask, y_mask, z, z_p, m_p, logs_p, m_q,
             logs_q) = generated_f0
            real_speech = commons.slice_segments(
                real_speech.unsqueeze(1), ids_slice * self.generator.hop_length,
                self.generator.segment_size).squeeze(1)
            generated_speech = generated_speech.squeeze(1)

        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)
        # 3. calculate discriminator losses, tpr losses [Optional]
        loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)
        loss = loss_disc + self.tpr_loss_weight * loss_tpr
        return {'loss': loss, 'loss_disc': loss_disc, 'loss_tpr': loss_tpr}
