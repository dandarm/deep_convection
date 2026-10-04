"""Seven-channel adaptation of VideoMAE V2 dual masking.

Uses HF VideoMAE transformer components, with the dual-mask forward and
running-cell sampler from https://github.com/OpenGVLab/VideoMAEv2.
The atmospheric variant retains global normalization and ViT-S dimensions.
"""
from dataclasses import dataclass

import torch
from transformers import VideoMAEForPreTraining
from transformers.utils import ModelOutput

from .videomae import patchify_videomae_pixels, videomae_small_7ch_config


def running_cell_mask(config, batch_size, ratio=.5, *, device='cpu', generator=None):
    """True means dropped from decoder; rotate a tiled 2x2 cell over time.

    Follows the official generator's four possible phase offsets. Supports
    exact cell ratios 25/50/75%, not arbitrary rounded global ratios.
    """
    height = width = int(config.image_size) // int(config.patch_size)
    time = int(config.num_frames) // int(config.tubelet_size)
    if height % 2 or width % 2 or ratio not in (.25, .5, .75):
        raise ValueError('Running cells require an even patch grid and ratio 25/50/75%')
    phases = torch.randint(4, (batch_size,), generator=generator, device=device)
    offsets = torch.arange(time, device=device)[None, :, None, None] + phases[:, None, None, None] + 1
    cell_positions = (torch.arange(height, device=device)[:, None] % 2) * 2 + torch.arange(width, device=device)[None, :] % 2
    return ((cell_positions + offsets) % 4 < int(4 * ratio)).reshape(batch_size, -1)


def fixed_running_cell_mask(config, indices, ratio, seed, device):
    """Reproducible per-index decoder mask independent of training RNG/batches."""
    return torch.cat([running_cell_mask(config, 1, ratio,
        generator=torch.Generator().manual_seed(seed + int(index))) for index in indices]).to(device)


@dataclass
class DualMaskOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    prediction_mask: torch.Tensor = None
    loss_mask: torch.Tensor = None


class VideoMAEV2ForPreTraining(VideoMAEForPreTraining):
    """ViT encoder plus decoder on visible tokens and sampled reconstruction tokens.

    Decoder positions may overlap encoder-visible positions, as in the official
    implementation. These duplicated positions are excluded from the loss.
    With decode_mask=None, reconstruct all encoder-hidden tokens for diagnostics.
    """
    def forward(self, pixel_values, bool_masked_pos, decode_mask=None, **kwargs):
        if self.config.norm_pix_loss or self.config.num_channels != 7:
            raise ValueError('Atmospheric V2 requires seven channels and no patch normalization')
        if bool_masked_pos.dtype != torch.bool or bool_masked_pos.ndim != 2:
            raise ValueError('Encoder mask must be a batched boolean mask')
        prediction_mask = bool_masked_pos if decode_mask is None else ~decode_mask
        if prediction_mask.shape != bool_masked_pos.shape or prediction_mask.dtype != torch.bool:
            raise ValueError('Decoder mask must match encoder mask shape and dtype')
        for mask in (bool_masked_pos, prediction_mask):
            if not torch.all(mask.sum(1) == mask.sum(1)[0]):
                raise ValueError('Each batch row must have an equal number of selected tokens')
        hidden = self.videomae(pixel_values, bool_masked_pos=bool_masked_pos).last_hidden_state
        visible = self.encoder_to_decoder(hidden)
        batch, _, dim = visible.shape
        positions = self.position_embeddings.expand(batch, -1, -1).to(visible)
        pos_visible = positions[~bool_masked_pos].reshape(batch, -1, dim)
        pos_reconstruct = positions[prediction_mask].reshape(batch, -1, dim)
        tokens = torch.cat((visible + pos_visible, self.mask_token + pos_reconstruct), dim=1)
        logits = self.decoder(tokens, pos_reconstruct.shape[1]).logits
        targets = patchify_videomae_pixels(pixel_values, self.config)[prediction_mask].reshape_as(logits)
        loss_mask = bool_masked_pos[prediction_mask].reshape(batch, -1)
        if not loss_mask.any():
            raise ValueError('No encoder-hidden reconstruction targets selected')
        loss = (logits.float() - targets.float()).square()[loss_mask].mean()
        return DualMaskOutput(loss=loss, logits=logits,
                              prediction_mask=prediction_mask, loss_mask=loss_mask)


def build_videomae_v2_small_7ch_from_scratch(**overrides):
    values=dict(videomae_version='v2', decoder_mask_ratio=.5, layer_norm_eps=1e-6)
    values.update(overrides)
    return VideoMAEV2ForPreTraining(videomae_small_7ch_config(**values))
