"""Deterministic sub-patch 3D PixelShuffle decoder for frozen VideoMAE probing.

Replaces the low-pass trilinear interpolation of LightweightFPN3D with learned
channel-to-space/time PixelShuffle-3D stages and direct sub-patch skip
projections from early Transformer blocks (blocks 3 and 6), enabling true
sub-16px spatial and sub-2-frame temporal decoding from frozen patch embeddings.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from .three_head_data import CLASS_NAMES


def pixel_shuffle_3d(x: torch.Tensor, rt: int, rh: int, rw: int) -> torch.Tensor:
    """Rearrange (B, C*rt*rh*rw, T, H, W) -> (B, C, T*rt, H*rh, W*rw)."""
    b, c_all, t, h, w = x.shape
    factor = rt * rh * rw
    if c_all % factor != 0:
        raise ValueError(f"Channel dimension {c_all} must be divisible by rt*rh*rw={factor}")
    c = c_all // factor
    return (
        x.view(b, c, rt, rh, rw, t, h, w)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .contiguous()
        .view(b, c, t * rt, h * rh, w * rw)
    )


def _icnr_init_conv3d(conv: nn.Conv3d, factor: int) -> None:
    """Initialize sub-voxel projection weights with ICNR to avoid checkerboard artifacts."""
    with torch.no_grad():
        out_ch, in_ch, kt, kh, kw = conv.weight.shape
        if out_ch % factor != 0:
            return
        sub_out = out_ch // factor
        base = torch.empty(sub_out, in_ch, kt, kh, kw, device=conv.weight.device, dtype=conv.weight.dtype)
        nn.init.kaiming_normal_(base, nonlinearity="relu")
        # Repeat identical weights across the `factor` sub-voxel slots per output channel
        repeated = base.unsqueeze(1).repeat(1, factor, 1, 1, 1, 1).view(out_ch, in_ch, kt, kh, kw)
        conv.weight.copy_(repeated)
        if conv.bias is not None:
            conv.bias.zero_()


class DepthwiseSeparableRes3D(nn.Module):
    """3x3x3 depthwise spatial-temporal seam mixing + 1x1x1 cross-channel mixing."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        gn_groups = min(groups, channels)
        while channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SubPatchFPN3D(nn.Module):
    """Progressive 3D PixelShuffle decoder from (8, 14, 14) tokens to (16, 112, 112) shared features.

    - Stage 1: f12 + f9 -> (8, 28, 28), 96 ch  [stride 8 px, 2x2 spatial PixelShuffle]
    - Stage 2: Stage 1 + f6 (4x4 PixelShuffle) -> (8, 56, 56), 64 ch  [stride 4 px]
    - Stage 3: Stage 2 (2x2x2 PixelShuffle) + f3 (2x8x8 PixelShuffle) -> (16, 112, 112), 48 ch  [stride 2 px, T=16]
    """

    def __init__(self, dimension: int, out_channels: int = 48):
        super().__init__()
        self.out_channels = out_channels

        # Stage 1: semantic fusion of blocks 12 and 9 -> (8, 28, 28)
        self.stage1_pre = nn.Sequential(
            nn.Conv3d(2 * dimension, 192, kernel_size=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
        )
        self.stage1_up = nn.Conv3d(192, 96 * 4, kernel_size=1)
        _icnr_init_conv3d(self.stage1_up, factor=4)
        self.stage1_refine = DepthwiseSeparableRes3D(96)

        # Stage 2: upsample stage 1 + unpack block 6 directly into 4x4 sub-patches -> (8, 56, 56)
        self.stage2_up = nn.Conv3d(96, 64 * 4, kernel_size=1)
        _icnr_init_conv3d(self.stage2_up, factor=4)
        self.skip6_bottleneck = nn.Sequential(
            nn.Conv3d(dimension, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.skip6_unpack = nn.Conv3d(128, 32 * 16, kernel_size=1)
        _icnr_init_conv3d(self.skip6_unpack, factor=16)
        self.stage2_fuse = nn.Sequential(
            nn.Conv3d(64 + 32, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.stage2_refine = DepthwiseSeparableRes3D(64)

        # Stage 3: upsample stage 2 (rt=2, rh=2, rw=2) + unpack block 3 directly into 2x8x8 sub-tubelets -> (16, 112, 112)
        self.stage3_up = nn.Conv3d(64, 32 * 8, kernel_size=1)
        _icnr_init_conv3d(self.stage3_up, factor=8)
        self.skip3_bottleneck = nn.Sequential(
            nn.Conv3d(dimension, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.skip3_unpack = nn.Conv3d(96, 16 * 128, kernel_size=1)
        _icnr_init_conv3d(self.skip3_unpack, factor=128)
        self.stage3_fuse = nn.Sequential(
            nn.Conv3d(32 + 16, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.stage3_refine = DepthwiseSeparableRes3D(out_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("SubPatchFPN3D requires encoder blocks 3, 6, 9, and 12")
        f3, f6, f9, f12 = features

        # Stage 1: (8, 14, 14) -> (8, 28, 28)
        x = self.stage1_pre(torch.cat([f12, f9], dim=1))
        x = pixel_shuffle_3d(self.stage1_up(x), 1, 2, 2)
        x = self.stage1_refine(x)

        # Stage 2: (8, 28, 28) -> (8, 56, 56)
        x_up = pixel_shuffle_3d(self.stage2_up(x), 1, 2, 2)
        s6 = pixel_shuffle_3d(self.skip6_unpack(self.skip6_bottleneck(f6)), 1, 4, 4)
        x = self.stage2_refine(self.stage2_fuse(torch.cat([x_up, s6], dim=1)))

        # Stage 3: (8, 56, 56) -> (16, 112, 112)
        x_up = pixel_shuffle_3d(self.stage3_up(x), 2, 2, 2)
        s3 = pixel_shuffle_3d(self.skip3_unpack(self.skip3_bottleneck(f3)), 2, 8, 8)
        x = self.stage3_refine(self.stage3_fuse(torch.cat([x_up, s3], dim=1)))
        return x


class SubPatchThreeHeadProbe(nn.Module):
    """Frozen VideoMAE encoder + SubPatchFPN3D + sub-pixel 2x2 PixelShuffle hierarchical heads."""

    def __init__(self, encoder, heads=("cma", "ct", "rdt"), out_channels: int = 48, encoder_mode: str = "frozen"):
        super().__init__()
        if not heads or any(head not in CLASS_NAMES for head in heads):
            raise ValueError("Unknown or empty probe tasks")
        if encoder.config.num_hidden_layers != 12:
            raise ValueError("Multilevel protocol requires a twelve-block encoder")
        self.encoder = encoder
        self.head_names = tuple(heads)
        self.fpn = SubPatchFPN3D(encoder.config.hidden_size, out_channels=out_channels)

        # Each head projects (16, 112, 112) -> 4 * num_classes -> PixelShuffle(1, 2, 2) -> (16, 224, 224)
        self.heads = nn.ModuleDict()
        self.cond_heads = nn.ModuleDict()
        for name in self.head_names:
            n_out = 1 if name == "cma" else 5
            proj = nn.Conv3d(out_channels, n_out * 4, kernel_size=1)
            _icnr_init_conv3d(proj, factor=4)
            self.heads[name] = proj

            cond_ch = int(name in ("ct", "rdt") and "cma" in self.head_names)
            cond_ch += 5 * int(name == "rdt" and "ct" in self.head_names)
            if cond_ch > 0:
                cond_layer = nn.Conv3d(cond_ch, n_out, kernel_size=1, bias=False)
                nn.init.zeros_(cond_layer.weight)
                self.cond_heads[name] = cond_layer

        self.auxiliary_mae = None
        self.set_encoder_mode(encoder_mode)

    def set_encoder_mode(self, mode: str) -> None:
        if mode not in ("frozen", "last4", "full"):
            raise ValueError("Encoder mode must be frozen, last4 or full")
        self.encoder_mode = mode
        self.encoder.requires_grad_(mode == "full")
        if mode == "last4":
            for block in self.encoder.encoder.layer[-4:]:
                block.requires_grad_(True)
            if getattr(self.encoder, "layernorm", None) is not None:
                self.encoder.layernorm.requires_grad_(True)
        if mode == "frozen":
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_mode == "frozen":
            self.encoder.eval()
        return self

    def encode(self, clips: torch.Tensor) -> list[torch.Tensor]:
        cfg = self.encoder.config
        if tuple(clips.shape[1:]) != (cfg.num_frames, 7, cfg.image_size, cfg.image_size):
            raise ValueError("Probe expects complete canonical sixteen-frame seven-channel clips")
        context = torch.no_grad() if self.encoder_mode == "frozen" else nullcontext()
        with context:
            states = self.encoder(clips, output_hidden_states=True).hidden_states
        t, grid = cfg.num_frames // cfg.tubelet_size, cfg.image_size // cfg.patch_size
        return [
            states[i].reshape(len(clips), t, grid, grid, cfg.hidden_size).permute(0, 4, 1, 2, 3).contiguous()
            for i in (3, 6, 9, 12)
        ]

    def decode(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.fpn(features)  # (B, 48, 16, 112, 112)
        output: dict[str, torch.Tensor] = {}
        for head in ("cma", "ct", "rdt"):
            if head not in self.head_names:
                continue
            logits = pixel_shuffle_3d(self.heads[head](shared), 1, 2, 2)  # (B, n_out, 16, 224, 224)
            parts = []
            if head in ("ct", "rdt") and "cma" in output:
                parts.append(output["cma"].sigmoid())
            if head == "rdt" and "ct" in output:
                parts.append(output["ct"].softmax(1))
            if parts and head in self.cond_heads:
                logits = logits + self.cond_heads[head](torch.cat(parts, dim=1))
            output[head] = logits
        return output

    def forward(self, clips: torch.Tensor | None = None, *, features: list[torch.Tensor] | None = None):
        return self.decode(self.encode(clips) if features is None else features)
