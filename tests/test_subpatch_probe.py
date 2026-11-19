"""Unit tests for the deterministic SubPatchFPN3D and SubPatchThreeHeadProbe."""
from __future__ import annotations

import torch
from transformers import VideoMAEConfig, VideoMAEModel

from emma_gpm.cloud_probe import state_digest
from emma_gpm.subpatch_probe import SubPatchFPN3D, SubPatchThreeHeadProbe, pixel_shuffle_3d


def test_pixel_shuffle_3d_exact_rearrangement():
    x = torch.arange(1 * 16 * 2 * 3 * 3, dtype=torch.float32).view(1, 16, 2, 3, 3)
    y = pixel_shuffle_3d(x, rt=2, rh=2, rw=2)
    assert y.shape == (1, 2, 4, 6, 6)
    # Channel 0, sub-voxel (rt=0, rh=0, rw=0) at (t=0, h=0, w=0) equals x[0, 0, 0, 0, 0]
    assert y[0, 0, 0, 0, 0] == x[0, 0, 0, 0, 0]
    # Channel 0, sub-voxel (rt=1, rh=1, rw=1) at (t=1, h=1, w=1) equals x[0, 7, 0, 0, 0]
    assert y[0, 0, 1, 1, 1] == x[0, 7, 0, 0, 0]


def test_subpatch_probe_shapes_and_frozen_backbone_invariance():
    cfg = VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=7,
        num_frames=16,
        tubelet_size=2,
        hidden_size=48,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=96,
        norm_pix_loss=False,
    )
    encoder = VideoMAEModel(cfg)
    probe = SubPatchThreeHeadProbe(encoder, heads=("cma", "ct", "rdt"), encoder_mode="frozen")
    before = state_digest(probe.encoder)

    feats = [torch.randn(2, 48, 8, 14, 14) for _ in range(4)]
    out = probe(features=feats)
    assert out["cma"].shape == (2, 1, 16, 224, 224)
    assert out["ct"].shape == (2, 5, 16, 224, 224)
    assert out["rdt"].shape == (2, 5, 16, 224, 224)

    # Verify gradients flow through head parameters only, never touching frozen encoder
    loss = out["cma"].square().mean() + out["ct"].square().mean() + out["rdt"].square().mean()
    loss.backward()
    assert all(p.grad is None for p in probe.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in probe.fpn.parameters())
    assert state_digest(probe.encoder) == before
