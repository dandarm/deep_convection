import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from emma_gpm.videomae import (  # noqa: E402
    build_videomae_small_7ch_from_scratch,
    forward_seviri_mae,
    patchify_videomae_pixels,
)


def test_seven_channel_forward_uses_non_patch_normalized_targets():
    model = build_videomae_small_7ch_from_scratch(
        image_size=32,
        num_frames=4,
        hidden_size=48,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=96,
        decoder_hidden_size=24,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=3,
        decoder_intermediate_size=48,
    )
    pixels = torch.randn(1, 4, 7, 32, 32)
    sequence_length = (4 // 2) * (32 // 16) ** 2
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    assert mask.shape == (1, sequence_length)

    patches = patchify_videomae_pixels(pixels, model.config)
    output = forward_seviri_mae(model, pixels, mask)

    assert patches.shape == (1, 8, 3584)
    assert output.logits.shape == (1, 4, 3584)
    assert torch.isfinite(output.loss)
    assert model.config.norm_pix_loss is False
