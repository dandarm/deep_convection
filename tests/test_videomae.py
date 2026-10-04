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


def test_unpatchify_roundtrip_and_masked_reconstruction_keep_visible_pixels():
    from emma_gpm.videomae import (
        videomae_small_7ch_config, unpatchify_videomae_pixels, restore_masked_videomae_pixels,
    )
    config = videomae_small_7ch_config(image_size=32, num_frames=4)
    pixels = torch.arange(4 * 7 * 32 * 32, dtype=torch.float32).reshape(1, 4, 7, 32, 32)
    patches = patchify_videomae_pixels(pixels, config)
    assert torch.equal(unpatchify_videomae_pixels(patches, config), pixels)
    mask = torch.tensor([[True, False, False, True] * 2])
    logits = patches[mask].reshape(1, 4, -1) + 10
    reconstructed, pixel_mask = restore_masked_videomae_pixels(pixels, logits, mask, config)
    expected = pixels + pixel_mask.expand_as(pixels).float() * 10
    assert torch.equal(reconstructed, expected)
    assert torch.equal(pixel_mask[:, 0], pixel_mask[:, 1])
