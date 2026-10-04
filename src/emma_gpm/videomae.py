"""Seven-channel VideoMAE construction and RGB checkpoint adaptation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .seviri import THERMAL_CHANNELS


NUM_SEVIRI_CHANNELS = len(THERMAL_CHANNELS)


def _transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import VideoMAEConfig, VideoMAEForPreTraining, VideoMAEModel
    except ImportError as error:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "Install requirements-training.txt to construct VideoMAE models."
        ) from error
    return VideoMAEConfig, VideoMAEForPreTraining, VideoMAEModel


def videomae_small_7ch_config(**overrides: Any) -> Any:
    """Return a ViT-S VideoMAE configuration for seven SEVIRI channels."""

    VideoMAEConfig, _, _ = _transformers()
    values: dict[str, Any] = {
        "image_size": 224,
        "patch_size": 16,
        "num_channels": NUM_SEVIRI_CHANNELS,
        "num_frames": 16,
        "tubelet_size": 2,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "decoder_hidden_size": 192,
        "decoder_num_hidden_layers": 4,
        "decoder_num_attention_heads": 3,
        "decoder_intermediate_size": 768,
        # Absolute brightness temperature and inter-channel differences carry
        # physical information.  One shared affine transform is fitted on the
        # training split; there is no per-channel, per-clip, or per-patch norm.
        "norm_pix_loss": False,
    }
    values.update(overrides)
    if int(values["num_channels"]) != NUM_SEVIRI_CHANNELS:
        raise ValueError(
            f"This project requires {NUM_SEVIRI_CHANNELS} channels: {THERMAL_CHANNELS}."
        )
    return VideoMAEConfig(**values)


def build_videomae_small_7ch_from_scratch(**config_overrides: Any) -> Any:
    """Instantiate a randomly initialized seven-channel VideoMAE-Small MAE."""

    _, VideoMAEForPreTraining, _ = _transformers()
    return VideoMAEForPreTraining(videomae_small_7ch_config(**config_overrides))


def reconstruction_values_per_tubelet(config: Any) -> int:
    patch_size = config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_height, patch_width = int(patch_size[0]), int(patch_size[1])
    else:
        patch_height = patch_width = int(patch_size)
    return (
        int(config.num_channels)
        * int(config.tubelet_size)
        * patch_height
        * patch_width
    )


def patchify_videomae_pixels(pixel_values: Any, config: Any) -> Any:
    """Convert ``(B,T,C,H,W)`` inputs to VideoMAE reconstruction targets."""

    batch, time, channels, height, width = pixel_values.shape
    patch_size = config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_height, patch_width = int(patch_size[0]), int(patch_size[1])
    else:
        patch_height = patch_width = int(patch_size)
    tubelet = int(config.tubelet_size)
    if channels != int(config.num_channels):
        raise ValueError(f"Input has {channels} channels; model expects {config.num_channels}.")
    if time % tubelet or height % patch_height or width % patch_width:
        raise ValueError("Input dimensions must be divisible by tubelet and patch sizes.")
    patches = pixel_values.view(
        batch,
        time // tubelet,
        tubelet,
        channels,
        height // patch_height,
        patch_height,
        width // patch_width,
        patch_width,
    )
    patches = patches.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    return patches.view(
        batch,
        (time // tubelet) * (height // patch_height) * (width // patch_width),
        tubelet * patch_height * patch_width * channels,
    )


def forward_seviri_mae(model: Any, pixel_values: Any, bool_masked_pos: Any, decode_mask: Any = None) -> Any:
    """Run VideoMAE with physically appropriate seven-channel targets.

    Transformers 5.18 raises for non-RGB data when ``norm_pix_loss=False``
    despite its error text recommending that setting.  We let the upstream
    model produce logits with its supported path, then replace its RGB-oriented
    loss with MSE on globally standardized, non-patch-normalized SEVIRI
    targets.  This also keeps the intended behavior stable if upstream changes.
    """

    if getattr(model.config, 'videomae_version', 'v1') == 'v2':
        return model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, decode_mask=decode_mask)
    if decode_mask is not None:
        raise ValueError('Decoder masking requires VideoMAE V2')
    import torch.nn.functional as functional

    requested_norm_pix_loss = bool(model.config.norm_pix_loss)
    if int(model.config.num_channels) != NUM_SEVIRI_CHANNELS:
        raise ValueError(
            f"SEVIRI MAE requires {NUM_SEVIRI_CHANNELS} channels; "
            f"received {model.config.num_channels}."
        )
    try:
        # The logits do not depend on target normalization.  This flag only
        # selects the upstream label-construction branch.
        model.config.norm_pix_loss = True
        output = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
    finally:
        model.config.norm_pix_loss = requested_norm_pix_loss

    patches = patchify_videomae_pixels(pixel_values, model.config)
    batch = patches.shape[0]
    labels = patches[bool_masked_pos].reshape(
        batch, -1, reconstruction_values_per_tubelet(model.config)
    )
    output.loss = functional.mse_loss(output.logits, labels)
    return output


def unpatchify_videomae_pixels(patches: Any, config: Any) -> Any:
    """Invert patchify, retaining the exact temporal, spatial and channel order."""
    size = config.patch_size
    ph, pw = (map(int, size) if isinstance(size, (tuple, list)) else (int(size), int(size)))
    height = width = int(config.image_size)
    time, channels, tubelet = int(config.num_frames), int(config.num_channels), int(config.tubelet_size)
    batch = patches.shape[0]
    expected = ((time // tubelet) * (height // ph) * (width // pw), channels * tubelet * ph * pw)
    if tuple(patches.shape[1:]) != expected:
        raise ValueError(f"Unexpected patch shape {tuple(patches.shape)}; expected {expected}.")
    values = patches.reshape(batch, time // tubelet, height // ph, width // pw, tubelet, ph, pw, channels)
    return values.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous().reshape(batch, time, channels, height, width)


def restore_masked_videomae_pixels(pixel_values: Any, logits: Any, mask: Any, config: Any) -> tuple[Any, Any]:
    """Fill only masked tubelets with predictions; return reconstruction and pixel mask.

    Visible pixels are copied from the input and must never count as predicted
    pixels when evaluating reconstruction error.
    """
    patches = patchify_videomae_pixels(pixel_values, config).clone()
    patches[mask] = logits.reshape(-1, reconstruction_values_per_tubelet(config))
    reconstruction = unpatchify_videomae_pixels(patches, config)
    ph, pw = (map(int, config.patch_size) if isinstance(config.patch_size, (tuple, list))
              else (int(config.patch_size), int(config.patch_size)))
    batch, time, _, height, width = pixel_values.shape
    pixel_mask = mask.reshape(batch, time // int(config.tubelet_size), height // ph, width // pw)
    pixel_mask = pixel_mask.repeat_interleave(int(config.tubelet_size), 1).repeat_interleave(ph, 2).repeat_interleave(pw, 3)
    return reconstruction, pixel_mask.unsqueeze(2)


def inflate_rgb_patch_embedding(rgb_weight: Any, target_channels: int = 7) -> Any:
    """Inflate a 3-channel Conv3d kernel while preserving equal-input response."""

    if rgb_weight.ndim != 5 or rgb_weight.shape[1] != 3:
        raise ValueError(
            "Expected RGB Conv3d weights shaped (out_channels,3,t,h,w); "
            f"received {tuple(rgb_weight.shape)}."
        )
    if target_channels < 1:
        raise ValueError("target_channels must be positive.")
    rgb_mean = rgb_weight.mean(dim=1, keepdim=True)
    return rgb_mean.repeat(1, target_channels, 1, 1, 1) * (3.0 / target_channels)


def build_videomae_7ch_from_rgb_checkpoint(checkpoint: str) -> Any:
    """Create a seven-channel MAE and retain all compatible RGB encoder weights.

    The RGB patch projection is mean-inflated and scaled by ``3/7``.  The MAE
    decoder is intentionally initialized anew because its prediction head must
    emit 3584 rather than 1536 values per tubelet.
    """

    _, VideoMAEForPreTraining, VideoMAEModel = _transformers()
    rgb_encoder = VideoMAEModel.from_pretrained(checkpoint)
    if int(rgb_encoder.config.num_channels) != 3:
        raise ValueError(
            f"Checkpoint {checkpoint!r} has {rgb_encoder.config.num_channels} channels; "
            "RGB adaptation requires exactly three."
        )

    target_config = deepcopy(rgb_encoder.config)
    target_config.num_channels = NUM_SEVIRI_CHANNELS
    target = VideoMAEForPreTraining(target_config)

    source_state = rgb_encoder.state_dict()
    target_state = target.videomae.state_dict()
    projection_weight = "embeddings.patch_embeddings.projection.weight"
    compatible = {
        name: value
        for name, value in source_state.items()
        if name != projection_weight
        and name in target_state
        and target_state[name].shape == value.shape
    }
    target.videomae.load_state_dict(compatible, strict=False)

    source_projection = rgb_encoder.embeddings.patch_embeddings.projection
    target_projection = target.videomae.embeddings.patch_embeddings.projection
    inflated = inflate_rgb_patch_embedding(
        source_projection.weight.detach(), NUM_SEVIRI_CHANNELS
    )
    with __import__("torch").no_grad():
        target_projection.weight.copy_(inflated)
        if source_projection.bias is not None and target_projection.bias is not None:
            target_projection.bias.copy_(source_projection.bias)

    expected = NUM_SEVIRI_CHANNELS * int(target_config.tubelet_size) * 16 * 16
    actual = reconstruction_values_per_tubelet(target_config)
    if actual != expected:
        raise RuntimeError(f"Unexpected reconstruction width {actual}; expected {expected}.")
    return target
