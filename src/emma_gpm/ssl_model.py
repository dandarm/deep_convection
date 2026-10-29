"""Shared VideoMAE encoders for MVM, future, joint and advective residual SSL."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import VideoMAEModel

from .videomae import videomae_small_7ch_config, videomae_base_7ch_config
from .videomae_v2 import VideoMAEV2ForPreTraining


def history_config(window, model_size="small", **overrides):
    values = dict(num_frames=window.history_frames, layer_norm_eps=1e-6,
                  videomae_version="v2", decoder_mask_ratio=.5)
    values.update(overrides)
    factories = {"small": videomae_small_7ch_config, "base": videomae_base_7ch_config}
    if model_size not in factories:
        raise ValueError("Expected small or base VideoMAE model size")
    config = factories[model_size](**values)
    if config.num_frames != window.history_frames or config.tubelet_size != 2 or config.norm_pix_loss:
        raise ValueError("SSL history requires two-frame tubelets and global normalization")
    return config


class SpatialForecastHead(nn.Module):
    """Concatenate temporal features at each patch; decode all horizons together."""
    def __init__(self, config, horizons, width=192):
        super().__init__()
        self.time = config.num_frames // config.tubelet_size
        self.grid = config.image_size // config.patch_size
        self.horizons = horizons
        self.channels = config.num_channels
        self.network = nn.Sequential(
            nn.Conv2d(self.time * config.hidden_size, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width, horizons * config.num_channels,
                               config.patch_size, stride=config.patch_size),
        )

    def forward(self, tokens):
        batch, count, dim = tokens.shape
        if count != self.time * self.grid ** 2:
            raise ValueError("Forecast head requires all historical tokens in temporal/spatial order")
        features = tokens.reshape(batch, self.time, self.grid, self.grid, dim)
        features = features.permute(0, 1, 4, 2, 3).reshape(batch, self.time * dim, self.grid, self.grid)
        pixels = self.network(features)
        return pixels.reshape(batch, self.horizons, self.channels, *pixels.shape[-2:])


class SeviriSSL(nn.Module):
    def __init__(self, config, horizons=2, objective="future", head_width=192, freeze_encoder=False):
        super().__init__()
        if objective not in {"mvm", "future", "joint", "residual"}:
            raise ValueError("Unknown SSL objective")
        if freeze_encoder and objective != "future":
            raise ValueError("Frozen common-head probing uses the direct future objective")
        self.config = deepcopy(config)
        self.objective = objective
        self.freeze_encoder = freeze_encoder
        # Construct the canonical encoder first for every objective. Creating
        # an MAE wrapper first can consume a different RNG stream and silently
        # change the starting backbone despite an identical experiment seed.
        encoder = VideoMAEModel(config)
        self.mae = VideoMAEV2ForPreTraining(config) if objective in {"mvm", "joint"} else None
        if self.mae is not None:
            self.mae.videomae = encoder
        self.backbone = None if self.mae is not None else encoder
        # Head initialization must not depend on whether an MAE decoder was
        # constructed first. This also gives every frozen encoder the same head.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() + 101)
            self.head = None if objective == "mvm" else SpatialForecastHead(config, horizons, head_width)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    @property
    def encoder(self):
        return self.mae.videomae if self.mae is not None else self.backbone

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, history):
        if tuple(history.shape[1:]) != (self.config.num_frames, 7, self.config.image_size, self.config.image_size):
            raise ValueError("Encoder accepts only the configured historical frames")
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(history).last_hidden_state
        return self.encoder(history).last_hidden_state

    def forward(self, history, *, mask=None, decode_mask=None, advection=None):
        if history.ndim != 5 or history.shape[1] != self.config.num_frames:
            raise ValueError("Only historical frames may enter SSL forward")
        result = {}
        if self.mae is not None:
            if mask is None:
                raise ValueError("Masked modeling requires an explicit historical mask")
            result["mvm_loss"] = self.mae(history, mask, decode_mask=decode_mask).loss
        if self.head is not None:
            predicted = self.head(self.encode(history))
            if self.objective == "residual":
                if advection is None or advection.shape != predicted.shape:
                    raise ValueError("Residual prediction requires a fixed historical advection baseline")
                result["residual"] = predicted
                predicted = predicted + advection.detach()
            result["prediction"] = predicted
        return result


def ssl_loss(output, target, *, loss="mse", huber_delta=1.0, lambda_mvm=1.0, lambda_future=1.0):
    terms = {}
    if "mvm_loss" in output:
        terms["mvm"] = output["mvm_loss"].float()
    if "prediction" in output:
        prediction = output["prediction"].float()
        if prediction.shape != target.shape:
            raise ValueError("Forecast and target shapes differ")
        if loss == "mse":
            terms["future"] = F.mse_loss(prediction, target.float())
        elif loss == "huber" and huber_delta > 0:
            terms["future"] = F.huber_loss(prediction, target.float(), delta=huber_delta)
        else:
            raise ValueError("Use MSE or Huber with positive delta (normalized units)")
    if not terms or lambda_mvm < 0 or lambda_future < 0:
        raise ValueError("Invalid SSL loss weights")
    total = sum(value * (lambda_mvm if name == "mvm" else lambda_future) for name, value in terms.items())
    if not any((lambda_mvm if name == "mvm" else lambda_future) > 0 for name in terms):
        raise ValueError("At least one active objective must have positive weight")
    return total, terms


def load_encoder_weights(encoder, path):
    """Load an exported HF encoder/MAE or a trusted local atomic checkpoint.

Sixteen-frame fixed sine positions are rebuilt for the shorter historical
prefix. Learned convolution/transformer weights must match strictly.
"""
    path = Path(path)
    if path.is_dir():
        if (path / "encoder").is_dir():
            path = path / "encoder"
        elif (path / "model").is_dir():
            path = path / "model"
        from safetensors.torch import load_file
        if (path / "model.safetensors").exists():
            weight_path = path / "model.safetensors"
            source = load_file(str(weight_path))
        else:
            weight_path = path / "pytorch_model.bin"
            source = torch.load(weight_path, map_location="cpu", weights_only=True)
    else:
        weight_path = path
        source = torch.load(path, map_location="cpu", weights_only=False)
        source = source.get("model", source)
    keys = encoder.state_dict()
    selected = None
    for prefix in ("mae.videomae.", "backbone.", "videomae.", ""):
        candidate = {name[len(prefix):]: value for name, value in source.items() if name.startswith(prefix)}
        if "embeddings.patch_embeddings.projection.weight" in candidate:
            selected = candidate
            break
    if selected is None:
        raise ValueError("No VideoMAE encoder weights in checkpoint")
    # HF exports legacy q_bias/v_bias names, while recent in-memory models use
    # Linear.query/value.bias. Convert only when the destination expects it.
    for name in list(selected):
        translated = name.replace(".q_bias", ".query.bias").replace(".v_bias", ".value.bias")
        if name not in keys and translated in keys:
            if translated in selected:
                raise ValueError(f"Ambiguous attention bias in checkpoint: {translated}")
            selected[translated] = selected.pop(name)
    # HF fixed positions are usually a plain tensor; tolerate a persisted copy
    # only when it is not a learned parameter in this target implementation.
    ignored = {name for name in selected if "position_embeddings" in name and name not in keys}
    selected = {name: value for name, value in selected.items() if name not in ignored}
    encoder.load_state_dict(selected, strict=True)
    hasher = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"source": str(path.resolve()), "weights_sha256": hasher.hexdigest(),
            "history_frames": encoder.config.num_frames, "rebuilt_fixed_positions": sorted(ignored)}


def encoder_source_config(path, window):
    """Preserve source architecture, activation and LayerNorm epsilon for probes."""
    from transformers import VideoMAEConfig

    path = Path(path)
    if path.is_dir():
        if (path / "encoder").is_dir():
            path = path / "encoder"
        elif (path / "model").is_dir():
            path = path / "model"
        values = json.loads((path / "config.json").read_text())
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = payload.get("identity", {})
        values = identity.get("model_config", identity.get("config"))
        if values is None:
            raise ValueError("Checkpoint has no encoder configuration; use a HF export or an atomic training checkpoint")
    config = VideoMAEConfig.from_dict(values)
    if config.num_channels != 7 or config.tubelet_size != 2 or config.image_size != 224 or config.patch_size != 16:
        raise ValueError("Real-data encoder initialization requires the seven-channel SEVIRI 224px/16px/2-frame architecture")
    config.num_frames = window.history_frames
    config.norm_pix_loss = False
    config.videomae_version = "v2"
    config.decoder_mask_ratio = .5
    return config
