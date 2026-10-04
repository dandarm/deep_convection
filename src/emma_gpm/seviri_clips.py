"""Strict seven-channel SEVIRI clip loading for VideoMAE.

The regional Zarr archive is stored as ``(time, channel, y, x)``.  VideoMAE's
public Hugging Face interface expects one sample as ``(time, channel, y, x)``
and a batch as ``(batch, time, channel, y, x)``.  This module keeps that
boundary explicit and refuses incomplete or reordered spectral cubes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .seviri import THERMAL_CHANNELS


VIDEO_MAE_CHANNELS = THERMAL_CHANNELS
VIDEO_MAE_NUM_CHANNELS = len(VIDEO_MAE_CHANNELS)


class IncompleteSeviriClipError(ValueError):
    """Raised when a requested clip has missing times, channels, or pixels."""


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def normalize_seviri_clip(
    clip_tchw: np.ndarray,
    global_mean: float,
    global_std: float,
) -> np.ndarray:
    """Apply one training-split affine transform to every pixel and channel.

    A shared scalar mean and standard deviation preserve absolute brightness-
    temperature offsets and spectral differences.  Per-channel and per-clip
    standardization are intentionally rejected.
    """

    mean = np.asarray(global_mean, dtype=np.float32)
    std = np.asarray(global_std, dtype=np.float32)
    if mean.ndim != 0 or std.ndim != 0:
        raise ValueError("Global normalization mean and std must be scalars.")
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Global normalization statistics must be finite.")
    if std <= 0:
        raise ValueError("Global normalization standard deviation must be positive.")
    return (clip_tchw.astype(np.float32, copy=False) - mean) / std


class SeviriZarrClipReader:
    """Read complete north-up, west-left crops from daily Zarr stores.

    ``origin_y`` and ``origin_x`` refer to native Zarr indices.  The returned
    tensor is reoriented when necessary so row zero is north and column zero
    is west, independently of the coordinate order used on disk.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        num_frames: int = 16,
        cadence_minutes: int = 5,
        crop_size: int = 224,
        channels: Sequence[str] = VIDEO_MAE_CHANNELS,
        require_finite: bool = True,
    ) -> None:
        self.root = Path(root)
        self.num_frames = int(num_frames)
        self.cadence_minutes = int(cadence_minutes)
        self.crop_size = int(crop_size)
        self.channels = tuple(channels)
        self.require_finite = bool(require_finite)
        self._stores: dict[Path, xr.Dataset] = {}

        if self.channels != VIDEO_MAE_CHANNELS:
            raise ValueError(
                "VideoMAE channel order must be exactly "
                f"{VIDEO_MAE_CHANNELS}; received {self.channels}."
            )
        if self.num_frames < 1 or self.cadence_minutes < 1 or self.crop_size < 1:
            raise ValueError("num_frames, cadence_minutes, and crop_size must be positive.")

    def __enter__(self) -> "SeviriZarrClipReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for dataset in self._stores.values():
            dataset.close()
        self._stores.clear()

    def _path_for_time(self, timestamp: pd.Timestamp) -> Path:
        return self.root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr"

    def _open_store(self, path: Path) -> xr.Dataset:
        if path not in self._stores:
            if not path.exists():
                raise IncompleteSeviriClipError(f"Missing daily SEVIRI store: {path}")
            # Let Zarr fetch the selected chunks directly. Dask graphs for the
            # entire daily array add substantial overhead to these small crops.
            dataset = xr.open_zarr(path, consolidated=True, chunks=None).sortby("time")
            if "seviri" not in dataset:
                dataset.close()
                raise IncompleteSeviriClipError(f"No 'seviri' array in {path}")
            actual_channels = tuple(str(value) for value in dataset.channel.values)
            if actual_channels != self.channels:
                dataset.close()
                raise IncompleteSeviriClipError(
                    f"Unexpected channel order in {path}: {actual_channels}; "
                    f"expected {self.channels}."
                )
            self._stores[path] = dataset
        return self._stores[path]

    def load(
        self,
        start_time: Any,
        *,
        origin_y: int,
        origin_x: int,
        global_mean: float | None = None,
        global_std: float | None = None,
    ) -> np.ndarray:
        """Return a VideoMAE sample in ``(T,C,H,W)`` order.

        Clips may cross midnight.  Every requested RSS time must exist exactly;
        no interpolation, frame repetition, or spectral substitution is used.
        """

        start = _utc_naive(start_time)
        expected_times = pd.date_range(
            start,
            periods=self.num_frames,
            freq=f"{self.cadence_minutes}min",
        )
        y0, x0 = int(origin_y), int(origin_x)
        y1, x1 = y0 + self.crop_size, x0 + self.crop_size
        if y0 < 0 or x0 < 0:
            raise IncompleteSeviriClipError("Crop origins must be non-negative.")

        pieces: list[np.ndarray] = []
        for path, times in pd.Series(expected_times, index=expected_times).groupby(
            expected_times.normalize(), sort=False
        ):
            day_times = pd.DatetimeIndex(times.to_numpy())
            store_path = self._path_for_time(pd.Timestamp(path))
            dataset = self._open_store(store_path)
            available = pd.DatetimeIndex(dataset.time.values)
            indices = available.get_indexer(day_times)
            if np.any(indices < 0):
                missing = day_times[indices < 0]
                raise IncompleteSeviriClipError(
                    f"Missing RSS frames in {store_path}: "
                    + ", ".join(value.isoformat() for value in missing)
                )
            height = int(dataset.sizes["y"])
            width = int(dataset.sizes["x"])
            if y1 > height or x1 > width:
                raise IncompleteSeviriClipError(
                    f"Crop [{y0}:{y1}, {x0}:{x1}] exceeds store shape "
                    f"({height}, {width}) in {store_path}."
                )
            piece = (
                dataset["seviri"]
                .isel(time=indices, y=slice(y0, y1), x=slice(x0, x1))
                .transpose("time", "channel", "y", "x")
                .values
            )
            selected_y = np.asarray(dataset.y.values[y0:y1])
            selected_x = np.asarray(dataset.x.values[x0:x1])
            y_difference = np.diff(selected_y)
            x_difference = np.diff(selected_x)
            if not (np.all(y_difference > 0) or np.all(y_difference < 0)):
                raise IncompleteSeviriClipError(
                    f"Non-monotonic y coordinates in {store_path}."
                )
            if not (np.all(x_difference > 0) or np.all(x_difference < 0)):
                raise IncompleteSeviriClipError(
                    f"Non-monotonic x coordinates in {store_path}."
                )
            # Image convention: north at the top and west on the left.
            if selected_y[0] < selected_y[-1]:
                piece = piece[..., ::-1, :]
            if selected_x[0] > selected_x[-1]:
                piece = piece[..., ::-1]
            pieces.append(np.asarray(piece, dtype=np.float32))

        clip = np.concatenate(pieces, axis=0)
        expected_shape = (
            self.num_frames,
            VIDEO_MAE_NUM_CHANNELS,
            self.crop_size,
            self.crop_size,
        )
        if clip.shape != expected_shape:
            raise IncompleteSeviriClipError(
                f"Loaded clip shape {clip.shape}; expected {expected_shape}."
            )
        if self.require_finite and not np.all(np.isfinite(clip)):
            raise IncompleteSeviriClipError("Clip contains NaN or infinite values.")
        if (global_mean is None) != (global_std is None):
            raise ValueError("global_mean and global_std must be provided together.")
        if global_mean is not None and global_std is not None:
            clip = normalize_seviri_clip(clip, global_mean, global_std)
        return clip


class SeviriVideoMAEDataset:
    """Minimal PyTorch-compatible dataset backed by a clip manifest.

    The manifest must contain ``start_time``, ``origin_y``, and ``origin_x``.
    PyTorch is imported lazily so catalogue and preprocessing tools remain
    usable in environments that do not have the training stack installed.
    """

    REQUIRED_COLUMNS = ("start_time", "origin_y", "origin_x")

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        reader: SeviriZarrClipReader,
        *,
        global_mean: float,
        global_std: float,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
        else:
            frame = pd.read_csv(manifest)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame]
        if missing:
            raise ValueError(f"Clip manifest is missing columns: {missing}")
        self.manifest = frame.reset_index(drop=True)
        self.reader = reader
        self.global_mean = float(global_mean)
        self.global_std = float(global_std)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - training dependency
            raise RuntimeError(
                "Install requirements-training.txt to use SeviriVideoMAEDataset."
            ) from error
        row = self.manifest.iloc[index]
        clip = self.reader.load(
            row.start_time,
            origin_y=int(row.origin_y),
            origin_x=int(row.origin_x),
            global_mean=self.global_mean,
            global_std=self.global_std,
        )
        return {"pixel_values": torch.from_numpy(np.ascontiguousarray(clip))}


class MaterializedSeviriDataset:
    """Load losslessly materialized, north-up Kelvin samples in manifest order."""

    def __init__(self, root, manifest_path, *, zarr_root, global_mean, global_std):
        self.root = Path(root)
        metadata = json.loads((self.root / "metadata.json").read_text())
        manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if metadata.get("manifest_sha256") != manifest_hash:
            raise ValueError("Materialized samples belong to a different manifest.")
        expected = dict(channels=list(VIDEO_MAE_CHANNELS), shape=[16, 7, 224, 224],
                        dtype="float16", units="K", orientation="north-up west-left",
                        cadence_minutes=5, lossless=True)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Unexpected materialized sample metadata: {key}")
        if Path(metadata["zarr_root"]).resolve() != Path(zarr_root).resolve():
            raise ValueError("Materialized samples belong to a different Zarr root.")
        self.samples = metadata["samples"]
        if len(self.samples) != len(pd.read_csv(manifest_path)):
            raise ValueError("Materialized sample count does not match the manifest.")
        for index, sample in enumerate(self.samples):
            expected_name = f"sample_{index:06d}.npy"
            path = self.root / expected_name
            if sample["file"] != expected_name or path.stat().st_size != sample["bytes"]:
                raise ValueError(f"Missing or truncated sample: {expected_name}")
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        clip = np.load(self.root / self.samples[index]["file"], allow_pickle=False)
        if clip.shape != (16, 7, 224, 224) or clip.dtype != np.float16 or not clip.flags.c_contiguous:
            raise IncompleteSeviriClipError("Unexpected materialized sample shape or dtype.")
        if not np.isfinite(clip).all():
            raise IncompleteSeviriClipError("Materialized clip contains non-finite values.")
        clip = normalize_seviri_clip(clip, self.global_mean, self.global_std)
        return {"pixel_values": torch.from_numpy(clip)}
