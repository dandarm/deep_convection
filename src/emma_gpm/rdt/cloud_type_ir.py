"""Causal SEVIRI thermal-IR cloud type used before RDT segmentation.

The operational RDT-CW processor consumes the external NWC SAF Cloud Type
product to exclude clear pixels before the IR10.8 contour hierarchy is built.
This module supplies the corresponding *pure SEVIRI* ancillary: it uses only
the seven thermal/WV channels already present in the VideoMAE cube and never
uses NWP, visible imagery, a land mask, or a numerical weather model.

The classification is intentionally conservative.  Deep/high opaque cloud is
identified from mutually independent upper-tropospheric channel contrasts.
Warm opaque cloud additionally requires a causal per-pixel clear-sky surface
reference, because a single thermal image cannot in general distinguish a
warm cloud from a cold surface.  The reference is initialized from the first
observation, updated only where the multispectral signature is surface-like,
and otherwise relaxes slowly.  Consequently, cold clear land is classified as
clear immediately, while warm-cloud retrieval remains explicitly dependent on
past clear observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


CLOUD_TYPE_NAMES = {
    0: "clear_or_unclassified",
    1: "low_warm_opaque",
    2: "medium_opaque",
    3: "high_opaque",
    4: "very_high_opaque",
    5: "semitransparent_cirrus",
}


@dataclass(frozen=True)
class IrCloudTypeConfig:
    """Explicit thresholds for the seven-channel thermal-IR CT proxy."""

    version: str = "seviri_ir_cloud_type_v1"
    expected_cadence_minutes: float = 5.0

    # A clear surface is seen strongly by the window channels but weakly by
    # upper-tropospheric WV/CO2/O3 channels.  At least three independent tests
    # must agree; no single BTD can declare a pixel clear.
    clear_wv062_ir108_max_k: float = -20.0
    clear_wv073_ir108_max_k: float = -7.0
    clear_ir097_ir108_max_k: float = -15.0
    clear_ir134_ir108_max_k: float = -12.0
    clear_votes_required: int = 3

    # Cloud-like spectral tests.  They deliberately sit on the cloudy side of
    # the observed separation between winter Sahara and opaque high cloud.
    cloud_wv062_ir108_min_k: float = -18.0
    cloud_wv073_ir108_min_k: float = -9.0
    cloud_ir087_ir108_min_k: float = -2.0
    cloud_ir097_ir108_min_k: float = -14.0
    cloud_ir134_ir108_min_k: float = -11.5
    cold_cloud_votes_required: int = 3
    warm_cloud_votes_required: int = 2

    # Causal clear-sky surface reference.  Clear-like observations update it
    # relatively quickly; under cloud it may cool only slowly, preventing the
    # cloud top itself from becoming the estimated surface.
    surface_clear_update_alpha: float = 0.35
    surface_cloud_relaxation_k_per_hour: float = 0.25
    warm_cloud_surface_depression_k: float = 6.0

    # Temperature categories follow the Cloud Type bins used by the local
    # NWC SAF reference implementation.
    very_high_limit_k: float = 220.15
    high_limit_k: float = 240.15
    medium_limit_k: float = 260.15
    warm_detection_limit_k: float = 278.15

    # Split-window semitransparent-cirrus curve.
    cirrus_min_ir108_k: float = 228.15
    cirrus_max_ir108_k: float = 268.15
    cirrus_split_base_k: float = 2.2
    cirrus_split_slope: float = 0.025
    cirrus_wv062_ir108_max_k: float = -11.0
    cirrus_ir087_ir108_min_k: float = -1.5

    def __post_init__(self) -> None:
        if self.version != "seviri_ir_cloud_type_v1":
            raise ValueError("Unsupported IR cloud-type version")
        if self.expected_cadence_minutes <= 0:
            raise ValueError("expected_cadence_minutes must be positive")
        if not 0 < self.surface_clear_update_alpha <= 1:
            raise ValueError("surface_clear_update_alpha must be in (0, 1]")
        if self.surface_cloud_relaxation_k_per_hour < 0:
            raise ValueError("surface relaxation must be nonnegative")
        if not 1 <= self.clear_votes_required <= 4:
            raise ValueError("clear_votes_required must be between 1 and 4")
        if not 1 <= self.cold_cloud_votes_required <= 5:
            raise ValueError("cold_cloud_votes_required must be between 1 and 5")
        if not 1 <= self.warm_cloud_votes_required <= 5:
            raise ValueError("warm_cloud_votes_required must be between 1 and 5")
        if not (
            self.very_high_limit_k
            < self.high_limit_k
            < self.medium_limit_k
            < self.warm_detection_limit_k
        ):
            raise ValueError("Cloud-type temperature limits must be increasing")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class IrCloudTypeResult:
    ct_class: np.ndarray
    opaque_cloud_mask: np.ndarray
    thin_cirrus_mask: np.ndarray
    surface_like_mask: np.ndarray
    surface_reference_k: np.ndarray
    surface_depression_k: np.ndarray
    spectral_cloud_votes: np.ndarray


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A finite timestamp is required")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


class SeviriIrCloudTypeTracker:
    """Stateful, causal seven-channel Cloud Type proxy."""

    def __init__(self, config: IrCloudTypeConfig | None = None) -> None:
        self.config = config or IrCloudTypeConfig()
        self.reset()

    def reset(self) -> None:
        self.surface_reference_k: np.ndarray | None = None
        self.previous_time: pd.Timestamp | None = None

    def _validate_time(self, timestamp: Any) -> pd.Timestamp:
        stamp = _normalise_timestamp(timestamp)
        if self.previous_time is not None:
            expected = pd.to_timedelta(self.config.expected_cadence_minutes, unit="m")
            if stamp - self.previous_time != expected:
                raise ValueError(
                    "IR cloud type requires an exact, strictly increasing cadence; "
                    f"received {self.previous_time.isoformat()} -> {stamp.isoformat()}"
                )
        return stamp

    def step(self, frame_7ch: np.ndarray, timestamp: Any) -> IrCloudTypeResult:
        stamp = self._validate_time(timestamp)
        frame = np.asarray(frame_7ch, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[0] != 7:
            raise ValueError(f"Expected frame shape (7,H,W), got {frame.shape}")

        cfg = self.config
        wv062, wv073, ir087, ir097, ir108, ir120, ir134 = frame
        finite = np.isfinite(frame).all(axis=0) & ((frame >= 100.0) & (frame <= 400.0)).all(axis=0)

        btd_062_108 = wv062 - ir108
        btd_073_108 = wv073 - ir108
        btd_087_108 = ir087 - ir108
        btd_097_108 = ir097 - ir108
        btd_108_120 = ir108 - ir120
        btd_134_108 = ir134 - ir108

        clear_votes = (
            (btd_062_108 <= cfg.clear_wv062_ir108_max_k).astype(np.uint8)
            + (btd_073_108 <= cfg.clear_wv073_ir108_max_k).astype(np.uint8)
            + (btd_097_108 <= cfg.clear_ir097_ir108_max_k).astype(np.uint8)
            + (btd_134_108 <= cfg.clear_ir134_ir108_max_k).astype(np.uint8)
        )
        surface_like = finite & (clear_votes >= cfg.clear_votes_required)

        cloud_votes = (
            (btd_062_108 >= cfg.cloud_wv062_ir108_min_k).astype(np.uint8)
            + (btd_073_108 >= cfg.cloud_wv073_ir108_min_k).astype(np.uint8)
            + (btd_087_108 >= cfg.cloud_ir087_ir108_min_k).astype(np.uint8)
            + (btd_097_108 >= cfg.cloud_ir097_ir108_min_k).astype(np.uint8)
            + (btd_134_108 >= cfg.cloud_ir134_ir108_min_k).astype(np.uint8)
        )

        if self.surface_reference_k is None:
            surface_reference = np.where(finite, ir108, np.nan).astype(np.float32)
        else:
            if self.surface_reference_k.shape != ir108.shape:
                raise ValueError("IR cloud-type grid shape changed within one sequence")
            relaxation = (
                cfg.surface_cloud_relaxation_k_per_hour
                * cfg.expected_cadence_minutes
                / 60.0
            )
            surface_reference = np.where(
                np.isfinite(self.surface_reference_k),
                self.surface_reference_k - relaxation,
                ir108,
            ).astype(np.float32)
            clear_update = (
                (1.0 - cfg.surface_clear_update_alpha) * surface_reference
                + cfg.surface_clear_update_alpha * ir108
            )
            surface_reference = np.where(surface_like, clear_update, surface_reference)

        surface_depression = surface_reference - ir108
        cold_cloud_like = finite & (~surface_like) & (
            cloud_votes >= cfg.cold_cloud_votes_required
        )
        warm_cloud_like = finite & (~surface_like) & (
            cloud_votes >= cfg.warm_cloud_votes_required
        ) & (surface_depression >= cfg.warm_cloud_surface_depression_k)

        cirrus_curve = cfg.cirrus_split_base_k + cfg.cirrus_split_slope * np.clip(
            ir108 - 230.0, 0.0, 45.0
        )
        thin_cirrus = (
            finite
            & (~surface_like)
            & (ir108 > cfg.cirrus_min_ir108_k)
            & (ir108 < cfg.cirrus_max_ir108_k)
            & (btd_108_120 > cirrus_curve)
            & (btd_062_108 < cfg.cirrus_wv062_ir108_max_k)
            & (btd_087_108 > cfg.cirrus_ir087_ir108_min_k)
        )

        opaque = (
            (
                (ir108 < cfg.medium_limit_k)
                & cold_cloud_like
            )
            | (
                (ir108 >= cfg.medium_limit_k)
                & (ir108 < cfg.warm_detection_limit_k)
                & warm_cloud_like
            )
        ) & (~thin_cirrus)

        ct_class = np.zeros(ir108.shape, dtype=np.uint8)
        ct_class[opaque & (ir108 >= cfg.medium_limit_k)] = 1
        ct_class[
            opaque & (ir108 >= cfg.high_limit_k) & (ir108 < cfg.medium_limit_k)
        ] = 2
        ct_class[
            opaque & (ir108 >= cfg.very_high_limit_k) & (ir108 < cfg.high_limit_k)
        ] = 3
        ct_class[opaque & (ir108 < cfg.very_high_limit_k)] = 4
        ct_class[thin_cirrus] = 5

        self.surface_reference_k = surface_reference.copy()
        self.previous_time = stamp
        return IrCloudTypeResult(
            ct_class=ct_class,
            opaque_cloud_mask=(ct_class >= 1) & (ct_class <= 4),
            thin_cirrus_mask=thin_cirrus,
            surface_like_mask=surface_like,
            surface_reference_k=surface_reference,
            surface_depression_k=surface_depression.astype(np.float32),
            spectral_cloud_votes=cloud_votes,
        )


__all__ = [
    "CLOUD_TYPE_NAMES",
    "IrCloudTypeConfig",
    "IrCloudTypeResult",
    "SeviriIrCloudTypeTracker",
]
