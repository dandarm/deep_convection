"""Explicit exploratory choices for the full-domain SEVIRI teacher."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class RDTTeacherConfig:
    version: str = "multispectral_utls_v0"
    smoothing_sigma_pixels: float = 0.7
    prominence_k: float = 6.0
    contour_step_k: float = 0.5
    min_area_km2: float = 100.0
    support_gradient_k_pixel: float = 0.4
    support_area_score_exponent: float = 0.5
    spectral_mad_floor_k: float = 0.75
    spectral_outlier_mad: float = 4.5
    plateau_core_quantile: float = 0.1
    plateau_gradient_quantile: float = 0.6
    plateau_edge_band_k: float = 1.0
    plateau_min_pixels: int = 20
    plateau_min_area_km2: float = 200.0
    plateau_dispersion_scale_k: float = 3.0
    proxy_window_frames: int = 3
    proxy_jump_scale_k: float = 3.0
    depth_min_denominator_k: float = 2.0
    min_proxy_confidence: float = 0.35
    min_aor: float = 0.15
    max_motion_pixels: int = 12
    contrast_k: float = 10.0
    cooling_15min_k: float = -2.5
    ot_btd_062_108_k: float = 1.0
    ot_anvil_contrast_k: float = 3.0
    ot_local_contrast_k: float = 1.0
    ot_local_radius_pixels: int = 3
    ot_spectral_outlier_mad: float = 6.0
    ot_min_area_km2: float = 0.0
    ot_max_area_km2: float = 1000.0
    ot_max_axis_ratio: float = 5.0
    ot_persistence_frames: int = 2

    def __post_init__(self):
        if self.version != "multispectral_utls_v0":
            raise ValueError("Use the existing rdt_teacher module for legacy_fixed_2ch")
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite config parameter: {key}")
        for key in ("prominence_k", "contour_step_k", "min_area_km2", "support_gradient_k_pixel",
                    "spectral_mad_floor_k", "spectral_outlier_mad", "plateau_min_area_km2",
                    "plateau_dispersion_scale_k", "proxy_jump_scale_k", "depth_min_denominator_k",
                    "ot_anvil_contrast_k", "ot_local_contrast_k", "ot_max_area_km2", "ot_max_axis_ratio"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Expected positive {key}")
        for key in ("plateau_min_pixels", "proxy_window_frames", "ot_local_radius_pixels", "ot_persistence_frames"):
            value = getattr(self, key)
            if value < 1 or int(value) != value:
                raise ValueError(f"Expected positive integer {key}")
        if self.max_motion_pixels < 0 or int(self.max_motion_pixels) != self.max_motion_pixels:
            raise ValueError("Invalid motion bound")
        if self.smoothing_sigma_pixels < 0 or self.plateau_edge_band_k < 0 or self.ot_min_area_km2 < 0 or self.support_area_score_exponent < 0:
            raise ValueError("Invalid nonnegative scale")
        if self.ot_min_area_km2 > self.ot_max_area_km2:
            raise ValueError("Inverted OT area bounds")
        if not (0 < self.min_aor <= 1 and 0 < self.plateau_core_quantile < self.plateau_gradient_quantile < 1
                and 0 <= self.min_proxy_confidence <= 1):
            raise ValueError("Invalid fraction")

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
