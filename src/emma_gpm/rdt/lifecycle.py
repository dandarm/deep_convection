"""Actual 15-minute tendencies, causal phases and conservative OT candidates."""
import numpy as np
from scipy import ndimage
from .features import spectral_distance


def lifecycle(now, before, history, event, config):
    known = len(history) >= 3 and not now["border"] and not any(r["border"] for r in history[-3:])
    cooling = now["min_k"]-history[-3]["min_k"] if known else np.nan
    growth_valid = known and event == "continuation" and all(r["event"] == "continuation" for r in history[-2:])
    growth = now["area_km2"]-history[-3]["area_km2"] if growth_valid else np.nan
    contrast = now["mean_k"]-now["min_k"]
    developing = known and growth_valid and contrast >= config.contrast_k and cooling <= config.cooling_15min_k and growth > 0
    inherited = before is not None and before.get("convective", False) and not now["border"] and not before["border"]
    category = 2 if developing else 3 if inherited else 1
    phase_valid = known and (not inherited or developing or cooling > config.cooling_15min_k)
    return dict(class_id=category, convective=category in (2, 3), contrast_k=contrast,
                cooling_k_15min=cooling, growth_km2_15min=growth,
                temporal_valid=known, growth_valid=growth_valid, phase_valid=phase_valid,
                reason="development_criteria" if developing else "inherited_convective" if inherited else
                "temporal_unknown" if not known else "criteria_not_met")


def ot_candidates(ir, features, mask, area, depth, proxy, config, *, convective):
    candidate = np.zeros(mask.shape, bool)
    if not convective or proxy.confidence < config.min_proxy_confidence or proxy.reason != "valid":
        return candidate
    radius = 2*config.ot_local_radius_pixels+1
    local_min = ndimage.minimum_filter(ir, size=radius, mode="nearest")
    local_mean = ndimage.uniform_filter(ir, size=radius, mode="nearest")
    # All six features gate outliers; WV/O3/CO2 explicitly included. This is
    # relative spectral consistency, not a calibrated sign model for O3/CO2.
    coherent = spectral_distance(features, proxy.spectral_centre, proxy.spectral_scale) <= config.ot_spectral_outlier_mad
    raw = mask & (depth > 1) & (proxy.filtered_k-ir >= config.ot_anvil_contrast_k)
    raw &= (ir <= local_min+config.ot_local_contrast_k) & (local_mean-ir >= config.ot_local_contrast_k)
    raw &= (features[0] >= config.ot_btd_062_108_k) & coherent
    components, n = ndimage.label(raw, structure=np.ones((3, 3)))
    for c, bbox in enumerate(ndimage.find_objects(components), 1):
        part = components[bbox] == c
        size = float(area[bbox][part].sum())
        yy, xx = np.where(part)
        axis_ratio = max(np.ptp(yy)+1, np.ptp(xx)+1)/min(np.ptp(yy)+1, np.ptp(xx)+1)
        if config.ot_min_area_km2 <= size <= config.ot_max_area_km2 and axis_ratio <= config.ot_max_axis_ratio:
            candidate[bbox][part] = True
    return candidate
