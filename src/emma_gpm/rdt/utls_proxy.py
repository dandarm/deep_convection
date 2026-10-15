"""SEVIRI-derived UTLS/anvil proxy v0; not a WMO tropopause retrieval."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from .features import robust_spectral_model, spectral_distance


@dataclass(frozen=True)
class ProxyEstimate:
    instant_k: float
    filtered_k: float
    confidence: float
    plateau: np.ndarray
    area_km2: float
    dispersion_k: float
    spectral_centre: np.ndarray
    spectral_scale: np.ndarray
    reason: str


def spectral_correction(features):
    """Zero by design; nonzero correction requires separate calibration."""
    return 0.0


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum()/2)])


def estimate_proxy(ir, features, mask, area, edge, config, *, history=(), tendency=None, support_confidence=1.):
    empty = np.zeros(mask.shape, bool)
    missing = lambda reason: ProxyEstimate(np.nan, np.nan, 0., empty, 0., np.nan, np.full(6, np.nan), np.full(6, np.nan), reason)
    if mask.sum() < config.plateau_min_pixels:
        return missing("insufficient_cell_pixels")
    # A one-pixel-wide cell has no observable 2D plateau gradient in its
    # local support. Abstain explicitly; do not invent a zero derivative
    # or substitute temperatures from neighboring objects.
    if min(ir.shape) < 2:
        return missing("insufficient_spatial_support")
    gy, gx = np.gradient(ir)
    gradient = np.hypot(gy, gx)
    # Retain ties at the quantile: a broad flat anvil must not be discarded
    # wholesale just because the cold core occupies less than one decile.
    central = mask & (ir >= np.quantile(ir[mask], config.plateau_core_quantile)) & (ir < edge-config.plateau_edge_band_k)
    if central.sum() < config.plateau_min_pixels:
        return missing("no_central_plateau")
    centre, scale = robust_spectral_model(features, central, config.spectral_mad_floor_k)
    distance = spectral_distance(features, centre, scale)
    plateau = central & (gradient <= np.quantile(gradient[central], config.plateau_gradient_quantile)) & (distance <= config.spectral_outlier_mad)
    if tendency is not None:
        finite = np.isfinite(tendency) & plateau
        if finite.sum() >= config.plateau_min_pixels:
            plateau &= np.isfinite(tendency) & (np.abs(tendency) <= np.quantile(np.abs(tendency[finite]), config.plateau_gradient_quantile))
    components, n = ndimage.label(plateau, structure=np.ones((3, 3)))
    if not n:
        return missing("no_coherent_plateau")
    sizes = np.bincount(components.ravel(), weights=area.ravel(), minlength=n+1)
    sizes[0] = 0.
    plateau = components == int(sizes.argmax())
    physical_area = float(area[plateau].sum())
    if plateau.sum() < config.plateau_min_pixels or physical_area < config.plateau_min_area_km2:
        return missing("insufficient_connected_plateau")
    weights = 1/(1+gradient[plateau])
    if tendency is not None:
        weights /= 1+np.where(np.isfinite(tendency[plateau]), np.abs(tendency[plateau]), 0.)
    # Bound non-area reliability weights: geodesic pixel size is retained.
    weights = np.clip(weights, .1, 1.)*area[plateau]
    instant = weighted_median(ir[plateau], weights)+spectral_correction(features[:, plateau])
    past = [v for v in history if np.isfinite(v)][-(config.proxy_window_frames-1):] if config.proxy_window_frames > 1 else []
    filtered = float(np.median([*past, instant]))
    dispersion = float(1.4826*np.median(np.abs(ir[plateau]-instant)))
    spectral_spread = float(np.median(np.mean(np.abs(features[:, plateau]-centre[:, None])/config.spectral_mad_floor_k, axis=0)))
    confidence = min(1., physical_area/(2*config.plateau_min_area_km2))*min(1., plateau.sum()/(2*config.plateau_min_pixels))
    confidence /= (1+dispersion/config.plateau_dispersion_scale_k)*(1+spectral_spread)
    if past:
        confidence /= 1+abs(instant-past[-1])/config.proxy_jump_scale_k
    denominator = edge-filtered
    confidence *= support_confidence*min(1., max(0., denominator)/(2*config.depth_min_denominator_k))
    reason = "valid" if denominator >= config.depth_min_denominator_k else "unstable_depth_denominator"
    if reason != "valid":
        confidence = 0.
    return ProxyEstimate(instant, filtered, float(np.clip(confidence, 0, 1)), plateau, physical_area, dispersion, centre, scale, reason)


def thermal_depth(ir, mask, edge, proxy, config):
    depth = np.full(ir.shape, np.nan, np.float32)
    denominator = edge-proxy
    if np.isfinite(denominator) and denominator >= config.depth_min_denominator_k:
        depth[mask] = (edge-ir[mask])/denominator
    return depth
