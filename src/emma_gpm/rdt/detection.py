"""Full-field h-minima/watershed, with explicit closed-support uncertainty.

The multispectral support below is a relative coherence heuristic, not an
operational ice/cloud mask. No absolute brightness-temperature cutoff is used.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import h_minima
from skimage.segmentation import watershed
from .features import robust_spectral_model, spectral_distance

CONNECTIVITY = np.ones((3, 3), dtype=bool)


def _saddles(topology, basins):
    saddle = np.full(int(basins.max())+1, np.inf)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys = slice(0, -dy) if dy else slice(None)
        yt = slice(dy, None) if dy else slice(None)
        xs = slice(0, -dx) if dx > 0 else slice(-dx, None) if dx < 0 else slice(None)
        xt = slice(dx, None) if dx > 0 else slice(0, dx) if dx < 0 else slice(None)
        a, b = basins[ys, xs], basins[yt, xt]
        different = (a > 0) & (b > 0) & (a != b)
        height = np.maximum(topology[ys, xs], topology[yt, xt])[different]
        np.minimum.at(saddle, a[different], height)
        np.minimum.at(saddle, b[different], height)
    return saddle


def detect_cells(ir, features, area, valid, config):
    """Return disjoint full-domain labels and compact object masks."""
    if not valid.any():
        return np.zeros(ir.shape, np.int32), {}
    # Only topology gets smoothing. Invalid observations never become pixels
    # in an object: normalized convolution is confined to the topology copy.
    sigma = config.smoothing_sigma_pixels
    values = np.where(valid, ir, 0.)
    if sigma:
        denominator = ndimage.gaussian_filter(valid.astype(float), sigma)
        topology = ndimage.gaussian_filter(values, sigma)/np.maximum(denominator, 1e-12)
    else:
        topology = values.copy()
    topology[~valid] = float(topology[valid].max()+config.prominence_k+1)
    minima = h_minima(topology, config.prominence_k, footprint=CONNECTIVITY) & valid
    markers, _ = ndimage.label(minima, CONNECTIVITY)
    if not markers.any():
        return np.zeros(ir.shape, np.int32), {}
    basins = watershed(topology, markers, connectivity=CONNECTIVITY, mask=valid)
    saddles = _saddles(topology, basins)
    gy, gx = np.gradient(topology)
    gradient = np.hypot(gy, gx)
    labels = np.zeros(ir.shape, np.int32)
    cells = {}
    invalid_rim = ndimage.binary_dilation(~valid, structure=CONNECTIVITY)
    for marker, bbox in enumerate(ndimage.find_objects(basins), 1):
        if bbox is None:
            continue
        basin = basins[bbox] == marker
        topo = topology[bbox]
        local_ir, local_area = ir[bbox], area[bbox]
        local_features = features[:, bbox[0], bbox[1]]
        seed = np.unravel_index(np.argmin(np.where(basin, topo, np.inf)), basin.shape)
        minimum = float(topo[seed])
        saddle = float(saddles[marker])
        support_score = 1.0
        if np.isfinite(saddle):
            # Immediate colder discrete level relative to this minimum,
            # so uniform temperature translation preserves the geometry.
            steps = np.ceil((saddle-minimum)/config.contour_step_k)-1
            edge = minimum+steps*config.contour_step_k
            edge_method = "pre_merge_saddle"
        else:
            # Isolated tower: strongest CLOSED thermal rim, with relative
            # multispectral agreement to the interior. A domain-edge basin
            # without a resolved rim is rejected, not capped at a fixed K.
            core = basin & (topo <= np.quantile(topo[basin], .5))
            centre, scale = robust_spectral_model(local_features, core, config.spectral_mad_floor_k)
            coherent = spectral_distance(local_features, centre, scale) <= config.spectral_outlier_mad
            maximum = float(topo[basin].max())
            bins = np.floor((topo-minimum)/config.contour_step_k).astype(int)
            best = None
            for step in range(int(np.ceil(config.prominence_k/config.contour_step_k)), int(np.ceil((maximum-minimum)/config.contour_step_k))):
                contour = basin & (bins <= step)
                if contour[0].any() or contour[-1].any() or contour[:, 0].any() or contour[:, -1].any():
                    break
                rim = contour & ~ndimage.binary_erosion(contour, structure=CONNECTIVITY)
                if not rim.any():
                    continue
                coherence = float(coherent[rim].mean())
                strength = float(np.median(gradient[bbox][rim]))
                # Prefer an extended cloud-support rim over a tiny embedded
                # cold-core rim; all areas are geodesic, with an explicit
                # configurable exponent. This support choice needs validation.
                score = strength*coherence*float(local_area[contour].sum())**config.support_area_score_exponent
                if coherence >= .5 and strength >= config.support_gradient_k_pixel and (best is None or score > best[0]):
                    best = (score, step, strength*coherence)
            if best is None:
                continue
            edge = minimum+(best[1]+1)*config.contour_step_k
            support_score = min(1., best[2]/(2*config.support_gradient_k_pixel))
            edge_method = "closed_gradient_spectral_support"
        candidate = basin & (topo < edge)
        connected, _ = ndimage.label(candidate, CONNECTIVITY)
        part = int(connected[seed])
        if not part:
            continue
        mask = connected == part
        physical_area = float(local_area[mask].sum())
        if physical_area < config.min_area_km2 or edge-minimum < config.prominence_k:
            continue
        yy, xx = np.where(mask)
        y0, x0 = bbox[0].start, bbox[1].start
        border = bool((yy+y0 == 0).any() or (yy+y0 == ir.shape[0]-1).any()
                      or (xx+x0 == 0).any() or (xx+x0 == ir.shape[1]-1).any()
                      or invalid_rim[bbox][mask].any())
        component = len(cells)+1
        labels[bbox][mask] = component
        weights = local_area[mask]*np.maximum(edge-local_ir[mask], 1e-6)
        cells[component] = dict(component=component, bbox=bbox, mask=mask,
            area_km2=physical_area, min_k=float(local_ir[mask].min()), mean_k=float(np.average(local_ir[mask], weights=local_area[mask])),
            t_edge_k=edge, edge_method=edge_method, support_confidence=support_score,
            border=border, centroid=np.array([y0+np.average(yy, weights=weights), x0+np.average(xx, weights=weights)]))
    return labels, cells
