"""Primitive SEVIRI observables for evaluation only, never pretraining labels.

Temporal derivatives are Eulerian (fixed pixels), not tracked cloud cooling.
Threshold areas are fractions of pixels, not geographic areas or cloud classes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .seviri import THERMAL_CHANNELS


def linear_tendency(values, cadence_minutes=5.0):
    """OLS tendency per 15 minutes, preserving its sign and physical units."""
    values = np.asarray(values, dtype=np.float64)
    time = np.arange(len(values), dtype=np.float64) * cadence_minutes
    time -= time.mean()
    return float(15 * np.dot(time, values) / np.dot(time, time))


def describe_clip(clip):
    """Return scalar observables and a unit/group/formula dictionary.

    Input must be the canonical raw north-up Kelvin (16,7,224,224) cube.
    No input normalization is performed. No object tracking/classification.
    """
    clip = np.asarray(clip)
    if clip.shape != (16, 7, 224, 224) or not np.isfinite(clip).all():
        raise ValueError('Expected a finite canonical (16,7,224,224) Kelvin clip')
    clip = clip.astype(np.float32, copy=False)
    result, schema = {}, {}

    def put(name, value, group, unit, definition):
        result[name] = float(value)
        schema[name] = dict(group=group, unit=unit, definition=definition)

    for index, channel in enumerate(THERMAL_CHANNELS):
        series = clip[:, index].mean((1, 2), dtype=np.float64)
        put(f'{channel}_mean_k', series.mean(), 'thermal', 'K', 'Mean over time and pixels')
        put(f'{channel}_tendency_k_15min', linear_tendency(series), 'dynamic', 'K/15min',
            'OLS slope of spatial mean at 5-minute cadence, multiplied by 15')

    ir = clip[:, 4]
    quantiles = np.quantile(ir, [.1, .5, .9], axis=(1, 2))
    for q, series in zip((10, 50, 90), quantiles):
        put(f'IR_108_q{q}_k', series.mean(), 'thermal', 'K', 'Time mean of frame spatial quantile')
    put('IR_108_q10_tendency_k_15min', linear_tendency(quantiles[0]), 'dynamic', 'K/15min',
        'OLS slope of frame spatial 10th percentile; no tracking')
    put('IR_108_spatial_std_k', ir.std((1, 2), dtype=np.float64).mean(), 'spatial', 'K',
        'Time mean of within-frame spatial standard deviation')
    for axis, direction in ((2, 'east'), (1, 'south')):
        put(f'IR_108_gradient_{direction}_k_pixel', np.abs(np.diff(ir, axis=axis)).mean(),
            'spatial', 'K/pixel', 'Mean absolute adjacent-pixel difference; projected grid')
    differences = np.diff(ir, axis=0)
    put('IR_108_temporal_abs_k_5min', np.abs(differences).mean(), 'dynamic', 'K/5min',
        'Mean absolute difference between consecutive frames at fixed pixels')
    put('IR_108_temporal_pattern_std_k_5min', differences.std((1, 2), dtype=np.float64).mean(),
        'dynamic', 'K/5min', 'Time mean spatial std of consecutive-frame differences')

    for threshold in (240, 260):
        fraction = (ir <= threshold).mean((1, 2)) * 100
        put(f'IR_108_below{threshold}_pct', fraction.mean(), 'thermal', '% pixels',
            f'Time mean pixel percentage with BT <= {threshold} K; not a cloud label')
        put(f'IR_108_below{threshold}_tendency_pp_15min', linear_tendency(fraction),
            'dynamic', 'percentage points/15min', 'OLS slope of threshold pixel percentage')
    counts, largest = [], []
    for frame in ir:
        labels, count = ndimage.label(frame <= 260, structure=np.ones((3, 3)))
        counts.append(count)
        sizes = np.bincount(labels.ravel())[1:]
        largest.append(100 * sizes.max() / frame.size if count else 0.)
    put('IR_108_below260_components', np.mean(counts), 'spatial', 'count',
        'Time mean number of 8-connected <=260 K regions; no size filtering/tracking')
    put('IR_108_below260_largest_pct', np.mean(largest), 'spatial', '% pixels',
        'Time mean largest 8-connected <=260 K region as percentage of crop pixels')
    put('IR_108_below260_largest_tendency_pp_15min', linear_tendency(largest),
        'dynamic', 'percentage points/15min', 'OLS slope of largest-region percentage; no tracking')

    for left, right in ((0, 4), (0, 1), (2, 4), (4, 5), (4, 6)):
        name = f'{THERMAL_CHANNELS[left]}_minus_{THERMAL_CHANNELS[right]}'
        difference = clip[:, left] - clip[:, right]
        series = difference.mean((1, 2), dtype=np.float64)
        put(f'{name}_mean_k', series.mean(), 'spectral', 'K', 'Mean brightness-temperature difference')
        put(f'{name}_spatial_std_k', difference.std((1, 2), dtype=np.float64).mean(),
            'spectral', 'K', 'Time mean spatial std of brightness-temperature difference')
        put(f'{name}_tendency_k_15min', linear_tendency(series), 'dynamic', 'K/15min',
            'OLS slope of spatial mean brightness-temperature difference')
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Nonfinite descriptor')
    return result, schema
