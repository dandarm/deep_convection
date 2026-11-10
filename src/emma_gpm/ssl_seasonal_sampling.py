"""Uniform calendar sampling and exact crop moments for native SEVIRI arrays."""
import numpy as np
import pandas as pd


def equal_quotas(keys, count, rng):
    keys = sorted(keys)
    if not keys or count < len(keys):
        raise ValueError('At least one sample per sampling group is required')
    values = np.full(len(keys), count // len(keys), dtype=int)
    values[rng.permutation(len(keys))[:count % len(keys)]] += 1
    return dict(zip(keys, map(int, values), strict=True))


def calendar_quotas(days, count, seed):
    """Equal month totals, then equal dates and six-hour bins within each month."""
    days = sorted(set(pd.to_datetime(days).strftime('%Y-%m-%d')))
    rng = np.random.default_rng(seed)
    months = sorted({day[:7] for day in days})
    result = {}
    for month, total in equal_quotas(months, count, rng).items():
        for day, daily in equal_quotas([d for d in days if d.startswith(month)], total, rng).items():
            result[day] = equal_quotas(range(4), daily, rng)
    return result


def crop_sums(values, crop):
    """Sum every spatial square with float64 accumulation."""
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64)
    return integral[crop:, crop:] - integral[:-crop, crop:] - integral[crop:, :-crop] + integral[:-crop, :-crop]


def valid_crop_origins(clip, geographic, physical=None, crop=224):
    """Require every value in every channel/frame to be observed and in 100..400 K."""
    valid = np.all(np.isfinite(clip) & (clip >= 100) & (clip <= 400), axis=(0, 1))
    return valid_crop_origins_from_pixels(valid, geographic, physical, crop)


def frame_pixel_validity(values):
    """Check each native observation once when many windows share its pixels."""
    valid = np.empty((len(values), *values.shape[-2:]), dtype=bool)
    for i, frame in enumerate(values):
        valid[i] = np.all(np.isfinite(frame) & (frame >= 100) & (frame <= 400), axis=0)
    return valid


def valid_crop_origins_from_pixels(valid, geographic, physical=None, crop=224):
    """Apply whole-footprint quality to a temporal conjunction of pixel masks."""
    if physical is not None:
        valid = valid & physical
    return geographic & (crop_sums(valid, crop) == crop * crop)


def crop_moments(clip, origins, crop=224):
    """All-channel/time population moments for equal-size selected crops."""
    sums = np.zeros(clip.shape[-2:], dtype=np.float64)
    squares = np.zeros_like(sums)
    invalid = np.zeros(clip.shape[-2:], dtype=bool)
    for frame in clip:
        values = frame.astype(np.float64)
        invalid |= ~np.isfinite(values).all(axis=0)
        # Missing pixels outside the selected crops must not contaminate the
        # prefix sums. Only this temporary accumulator is cleared; observations
        # stay unchanged, and any selected crop touching a missing value fails.
        np.nan_to_num(values, copy=False, nan=0., posinf=0., neginf=0.)
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
    y, x = np.asarray(origins, dtype=int).T
    if (crop_sums(invalid, crop)[y, x] > 0).any():
        raise ValueError('Crop moments require complete selected observations')
    count = clip.shape[0] * clip.shape[1] * crop * crop
    mean = crop_sums(sums, crop)[y, x] / count
    variance = crop_sums(squares, crop)[y, x] / count - np.square(mean)
    return mean, np.maximum(variance, 0)


def selected_time_indices(times, starts, frames=16):
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Unique chronological source observations required')
    indices = []
    for start in starts:
        wanted = pd.date_range(pd.Timestamp(start), periods=frames, freq='5min')
        found = times.get_indexer(wanted)
        if (found < 0).any():
            raise ValueError('A selected window has missing RSS observations')
        indices.extend(found)
    return np.unique(indices)
