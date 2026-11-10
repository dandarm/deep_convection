"""Dataset statistics and calendar selection for overlap-constrained clips."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ssl_seasonal_sampling import crop_sums


def asynchronous_start_counts(frame_count, *, seed, streams=12, clip_frames=16,
                              min_gap_frames=0, max_gap_frames=16):
    """Independent streams with initial jitter and random gaps after each clip.

    A start occupies ``clip_frames`` observed frames. Each stream then waits
    zero to sixteen additional frame intervals by default. Schedule generation
    has its own RNG, independent of spatial rejection and crop materialization.
    """
    parameters = (frame_count, streams, clip_frames, min_gap_frames, max_gap_frames)
    if (any(not isinstance(v, (int, np.integer)) for v in parameters)
            or frame_count < 0 or streams < 1 or clip_frames < 1
            or not 0 <= min_gap_frames <= max_gap_frames):
        raise ValueError('Invalid asynchronous temporal sampling parameters')
    counts = np.zeros(max(0, frame_count-clip_frames+1), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(streams):
        start = int(rng.integers(clip_frames))
        while start < len(counts):
            counts[start] += 1
            start += clip_frames + int(rng.integers(min_gap_frames, max_gap_frames+1))
    return counts


def window_crop_moments(values, windows, *, crop=224):
    """Yield exact population moments, reading each needed frame once.

    Windows are chronological (indices, origins) pairs. Temporary accumulators
    clear invalid pixels outside accepted crops; the observations remain intact.
    The caller must have validated every selected crop in every channel/frame.
    """
    cache = {}
    total = np.zeros(values.shape[-2:], np.float64)
    squared = np.zeros_like(total)
    previous = -1
    for indices, origins in windows:
        indices = np.asarray(indices, dtype=int)
        if (not len(indices) or indices[0] <= previous or (np.diff(indices) != 1).any()
                or indices[0] < 0 or indices[-1] >= len(values)):
            raise ValueError('Increasing, complete contiguous windows are required')
        previous = int(indices[0])
        wanted = set(map(int, indices))
        for index in list(cache):
            if index not in wanted:
                sums, squares = cache.pop(index)
                total -= sums
                squared -= squares
        for index in indices:
            index = int(index)
            if index not in cache:
                frame = values[index].astype(np.float64)
                np.nan_to_num(frame, copy=False, nan=0., posinf=0., neginf=0.)
                sums, squares = frame.sum(axis=0), np.square(frame).sum(axis=0)
                total += sums
                squared += squares
                cache[index] = sums, squares
        y, x = np.asarray(origins, dtype=int).T
        count = len(indices) * values.shape[1] * crop * crop
        mean = crop_sums(total, crop)[y, x] / count
        variance = crop_sums(squared, crop)[y, x] / count - mean**2
        yield mean, np.maximum(variance, 0.)


def balanced_subset(frame, target, seed):
    """Equal month totals, with capacity-aware date and six-hour allocations.

    The budget is capped by the smallest monthly candidate pool. Within each
    month the algorithm fills dates evenly, reallocating only unavailable slots;
    the same rule applies to the four time-of-day bins within each date.
    """
    if target < 1 or frame.empty:
        raise ValueError('A positive target and nonempty candidate manifest are required')
    frame = frame.copy()
    frame['month'] = frame.day.str[:7]
    rng = np.random.default_rng(seed)
    months = sorted(frame.month.unique())
    counts = frame.groupby('month').size()
    per_month = min(target // len(months), int(counts.min()))
    total = min(target, per_month * len(months) + min(target % len(months),
        int((counts > per_month).sum())))

    def allocations(capacities, count):
        keys = sorted(capacities)
        capacity = np.array([capacities[key] for key in keys], dtype=int)
        allocated = np.zeros(len(keys), dtype=int)
        if count > capacity.sum():
            raise ValueError('Allocation exceeds available candidates')
        while count:
            choices = np.flatnonzero(allocated < capacity)
            choices = choices[rng.permutation(len(choices))]
            take = choices[:count]
            allocated[take] += 1
            count -= len(take)
        return dict(zip(keys, map(int, allocated), strict=True))

    chosen = []
    for month, n_month in allocations(counts.to_dict(), total).items():
        monthly = frame[frame.month == month]
        for day, n_day in allocations(monthly.groupby('day').size().to_dict(), n_month).items():
            daily = monthly[monthly.day == day]
            for time_bin, n_bin in allocations(daily.groupby('time_bin').size().to_dict(), n_day).items():
                group = daily[daily.time_bin == time_bin]
                chosen.extend(rng.choice(group.index.to_numpy(), n_bin, replace=False))
    return frame.loc[chosen].drop(columns='month').sort_values(
        ['start_time', 'origin_y', 'origin_x']).reset_index(drop=True)


def active_clip_rows(manifest, timestamp, *, frames=16, cadence_minutes=5):
    """Select the actual accepted clips containing this exact observation."""
    timestamp = pd.Timestamp(timestamp)
    starts = pd.to_datetime(manifest.start_time, utc=True)
    offset = (timestamp - starts).dt.total_seconds() / (60 * cadence_minutes)
    return manifest.loc[(offset >= 0) & (offset < frames) & (offset % 1 == 0)]
