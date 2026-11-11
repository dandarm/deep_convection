"""Exact reuse of observed pixel/frame positions in a crop manifest."""
import numpy as np
import pandas as pd


def rectangle_coverage(y, x, size):
    """Union area and maximum multiplicity on coordinate-compressed rectangles."""
    if not len(y):
        return 0, 0
    ys, xs = np.unique(np.r_[y, y+size]), np.unique(np.r_[x, x+size])
    difference = np.zeros((len(ys), len(xs)), np.int32)
    for dy, dx, sign in ((0, 0, 1), (size, 0, -1), (0, size, -1), (size, size, 1)):
        np.add.at(difference, (np.searchsorted(ys, y+dy), np.searchsorted(xs, x+dx)), sign)
    count = difference.cumsum(0).cumsum(1)[:-1, :-1]
    area = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return int(area[count > 0].sum()), int(count.max())


def overlap_summary(manifest, *, size=224, frames=16):
    """Count geometric reuse and nearest-overlap fractions across actual clips.

    Each position represents one spatial pixel at one observed timestamp. All
    channels share those positions. These counts describe geometric coverage;
    atmospheric correlation between different observations is a separate issue.
    """
    from .ssl_continuous_sampling import temporal_context

    if manifest.empty:
        raise ValueError('A nonempty clip manifest is required')
    manifest = manifest.copy()
    starts = pd.to_datetime(manifest.start_time, utc=True)
    manifest['day'] = starts.dt.strftime('%Y-%m-%d')
    ends = starts+pd.Timedelta(minutes=5*(frames-1))
    dates = sorted(set(starts.dt.normalize()) | set(ends.dt.normalize()))
    daily, neighbors = [], []
    for date in dates:
        day = date.strftime('%Y-%m-%d')
        group = manifest[manifest.day == day]
        context = temporal_context(manifest, day, frames=frames)
        times = pd.DatetimeIndex(pd.to_datetime(context.start_time, utc=True)).as_unit('ns')
        t = times.asi8 // 300_000_000_000
        y, x = context.origin_y.to_numpy(int), context.origin_x.to_numpy(int)
        owned = np.flatnonzero(context.day.to_numpy() == day)
        # Each owned clip sees both previous-day and following-day starts.
        shared = np.maximum(0, frames-np.abs(t[owned, None]-t[None, :]))
        shared[np.arange(len(owned)), owned] = 0
        intersection = np.maximum(0, size-np.abs(y[owned, None]-y[None, :])) * np.maximum(
            0, size-np.abs(x[owned, None]-x[None, :]))
        iou = np.where(shared > 0, intersection/(2*size*size-intersection), 0.)
        fraction = intersection/(size*size) * shared/frames
        neighbors.append(context.iloc[owned][['start_time', 'origin_y', 'origin_x', 'day']].assign(
            maximum_active_iou=iou.max(axis=1), maximum_shared_observation_fraction=fraction.max(axis=1)))
        first = date.value//300_000_000_000
        last = first+288
        # Partition observed time, so midnight observations are counted once.
        boundaries = np.unique(np.r_[first, last, np.clip(t, first, last),
                                      np.clip(t+frames, first, last)])
        unique, nframes, maximum_active, maximum_reuse = 0, 0, 0, 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            active = (t <= start) & (t+frames > start)
            area, multiplicity = rectangle_coverage(y[active], x[active], size)
            unique += area*int(end-start)
            nframes += int(end-start) if area else 0
            maximum_active = max(maximum_active, int(active.sum()))
            maximum_reuse = max(maximum_reuse, multiplicity)
        daily.append(dict(day=day, clips=len(group), start_times=group.start_time.nunique(),
            distinct_frames=nframes, unique_pixel_frames=unique,
            maximum_active_clips=maximum_active, maximum_pixel_frame_reuse=maximum_reuse,
            maximum_active_iou=float(iou.max(initial=0))))
    daily = pd.DataFrame(daily)
    neighbors = pd.concat(neighbors, ignore_index=True)
    sampled = len(manifest)*frames*size*size
    unique = int(daily.unique_pixel_frames.sum())
    per_start = manifest.groupby('start_time').size()
    summary = dict(clips=len(manifest), dates=len(daily), distinct_starts=manifest.start_time.nunique(),
        distinct_native_frames=int(daily.distinct_frames.sum()), sampled_pixel_frames=sampled,
        unique_pixel_frames=unique, repeated_pixel_frame_fraction=1-unique/sampled,
        crops_per_start_mean=float(per_start.mean()), crops_per_start_max=int(per_start.max()),
        maximum_active_iou=float(daily.maximum_active_iou.max()),
        maximum_active_clips=int(daily.maximum_active_clips.max()),
        median_maximum_shared_observation_fraction=float(neighbors.maximum_shared_observation_fraction.median()),
        clips_with_more_than_80_percent_shared_observations=float((neighbors.maximum_shared_observation_fraction > .8).mean()),
        interpretation='Exact geometric reuse of observed pixel/frame positions across clips.')
    return summary, daily, neighbors
