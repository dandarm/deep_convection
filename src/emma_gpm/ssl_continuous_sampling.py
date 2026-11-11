"""Continuous UTC windows and bounded temporal context for SEVIRI sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_NS = 300_000_000_000


def training_time_mask(times, reserved_days, separation_hours=24):
    """Admissible actual observations, including the full reserved-day buffer."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit('ns')
    if times.hasnans or separation_hours < 0:
        raise ValueError('Finite timestamps and a nonnegative separation are required')
    allowed = np.ones(len(times), bool)
    gap = pd.Timedelta(hours=separation_hours)
    for day in pd.to_datetime(list(reserved_days), utc=True).normalize():
        allowed &= ~((times >= day-gap) &
                     (times <= day+pd.Timedelta(hours=23, minutes=55)+gap))
    return allowed


def continuous_windows(times, owner_day, *, allowed=None, frames=16):
    """Map absolute five-minute start slots to complete native array indices.

    A day owns clip starts only. The supplied observations can extend into the
    following day or year; all observations must be present and admissible.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit('ns')
    if (times.hasnans or not times.is_unique or not times.is_monotonic_increasing
            or (times.asi8 % SLOT_NS).any() or frames < 1):
        raise ValueError('Unique increasing five-minute timestamps are required')
    allowed = np.ones(len(times), bool) if allowed is None else np.asarray(allowed, bool)
    if allowed.shape != (len(times),):
        raise ValueError('One eligibility flag is required for each observation')
    day = pd.Timestamp(owner_day)
    day = (day.tz_localize('UTC') if day.tzinfo is None else day.tz_convert('UTC')).normalize()
    windows = {}
    for i in np.flatnonzero((times >= day) & (times < day+pd.Timedelta(days=1))):
        indices = np.arange(i, i+frames)
        if (indices[-1] < len(times) and allowed[indices].all()
                and np.all(np.diff(times.asi8[indices]) == SLOT_NS)):
            windows[int(times.asi8[i]//SLOT_NS)] = indices
    return windows


def temporal_context(manifest, day, *, frames=16):
    """Existing clips that can intersect any start owned by this UTC day."""
    day = pd.Timestamp(day, tz='UTC')
    times = pd.to_datetime(manifest.start_time, utc=True)
    reach = pd.Timedelta(minutes=5*(frames-1))
    return manifest[(times >= day-reach) & (times < day+pd.Timedelta(days=1)+reach)]


def event_array(manifest):
    times = pd.DatetimeIndex(pd.to_datetime(manifest.start_time, utc=True)).as_unit('ns')
    return np.column_stack([times.asi8//SLOT_NS, manifest.origin_y, manifest.origin_x]).astype(np.int64)


def preparation_phase(day):
    """Nonadjacent days may run concurrently; neighboring days run afterwards."""
    return pd.Timestamp(day).toordinal() % 2
