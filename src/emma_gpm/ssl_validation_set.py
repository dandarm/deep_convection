"""Selection and temporal isolation of a fixed, multi-day SSL holdout."""
import numpy as np
import pandas as pd


def assert_day_separation(training, days, separation_hours=24):
    """Reserve complete UTC days and a gap on both sides of each observation."""
    starts = pd.DatetimeIndex(pd.to_datetime(training.start_time, utc=True)).unique()
    if starts.empty or starts.hasnans or separation_hours < 0:
        raise ValueError('Finite training timestamps and a nonnegative separation required')
    ends = starts + pd.Timedelta(minutes=75)
    gap = pd.Timedelta(hours=separation_hours)
    for day in pd.to_datetime(list(days), utc=True).normalize():
        last = day + pd.Timedelta(hours=23, minutes=55)
        if ((starts <= last + gap) & (ends + gap >= day)).any():
            raise ValueError(f'Training observations intersect reserved validation day or gap: {day.date()}')


def select_balanced_validation(pool, days, samples_per_day=512, seed=20261102):
    """Equal weight for each UTC day and each six-hour bin after validity checks."""
    if samples_per_day < 4 or samples_per_day % 4:
        raise ValueError('Samples per day must be a positive multiple of four')
    if pool.duplicated(['start_time', 'origin_y', 'origin_x']).any():
        raise ValueError('Duplicate validation proposals')
    rng = np.random.default_rng(seed)
    selected = []
    for day in sorted(pd.to_datetime(list(days)).strftime('%Y-%m-%d')):
        for time_bin in range(4):
            group = pool.loc[(pool.day == day) & (pool.time_bin == time_bin)]
            if len(group) < samples_per_day // 4:
                raise ValueError(f'Insufficient valid proposals for {day}, six-hour bin {time_bin}')
            selected.append(group.iloc[rng.choice(len(group), samples_per_day // 4, replace=False)])
    if not selected:
        raise ValueError('At least one validation day required')
    return pd.concat(selected).sort_values(['day', 'start_time', 'origin_y', 'origin_x']).reset_index(drop=True)
