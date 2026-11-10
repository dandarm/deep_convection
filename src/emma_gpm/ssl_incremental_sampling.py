"""Append random crops to a fixed manifest with explicit pairwise reuse limits."""
from __future__ import annotations

import numpy as np
import pandas as pd


class IncrementalCropSampler:
    """Index every fixed event, including those later than a new proposal.

    Events are (start_frame, native_y, native_x). The additional observation
    reuse constraint applies to pairs containing at least one appended event.
    """

    def __init__(self, fixed, *, size=224, frames=16, max_iou=.2, max_shared_fraction=.2):
        fixed = np.asarray(fixed)
        if fixed.size == 0:
            fixed = np.empty((0, 3), dtype=np.int64)
        if (fixed.ndim != 2 or fixed.shape[1] != 3 or not np.isfinite(fixed).all()
                or not np.equal(fixed, fixed.astype(np.int64)).all() or (fixed < 0).any()
                or len(np.unique(fixed, axis=0)) != len(fixed)):
            raise ValueError('Fixed events must be unique nonnegative integer triples')
        if size < 1 or frames < 1 or not 0 <= max_iou < 1 or not 0 <= max_shared_fraction <= 1:
            raise ValueError('Invalid crop or overlap parameters')
        self.size, self.frames = size, frames
        self.max_iou, self.max_shared_fraction = max_iou, max_shared_fraction
        self.fixed_count = self.count = len(fixed)
        self._events = np.empty((max(1024, 2*len(fixed)), 3), dtype=np.int64)
        self._events[:self.count] = fixed

    @property
    def events(self):
        return self._events[:self.count].copy()

    def try_add(self, event):
        event = np.asarray(event)
        if (event.shape != (3,) or not np.isfinite(event).all()
                or not np.equal(event, event.astype(np.int64)).all() or (event < 0).any()):
            raise ValueError('A proposal must be a nonnegative integer triple')
        event = event.astype(np.int64)
        previous = self._events[:self.count]
        shared = self.frames - np.abs(previous[:, 0] - event[0])
        relevant = shared > 0
        other = previous[relevant]
        intersection = np.maximum(0, self.size-np.abs(other[:, 1]-event[1])) * np.maximum(
            0, self.size-np.abs(other[:, 2]-event[2]))
        area = self.size*self.size
        if np.any(intersection/(2*area-intersection) > self.max_iou+1e-12):
            return 'spatial_iou'
        if np.any(intersection*shared[relevant]/(area*self.frames) > self.max_shared_fraction+1e-12):
            return 'shared_observations'
        if self.count == len(self._events):
            self._events = np.vstack([self._events, np.empty_like(self._events)])
        self._events[self.count] = event
        self.count += 1
        return 'accepted'


def augment_events(fixed, origins_by_start, origin_width, *, seed, size=224, frames=16,
                   max_iou=.2, max_shared_fraction=.2, block_size=1024,
                   minimum_acceptance=.1, patience=3, max_proposals=32768):
    """Uniform start times and admissible origins, stopping at low acceptance.

    Rejections count only complete, geographically/radiometrically admissible
    proposals. Every proposal is evaluated once, with no hidden spatial retries.
    """
    if (block_size < 1 or patience < 1 or max_proposals < block_size
            or not 0 <= minimum_acceptance <= 1 or origin_width < 1):
        raise ValueError('Invalid rejection stopping parameters')
    choices = {int(t): np.asarray(origins, dtype=np.int64)
               for t, origins in origins_by_start.items() if len(origins)}
    if not choices or any((v < 0).any() for v in choices.values()):
        raise ValueError('At least one complete window with admissible origins is required')
    sampler = IncrementalCropSampler(fixed, size=size, frames=frames, max_iou=max_iou,
                                     max_shared_fraction=max_shared_fraction)
    rng = np.random.default_rng(seed)
    starts = np.array(sorted(choices), dtype=np.int64)
    blocks, bad, tried = [], 0, 0
    while tried < max_proposals:
        count = min(block_size, max_proposals-tried)
        outcomes = dict(accepted=0, spatial_iou=0, shared_observations=0)
        for start in rng.choice(starts, size=count):
            positions = choices[int(start)]
            y, x = divmod(int(positions[rng.integers(len(positions))]), origin_width)
            outcomes[sampler.try_add((start, y, x))] += 1
        tried += count
        rate = outcomes['accepted']/count
        blocks.append(dict(proposals=count, cumulative_proposals=tried,
            total_clips=sampler.count, acceptance_rate=rate, **outcomes))
        bad = bad+1 if rate < minimum_acceptance else 0
        if bad >= patience:
            break
    return sampler.events[sampler.fixed_count:], blocks, (
        'low_acceptance' if bad >= patience else 'proposal_limit')


def audit_added_pairs(events, fixed_count, *, size=224, frames=16):
    """Independent matrix audit distinguishing old pairs from appended pairs."""
    events = np.asarray(events, dtype=np.int64)
    shared = np.maximum(0, frames-np.abs(events[:, 0, None]-events[None, :, 0]))
    intersection = np.maximum(0, size-np.abs(events[:, 1, None]-events[None, :, 1])) * np.maximum(
        0, size-np.abs(events[:, 2, None]-events[None, :, 2]))
    pairs = np.triu(shared > 0, k=1)
    added = np.arange(len(events)) >= fixed_count
    new_pairs = pairs & (added[:, None] | added[None, :])
    fraction = intersection*shared/(size*size*frames)
    iou = intersection/(2*size*size-intersection)
    return dict(maximum_active_iou=float(iou[pairs].max(initial=0)),
        maximum_new_pair_shared_fraction=float(fraction[new_pairs].max(initial=0)),
        maximum_fixed_pair_shared_fraction=float(fraction[pairs & ~new_pairs].max(initial=0)),
        new_pairs_checked=int(new_pairs.sum()))


def balanced_extension(fixed, candidates, *, seed, world_size=4):
    """Keep every fixed clip and balance the expanded total over calendar months."""
    if fixed.empty or fixed.clip_id.duplicated().any() or candidates.clip_id.duplicated().any():
        raise ValueError('Unique clip identities and a nonempty fixed manifest are required')
    if not set(fixed.clip_id) <= set(candidates.clip_id):
        raise ValueError('Every fixed clip must be present in the candidate pool')
    fixed = fixed.copy(); candidates = candidates.copy()
    fixed['month'] = fixed.day.str[:7]; candidates['month'] = candidates.day.str[:7]
    months = sorted(fixed.month.unique())
    if set(months) != set(candidates.month.unique()):
        raise ValueError('Preserve the original observation months')
    quantum = world_size//np.gcd(world_size, len(months))
    per_month = int(candidates.groupby('month').size().min())//quantum*quantum
    if per_month < fixed.groupby('month').size().max():
        raise ValueError('Balanced extension cannot discard fixed clips')
    rng = np.random.default_rng(seed)

    def allocate(capacity, existing, budget):
        keys = sorted(capacity)
        left = np.array([capacity[k] for k in keys], dtype=int)
        current = np.array([existing.get(k, 0) for k in keys], dtype=int)
        selected = np.zeros(len(keys), dtype=int)
        if budget < 0 or budget > left.sum():
            raise ValueError('Requested allocation exceeds remaining candidates')
        while budget:
            eligible = np.flatnonzero(left > 0)
            tied = eligible[current[eligible] == current[eligible].min()]
            take = rng.permutation(tied)[:budget]
            selected[take] += 1; current[take] += 1; left[take] -= 1
            budget -= len(take)
        return dict(zip(keys, map(int, selected), strict=True))

    additional = candidates[~candidates.clip_id.isin(fixed.clip_id)]
    chosen = []
    for month in months:
        old = fixed[fixed.month == month]
        available = additional[additional.month == month]
        daily = allocate(available.groupby('day').size().to_dict(), old.groupby('day').size().to_dict(),
                         per_month-len(old))
        for day, budget in daily.items():
            old_day = old[old.day == day]; new_day = available[available.day == day]
            bins = allocate(new_day.groupby('time_bin').size().to_dict(),
                old_day.groupby('time_bin').size().to_dict(), budget)
            for time_bin, count in bins.items():
                group = new_day[new_day.time_bin == time_bin]
                chosen.extend(rng.choice(group.index.to_numpy(), count, replace=False))
    fixed_rows = candidates[candidates.clip_id.isin(fixed.clip_id)]
    result = pd.concat([fixed_rows, candidates.loc[chosen]], ignore_index=True).drop(columns='month')
    return result.sort_values(['start_time', 'origin_y', 'origin_x']).reset_index(drop=True)
