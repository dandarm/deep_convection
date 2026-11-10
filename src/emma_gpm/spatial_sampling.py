"""Random SEVIRI crops with strict overlap limits for concurrent clips.

Inspired by Demetra's random_tiles notebook (active IoU and coverage bias).
Unlike its fallback, an infeasible candidate is never accepted.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropEvent:
    start: pd.Timestamp
    y: int
    x: int


def crop_iou(a, b, size):
    intersection = max(0, size - abs(a.y - b.y)) * max(0, size - abs(a.x - b.x))
    return intersection / (2 * size * size - intersection)


def shared_frames(a, b, duration_minutes=75):
    # Both endpoints are actual observed frames, so touching times overlap.
    return abs(a.start - b.start) <= pd.Timedelta(minutes=duration_minutes)


def verify_overlap(events, size=224, max_iou=0.):
    active = []
    pairs = 0
    worst = 0.
    for event in sorted(events, key=lambda e: e.start):
        active = [other for other in active if shared_frames(event, other)]
        for other in active:
            iou = crop_iou(event, other, size)
            pairs += 1
            worst = max(worst, iou)
            if iou > max_iou + 1e-12:
                raise ValueError('Spatial overlap exceeds limit between clips sharing frames')
        active.append(event)
    return dict(concurrent_pairs_checked=pairs, maximum_concurrent_iou=worst)


def choose_crop(rng, timestamp, active, coverage, size, *, attempts=2048,
                max_iou=0., coverage_weight=1., border_boost=.25,
                min_gap_pixels=0, center_allowed=None, origin_mask=None):
    """Choose among feasible random origins; return None if search finds none."""
    height, width = coverage.shape
    max_y, max_x = height - size, width - size
    origins = rng.integers([max_y + 1, max_x + 1], size=(attempts, 2))
    ys, xs = origins[:, 0], origins[:, 1]
    feasible = np.ones(attempts, dtype=bool)
    if origin_mask is not None:
        if origin_mask.shape != (max_y + 1, max_x + 1) or origin_mask.dtype != bool:
            raise ValueError('Origin mask must match all possible crop origins')
        feasible &= origin_mask[ys, xs]
    if center_allowed is not None:
        feasible &= np.array([center_allowed(int(y), int(x)) for y, x in origins])
    if active:
        other_y = np.array([event.y for event in active])
        other_x = np.array([event.x for event in active])
        delta_y = np.abs(ys[:, None] - other_y)
        delta_x = np.abs(xs[:, None] - other_x)
        intersection = np.maximum(0, size - delta_y) * np.maximum(0, size - delta_x)
        iou = intersection / (2 * size * size - intersection)
        feasible &= (iou <= max_iou + 1e-12).all(axis=1)
        if min_gap_pixels:
            feasible &= ~((delta_x < size + min_gap_pixels) &
                          (delta_y < size + min_gap_pixels)).any(axis=1)
    choices = np.flatnonzero(feasible)
    if not len(choices):
        return None
    ys, xs = ys[choices], xs[choices]
    probes = np.stack([coverage[ys, xs], coverage[ys, xs+size-1],
        coverage[ys+size-1, xs], coverage[ys+size-1, xs+size-1],
        coverage[ys+size//2, xs+size//2]])
    score = coverage_weight * probes.mean(axis=0)
    edge_distance = np.minimum.reduce([xs, max_x-xs, ys, max_y-ys])
    score -= border_boost * (1-edge_distance/max(1,max(max_x,max_y)))
    selected = int(np.argmin(score))
    return CropEvent(timestamp, int(ys[selected]), int(xs[selected]))


class ActiveCropSampler:
    """Chronological proposals with strict active IoU and Demetra coverage bias.

    The caller supplies whole-crop geographic and radiometric validity for each
    exact RSS window. Every accepted event enters the returned dataset; a
    renderer can replay that manifest without generating a second simulation.
    Coverage and the random generator persist across windows within a day.
    """

    def __init__(self, shape, *, seed, max_iou=.2, size=224, attempts=2048,
                 proposals_per_start=16, coverage_weight=1., border_boost=.25):
        if (len(shape) != 2 or min(shape) < size or size < 1
                or not 0 <= max_iou < 1 or attempts < 1 or proposals_per_start < 1):
            raise ValueError('Invalid chronological crop sampling configuration')
        self.rng = np.random.default_rng(seed)
        self.coverage = np.zeros(shape, dtype=np.uint32)
        self.max_iou, self.size = max_iou, size
        self.attempts, self.proposals_per_start = attempts, proposals_per_start
        self.coverage_weight, self.border_boost = coverage_weight, border_boost
        self.active, self.events = [], []
        self.previous = None
        self.exhausted_windows = 0

    def sample_window(self, timestamp, origin_mask):
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp) or (self.previous is not None and timestamp <= self.previous):
            raise ValueError('Sampling windows must have unique increasing timestamps')
        self.previous = timestamp
        probe = CropEvent(timestamp, 0, 0)
        self.active = [event for event in self.active if shared_frames(probe, event)]
        selected = []
        for _ in range(self.proposals_per_start):
            event = choose_crop(self.rng, timestamp, self.active, self.coverage, self.size,
                attempts=self.attempts, max_iou=self.max_iou,
                coverage_weight=self.coverage_weight, border_boost=self.border_boost,
                origin_mask=origin_mask)
            if event is None:
                self.exhausted_windows += 1
                break
            self.active.append(event)
            self.events.append(event)
            selected.append(event)
            self.coverage[event.y:event.y+self.size, event.x:event.x+self.size] += 1
        return selected


def pixel_frame_coverage(manifest, height, width, *, size=224, frames=16, cadence_minutes=5):
    """Exact geometric reuse of observed pixel/frame positions across samples."""
    rectangles = {}
    for row in manifest.itertuples(index=False):
        start = pd.Timestamp(row.start_time)
        for timestamp in pd.date_range(start, periods=frames, freq=f'{cadence_minutes}min'):
            rectangles.setdefault(timestamp, []).append((int(row.origin_y), int(row.origin_x)))
    unique = 0
    occupancy = np.zeros((height, width), dtype=bool)
    for origins in rectangles.values():
        occupancy.fill(False)
        for y, x in origins:
            if y < 0 or x < 0 or y+size > height or x+size > width:
                raise ValueError('Crop outside domain')
            occupancy[y:y+size, x:x+size] = True
        unique += int(occupancy.sum())
    sampled = len(manifest) * frames * size * size
    return dict(samples=len(manifest), sampled_pixel_frames=sampled, unique_pixel_frames=unique,
                repeated_pixel_frame_fraction=1-unique/sampled if sampled else 0.,
                distinct_frames=len(rectangles))
