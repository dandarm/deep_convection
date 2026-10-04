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
                min_gap_pixels=0, center_allowed=None):
    """Choose among feasible random origins; return None if search finds none."""
    height, width = coverage.shape
    max_y, max_x = height - size, width - size
    origins = rng.integers([max_y + 1, max_x + 1], size=(attempts, 2))
    ys, xs = origins[:, 0], origins[:, 1]
    feasible = np.ones(attempts, dtype=bool)
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
