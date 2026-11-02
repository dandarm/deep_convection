"""Pixel-resolution random crop origins over physically present SEVIRI chunks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


def geographic_origin_mask(x, y, projection, domain, crop=224):
    """Accept origins only when every crop pixel centre is inside lon/lat bounds."""
    west, east, south, north = domain
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90) or crop < 1:
        raise ValueError('Ordered geographic bounds and a positive crop size required')
    x, y = np.asarray(x), np.asarray(y)
    if min(len(x), len(y)) < crop:
        raise ValueError('Geographic grid is smaller than the crop')
    transformer = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    inside = (np.isfinite(lon) & np.isfinite(lat) & (lon >= west) & (lon <= east)
              & (lat >= south) & (lat <= north))
    integral = np.pad(inside.astype(np.int32), ((1,0),(1,0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
    counts = integral[crop:,crop:]-integral[:-crop,crop:]-integral[crop:,:-crop]+integral[:-crop,:-crop]
    return counts == crop*crop


def origin_rectangles(complete, height, width, cy, cx, crop=224):
    """Partition valid integer origins into disjoint half-open rectangles.

    Chunk support changes at chunk boundaries and when the far crop edge
    crosses one. Every origin inside a returned rectangle has complete support.
    Rectangle area counts integer origins, enabling unbiased random selection.
    """
    if min(height, width, cy, cx, crop) < 1:
        raise ValueError('Positive image, chunk and crop dimensions required')
    if height < crop or width < crop:
        return np.empty((0, 4), dtype=np.int64)

    def boundaries(length, chunk):
        end = length-crop+1
        edges = {0, end}
        for boundary in range(0, length+chunk, chunk):
            edges.update(v for v in (boundary, boundary-crop+1) if 0 < v < end)
        return sorted(edges)

    ys, xs = boundaries(height, cy), boundaries(width, cx)
    rectangles = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if complete[y0//cy:(y0+crop-1)//cy+1, x0//cx:(x0+crop-1)//cx+1].all():
                rectangles.append((y0, y1, x0, x1))
    return np.asarray(rectangles, dtype=np.int64).reshape(-1, 4)


def random_candidates(inventory, count, seed=20261024, *, crop=224, frames=16, origin_masks=None):
    """Balance UTC dates/six-hour bins; sample unique time/origin tuples.

    Within each date/bin, all physically supported (start, y, x) tuples have
    equal proposal probability. Coordinates use individual native-grid pixels.
    Physical support is checked for every frame; pixel values are checked by
    the dataset builder. Spatial and temporal overlap within training is allowed.
    """
    if count < 1 or frames < 1:
        raise ValueError('Positive candidate and frame counts required')
    groups = {}
    for record in sorted(inventory['stores'], key=lambda r:r['store']):
        times = pd.DatetimeIndex(pd.to_datetime(record['times'], utc=True))
        if times.hasnans or not times.is_unique:
            raise ValueError('Unique finite observation timestamps required')
        height, width = record['shape'][2:]
        origin_shape = (height-crop+1, width-crop+1)
        if min(origin_shape) < 1:
            continue
        allowed = np.ones(origin_shape, dtype=bool) if origin_masks is None else origin_masks[record['store']]
        if allowed.shape != origin_shape or allowed.dtype != bool:
            raise ValueError('Geographic origin mask has incompatible shape or dtype')
        cy, cx = record['chunks'][2:]
        present = np.zeros((len(times), (height+cy-1)//cy, (width+cx-1)//cx), dtype=bool)
        for name in record['present']:
            try:
                t, channel, y, x = map(int, name.split('.'))
                if channel == 0 and min(t, y, x) >= 0:
                    present[t, y, x] = True
            except (ValueError, IndexError):
                continue
        geometries = {}
        for start in times.sort_values():
            stamps = pd.date_range(start, periods=frames, freq='5min')
            indices = times.get_indexer(stamps)
            if (indices < 0).any() or stamps[-1].normalize() != start.normalize():
                continue
            complete = present[indices].all(axis=0)
            key = complete.tobytes()
            if key not in geometries:
                rectangles = origin_rectangles(complete, height, width, cy, cx, crop)
                valid = np.zeros(origin_shape, dtype=bool)
                for y0,y1,x0,x1 in rectangles:
                    valid[y0:y1,x0:x1] = True
                geometries[key] = np.flatnonzero(valid & allowed)
            locations = geometries[key]
            if not len(locations):
                continue
            group = (start.strftime('%Y-%m-%d'), start.hour//6)
            groups.setdefault(group, []).append((record['store'], start, locations, origin_shape[1]))
    if not groups:
        raise ValueError('No complete physical windows for random sampling')
    rng = np.random.default_rng(seed)
    group_keys = sorted(groups)
    quotas = np.full(len(groups), count//len(groups), dtype=np.int64)
    quotas[rng.permutation(len(groups))[:count % len(groups)]] += 1
    rows = []
    for group, quota in zip(group_keys, quotas, strict=True):
        windows = groups[group]
        cumulative = np.cumsum([len(window[2]) for window in windows])
        capacity = int(cumulative[-1])
        if quota > capacity:
            raise ValueError(f'Insufficient unique physical origins in {group}: {capacity}')
        # Sample integer ranks without replacement across all supported tuples.
        # Generator.choice would allocate the huge population for some regimes.
        ranks = set()
        while len(ranks) < quota:
            ranks.update(map(int, rng.integers(capacity, size=int(quota-len(ranks)))))
        for rank in sorted(ranks):
            index = int(np.searchsorted(cumulative, rank, side='right'))
            offset = rank - (int(cumulative[index-1]) if index else 0)
            store, start, locations, origin_width = windows[index]
            y, x = map(int, divmod(locations[offset], origin_width))
            rows.append(dict(start_time=start.isoformat().replace('+00:00','Z'),
                origin_y=y, origin_x=x, day=group[0], time_bin=group[1], store=store))
    # Day/time locality speeds validation; training uses its own per-epoch shuffle.
    frame = pd.DataFrame(rows).sort_values(['store','start_time','origin_y','origin_x']).reset_index(drop=True)
    if len(frame) != count or frame.duplicated(['start_time','origin_y','origin_x']).any():
        raise ValueError('Random candidates must have unique time/origin identities')
    return frame
