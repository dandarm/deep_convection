"""Independent, temporally separated cloud-mask targets from the repository CMA."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from .rdt.storage import DomainFrameReader, atomic_json, sha256
from .seviri import THERMAL_CHANNELS
from .seviri_frame_cache import source_stamp


def canonical_axes(x, y):
    x, y = np.asarray(x), np.asarray(y)
    for axis in (x, y):
        delta = np.diff(axis)
        if not np.isfinite(axis).all() or not ((delta > 0).all() or (delta < 0).all()):
            raise ValueError("Coordinates must be finite and strictly monotonic")
    return (x[::-1] if x[0] > x[-1] else x), (y[::-1] if y[0] < y[-1] else y)


def check_times(times):
    times = pd.DatetimeIndex(times)
    if times.hasnans or times.has_duplicates or len(times) < 16:
        raise ValueError("Missing or duplicate timestamps")
    if not (np.diff(times.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError("CMA temporal history requires exact increasing five-minute cadence")
    return times


def check_split_days(splits, excluded_days):
    flat = [str(pd.Timestamp(d).date()) for days in splits.values() for d in days]
    excluded = {str(pd.Timestamp(d).date()) for d in excluded_days}
    if len(flat) != len(set(flat)) or set(flat) & excluded:
        raise ValueError("Probe days overlap each other or pretraining/validation days")
    for left, right in (("train", "validation"), ("validation", "test")):
        gap = min(pd.Timestamp(d) for d in splits[right]) - max(pd.Timestamp(d) for d in splits[left])
        if gap < pd.Timedelta(days=2):
            raise ValueError("At least one unused calendar day is required between probe splits")


def binary_target(cma):
    cma = np.asarray(cma)
    if not np.isin(cma, [0, 1, 255]).all():
        raise ValueError("Expected binary cma (0 clear, 1 cloudy, 255 missing)")
    return cma.astype(np.uint8, copy=True)


def land_geometry(x, y, projection):
    from cartopy.io import shapereader
    import shapely
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True).transform(xx, yy)
    finite = np.isfinite(lon) & np.isfinite(lat)
    path = Path(shapereader.natural_earth(resolution="10m", category="physical", name="land"))
    bounds = shapely.box(lon[finite].min(), lat[finite].min(), lon[finite].max(), lat[finite].max())
    polygons = [g for g in shapereader.Reader(path).geometries() if g.intersects(bounds)]
    land = shapely.union_all(polygons)
    shapely.prepare(land)
    mask = np.zeros(lon.shape, dtype=bool)
    mask[finite] = shapely.contains_xy(land, lon[finite], lat[finite])
    # Outside-Earth pixels remain missing radiometrically in the teacher.
    lon[~finite], lat[~finite] = np.nan, np.nan
    ancillary = dict(land_source="Natural Earth 1:10m land polygons, pixel-centre classification",
                     land_files={p.name: sha256(p) for p in sorted(path.parent.glob(path.stem+'.*'))},
                     elevation="default zero metres", snow_ice="default absent",
                     solar_geometry="repository IR-only default", nwp="autonomous seven-channel estimates")
    return lat, lon, mask, ancillary


def build_cloud_dataset(root, output, plan, code_root):
    """Full-domain causal CMA, canonical 16-frame inputs, label at observed frame 9.

    Each day is committed independently. An interrupted day is recomputed from
    its first observation; published days and all hashes are checked on resume.
    """
    from cma_nwcsaf import NwcSafCmaTracker, NwcSafCmaConfig
    from .file_lock import exclusive_file_lock
    root, output, code_root = Path(root).resolve(), Path(output).resolve(), Path(code_root)
    output.mkdir(parents=True, exist_ok=True)
    check_split_days(plan['splits'], plan['excluded_pretraining_days'])
    sources = {str(p.relative_to(code_root)): sha256(p) for p in sorted((code_root/'CMA_original/cma_nwcsaf').glob('*.py'))}
    for name in ['src/emma_gpm/cloud_probe_data.py', 'src/emma_gpm/rdt/storage.py', 'src/emma_gpm/seviri_clips.py']:
        sources[name] = sha256(code_root/name)
    cfg = NwcSafCmaConfig()
    protocol = dict(version=1, plan=plan, sources=sources, cma_config=asdict(cfg),
                    zarr_root=str(root), channels=list(THERMAL_CHANNELS), units='K',
                    clip_shape=[16, 7, 224, 224], history_frames=10, target_frame=9,
                    target='cma: 0 clear, 1 cloudy, 255 excluded', teacher_domain='full archive rectangle',
                    orientation='north-up, west-left', warmup_minutes=60)
    with exclusive_file_lock(output/'build.lock'):
        if (output/'protocol.json').exists():
            if json.loads((output/'protocol.json').read_text()) != protocol:
                raise ValueError('Dataset protocol changed: choose a new output directory')
        else:
            atomic_json(output/'protocol.json', protocol)
        rows, day_records = [], {}
        start_wall = time.monotonic()
        reader = DomainFrameReader(root)
        geometry = None
        try:
            for split, days in plan['splits'].items():
                for day in days:
                    day_key = pd.Timestamp(day).strftime('%Y%m%d')
                    target_dir = output/day_key
                    if (target_dir/'complete.json').exists():
                        old = json.loads((target_dir/'complete.json').read_text())
                        for row in old['samples']:
                            for key in ['clip', 'label', 'quality']:
                                if sha256(output/row[key]) != row[key+'_sha256']:
                                    raise ValueError('Published dataset file changed')
                        rows.extend(old['samples']); day_records[day_key] = old['source']
                        continue
                    target_dir.mkdir(exist_ok=True)
                    source = root/day_key[:4]/(day_key+'.zarr')
                    stamp = source_stamp(source)
                    ds = reader.reader._open_store(source)
                    times = check_times(ds.time.values)
                    first = reader.load(times[0])
                    x, y = canonical_axes(ds.x.values, ds.y.values)
                    grid = (x.tolist(), y.tolist(), ds.attrs['projection_wkt'])
                    if geometry is None:
                        geometry = grid
                        lat, lon, land, ancillary = land_geometry(x, y, grid[2])
                        atomic_json(output/'ancillary.json', ancillary)
                    elif geometry != grid:
                        raise ValueError('Probe domain geometry changes between days')
                    origins = [(iy, ix) for iy in plan['crop_rows'] for ix in plan['crop_columns']]
                    if any(iy < 0 or ix < 0 or iy+224 > len(y) or ix+224 > len(x) for iy, ix in origins):
                        raise ValueError('Crop outside domain')
                    starts = plan['start_indices'][split]
                    ends = {i+15: i for i in starts}
                    targets = {i+9: i for i in starts}
                    if min(targets) < 12 or max(ends) >= len(times) or min(np.diff(starts)) < 16:
                        raise ValueError('Insufficient warm-up, overlapping clips, or incomplete day')
                    tracker = NwcSafCmaTracker(cfg, lat_deg=lat, lon_deg=lon, is_land=land)
                    ring, saved_targets, samples = deque(maxlen=16), {}, []
                    frame_digest = hashlib.sha256()
                    for index in range(max(ends)+1):
                        frame = first if index == 0 else reader.load(times[index])
                        # Raw Zarr is float16: preserve every finite value exactly.
                        native = frame.astype(np.float16)
                        if not np.array_equal(native.astype(np.float32), frame, equal_nan=True):
                            raise ValueError('Source cannot be represented losslessly as float16')
                        frame_digest.update(native.tobytes())
                        ring.append(native)
                        # Explicit invalid-domain mask, shared by all seven channels.
                        valid = np.isfinite(frame).all(0) & (frame >= 100).all(0) & (frame <= 400).all(0)
                        frame[:, ~valid] = np.nan
                        cma, _ = tracker.step(frame, timestamp=times[index].to_pydatetime())
                        if index in targets:
                            saved_targets[targets[index]] = [(binary_target(cma.cma[iy:iy+224, ix:ix+224]),
                                cma.cma_quality[iy:iy+224, ix:ix+224].copy()) for iy, ix in origins]
                        if index in ends:
                            initial = ends[index]
                            for crop_index, (iy, ix) in enumerate(origins):
                                clip = np.stack([f[:, iy:iy+224, ix:ix+224] for f in ring])
                                if clip.shape != (16,7,224,224) or not np.isfinite(clip).all() or (clip < 100).any() or (clip > 400).any():
                                    raise ValueError(f'Incomplete radiometry: {day_key}/{initial}/{iy}/{ix}')
                                label, quality = saved_targets[initial][crop_index]
                                if not np.isin(label,[0,1]).any():
                                    raise ValueError('No valid target pixels')
                                key = f'{day_key}_{initial:03d}_{iy}_{ix}'
                                row = dict(id=key, split=split, day=str(pd.Timestamp(day).date()),
                                    start_time=times[initial].isoformat(), end_time=times[index].isoformat(),
                                    target_time=times[initial+9].isoformat(), canonical_y=iy, canonical_x=ix,
                                    native_y=len(y)-iy-224 if ds.y.values[0] < ds.y.values[-1] else iy,
                                    native_x=len(x)-ix-224 if ds.x.values[0] > ds.x.values[-1] else ix,
                                    valid_pixels=int((label != 255).sum()), cloud_pixels=int((label == 1).sum()))
                                for field, array in [('clip',clip),('label',label),('quality',quality)]:
                                    path = target_dir/f'{key}_{field}.npy'
                                    with path.with_suffix('.part').open('wb') as stream:
                                        np.save(stream,array,allow_pickle=False)
                                    path.with_suffix('.part').replace(path)
                                    row[field], row[field+'_sha256'] = str(path.relative_to(output)), sha256(path)
                                samples.append(row)
                            del saved_targets[initial]
                        if index % 12 == 0 or index == max(ends):
                            progress = dict(status='building',pid=os.getpid(),updated_unix=time.time(),day=day_key,
                                frame=index+1,frames=max(ends)+1,samples=len(rows)+len(samples),elapsed_seconds=time.monotonic()-start_wall)
                            atomic_json(output/'progress.json',progress)
                            print(json.dumps(progress),flush=True)
                    if source_stamp(source) != stamp:
                        raise ValueError(f'Source changed while reading {day_key}; rerun this day')
                    record = dict(path=str(source),stamp=stamp,processed_frames=max(ends)+1,
                        oriented_float16_frames_sha256=frame_digest.hexdigest(),first_time=times[0].isoformat(),
                        last_time=times[max(ends)].isoformat(),projection_wkt=grid[2])
                    atomic_json(target_dir/'complete.json',dict(samples=samples,source=record))
                    rows.extend(samples); day_records[day_key] = record
            manifest = dict(protocol_sha256=sha256(output/'protocol.json'),ancillary_sha256=sha256(output/'ancillary.json'),
                            days=day_records,samples=rows)
            atomic_json(output/'manifest.json',manifest)
            atomic_json(output/'progress.json',dict(status='completed',pid=os.getpid(),updated_unix=time.time(),
                samples=len(rows),splits={s:sum(r['split']==s for r in rows) for s in plan['splits']},
                elapsed_seconds=time.monotonic()-start_wall))
            return manifest
        finally:
            reader.close()
