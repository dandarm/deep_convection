"""Freeze readable SEVIRI upload chunks and sample complete experimental clips."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xarray as xr

from .seviri import THERMAL_CHANNELS


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_chunks(store, name, metadata):
    spec = metadata[f"{name}/.zarray"]
    if spec["zarr_format"] != 2 or spec.get("dimension_separator", ".") != ".":
        raise ValueError("Expected dot-separated Zarr v2 chunks")
    counts = [(n + c - 1) // c for n, c in zip(spec["shape"], spec["chunks"])]
    return [store / name / ".".join(map(str, index))
            for index in itertools.product(*(range(n) for n in counts))]


def inventory_upload(root, stable_seconds=5):
    """Read coordinates only after every physical coordinate chunk is stable.

    ctime, not mtime, detects freshly received files when rsync preserves the
    original modification date. Absent radiance chunks are never observations.
    """
    root = Path(root)
    stores, skipped = [], []
    cutoff = time.time() - stable_seconds
    for store in sorted(root.glob("[0-9][0-9][0-9][0-9]/*.zarr")):
        try:
            metadata = json.loads((store / ".zmetadata").read_text())["metadata"]
            headers = [store / key for key in (".zmetadata", ".zgroup", ".zattrs")]
            for name in ("time", "channel", "x", "y", "seviri"):
                headers.extend(store / name / key for key in (".zarray", ".zattrs"))
                if name != "seviri":
                    headers.extend(array_chunks(store, name, metadata))
            if not all(p.is_file() and p.stat().st_size > 0 and p.stat().st_ctime <= cutoff for p in headers):
                raise ValueError("coordinate or metadata files not yet complete/stable")
            hashes = {str(p.relative_to(root)): file_digest(p) for p in headers}
            spec = metadata["seviri/.zarray"]
            if spec["chunks"][:2] != [1, 7] or spec["shape"][1] != 7:
                raise ValueError("Expected one instant and seven channels per chunk")
            if metadata["seviri/.zattrs"].get("_ARRAY_DIMENSIONS") != ["time", "channel", "y", "x"]:
                raise ValueError("Unexpected radiance dimensions")
            with xr.open_zarr(store, consolidated=True, chunks=None) as ds:
                times = pd.DatetimeIndex(ds.time.values).as_unit("ns")
                if not times.is_unique or times.hasnans or (times.asi8 % (300 * 10**9)).any():
                    raise ValueError("Invalid or duplicate five-minute timestamps")
                if list(map(str, ds.channel.values)) != list(THERMAL_CHANNELS):
                    raise ValueError("Unexpected channel order")
                coords = {axis: np.asarray(ds[axis].values) for axis in ("x", "y")}
                if any(not (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)) for v in coords.values()):
                    raise ValueError("Nonmonotonic spatial coordinates")
            if any(file_digest(root / name) != value for name, value in hashes.items()):
                raise ValueError("Coordinates changed while being read")
            present = set()
            for p in (store / "seviri").iterdir():
                if p.name.startswith(".") or not p.is_file():
                    continue
                st = p.stat()
                if st.st_size > 0 and st.st_ctime <= cutoff:
                    present.add(p.name)
            stores.append(dict(store=str(store.relative_to(root)), shape=spec["shape"],
                chunks=spec["chunks"], times=[t.isoformat() for t in times],
                present=sorted(present), headers=hashes,
                x_increasing=bool(coords["x"][-1] > coords["x"][0]),
                y_increasing=bool(coords["y"][-1] > coords["y"][0])))
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            skipped.append(dict(store=str(store.relative_to(root)), reason=str(exc)))
    return dict(stores=stores, skipped=skipped, stable_seconds=stable_seconds)


def crop_chunk_names(record, start_time, y, x, crop=224, frames=16):
    native = pd.DatetimeIndex(pd.to_datetime(record["times"], utc=True))
    start = pd.to_datetime(start_time, utc=True)
    indices = native.get_indexer(pd.date_range(start, periods=frames, freq="5min"))
    if (indices < 0).any():
        raise ValueError("Incomplete time window")
    cy, cx = record["chunks"][2:]
    return [f"{t}.0.{iy}.{ix}" for t in indices
            for iy in range(y // cy, (y + crop - 1) // cy + 1)
            for ix in range(x // cx, (x + crop - 1) // cx + 1)]


def spatial_axis(length, crop=224, stride=112):
    if length < crop or stride < 1:
        return []
    return sorted(set(range(0, length - crop + 1, stride)) | {length - crop})


def available_candidates(inventory, stride=112):
    """Within-day windows only; gaps and absent chunks cannot be crossed."""
    rows = []
    for record in inventory["stores"]:
        times = pd.DatetimeIndex(record["times"])
        sorted_times = times.sort_values()
        cy, cx = record["chunks"][2:]
        height, width = record["shape"][2:]
        present = np.zeros((len(times), (height + cy - 1)//cy, (width + cx - 1)//cx), dtype=bool)
        for name in record["present"]:
            try:
                t, channel, y, x = map(int, name.split("."))
                if channel == 0:
                    present[t, y, x] = True
            except (ValueError, IndexError):
                continue
        origins = list(itertools.product(spatial_axis(height, stride=stride), spatial_axis(width, stride=stride)))
        for start in sorted_times[:-15]:
            stamps = pd.date_range(start, periods=16, freq="5min")
            indices = times.get_indexer(stamps)
            if (indices < 0).any() or stamps[-1].normalize() != start.normalize():
                continue
            complete = present[indices].all(axis=0)
            for y, x in origins:
                if complete[y//cy:(y+223)//cy+1, x//cx:(x+223)//cx+1].all():
                    rows.append(dict(start_time=start.isoformat() + "Z", origin_y=y, origin_x=x,
                                     day=start.strftime("%Y-%m-%d"), time_bin=start.hour//6,
                                     store=record["store"]))
    return pd.DataFrame(rows)


def temporal_partition(candidates, validation_hours=12, embargo_hours=24):
    if candidates.empty:
        raise ValueError("No complete physical 16-frame candidates")
    times = pd.to_datetime(candidates.start_time, utc=True)
    # Use the most recent available interval, with a strict embargo measured
    # after the final training observation, not after its starting timestamp.
    boundary = (times.max() + pd.Timedelta(minutes=80) - pd.Timedelta(hours=validation_hours)).ceil("5min")
    training = candidates[times + pd.Timedelta(minutes=75, hours=embargo_hours) < boundary].copy()
    validation = candidates[times >= boundary].copy()
    reserved = pd.date_range("2020-03-03", "2020-03-09", tz="UTC").strftime("%Y-%m-%d")
    training = training[~training.day.isin(reserved)]
    if training.empty or validation.empty:
        raise ValueError("Not enough uploaded timestamps for nonempty splits and the 24h embargo")
    return training.reset_index(drop=True), validation.reset_index(drop=True), boundary.isoformat()


def balanced_candidate_order(frame, seed):
    """Round-robin geographic sites, UTC days and six-hour bins, with seeded ties."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(indices) for indices in
              frame.groupby(["day", "origin_y", "origin_x", "time_bin"], sort=True).indices.values()]
    ordering = []
    for level in range(max(map(len, groups), default=0)):
        active = [i for i, group in enumerate(groups) if level < len(group)]
        for i in rng.permutation(active):
            ordering.append(groups[i][level])
    return frame.iloc[ordering].reset_index(drop=True)


def source_files_for_manifests(inventory, frames):
    records = {r["store"]: r for r in inventory["stores"]}
    paths = set()
    for frame in frames:
        for row in frame.itertuples(index=False):
            record = records[row.store]
            paths.update(record["headers"])
            paths.update(f"{row.store}/seviri/{name}" for name in
                         crop_chunk_names(record, row.start_time, row.origin_y, row.origin_x))
    return sorted(paths)


def verify_snapshot(root, hashes):
    for name, expected in hashes.items():
        if file_digest(Path(root) / name) != expected:
            raise ValueError(f"Frozen dataset source changed: {name}")
