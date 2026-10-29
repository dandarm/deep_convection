"""Lossless native-frame NPY cache for repeated SEVIRI crop access.

Frames retain their native coordinates and times. SeviriFrameCacheReader uses
the existing clip reader's temporal, spectral, orientation and finite checks.
The source files remain authoritative; cache provenance is checked per store.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xarray as xr

from .seviri_clips import SeviriZarrClipReader, VIDEO_MAE_CHANNELS, IncompleteSeviriClipError


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def source_stamp(path):
    """Seal the list, sizes and modification times of published source files."""
    records = []
    for file in sorted(Path(path).rglob("*")):
        if file.is_file() and ".rsync-partial" not in file.parts and not file.name.endswith(".part"):
            info = file.stat()
            records.append((str(file.relative_to(path)), info.st_size, info.st_mtime_ns))
    if not records:
        raise IncompleteSeviriClipError(f"Empty or missing source store: {path}")
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def manifest_days(manifest, num_frames=16, cadence_minutes=5):
    days = set()
    for start in pd.to_datetime(manifest.start_time, utc=True):
        end = start + pd.Timedelta(minutes=(num_frames - 1) * cadence_minutes)
        days.update(day.strftime("%Y%m%d") for day in pd.date_range(start.normalize(), end.normalize()))
    return sorted(days)


def build_frame_cache(zarr_root, output, days, *, frames_per_read=8):
    """Write every native frame once; verify exact float16 bits on reread."""
    zarr_root, output = Path(zarr_root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frame cache output must be an empty directory")
    if frames_per_read < 1 or not days:
        raise ValueError("At least one day and a positive read size are required")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = {}
    with SeviriZarrClipReader(zarr_root) as reader:
        for day in sorted(set(days)):
            timestamp = pd.Timestamp(day)
            source = reader._path_for_time(timestamp)
            stamp = source_stamp(source)
            ds = reader._open_store(source)
            array = ds.seviri.transpose("time", "channel", "y", "x")
            if array.dtype != np.float16:
                raise ValueError("This cache preserves a float16 source without conversion")
            times = pd.DatetimeIndex(ds.time.values)
            if times.has_duplicates or times.hasnans or not times.is_monotonic_increasing:
                raise IncompleteSeviriClipError("Source times must be unique and increasing")
            target = output / f"{day}.npy"
            temporary = target.with_suffix(".npy.part")
            cached = np.lib.format.open_memmap(temporary, mode="w+", dtype=array.dtype, shape=array.shape)
            for first in range(0, len(times), frames_per_read):
                last = min(len(times), first + frames_per_read)
                original = array.isel(time=slice(first, last)).values
                cached[first:last] = original
                if not np.array_equal(cached[first:last].view(np.uint16), original.view(np.uint16)):
                    raise ValueError("Cache differs from source float16 values")
            cached.flush()
            del cached
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            checksum = file_digest(temporary)
            if source_stamp(source) != stamp:
                raise ValueError(f"Source changed during cache creation: {source}")
            temporary.replace(target)
            records[str(source.relative_to(zarr_root))] = dict(
                file=target.name, bytes=target.stat().st_size, sha256=checksum,
                source_stamp=stamp, shape=list(array.shape), dtype=str(array.dtype),
                time=[value.isoformat() for value in times],
                channel=[str(value) for value in ds.channel.values],
                y=ds.y.values.tolist(), x=ds.x.values.tolist())
            print(f"Cached {day}: {target.stat().st_size / 2**30:.1f} GiB", flush=True)
    metadata = dict(version=1, units="K", layout="native time,channel,y,x", lossless=True,
                    zarr_root=str(zarr_root), stores=records,
                    elapsed_seconds=time.perf_counter()-started,
                    total_bytes=sum(record["bytes"] for record in records.values()))
    temporary = output / "metadata.json.part"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(output / "metadata.json")
    return metadata


class SeviriFrameCacheReader(SeviriZarrClipReader):
    """Replace compressed storage access while retaining the original checks."""

    def __init__(self, root, cache_root, **kwargs):
        super().__init__(root, **kwargs)
        self.cache_root = Path(cache_root)
        metadata = json.loads((self.cache_root / "metadata.json").read_text())
        if (metadata.get("version") != 1 or metadata.get("units") != "K"
                or metadata.get("layout") != "native time,channel,y,x" or metadata.get("lossless") is not True):
            raise ValueError("Unexpected native-frame cache metadata")
        if Path(metadata["zarr_root"]).resolve() != self.root.resolve():
            raise ValueError("Frame cache belongs to another Zarr root")
        self.entries = metadata["stores"]

    def load_native_into(self, start_time, *, origin_y, origin_x, out):
        """Copy a Kelvin float16 clip directly into a supplied contiguous buffer.

        Native source values, exact RSS times, channel order, bounds and
        orientation are checked before the buffer is returned. Finiteness
        follows require_finite; the raw benchmark can instead validate the
        complete tensor in ForecastWindow.split on the GPU before model use.
        This path is used to measure direct pinned-memory batch preparation.
        """
        expected_shape = (self.num_frames, len(self.channels), self.crop_size, self.crop_size)
        if out.shape != expected_shape or out.dtype != np.float16 or not out.flags.c_contiguous:
            raise ValueError("Destination must be a contiguous float16 clip buffer")
        start = pd.Timestamp(start_time)
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        expected = pd.date_range(start, periods=self.num_frames, freq=f"{self.cadence_minutes}min")
        y0, x0 = int(origin_y), int(origin_x)
        y1, x1 = y0 + self.crop_size, x0 + self.crop_size
        if min(y0, x0) < 0:
            raise IncompleteSeviriClipError("Crop origins must be non-negative")
        offset = 0
        for day, times in pd.Series(expected, index=expected).groupby(expected.normalize(), sort=False):
            ds = self._open_store(self._path_for_time(pd.Timestamp(day)))
            indices = pd.DatetimeIndex(ds.time.values).get_indexer(pd.DatetimeIndex(times.to_numpy()))
            if (indices < 0).any():
                raise IncompleteSeviriClipError("Missing exact RSS observation in frame cache")
            if y1 > ds.sizes["y"] or x1 > ds.sizes["x"]:
                raise IncompleteSeviriClipError("Crop exceeds cached frame bounds")
            y, x = np.asarray(ds.y.values[y0:y1]), np.asarray(ds.x.values[x0:x1])
            if not (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)):
                raise IncompleteSeviriClipError("Non-monotonic y coordinates")
            if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
                raise IncompleteSeviriClipError("Non-monotonic x coordinates")
            selection = slice(indices[0], indices[-1] + 1) if np.all(np.diff(indices) == 1) else indices
            piece = ds.seviri.data[selection, :, y0:y1, x0:x1]
            if y[0] < y[-1]:
                piece = piece[..., ::-1, :]
            if x[0] > x[-1]:
                piece = piece[..., ::-1]
            np.copyto(out[offset:offset + len(indices)], piece)
            offset += len(indices)
        if offset != self.num_frames:
            raise IncompleteSeviriClipError("Incomplete cached temporal window")
        if self.require_finite and not np.isfinite(out).all():
            raise IncompleteSeviriClipError("Clip contains NaN or infinite values")
        return out

    def load_native(self, start_time, *, origin_y, origin_x):
        out = np.empty((self.num_frames, len(self.channels), self.crop_size, self.crop_size), dtype=np.float16)
        return self.load_native_into(start_time, origin_y=origin_y, origin_x=origin_x, out=out)

    def _open_store(self, path):
        path = Path(path)
        if path not in self._stores:
            key = str(path.relative_to(self.root))
            if key not in self.entries:
                raise IncompleteSeviriClipError(f"Day missing from frame cache: {key}")
            entry = self.entries[key]
            if tuple(entry["channel"]) != VIDEO_MAE_CHANNELS:
                raise IncompleteSeviriClipError("Unexpected cached channel order")
            if entry["file"] != f"{path.stem}.npy":
                raise ValueError("Unexpected cached frame filename")
            cached_path = self.cache_root / entry["file"]
            if cached_path.stat().st_size != entry["bytes"]:
                raise IncompleteSeviriClipError("Truncated cached frame array")
            if source_stamp(path) != entry["source_stamp"]:
                raise ValueError(f"Frame cache source has changed: {path}")
            values = np.load(cached_path, mmap_mode="r", allow_pickle=False)
            if values.dtype != np.float16 or list(values.shape) != entry["shape"]:
                raise IncompleteSeviriClipError("Unexpected cached shape or dtype")
            ds = xr.Dataset({"seviri": (("time", "channel", "y", "x"), values)},
                            coords=dict(time=pd.to_datetime(entry["time"]), channel=entry["channel"],
                                        y=entry["y"], x=entry["x"]))
            ds.set_close(values._mmap.close)
            self._stores[path] = ds
        return self._stores[path]
