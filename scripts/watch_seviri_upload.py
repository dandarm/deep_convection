#!/usr/bin/env python3
"""Wait for stable, physically present Zarr chunks; run one real CUDA smoke.

The watcher never writes to the archive. Missing chunks are not interpreted
as Zarr fill values. Only the verified clip is used, not an incomplete day.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_clips import IncompleteSeviriClipError, SeviriZarrClipReader
from scripts.pretrain_seviri_ssl import digest, write_json


def chunk_paths(store, metadata, name):
    array = metadata[f"{name}/.zarray"]
    if array.get("zarr_format") != 2 or array.get("dimension_separator", ".") != ".":
        raise ValueError("Watcher currently supports the project's dot-key Zarr v2 archive")
    counts = [(size + chunk - 1) // chunk for size, chunk in zip(array["shape"], array["chunks"])]
    return [store / name / ".".join(map(str, indices)) for indices in itertools.product(*(range(n) for n in counts))]


def stable(paths, seconds):
    now = time.time()
    try:
        return all(path.is_file() and path.stat().st_size > 0 and now - path.stat().st_mtime >= seconds for path in paths)
    except FileNotFoundError:
        return False


def ready_candidate(root, stable_seconds=30):
    for store in sorted(root.glob("[0-9][0-9][0-9][0-9]/*.zarr")):
        try:
            metadata = json.loads((store / ".zmetadata").read_text())["metadata"]
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        headers = [store / name for name in (".zmetadata", ".zattrs", ".zgroup")]
        coordinates = []
        for name in ("channel", "time", "x", "y"):
            coordinates.extend(chunk_paths(store, metadata, name))
            headers.extend([store / name / ".zarray", store / name / ".zattrs"])
        headers.extend([store / "seviri/.zarray", store / "seviri/.zattrs"])
        if not stable(headers + coordinates, stable_seconds):
            continue
        with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
            native_times = pd.DatetimeIndex(dataset.time.values)
            if not native_times.is_unique or native_times.hasnans:
                continue
            # Historical stores append newly acquired observations after an
            # earlier partial day. The strict reader selects by timestamp and
            # sorts that index; map each selected time back to its physical
            # chunk here instead of assuming storage indices are chronological.
            times = native_times.sort_values()
            array = metadata["seviri/.zarray"]
            if array["shape"][1] != 7 or array["chunks"][:2] != [1, 7]:
                raise ValueError("Unexpected spectral or time chunk layout")
            _, _, height, width = array["shape"]
            _, _, cy, cx = array["chunks"]
            # Prefer an interior crop; then search all complete 224px tiles.
            origins = [(224, 672)] + list(itertools.product(range(0, height - 223, 224), range(0, width - 223, 224)))
            for start in range(len(times) - 15):
                expected = pd.date_range(times[start], periods=16, freq="5min")
                if not times[start:start + 16].equals(expected):
                    continue
                native_indices = native_times.get_indexer(expected)
                for y, x in dict.fromkeys(origins):
                    if y + 224 > height or x + 224 > width:
                        continue
                    chunks = [store / "seviri" / f"{t}.0.{iy}.{ix}"
                        for t in native_indices
                        for iy in range(y // cy, (y + 223) // cy + 1)
                        for ix in range(x // cx, (x + 223) // cx + 1)]
                    paths = headers + coordinates + chunks
                    if not stable(paths, stable_seconds):
                        continue
                    before = {str(path.relative_to(root)): digest(path) for path in paths}
                    try:
                        with SeviriZarrClipReader(root) as reader:
                            raw = reader.load(times[start], origin_y=y, origin_x=x)
                    except (IncompleteSeviriClipError, ValueError, OSError):
                        continue
                    if raw.min() < 100 or raw.max() > 400 or not np.isfinite(raw).all():
                        continue
                    after = {str(path.relative_to(root)): digest(path) for path in paths}
                    if before == after:
                        return dict(start_time=times[start].isoformat() + "Z", origin_y=y, origin_x=x,
                                    source_sha256=after, store=str(store))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stable-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.stable_seconds < 0:
        parser.error("Invalid polling/stability interval")
    with exclusive_file_lock(args.output_dir / ".lock"):
        report_path = args.output_dir / "real_cuda_preflight.json"
        if report_path.exists():
            print(f"Smoke report already exists: {report_path}", flush=True)
            return
        while True:
            candidate = ready_candidate(args.zarr_root, args.stable_seconds)
            if candidate:
                write_json(args.output_dir / "sample.json", candidate)
                write_json(args.output_dir / "status.json", dict(status="smoke_running", sample=candidate["start_time"]))
                command = [sys.executable, str(ROOT / "scripts/smoke_seviri_ssl.py"),
                    "--zarr-root", str(args.zarr_root), "--start-time", candidate["start_time"],
                    "--origin-y", str(candidate["origin_y"]), "--origin-x", str(candidate["origin_x"]),
                    "--output", str(report_path)]
                print("Running verified real-data smoke: " + " ".join(command), flush=True)
                subprocess.run(command, check=True, cwd=ROOT)
                for name, expected in candidate["source_sha256"].items():
                    if digest(args.zarr_root / name) != expected:
                        report_path.rename(args.output_dir / "invalidated_changed_source.json")
                        raise RuntimeError("Source changed while smoke was running")
                write_json(args.output_dir / "status.json", dict(status="completed", sample=candidate["start_time"], report=str(report_path)))
                return
            stores = list(args.zarr_root.glob("[0-9][0-9][0-9][0-9]/*.zarr"))
            state = dict(status="waiting_for_complete_stable_clip", stores_seen=len(stores),
                         checked_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "status.json", state)
            print(json.dumps(state), flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
