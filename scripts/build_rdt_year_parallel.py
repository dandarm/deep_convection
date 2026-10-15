#!/usr/bin/env python3
"""Parallel full-domain RDT: independent tasks ONLY at real observation gaps.

Within each task every available five-minute frame is processed causally.
Published frames are atomic, ZIP-contained lossless Zarr; tables stay partitioned.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import signal
import time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from emma_gpm.rdt_preview import archive_inventory
from emma_gpm.rdt.storage import atomic_json, sha256, frame_key, read_table


def split_sequences(times):
    """Midnight is not a boundary. Reject unordered/duplicate timestamps."""
    times = pd.DatetimeIndex(times)
    if times.empty or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Require increasing unique observation timestamps")
    breaks = np.r_[0, np.flatnonzero(np.diff(times.asi8) != pd.Timedelta(minutes=5).value)+1, len(times)]
    return [dict(id=i, start=times[a].isoformat(), end_exclusive=(times[b-1]+pd.Timedelta(minutes=5)).isoformat(),
                 frames=int(b-a), frame_offset=int(a), track_offset=(i+1)*(1 << 32),
                 previous_time=times[a-1].isoformat() if a else None)
            for i, (a, b) in enumerate(zip(breaks[:-1], breaks[1:]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, default=Path("/media/fenrir/disk1/Datasets/MSG-SEVIRI"))
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/rdt_preview/domain_utls_2020_parallel")
    parser.add_argument("--inventory-file", type=Path, help="Trusted local inventory, for repeatable benchmarks")
    parser.add_argument("--max-frames-per-sequence", type=int, help="Integration/benchmark only; output remains partial")
    parser.add_argument("--sequence-limit", type=int, help="Integration/benchmark only")
    parser.add_argument("--crop-catalog", action="store_true", help="Opt in to the diagnostic regular crop lattice; default is full-domain only")
    parser.add_argument("--detector-workers", type=int, default=0, help="Extra detection processes per active sequence; useful for a small serial tail")
    parser.add_argument("--async-save", action="store_true")
    args = parser.parse_args()
    if args.detector_workers < 0:
        parser.error("Detection worker count must be nonnegative")
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f"Supervisor interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, stop_on_signal)
    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        parser.error("Invalid worker count")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence < 1:
        parser.error("Frame limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/"frames").mkdir(exist_ok=True)
    lock = (output/"scan.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), ROOT/"scripts/build_rdt_pseudolabels.py", *sorted((ROOT/"src/emma_gpm/rdt").glob("*.py")),
               ROOT/"src/emma_gpm/rdt_teacher.py", ROOT/"src/emma_gpm/rdt_preview.py", ROOT/"src/emma_gpm/seviri_clips.py"]
    fingerprint = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    plan_path = output/"parallel_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["sources"] != fingerprint or plan["root"] != str(args.zarr_root.resolve()) or plan["year"] != args.year:
            raise ValueError("Frozen source/archive/year changed; use a new output directory")
        if plan["max_frames_per_sequence"] != args.max_frames_per_sequence or plan["sequence_limit"] != args.sequence_limit:
            raise ValueError("Do not mix integration and annual results")
        inventory, times, geometry = pickle.loads((output/"inventory.pkl").read_bytes())
    else:
        inventory, available, geometry = pickle.loads(args.inventory_file.read_bytes()) if args.inventory_file else archive_inventory(args.zarr_root)
        times = available[available.year == args.year]
        segments = split_sequences(times)
        if args.sequence_limit:
            segments = segments[:args.sequence_limit]
        plan = dict(root=str(args.zarr_root.resolve()), year=args.year, frames=len(times),
                    first=times[0].isoformat(), last=times[-1].isoformat(), days=len(times.normalize().unique()),
                    sequences=segments, sources=fingerprint, max_frames_per_sequence=args.max_frames_per_sequence,
                    sequence_limit=args.sequence_limit, partition="only real gaps; continuous tracking across midnight",
                    track_ids="int64: (sequence_id+1)*2**32 + local_track_id; globally unique and deterministic")
        (output/"inventory.pkl").write_bytes(pickle.dumps((inventory, times, geometry), protocol=5))
        atomic_json(plan_path, plan)
    plan["crop_catalog_enabled"]=args.crop_catalog
    atomic_json(plan_path,plan)
    atomic_json(output/"request.json", dict(crop_catalog_enabled=args.crop_catalog,start=f"{args.year}-01-01T00:00:00", end_exclusive=f"{args.year+1}-01-01T00:00:00", frames=plan["frames"]))
    segments = sorted(plan["sequences"], key=lambda s: -s["frames"])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", BLOSC_NTHREADS="1", ZARR_ASYNC_CONCURRENCY="2", ZARR_THREADING_MAX_WORKERS="1")
    active, pending, handles = {}, list(segments), []
    started = time.monotonic()
    old = json.loads((output/"progress.json").read_text()) if (output/"progress.json").exists() else {}
    elapsed_before = old.get("elapsed_seconds", 0.)
    seen, records = set(), {}
    error = None
    last_publish = -float("inf")

    def publish(status):
        statuses, catalogs = [], []
        for segment in plan["sequences"]:
            directory = output/"segments"/f"{segment['id']:03d}"
            state_path = directory/"progress.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="queued", processed=0)
            statuses.append(dict(sequence=segment["id"], frames=segment["frames"], **state))
            protocol_path = directory/"protocol.json"
            if protocol_path.exists() and not (output/"protocol.json").exists():
                protocol = json.loads(protocol_path.read_text())
                protocol.update(start_date=f"{args.year}-01-01T00:00:00", warmup_minutes=0,
                                track_id_offset=None, frame_index_offset=0, parallel_sources=plan["sources"])
                atomic_json(output/"protocol.json", protocol)
            metrics = directory/"frame_metrics.csv"
            if metrics.exists():
                for record in read_table(metrics).to_dict("records"):
                    stamp = pd.Timestamp(record["time"])
                    key = frame_key(stamp)
                    records[key] = record
                    if key not in seen:
                        target = directory/"frames"/key
                        link = output/"frames"/key
                        if not link.exists():
                            link.symlink_to(target, target_is_directory=True)
                        seen.add(key)
            catalog_path = directory/"catalog.csv"
            if args.crop_catalog and catalog_path.exists():
                catalogs.append(pd.read_csv(catalog_path))
        if records:
            path = output/"frame_metrics.csv.part"
            pd.DataFrame([records[k] for k in sorted(records)]).to_csv(path, index=False)
            path.replace(output/"frame_metrics.csv")
        if catalogs:
            catalog = pd.concat(catalogs, ignore_index=True)
            path = output/"catalog.csv.part"
            catalog.to_csv(path, index=False)
            path.replace(output/"catalog.csv")
            atomic_json(output/"catalog_status.json", dict(clips=len(catalog), valid=int((catalog.status=="ok").sum()), windows=int(catalog.start_time.nunique())))
        elapsed = elapsed_before+time.monotonic()-started
        processed = sum(s.get("processed", 0) for s in statuses)
        progress = dict(status=status, processed=processed, requested=plan["frames"], day_frames=processed,
                        elapsed_seconds=elapsed, frames_per_second=processed/elapsed if elapsed else 0,
                        crop_catalog_enabled=args.crop_catalog,workers=args.workers, active_workers=len(active),
                        detector_workers_per_sequence=args.detector_workers, active_detection_workers=len(active)*args.detector_workers,
                        async_save=args.async_save, completed_sequences=sum(s["status"]=="complete" for s in statuses),
                        sequences=len(statuses), last_time=max((s.get("last_time") or "" for s in statuses), default="") or None,
                        cells=sum(s.get("cells",0) for s in statuses), convective_objects=sum(s.get("convective_objects",0) for s in statuses),
                        ot_pixels=sum(s.get("ot_pixels",0) for s in statuses),
                        invalid_radiometry_pixels=sum(s.get("invalid_radiometry_pixels",0) for s in statuses),
                        available_first=plan["first"], available_last=plan["last"], available_days=plan["days"],
                        note="Concurrent sequences: latest time does not imply all earlier days are complete", error=error)
        atomic_json(output/"progress.json", progress)
        atomic_json(output/"sequence_status.json", statuses)
        if status == "complete":
            atomic_json(output/"summary.json", progress)
        print(json.dumps(progress), flush=True)

    try:
        while pending or active:
            while pending and len(active) < args.workers:
                segment = pending.pop(0)
                directory = output/"segments"/f"{segment['id']:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                state = json.loads((directory/"progress.json").read_text()) if (directory/"progress.json").exists() else {}
                if state.get("status") == "complete":
                    continue
                command = [str(ROOT/".venv/bin/python"), str(ROOT/"scripts/build_rdt_pseudolabels.py"),
                           "--zarr-root", str(args.zarr_root), "--start-time", segment["start"], "--end-time", segment["end_exclusive"],
                           "--output-dir", str(directory), "--inventory-file", str(output/"inventory.pkl"),
                           "--track-id-offset", str(segment["track_offset"]), "--frame-index-offset", str(segment["frame_offset"]),
                           "--packed-fields", "--no-joined-table", "--catalog-interval", "288"]
                command += ["--detector-workers", str(args.detector_workers)]
                if args.async_save:
                    command += ["--async-save"]
                if not args.crop_catalog:
                    command += ["--no-crop-catalog"]
                if segment["previous_time"]:
                    command += ["--previous-time", segment["previous_time"]]
                if args.max_frames_per_sequence:
                    command += ["--max-frames", str(args.max_frames_per_sequence)]
                handle = (directory/"training.log").open("a")
                handles.append(handle)
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[segment["id"]] = proc
            if time.monotonic()-last_publish >= 15:
                publish("running")
                last_publish = time.monotonic()
            for segment_id, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    del active[segment_id]
                    if code:
                        raise RuntimeError(f"Sequence {segment_id:03d} failed (exit {code}); see segments/{segment_id:03d}/training.log")
            if active:
                time.sleep(.5)
        limited = bool(args.max_frames_per_sequence or args.sequence_limit)
        publish("partial" if limited else "complete")
    except BaseException as exc:
        error = str(exc)
        for proc in active.values():
            proc.terminate()
        for proc in active.values():
            proc.wait()
        publish("error")
        raise
    finally:
        for handle in handles:
            handle.close()
        lock.close()


if __name__ == "__main__":
    main()
