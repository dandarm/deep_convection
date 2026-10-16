#!/usr/bin/env python3
"""Incremental full-domain multispectral teacher; crop only published outputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pickle
import signal
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from emma_gpm.file_lock import acquire_file_lock
from emma_gpm.process_metrics import peak_rss_gib
from emma_gpm.rdt import RDTDomainTeacher, RDTTeacherConfig
from emma_gpm.rdt.storage import DomainFrameReader, atomic_json, restore_latest, save_frame, sha256, publish_domain_tables
from emma_gpm.rdt_teacher import projected_pixel_areas
from emma_gpm.rdt_preview import archive_inventory
from emma_gpm.rdt.catalog import publish_crop_catalog
from emma_gpm.rdt.execution import prepared_frames, process_frames


def publish(output, records, requested, start, started, status, prior_elapsed=0, error=None):
    target = [r for r in records if pd.Timestamp(r["time"]) >= start]
    summary = dict(status=status, processed=len(records), requested=requested, day_frames=len(target),
        last_time=records[-1]["time"] if records else None,
        elapsed_seconds=prior_elapsed+time.monotonic()-started,
        cells=sum(r["cells"] for r in target), convective_objects=sum(r["convective_objects"] for r in target),
        ot_pixels=sum(r["ot_pixels"] for r in target),
        mean_frame_seconds=float(np.mean([r["frame_seconds"] for r in target])) if target else None,
        peak_rss_gib=max((r["rss_gib"] for r in records), default=0),
        resets=sum(bool(r["reset_reason"]) for r in records),
        invalid_radiometry_pixels=sum(r.get("invalid_radiometry_pixels",0) for r in target), error=error)
    atomic_json(output/"progress.json", summary)
    if records and (len(records) % 16 == 0 or status != "running"):
        temp = output/"frame_metrics.csv.part"
        pd.DataFrame(records).drop(columns=[c for c in ["checkpoint_sha256", "objects_sha256", "links_sha256"] if c in records[0]]).to_csv(temp, index=False)
        temp.replace(output/"frame_metrics.csv")
    if status == "complete":
        atomic_json(output/"summary.json", summary)
    return summary


def accept_quality_mask_upgrade(path, candidate, audit_path):
    """Narrow, auditable compatibility exception for this validation repair.

    Previous committed frames passed the old global [100,400] guard on every
    observed pixel. The new per-pixel domain mask is therefore identical on
    those frames. An explicit report pins both old and new source hashes;
    changed thresholds, geometry, configuration or other sources still fail.
    """
    if audit_path is None:
        raise ValueError("Different source/config/geometry; use a new output directory")
    prior = json.loads(path.read_text())
    audit = json.loads(Path(audit_path).read_text())
    allowed = {"src/emma_gpm/rdt/pipeline.py", "src/emma_gpm/rdt/storage.py", "scripts/build_rdt_pseudolabels.py"}
    changed = {name for name in set(prior["sources"]) | set(candidate["sources"])
               if prior["sources"].get(name) != candidate["sources"].get(name)}
    if (sha256(path) != audit.get("previous_protocol_sha256") or
            not changed <= allowed or not changed or
            {k:v for k,v in prior.items() if k != "sources"} != {k:v for k,v in candidate.items() if k != "sources"} or
            prior["sources"] != audit.get("old_sources") or candidate["sources"] != audit.get("new_sources") or
            audit.get("compatibility_check") != "previous-good-frame-exact-equality" or
            not audit.get("previous_outputs_unchanged") or not audit.get("bad_pixels_excluded_without_repair")):
        raise ValueError("Quality-mask migration is not compatible with this protocol")


def main():
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f"Worker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-root", type=Path, default=Path(os.environ.get("SEVIRI_ZARR_ROOT", "/media/fenrir/disk1/Datasets/MSG-SEVIRI")))
    parser.add_argument("--start-date", default="2020-07-11")
    parser.add_argument("--end-date", default=None, help="Inclusive; extend this date to resume across days")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/rdt_preview/domain_utls_july2020")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--area-cache", type=Path, default=ROOT/"results/rdt_preview/annual8192")
    parser.add_argument("--max-frames", type=int, help="Optional integration limit; next invocation resumes")
    parser.add_argument("--quality-mask-upgrade", type=Path, help="Explicit source-pinned compatibility audit for the radiometric validation repair")
    parser.add_argument("--start-time", help="Exact independent sequence start; no invented warm-up")
    parser.add_argument("--end-time", help="Exclusive exact sequence end")
    parser.add_argument("--inventory-file", type=Path, help="Trusted local frozen archive inventory")
    parser.add_argument("--track-id-offset", type=int, default=0)
    parser.add_argument("--frame-index-offset", type=int, default=0)
    parser.add_argument("--previous-time", help="Previous observation across the real gap")
    parser.add_argument("--packed-fields", action="store_true")
    parser.add_argument("--no-joined-table", action="store_true", help="Keep object/lineage tables partitioned by frame")
    parser.add_argument("--catalog-interval", type=int, default=16)
    parser.add_argument("--no-crop-catalog", action="store_true", help="Produce full-domain fields only; study crops remain on demand")
    parser.add_argument("--detector-workers", type=int, default=0, help="Bounded current-frame detection processes; tracking stays serial")
    parser.add_argument("--async-save", action="store_true", help="Overlap one atomic save with the next causal step")
    args = parser.parse_args()
    if args.detector_workers < 0:
        parser.error("Detection worker count must be nonnegative")
    if args.packed_fields:
        import zarr
        from numcodecs import blosc
        zarr.config.set({"async.concurrency": 2, "threading.max_workers": 1})
        blosc.set_nthreads(1)
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date or args.start_date).normalize()+pd.Timedelta(days=1)
    if args.start_time:
        start = pd.Timestamp(args.start_time)
    if args.end_time:
        end = pd.Timestamp(args.end_time)
    if start >= end or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Invalid time range/frame limit")
    cfg = RDTTeacherConfig(**json.loads(args.config.read_text())) if args.config else RDTTeacherConfig()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_file_lock(output/"scan.lock")
    reader = DomainFrameReader(args.zarr_root)
    started = time.monotonic()
    records, prior_elapsed = [], 0.
    requested = json.loads((output/"request.json").read_text()).get("frames",0) if (output/"request.json").exists() else 0
    try:
        if args.inventory_file:
            inventory, available, geometry = pickle.loads(args.inventory_file.read_bytes())
        else:
            inventory, available, geometry = archive_inventory(args.zarr_root)
        warmup = 0 if args.start_time else 15
        times = available[(available >= start-pd.Timedelta(minutes=warmup)) & (available < end)]
        requested = len(times)
        if not len(times) or not (times >= start).any():
            raise ValueError("No published frames in requested range")
        first = reader.load(times[0])
        x, y, projection = geometry
        geometry_hash = hashlib.sha256(x.tobytes()+y.tobytes()+projection.encode()).hexdigest()
        sources = [*sorted((ROOT/"src/emma_gpm/rdt").glob("*.py")), Path(__file__),
                   ROOT/"src/emma_gpm/seviri_clips.py", ROOT/"src/emma_gpm/rdt_teacher.py"]
        protocol = dict(teacher_version=cfg.version, config=asdict(cfg), config_hash=cfg.config_hash,
            root=str(args.zarr_root.resolve()), start_date=start.isoformat(), warmup_minutes=warmup,
            track_id_offset=args.track_id_offset, frame_index_offset=args.frame_index_offset,
            field_storage="zarr-v2-lossless-zip" if args.packed_fields else "zarr-v2-directory",
            projection_wkt=projection, geometry_sha256=geometry_hash, shape=list(first.shape[1:]),
            orientation="north-up west-left", sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
            missing_observations="explicit invalid mask; touching objects uncertain; gaps reset tracking",
            temperature_reference="SEVIRI-derived UTLS/anvil temperature proxy v0, not WMO tropopause")
        path = output/"protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            accept_quality_mask_upgrade(path, protocol, args.quality_mask_upgrade)
        atomic_json(path, protocol)
        # Immutable metadata snapshots for the days already requested. Extending
        # the end date appends new days without changing existing snapshots.
        snapshot_path = output/"source_days.json"
        snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
        for day in times.normalize().unique():
            source = args.zarr_root/f"{day:%Y}"/f"{day:%Y%m%d}.zarr"/".zmetadata"
            key = str(source.resolve())
            current_hash = sha256(source)
            if key in snapshot and snapshot[key] != current_hash:
                raise ValueError("Published source metadata changed; use a new label output")
            snapshot[key] = current_hash
        atomic_json(snapshot_path, snapshot)
        area_path = output/"pixel_area_north_up.npy"
        if not area_path.exists():
            cached = args.area_cache/"pixel_area_native.npy"
            cache_protocol = args.area_cache/"protocol.json"
            if cached.exists() and cache_protocol.exists() and json.loads(cache_protocol.read_text()).get("geometry_sha256") == geometry_hash:
                area = np.load(cached)
            else:
                area = projected_pixel_areas(x, y, projection, allow_invalid=True)
            if y[0] < y[-1]:
                area = area[::-1]
            if x[0] > x[-1]:
                area = area[:, ::-1]
            temporary = area_path.with_suffix(".npy.part")
            with temporary.open("wb") as handle:
                np.save(handle, np.ascontiguousarray(area))
            temporary.replace(area_path)
        area = np.load(area_path)
        if (output/"progress.json").exists():
            # Checksum verification may take time on a month of outputs.
            state = json.loads((output/"progress.json").read_text())
            prior_elapsed = state.get("elapsed_seconds",0.)
            atomic_json(output/"progress.json", dict(state,status="verifying_checkpoint",requested=requested,error=None))
        teacher, records = restore_latest(output, cfg.config_hash)
        if teacher is None:
            teacher = RDTDomainTeacher(area, cfg, track_id_offset=args.track_id_offset)
            teacher.frame_index = args.frame_index_offset
            if args.previous_time:
                teacher.last_time = pd.Timestamp(args.previous_time)
        else:
            actual_last = reader.load(teacher.last_time)
            if hashlib.sha256(actual_last.tobytes()).hexdigest() != records[-1]["source_frame_sha256"]:
                raise ValueError("Last input frame changed since checkpoint")
            if records[-1]["time"] >= end.isoformat():
                raise ValueError("Cannot shrink an existing processed time range")
        atomic_json(output/"request.json", dict(start=start.isoformat(), end_exclusive=end.isoformat(), frames=len(times), crop_catalog_enabled=not args.no_crop_catalog, detector_workers=args.detector_workers, async_save=args.async_save, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        publish(output, records, len(times), start, started, "running", prior_elapsed)
        def frame_metadata(frame, result, frame_started):
            table = result.objects
            return dict(source_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                cells=len(table), convective_objects=int(table.class_id.isin([2, 3]).sum()) if len(table) else 0,
                ot_pixels=int(result.fields["ot_mask"].sum()), provisional_ot_pixels=int(result.fields["provisional_ot"].sum()),
                valid_fraction=float(result.fields["valid_observation"].mean()),
                invalid_radiometry_pixels=int(result.fields["invalid_radiometry"].sum()),
                valid_proxies=int((table.proxy_confidence >= cfg.min_proxy_confidence).sum()) if len(table) else 0,
                mean_proxy_confidence=float(table.proxy_confidence.mean()) if len(table) else None,
                valid_temporal_objects=int(table.temporal_valid.sum()) if len(table) else 0,
                reset_reason=result.reset_reason, frame_seconds=time.monotonic()-frame_started,
                rss_gib=peak_rss_gib())

        def on_commit(record):
            records.append(record)
            if not args.no_crop_catalog and (len(records) == 16 or len(records) % args.catalog_interval == 0):
                publish_crop_catalog(output, start, times, area, x, y)
            summary = publish(output, records, len(times), start, started, "running", prior_elapsed)
            print(json.dumps({k: summary[k] for k in ("processed", "requested", "last_time", "mean_frame_seconds", "peak_rss_gib")}), flush=True)

        remaining = times if teacher.last_time is None else times[times > teacher.last_time]
        selected = remaining[:args.max_frames] if args.max_frames else remaining
        process_frames(teacher, prepared_frames(args.zarr_root, selected, area, cfg, args.detector_workers),
            output, frame_metadata, on_commit, packed=args.packed_fields, async_save=args.async_save)
        if args.max_frames and len(selected) >= args.max_frames:
            publish(output, records, len(times), start, started, "partial", prior_elapsed)
            return 0
        if not args.no_crop_catalog:
            publish_crop_catalog(output, start, times, area, x, y)
        if not args.no_joined_table:
            publish_domain_tables(output)
        publish(output, records, len(times), start, started, "complete", prior_elapsed)
        return 0
    except BaseException as error:
        publish(output, records, requested, start, started, "error", prior_elapsed, str(error))
        raise
    finally:
        reader.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
