#!/usr/bin/env python3
"""Build a resumable activity index from public NWC SAF RDT-CW GIFs.

This is a discovery catalogue, not a numerical-product validation.  It counts
the rendered lifecycle contours and connected contour components in the map
panel while retaining orange split-trigger only as an excluded diagnostic.
GIF payloads are streamed, analysed and discarded; results are committed to
SQLite by the main process and can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import requests
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.nwcsaf_rdt_archive import (  # noqa: E402
    PHASE_RGB,
    archive_candidate_urls,
)


TARGET_PHASES = ("triggering", "growing", "mature", "decay")
ALL_PHASES = (*TARGET_PHASES, "triggering_from_split")
EXPECTED_LAYOUTS = {
    "v2021": {"size": (2221, 1119), "map_bbox": (0, 37, 2221, 1006)},
    "v2025_msg": {"size": (2221, 1308), "map_bbox": (0, 37, 2221, 1188)},
}
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ScanPeriod:
    name: str
    archive: str
    start: pd.Timestamp
    end: pd.Timestamp


def _utc_naive(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Finite timestamps are required")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def parse_period(values: list[str]) -> ScanPeriod:
    name, archive, start_raw, end_raw = values
    if archive not in EXPECTED_LAYOUTS:
        raise ValueError(f"Unknown archive {archive!r}")
    start, end = _utc_naive(start_raw), _utc_naive(end_raw)
    if start >= end or start.minute or start.second or start.microsecond:
        raise ValueError("Periods must be increasing and start on an exact UTC hour")
    return ScanPeriod(name=name, archive=archive, start=start, end=end)


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "deep-convection-research/1.0 "
                    "(public NWC SAF GIF activity indexing; contact via nwcsaf.org source page)"
                )
            }
        )
        _THREAD_LOCAL.session = session
    return session


def _candidate_urls(timestamp: pd.Timestamp, archive: str) -> tuple[str, ...]:
    urls = archive_candidate_urls(timestamp, archive=archive)
    # The public MSG archive is currently MSG3.  Keep all documented fallbacks,
    # but avoid one guaranteed MSG4 404 for almost every historical slot.
    return (urls[1], urls[0], urls[2], urls[3])


def _component_records(mask: np.ndarray, minimum_pixels: int = 4) -> list[dict[str, float | int]]:
    connected = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(connected, structure=np.ones((3, 3), dtype=np.uint8))
    if not count or not mask.any():
        return []
    height, width = mask.shape
    rows, columns = np.where(mask)
    identifiers = labels[rows, columns]
    pixels = np.bincount(identifiers, minlength=count + 1)
    row_sums = np.bincount(identifiers, weights=rows, minlength=count + 1)
    column_sums = np.bincount(identifiers, weights=columns, minlength=count + 1)
    keep = np.flatnonzero(pixels >= minimum_pixels)
    keep = keep[keep != 0]
    return [
        {
            "line_pixels": int(pixels[identifier]),
            "row_fraction": float(
                row_sums[identifier] / pixels[identifier] / max(1, height - 1)
            ),
            "column_fraction": float(
                column_sums[identifier] / pixels[identifier] / max(1, width - 1)
            ),
        }
        for identifier in keep
    ]


def analyse_gif_bytes(
    payload: bytes,
    archive: str,
    *,
    color_tolerance: float = 70.0,
) -> dict[str, Any]:
    layout = EXPECTED_LAYOUTS[archive]
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "GIF" or image.size != layout["size"]:
            raise ValueError(f"Unexpected {archive} GIF format/size: {image.format} {image.size}")
        image.seek(0)
        if image.mode == "P" and image.getpalette() is not None:
            colour_indices = np.asarray(image, dtype=np.uint8)
            raw_palette = np.asarray(image.getpalette(), dtype=np.uint8)
            palette_rgb = raw_palette.reshape(-1, 3)
            image_shape = (*colour_indices.shape, 3)
        else:
            # Defensive fallback for an unusual GIF mode.  The public archive
            # is palettized, so normal scans avoid expanding every pixel to RGB.
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            palette_rgb, colour_indices = np.unique(
                rgb.reshape(-1, 3), axis=0, return_inverse=True
            )
            colour_indices = colour_indices.reshape(rgb.shape[:2])
            image_shape = rgb.shape
    x0, y0, x1, y1 = layout["map_bbox"]
    map_indices = colour_indices[y0:y1, x0:x1]
    names = tuple(PHASE_RGB)
    phase_palette = np.asarray([PHASE_RGB[name] for name in names], dtype=np.int16)
    delta = palette_rgb.astype(np.int16)[:, None, :] - phase_palette[None, :, :]
    squared = np.sum(delta.astype(np.int32) ** 2, axis=-1)
    nearest_lut = np.argmin(squared, axis=-1)
    accepted_lut = (
        np.take_along_axis(squared, nearest_lut[:, None], axis=-1)[:, 0]
        <= color_tolerance**2
    )
    nearest = nearest_lut[map_indices]
    accepted = accepted_lut[map_indices]
    phase_records: dict[str, list[dict[str, float | int]]] = {}
    result: dict[str, Any] = {"width": image_shape[1], "height": image_shape[0]}
    for index, name in enumerate(names):
        line = accepted & (nearest == index)
        components = _component_records(line)
        phase_records[name] = components
        result[f"{name}_line_pixels"] = int(line.sum())
        result[f"{name}_components"] = len(components)
    result["target_line_pixels"] = int(
        sum(result[f"{name}_line_pixels"] for name in TARGET_PHASES)
    )
    result["target_components"] = int(
        sum(result[f"{name}_components"] for name in TARGET_PHASES)
    )
    result["phase_richness"] = int(
        sum(result[f"{name}_components"] > 0 for name in TARGET_PHASES)
    )
    objects = []
    for name in ALL_PHASES:
        objects.extend({"phase": name, **record} for record in phase_records[name])
    result["objects_json"] = json.dumps(objects, separators=(",", ":"))
    return result


def scan_slot(
    period: ScanPeriod,
    timestamp: pd.Timestamp,
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    started = time.monotonic()
    errors: list[str] = []
    for url in _candidate_urls(timestamp, period.archive):
        for attempt in range(retries + 1):
            try:
                response = _session().get(url, timeout=timeout_seconds)
                if response.status_code == 404:
                    break
                response.raise_for_status()
                if "image" not in response.headers.get("content-type", ""):
                    raise ValueError(f"Unexpected content type {response.headers.get('content-type')}")
                metrics = analyse_gif_bytes(response.content, period.archive)
                satellite = url.split("_RDT-CW_", 1)[1].split("_", 1)[0]
                return {
                    "period": period.name,
                    "archive": period.archive,
                    "time": timestamp.isoformat(),
                    "status": "ok",
                    "satellite": satellite,
                    "url": url,
                    "bytes": len(response.content),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error": None,
                    **metrics,
                }
            except (requests.RequestException, ValueError, OSError) as error:
                errors.append(f"{url} attempt={attempt}: {type(error).__name__}: {error}")
                if attempt < retries:
                    time.sleep(min(4.0, 0.5 * 2**attempt))
    return {
        "period": period.name,
        "archive": period.archive,
        "time": timestamp.isoformat(),
        "status": "missing",
        "satellite": None,
        "url": None,
        "bytes": 0,
        "sha256": None,
        "width": None,
        "height": None,
        "target_line_pixels": 0,
        "target_components": 0,
        "phase_richness": 0,
        **{f"{name}_{suffix}": 0 for name in ALL_PHASES for suffix in ("line_pixels", "components")},
        "objects_json": "[]",
        "elapsed_seconds": time.monotonic() - started,
        "error": " | ".join(errors[-4:]) if errors else "all candidates returned 404",
    }


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    columns = [
        "period TEXT NOT NULL",
        "archive TEXT NOT NULL",
        "time TEXT NOT NULL",
        "status TEXT NOT NULL",
        "satellite TEXT",
        "url TEXT",
        "bytes INTEGER NOT NULL",
        "sha256 TEXT",
        "width INTEGER",
        "height INTEGER",
        "target_line_pixels INTEGER NOT NULL",
        "target_components INTEGER NOT NULL",
        "phase_richness INTEGER NOT NULL",
    ]
    columns.extend(
        f"{name}_{suffix} INTEGER NOT NULL"
        for name in ALL_PHASES
        for suffix in ("line_pixels", "components")
    )
    columns.extend(
        [
            "objects_json TEXT NOT NULL",
            "elapsed_seconds REAL NOT NULL",
            "error TEXT",
            "scanned_at TEXT NOT NULL",
            "PRIMARY KEY(period, time)",
        ]
    )
    connection.execute(f"CREATE TABLE IF NOT EXISTS slots ({', '.join(columns)})")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    return connection


def _commit_rows(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    records = []
    scanned_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        records.append({**row, "scanned_at": scanned_at})
    keys = list(records[0])
    placeholders = ",".join("?" for _ in keys)
    updates = ",".join(f"{key}=excluded.{key}" for key in keys if key not in ("period", "time"))
    statement = (
        f"INSERT INTO slots ({','.join(keys)}) VALUES ({placeholders}) "
        f"ON CONFLICT(period,time) DO UPDATE SET {updates}"
    )
    with connection:
        connection.executemany(statement, [[record[key] for key in keys] for record in records])


def _export_csv(connection: sqlite3.Connection, path: Path) -> None:
    frame = pd.read_sql_query("SELECT * FROM slots ORDER BY period, time", connection)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        nargs=4,
        action="append",
        metavar=("NAME", "ARCHIVE", "START", "END"),
        help="Repeatable half-open UTC period; ARCHIVE is v2021 or v2025_msg",
    )
    parser.add_argument(
        "--period-csv",
        type=Path,
        help=(
            "Optional candidate table with archive/start/end_exclusive and either name or "
            "selection_rank; it may be combined with --period"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.cadence_minutes <= 0 or 60 % args.cadence_minutes or args.workers < 1:
        parser.error("Cadence must be a positive divisor of 60 and workers must be positive")
    periods = [parse_period(values) for values in (args.period or [])]
    if args.period_csv:
        table = pd.read_csv(args.period_csv)
        required = {"archive", "start", "end_exclusive"}
        if not required.issubset(table):
            parser.error(f"Period CSV lacks columns {sorted(required - set(table))}")
        for index, row in enumerate(table.itertuples(index=False), start=1):
            name = getattr(row, "name", None)
            if not isinstance(name, str) or not name:
                rank = getattr(row, "selection_rank", index)
                name = f"candidate_{int(rank):03d}"
            periods.append(
                parse_period([name, str(row.archive), str(row.start), str(row.end_exclusive)])
            )
    if not periods:
        parser.error("At least one --period or --period-csv is required")
    if len({period.name for period in periods}) != len(periods):
        parser.error("Period names must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "activity.sqlite"
    csv_path = args.output_dir / "activity.csv"
    progress_path = args.output_dir / "progress.json"
    connection = _connect(database_path)
    specification = {
        "periods": [
            {
                "name": period.name,
                "archive": period.archive,
                "start": period.start.isoformat(),
                "end_exclusive": period.end.isoformat(),
            }
            for period in periods
        ],
        "cadence_minutes": args.cadence_minutes,
        "target_phases": list(TARGET_PHASES),
        "excluded_phase": "triggering_from_split",
        "layouts": EXPECTED_LAYOUTS,
        "screening_semantics": "rendered contour pixels and connected contour components",
    }
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            ("scan_specification", json.dumps(specification, sort_keys=True)),
        )
    existing = {
        (row[0], row[1]): row[2]
        for row in connection.execute("SELECT period,time,status FROM slots")
    }
    tasks: list[tuple[ScanPeriod, pd.Timestamp]] = []
    for period in periods:
        timestamps = pd.date_range(
            period.start,
            period.end,
            freq=f"{args.cadence_minutes}min",
            inclusive="left",
        )
        for timestamp in timestamps:
            status = existing.get((period.name, timestamp.isoformat()))
            if status is None or (args.retry_missing and status == "missing"):
                tasks.append((period, timestamp))

    total_expected = sum(
        len(pd.date_range(period.start, period.end, freq=f"{args.cadence_minutes}min", inclusive="left"))
        for period in periods
    )
    started = time.monotonic()
    completed = 0
    buffered: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    scan_slot,
                    period,
                    timestamp,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                ): (period.name, timestamp)
                for period, timestamp in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                buffered.append(future.result())
                completed += 1
                if len(buffered) >= 25:
                    _commit_rows(connection, buffered)
                    buffered.clear()
                if completed % args.progress_every == 0 or completed == len(tasks):
                    _commit_rows(connection, buffered)
                    buffered.clear()
                    counts = dict(connection.execute("SELECT status,COUNT(*) FROM slots GROUP BY status"))
                    elapsed = time.monotonic() - started
                    progress = {
                        "status": "scanning" if completed < len(tasks) else "complete",
                        "completed_this_run": completed,
                        "remaining_this_run": len(tasks) - completed,
                        "catalogued_total": sum(counts.values()),
                        "expected_total": total_expected,
                        "status_counts": counts,
                        "elapsed_seconds_this_run": elapsed,
                        "slots_per_second": completed / elapsed if elapsed else None,
                    }
                    temporary = progress_path.with_suffix(".json.part")
                    temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                    temporary.replace(progress_path)
                    print(json.dumps(progress), flush=True)
    finally:
        _commit_rows(connection, buffered)
        _export_csv(connection, csv_path)
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
