#!/usr/bin/env python3
"""Download a verified, resumable interval of public NWC SAF RDT-CW GIFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.nwcsaf_rdt_archive import (  # noqa: E402
    archive_candidate_urls,
    archive_slots,
    sha256_file,
    validate_archive_gif,
)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _download_slot(
    stamp: pd.Timestamp,
    output: Path,
    archive: str,
    timeout_seconds: int,
    retries: int,
) -> dict:
    directory = output / f"{stamp:%Y}" / f"{stamp:%Y%m%d}"
    directory.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "deep-convection-nwcsaf-reference/1.0"})
        for url in archive_candidate_urls(stamp, archive=archive):
            filename = url.rsplit("/", 1)[-1]
            target = directory / filename
            if target.exists():
                try:
                    width, height = validate_archive_gif(target)
                    return {
                        "time": stamp.isoformat(),
                        "status": "cached",
                        "url": url,
                        "path": str(target.resolve()),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                        "width": width,
                        "height": height,
                    }
                except Exception:
                    target.unlink()
            for attempt in range(retries + 1):
                temporary = target.with_suffix(target.suffix + ".part")
                try:
                    with session.get(url, stream=True, timeout=(20, timeout_seconds)) as response:
                        if response.status_code == 404:
                            break
                        response.raise_for_status()
                        expected = response.headers.get("Content-Length")
                        with temporary.open("wb") as stream:
                            for block in response.iter_content(2**20):
                                if block:
                                    stream.write(block)
                    if expected is not None and temporary.stat().st_size != int(expected):
                        raise RuntimeError("Content-Length mismatch")
                    width, height = validate_archive_gif(temporary)
                    temporary.replace(target)
                    return {
                        "time": stamp.isoformat(),
                        "status": "downloaded",
                        "url": url,
                        "path": str(target.resolve()),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                        "width": width,
                        "height": height,
                    }
                except Exception as error:
                    temporary.unlink(missing_ok=True)
                    errors.append(f"{url} attempt={attempt + 1}: {type(error).__name__}: {error}")
                    if attempt < retries:
                        time.sleep(min(8, 2**attempt))
            # A 404 advances immediately to the next MSG satellite fallback.
    return {
        "time": stamp.isoformat(),
        "status": "missing",
        "url": None,
        "path": None,
        "bytes": 0,
        "sha256": None,
        "width": None,
        "height": None,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC quarter-hour slot")
    parser.add_argument("--end", required=True, help="Exclusive UTC end")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", choices=["v2021", "v2025_msg"], default="v2025_msg")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or args.timeout_seconds < 1:
        parser.error("workers/timeout must be positive and retries nonnegative")
    slots = archive_slots(args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _download_slot,
                stamp,
                args.output_dir,
                args.archive,
                args.timeout_seconds,
                args.retries,
            ): stamp
            for stamp in slots
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['time']} {row['status']} {row.get('url') or ''}", flush=True)
    rows.sort(key=lambda row: row["time"])
    frame = pd.DataFrame(rows)
    temporary = args.output_dir / "manifest.csv.part"
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output_dir / "manifest.csv")
    summary = {
        "archive": args.archive,
        "start": pd.Timestamp(args.start).isoformat(),
        "end_exclusive": pd.Timestamp(args.end).isoformat(),
        "requested": len(rows),
        "available": sum(row["status"] != "missing" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "satellite_fallback_order": ["MSG4", "MSG3", "MSG2", "MSG1"],
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
