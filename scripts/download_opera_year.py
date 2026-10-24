#!/usr/bin/env python3
"""Download one complete year of historical OPERA DBZH/QIND composites.

The historical ODYSSEY archive is sampled every 15 minutes.  Downloads are
resumable at file granularity and use an atomic ``.part`` file so interrupted
transfers are never mistaken for complete HDF5 products.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, UTC
import os
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.request

import h5py
import pandas as pd


ENDPOINT = "https://s3.waw3-1.cloudferro.com/openradar-archive"
PRODUCT = "DBZH_QIND"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Result:
    timestamp: pd.Timestamp
    status: str
    path: str
    bytes: int
    attempts: int
    message: str = ""


def opera_key(timestamp: pd.Timestamp, product: str = PRODUCT) -> str:
    return (
        f"{timestamp:%Y/%m/%d}/OPERA/COMP/"
        f"OPERA@{timestamp:%Y%m%dT%H%M}@0@{product}.h5"
    )


def validate_opera_file(path: Path) -> None:
    """Reject truncated files and products without both expected quantities."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("empty file")
    with h5py.File(path, "r") as handle:
        if int(handle["where"].attrs["xsize"]) != 1900:
            raise ValueError("unexpected xsize")
        if int(handle["where"].attrs["ysize"]) != 2200:
            raise ValueError("unexpected ysize")
        quantities: set[str] = set()
        for name in handle:
            if not name.startswith("dataset") or "what" not in handle[name]:
                continue
            value = handle[name]["what"].attrs.get("quantity", b"")
            if isinstance(value, bytes):
                value = value.decode("ascii")
            quantities.add(str(value))
        if not {"DBZH", "QIND"}.issubset(quantities):
            raise ValueError(f"missing quantities: {sorted(quantities)}")


def download_one(
    timestamp: pd.Timestamp,
    root: Path,
    *,
    retries: int,
    timeout: int,
    validate_existing: bool,
) -> Result:
    key = opera_key(timestamp)
    path = root / key
    if path.exists() and path.stat().st_size > 0:
        try:
            if validate_existing:
                validate_opera_file(path)
            return Result(timestamp, "existing", str(path), path.stat().st_size, 0)
        except Exception:
            path.unlink(missing_ok=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    url = f"{ENDPOINT}/{key}"
    last_error = ""
    for attempt in range(1, retries + 2):
        part.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "deep-convection-opera-downloader/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open(
                "wb"
            ) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            validate_opera_file(part)
            os.replace(part, path)
            return Result(
                timestamp, "downloaded", str(path), path.stat().st_size, attempt
            )
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}: {error.reason}"
            if error.code == 404:
                part.unlink(missing_ok=True)
                return Result(timestamp, "missing", str(path), 0, attempt, last_error)
        except Exception as error:  # network and HDF5 validation failures are retried
            last_error = f"{type(error).__name__}: {error}"
        part.unlink(missing_ok=True)
        if attempt <= retries:
            time.sleep(min(2 ** (attempt - 1), 8) + random.random())
    return Result(timestamp, "failed", str(path), 0, retries + 1, last_error)


def year_slots(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=1, day=1)
    stop = pd.Timestamp(year=year + 1, month=1, day=1)
    return pd.date_range(start, stop, freq="15min", inclusive="left")


def write_manifest(results: list[Result], path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [item.timestamp.isoformat() for item in results],
            "status": [item.status for item in results],
            "path": [item.path for item in results],
            "bytes": [item.bytes for item in results],
            "attempts": [item.attempts for item in results],
            "message": [item.message for item in results],
        }
    ).sort_values("timestamp_utc")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\raw\opera"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/opera_download_2020.csv"),
    )
    args = parser.parse_args()
    if args.year < 2000 or args.year > datetime.now(UTC).year:
        raise SystemExit("year is outside the supported archive range")
    if args.workers < 1 or args.retries < 0:
        raise SystemExit("workers must be positive and retries non-negative")

    slots = year_slots(args.year)
    results: list[Result] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                timestamp,
                args.root,
                retries=args.retries,
                timeout=args.timeout,
                validate_existing=args.validate_existing,
            ): timestamp
            for timestamp in slots
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if completed % args.progress_every == 0 or completed == len(slots):
                    counts = pd.Series([item.status for item in results]).value_counts()
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = completed / elapsed
                    with PRINT_LOCK:
                        print(
                            f"completed={completed}/{len(slots)} rate={rate:.1f}/s "
                            + " ".join(
                                f"{name}={int(counts.get(name, 0))}"
                                for name in ("downloaded", "existing", "missing", "failed")
                            ),
                            flush=True,
                        )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print("Interrupted; writing the partial manifest.", flush=True)
        finally:
            write_manifest(results, args.manifest)

    counts = pd.Series([item.status for item in results]).value_counts()
    failed = int(counts.get("failed", 0))
    print(f"manifest={args.manifest.resolve()}")
    print(counts.to_string())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
