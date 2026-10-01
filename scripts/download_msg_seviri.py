#!/usr/bin/env python3
"""Download co-temporal MSG/SEVIRI L1.5 native scenes from EUMETSAT.

Credentials are read from process environment or a local, Git-ignored ``.env``
file containing ``EUMETSAT_CONSUMER_KEY`` and ``EUMETSAT_CONSUMER_SECRET``.
Each MSG product is a single native ``.nat`` scene with all SEVIRI channels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime
import os
from pathlib import Path
from threading import local
import time

import eumdac
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = "EO:EUM:DAT:MSG:MSG15-RSS"
_THREAD_LOCAL = local()
_NATIVE_MAGIC = b"FormatName                  : NATIVE"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().removesuffix("Z"))


def credentials() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit("Missing EUMETSAT credentials in environment or .env")
    return key, secret


def is_valid_native(path: Path) -> bool:
    """Check that a cached file is a non-empty MSG native product."""
    if not path.exists() or path.stat().st_size <= len(_NATIVE_MAGIC):
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(len(_NATIVE_MAGIC)) == _NATIVE_MAGIC
    except OSError:
        return False


def thread_datastore(key: str, secret: str) -> eumdac.DataStore:
    """Keep one authenticated EUMDAC client per download worker."""
    datastore = getattr(_THREAD_LOCAL, "datastore", None)
    credentials_seen = getattr(_THREAD_LOCAL, "credentials", None)
    current_credentials = (key, secret)
    if datastore is None or credentials_seen != current_credentials:
        datastore = eumdac.DataStore(eumdac.AccessToken(current_credentials))
        _THREAD_LOCAL.datastore = datastore
        _THREAD_LOCAL.credentials = current_credentials
    return datastore


def download_interval(
    collection,
    start: datetime,
    end: datetime,
    output: Path,
    list_only: bool,
    *,
    collection_id: str,
    key: str,
    secret: str,
    workers: int,
    retries: int,
    read_timeout: int,
) -> int:
    products = list(collection.search(dtstart=start, dtend=end, set="brief"))
    print(f"interval={start.isoformat()}Z/{end.isoformat()}Z products={len(products)}")
    for product in products:
        print(f"{product} {product.sensing_start.isoformat()}Z")
    if list_only:
        return len(products)

    output.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []
    for product in products:
        product_id = str(product)
        product_dir = output / product_id
        product_dir.mkdir(parents=True, exist_ok=True)
        entries = [entry for entry in product.entries if entry.endswith(".nat")]
        if len(entries) != 1:
            raise RuntimeError(f"Expected one .nat entry for {product}, got {entries}")
        target = product_dir / Path(entries[0]).name
        if is_valid_native(target):
            print(f"skip existing {target.name}")
            continue
        pending.append(product_id)

    def download_one(product_id: str) -> Path:
        datastore = thread_datastore(key, secret)
        product = datastore.get_product(
            product_id=product_id,
            collection_id=collection_id,
        )
        entries = [entry for entry in product.entries if entry.endswith(".nat")]
        if len(entries) != 1:
            raise RuntimeError(f"Expected one .nat entry for {product_id}, got {entries}")
        entry = entries[0]
        target = output / product_id / Path(entry).name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                url = product.datastore.urls.get(
                    "datastore",
                    "download product",
                    vars={"collection_id": collection_id, "product_id": product_id},
                ) + "/entry"
                headers = eumdac.common.headers.copy()
                print(
                    f"downloading {target.name} attempt={attempt}/{retries + 1}",
                    flush=True,
                )
                with requests.get(
                    url,
                    auth=product.datastore.token.auth,
                    params={"name": entry},
                    stream=True,
                    headers=headers,
                    timeout=(30, read_timeout),
                ) as response:
                    response.raise_for_status()
                    expected_bytes = response.headers.get("Content-Length")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                if expected_bytes is not None and temporary.stat().st_size != int(expected_bytes):
                    raise RuntimeError(
                        f"Incomplete download for {target.name}: "
                        f"{temporary.stat().st_size} != {expected_bytes} bytes"
                    )
                temporary.replace(target)
                if not is_valid_native(target):
                    raise RuntimeError(f"Invalid MSG native payload: {target}")
                print(
                    f"downloaded {target.name} ({target.stat().st_size / 1024**2:.1f} MiB)",
                    flush=True,
                )
                return target
            except Exception as exc:
                last_error = exc
                if temporary.exists():
                    temporary.unlink()
                if attempt > retries:
                    break
                delay = min(30, 2 ** (attempt - 1))
                print(f"retry {target.name} in {delay}s after: {exc}", flush=True)
                time.sleep(delay)
        raise RuntimeError(f"Download failed for {product_id}: {last_error}")

    print(f"cached={len(products) - len(pending)} pending={len(pending)} workers={workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, product_id) for product_id in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return len(products)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="UTC, e.g. 2020-06-16T23:15:00")
    parser.add_argument("--end", help="UTC, exclusive end")
    parser.add_argument("--manifest", type=Path, help="CSV with slot_start and slot_end columns")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--read-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or args.read_timeout < 1:
        raise SystemExit("Require --workers >= 1, --retries >= 0, --read-timeout >= 1")
    if args.manifest and (args.start or args.end):
        raise SystemExit("Use --manifest or --start/--end, not both")
    if not args.manifest and (not args.start or not args.end):
        raise SystemExit("Provide --manifest or both --start and --end")

    key, secret = credentials()
    token = eumdac.AccessToken((key, secret))
    collection = eumdac.DataStore(token).get_collection(args.collection)
    print(f"collection={collection}")
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as stream:
            intervals = {(row["slot_start"], row["slot_end"]) for row in csv.DictReader(stream)}
    else:
        intervals = {(args.start, args.end)}
    total = 0
    for start_text, end_text in sorted(intervals):
        start, end = parse_utc(start_text), parse_utc(end_text)
        if end <= start:
            raise SystemExit(f"Invalid interval: {start_text}/{end_text}")
        total += download_interval(
            collection,
            start,
            end,
            args.out,
            args.list_only,
            collection_id=args.collection,
            key=key,
            secret=secret,
            workers=args.workers,
            retries=args.retries,
            read_timeout=args.read_timeout,
        )
    print(f"products_total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
