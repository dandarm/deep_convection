#!/usr/bin/env python3
"""Resume a PPS order safely from its generated Python download script.

The PPS script is imported only to obtain its authenticated opener and ordered
file list.  Its ``get`` function is replaced with a streaming, checksummed,
retrying implementation before the order is started.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import urllib.request


CHUNK_SIZE = 1024 * 1024


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_order_script(path: Path):
    spec = importlib.util.spec_from_file_location("pps_order", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load PPS order script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("order_script", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    order_script = args.order_script.resolve(strict=True)
    destination = args.destination.resolve(strict=True)
    order = load_order_script(order_script)
    order.forceHttps = True

    state = {"seen": 0, "downloaded": 0, "skipped": 0}

    def resumable_get(filepath: str, checksum: str | None = None) -> None:
        state["seen"] += 1
        filename = Path(filepath).name
        target = destination / filename
        partial = destination / f"{filename}.part"

        if target.exists() and checksum and sha1(target) == checksum:
            state["skipped"] += 1
            if state["seen"] % 25 == 0:
                print(f"Checked {state['seen']} existing order entries", flush=True)
            return

        if target.exists():
            target.unlink()

        url = "https://arthurhouhttps.pps.eosdis.nasa.gov" + filepath
        for attempt in range(1, args.retries + 1):
            try:
                digest = hashlib.sha1()
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
                        digest.update(chunk)
                if checksum and digest.hexdigest() != checksum:
                    raise OSError("SHA-1 checksum mismatch")
                os.replace(partial, target)
                state["downloaded"] += 1
                print(
                    f"Downloaded {state['seen']}: {filename} "
                    f"({target.stat().st_size / 1024**2:.1f} MiB)",
                    flush=True,
                )
                return
            except Exception as exc:
                partial.unlink(missing_ok=True)
                if attempt == args.retries:
                    raise
                delay = min(60, 2**attempt)
                print(
                    f"Retry {attempt}/{args.retries} for {filename} in {delay}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

    order.get = resumable_get
    try:
        order.getPpsFiles()
    except SystemExit as exc:
        code = int(exc.code or 0)
    else:
        code = 0

    print(
        f"Resume summary: downloaded={state['downloaded']}, "
        f"skipped={state['skipped']}, seen={state['seen']}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
