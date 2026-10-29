#!/usr/bin/env python3
"""Build an exact native-frame cache for the days referenced by a manifest."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import zarr
from numcodecs import blosc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emma_gpm.seviri_frame_cache import build_frame_cache, manifest_days


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 4})
    blosc.set_nthreads(1)
    result = build_frame_cache(args.zarr_root, args.output_dir, manifest_days(pd.read_csv(args.manifest)))
    print(json.dumps({k:v for k,v in result.items() if k!="stores"}, indent=2))


if __name__ == "__main__":
    main()
