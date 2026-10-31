#!/usr/bin/env python3
"""Build independent CMA labels and lossless canonical probe clips."""
import argparse
import json
from pathlib import Path
import sys
import os
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'CMA_original')]
from emma_gpm.cloud_probe_data import build_cloud_dataset
from emma_gpm.rdt.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/ssl_cloud_probe_dataset.json')
    args = parser.parse_args()
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency':2,'threading.max_workers':1})
    blosc.set_nthreads(1)
    try:
        build_cloud_dataset(args.zarr_root,args.output_dir,json.loads(args.plan.read_text()),ROOT)
    except BaseException as error:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/'progress.json',dict(status='failed',pid=os.getpid(),updated_unix=time.time(),error=repr(error)))
        raise

if __name__ == '__main__':
    main()
