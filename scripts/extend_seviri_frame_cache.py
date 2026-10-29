#!/usr/bin/env python3
"""Build missing daily arrays and share existing verified native-frame files."""
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import zarr
from numcodecs import blosc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_frame_cache import build_frame_cache, manifest_days, file_digest, source_stamp
from scripts.pretrain_seviri_ssl import write_json


def extend(root, output, reuse, days):
    previous = json.loads((reuse/'metadata.json').read_text())
    if Path(previous['zarr_root']).resolve() != root.resolve():
        raise ValueError('Reused frame cache belongs to another source archive')
    records = {}
    for day in days:
        key = f'{day[:4]}/{day}.zarr'
        if key in previous['stores']:
            entry = previous['stores'][key]
            source = reuse/entry['file']
        else:
            directory = output/'daily_builds'/day
            if not (directory/'metadata.json').exists():
                build_frame_cache(root, directory, [day])
            entry = json.loads((directory/'metadata.json').read_text())['stores'][key]
            source = directory/entry['file']
        if source_stamp(root/key) != entry['source_stamp'] or file_digest(source) != entry['sha256']:
            raise ValueError(f'Changed source or cached frame file: {day}')
        target = output/entry['file']
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError('Existing frame destination differs from its shared source')
        else:
            os.link(source, target)
        records[key] = entry
        print(f'Verified daily file {day}', flush=True)
    metadata = dict(previous, stores=records, total_bytes=sum(e['bytes'] for e in records.values()),
                    reused_from=str(reuse.resolve()))
    metadata.pop('elapsed_seconds', None)
    write_json(output/'metadata.json', metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    days = sorted({d for path in args.manifest for d in manifest_days(pd.read_csv(path))})
    zarr.config.set({'async.concurrency':8, 'threading.max_workers':4})
    blosc.set_nthreads(1)
    with exclusive_file_lock(args.output_dir/'.lock'):
        extend(args.zarr_root, args.output_dir, args.reuse_root, days)


if __name__ == '__main__':
    main()
