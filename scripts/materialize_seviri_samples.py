#!/usr/bin/env python3
"""Materialize a manifest into lossless north-up Kelvin float16 NPY samples."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emma_gpm.seviri_clips import SeviriZarrClipReader, VIDEO_MAE_CHANNELS


def initialize_materializer(zarr_root, output):
    global _reader, _output
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    _reader = SeviriZarrClipReader(zarr_root)
    _output = Path(output)


def save_sample(task):
    index, timestamp, y, x = task
    clip = _reader.load(timestamp, origin_y=y, origin_x=x)
    stored = np.ascontiguousarray(clip, dtype=np.float16)
    if not np.array_equal(stored.astype(np.float32), clip):
        raise ValueError('Float16 would change the source brightness temperatures.')
    name = f'sample_{index:06d}.npy'
    path = _output / name
    temporary = path.with_suffix('.npy.part')
    with temporary.open('wb') as f:
        np.save(f, stored, allow_pickle=False)
    restored = np.load(temporary, allow_pickle=False)
    if not np.array_equal(restored.astype(np.float32), clip):
        raise ValueError(f'Saved sample differs from source: {name}')
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    return dict(file=name, bytes=path.stat().st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-workers', type=int, default=16)
    args = parser.parse_args()
    if args.num_workers < 1:
        raise SystemExit('num-workers must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Output directory must be empty; existing samples are preserved.')
    frame = pd.read_csv(args.manifest)
    if frame.empty:
        raise SystemExit('Empty manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ((i, row.start_time, int(row.origin_y), int(row.origin_x))
             for i, row in enumerate(frame.itertuples(index=False)))
    started = time.perf_counter()
    samples = []
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_materializer,
                             initargs=(args.zarr_root, args.output_dir)) as executor:
        for sample in executor.map(save_sample, tasks, chunksize=1):
            samples.append(sample)
            if len(samples) % 256 == 0:
                print(f'saved={len(samples)}/{len(frame)}', flush=True)
    elapsed = time.perf_counter() - started
    metadata = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    zarr_root=str(args.zarr_root.resolve()), channels=list(VIDEO_MAE_CHANNELS),
                    shape=[16, 7, 224, 224], dtype='float16', units='K',
                    orientation='north-up west-left', cadence_minutes=5, lossless=True,
                    samples=samples, elapsed_seconds=elapsed,
                    total_sample_bytes=sum(s['bytes'] for s in samples))
    temporary = args.output_dir / 'metadata.json.part'
    temporary.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    temporary.replace(args.output_dir / 'metadata.json')
    print(f'completed samples={len(samples)} elapsed_seconds={elapsed:.2f} '
          f'gib={metadata["total_sample_bytes"]/2**30:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
