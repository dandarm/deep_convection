#!/usr/bin/env python3
"""Compare exact clip values and benchmark Base with the production frame loader."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.seviri_clip_files import ClipFileDataset, ClipBatchLoader, SHAPE
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from scripts.profile_seviri_ssl import model_and_optimizer, training_step, Timing
from scripts.pretrain_seviri_ssl import write_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--frame-cache',type=Path,required=True)
    parser.add_argument('--clip-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    stats=json.loads((args.dataset/'channel_stats.json').read_text())
    data=FrameFileDataset(args.frame_cache,args.dataset/'train.csv',zarr_root=stats['zarr_root'])
    old=ClipFileDataset(args.clip_root,args.dataset/'train.csv',zarr_root=stats['zarr_root'])
    a,b=np.empty(SHAPE,dtype=np.float16),np.empty(SHAPE,dtype=np.float16)
    indices=np.random.default_rng(20261026).choice(len(data),64,replace=False)
    for index in indices:
        data.decode_into(index,a);old.decode_into(index,b)
        if not np.array_equal(a.view(np.uint16),b.view(np.uint16)):
            raise ValueError(f'Native-frame and clip files differ at manifest row {index}')
    print('Verified exact float16 bits for 64 randomly selected clips',flush=True)
    results=dict(exact_comparison_clips=64,manifest_sha256=digest(args.dataset/'train.csv'),
        frame_metadata_sha256=data.metadata_sha256,script_sha256=digest(__file__),models={})
    for objective in ('future','joint'):
        model,optimizer=model_and_optimizer(objective,'base')
        loader=ClipBatchLoader(data,batch_size=32,workers=4,prefetch=4,
            mean=stats['global_mean_k'],std=stats['global_std_k'],device='cuda',seed=20261024,shuffle=True)
        iterator=iter(loader);rng=torch.Generator().manual_seed(20261025)
        for step in range(4):
            training_step(model,optimizer,next(iterator)['pixel_values'],step,1,Timing(),rng)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        timing=Timing();start=time.perf_counter();wait=0.
        for step in range(32):
            before=time.perf_counter();pixels=next(iterator)['pixel_values'];wait+=time.perf_counter()-before
            training_step(model,optimizer,pixels,step,1,timing,rng)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        results['models'][objective]=dict(parameters=sum(p.numel() for p in model.parameters()),
            encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
            steps=32,seconds_per_step=elapsed/32,loader_wait_seconds_per_step=wait/32,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,timings=timing.report())
        print(objective, f'{elapsed/32:.3f} s/update, {torch.cuda.max_memory_allocated()/2**30:.1f} GiB',flush=True)
        iterator.close();del model,optimizer,pixels;torch.cuda.empty_cache()
        write_json(args.output_dir/'result.json',results)
    data.close()


if __name__=='__main__':
    main()
