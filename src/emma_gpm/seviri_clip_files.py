"""Lossless per-clip files and ordered threaded delivery of raw Kelvin batches."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc, blosc

from .seviri_clips import VIDEO_MAE_CHANNELS, IncompleteSeviriClipError

SHAPE = (16, 7, 224, 224)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def clip_key(timestamp, y, x):
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize('UTC')
    else:
        stamp = stamp.tz_convert('UTC')
    return f'{stamp.isoformat()}|{int(y)}|{int(x)}'


def codec():
    return Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE)


class ClipFileDataset:
    """Manifest-bound Kelvin clips, decoded directly into supplied buffers."""
    def __init__(self, root, manifest, *, zarr_root):
        self.root = Path(root)
        self.manifest = pd.read_csv(manifest)
        raw = (self.root/'metadata.json').read_bytes()
        self.metadata_sha256 = digest_bytes(raw)
        metadata = json.loads(raw)
        expected = dict(version=1, shape=list(SHAPE), dtype='float16', units='K',
                        channels=list(VIDEO_MAE_CHANNELS), orientation='north-up west-left',
                        cadence_minutes=5, lossless=True, codec=codec().get_config())
        for key,value in expected.items():
            if metadata.get(key)!=value:
                raise ValueError(f'Unexpected clip-file metadata: {key}')
        if Path(metadata['zarr_root']).resolve()!=Path(zarr_root).resolve():
            raise ValueError('Clip files belong to another source archive')
        manifest_hash=digest_bytes(Path(manifest).read_bytes())
        if manifest_hash not in metadata['manifests']:
            raise ValueError('Clip files do not cover this exact manifest')
        self.records=[]
        for row,name in zip(self.manifest.itertuples(index=False),metadata['manifests'][manifest_hash]['files'],strict=True):
            record=metadata['clips'][name]
            if record['key']!=clip_key(row.start_time,row.origin_y,row.origin_x):
                raise ValueError('Clip-file ordering differs from the manifest')
            if name!=Path(name).name or not name.endswith('.blosc'):
                raise ValueError('Unexpected clip filename')
            if (self.root/name).stat().st_size!=record['bytes']:
                raise IncompleteSeviriClipError(f'Missing or truncated clip file: {name}')
            self.records.append((name,record['sha256']))
        blosc.set_nthreads(1)

    def __len__(self):
        return len(self.records)

    def decode_into(self,index,out):
        if out.shape!=SHAPE or out.dtype!=np.float16 or not out.flags.c_contiguous:
            raise ValueError('Clip destination must be contiguous float16 with canonical shape')
        name,expected=self.records[index]
        encoded=(self.root/name).read_bytes()
        if digest_bytes(encoded)!=expected:
            raise IncompleteSeviriClipError(f'Clip checksum differs: {name}')
        codec().decode(encoded,out=out)
        return out


class ClipBatchLoader:
    """Prepare pinned raw batches in manifest/sampler order and normalize on device.

    Temporal/spectral/crop validity is sealed at materialization. Checksums are
    checked on every read. The trainer and evaluator validate every normalized
    pixel with ForecastWindow.split before using any observation in a model.
    """
    def __init__(self,dataset,*,batch_size,workers,prefetch,mean,std,device,seed,shuffle):
        import torch
        from torch.utils.data import RandomSampler, SequentialSampler
        if batch_size<1 or workers<1 or prefetch<1:
            raise ValueError('Positive batch size, workers and prefetch are required')
        if not np.isfinite([mean,std]).all() or std<=0:
            raise ValueError('Global normalization must be finite with positive standard deviation')
        self.dataset=dataset;self.batch_size=batch_size;self.workers=workers;self.prefetch=prefetch
        self.mean=float(np.float32(mean));self.std=float(np.float32(std));self.device=torch.device(device)
        self.pin_memory=self.device.type=='cuda'
        self.generator=torch.Generator().manual_seed(seed)
        sampler_generator=torch.Generator().manual_seed(seed)
        torch.empty((),dtype=torch.int64).random_(generator=sampler_generator)
        self.sampler=(RandomSampler(dataset,generator=sampler_generator) if shuffle else SequentialSampler(dataset))
        self._initialized=False

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        import torch
        if not self._initialized:
            torch.empty((),dtype=torch.int64).random_(generator=self.generator)
            self._initialized=True
        order=list(self.sampler)
        groups=iter(order[start:start+self.batch_size] for start in range(0,len(order),self.batch_size))
        def prepare(indices):
            pixels=torch.empty((len(indices),*SHAPE),dtype=torch.float16,pin_memory=self.pin_memory)
            values=pixels.numpy()
            for slot,index in enumerate(indices):
                self.dataset.decode_into(index,values[slot])
            return pixels
        pool=ThreadPoolExecutor(max_workers=self.workers)
        pending=deque()
        try:
            for _ in range(self.prefetch):
                group=next(groups,None)
                if group is None:break
                pending.append(pool.submit(prepare,group))
            while pending:
                pixels=pending.popleft().result()
                group=next(groups,None)
                if group is not None:pending.append(pool.submit(prepare,group))
                normalized=(pixels.to(self.device,non_blocking=self.pin_memory).float()-self.mean)/self.std
                yield {'pixel_values':normalized}
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
