"""Exact data partitions and rank-local restart state for single-node DDP.

DDP options and per-rank RNG capture follow Demetra/moduli/videomae/utils.py.
Validation combines sufficient statistics and retains global mask indices.
"""
from datetime import timedelta
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


class GlobalBatchSampler:
    """Partition each global batch exactly once, with equal local tail sizes."""
    def __init__(self, count, batch_size, rank, world_size, seed, shuffle=True):
        if min(count,batch_size,world_size)<1 or not 0<=rank<world_size or count%world_size:
            raise ValueError('Sample count must divide evenly across ranks; padding is forbidden')
        self.count,self.batch_size,self.rank,self.world_size = count,batch_size,rank,world_size
        self.seed,self.shuffle,self.epoch = seed,shuffle,0

    def __len__(self):
        return (self.count+self.batch_size*self.world_size-1)//(self.batch_size*self.world_size)

    def __iter__(self):
        if self.shuffle:
            order=torch.randperm(self.count,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            step=self.batch_size*self.world_size
            for start in range(0,self.count,step):yield order[start:start+step][self.rank::self.world_size]
        else:
            size=self.count//self.world_size
            first=self.rank*size
            for start in range(first,first+size,self.batch_size):
                yield list(range(start,min(start+self.batch_size,first+size)))


def initialize(device_type):
    rank,local,world=(int(os.environ.get(k,default)) for k,default in (
        ('RANK','0'),('LOCAL_RANK','0'),('WORLD_SIZE','1')))
    device=torch.device('cuda',local) if device_type=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=45))
    return rank,world,device


def capture_rng(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device)


def save_distributed(path,model,optimizer,progress,identity,device):
    states=[None]*dist.get_world_size()
    dist.all_gather_object(states,capture_rng(device))
    if dist.get_rank()==0:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),progress=progress,
                     identity=identity,rank_rng=states)
        temporary=path.with_suffix('.pt.part')
        with temporary.open('wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    dist.barrier()


def restore_distributed(path,model,optimizer,identity,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']!=identity or len(saved['rank_rng'])!=dist.get_world_size():
        raise ValueError('Distributed checkpoint identity or topology differs')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rank_rng'][dist.get_rank()],device)
    return saved['progress']
