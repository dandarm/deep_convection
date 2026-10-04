"""Atomic epoch-boundary checkpoints for the SEVIRI integration trainer."""
import os
from pathlib import Path
import random

import numpy as np


def save_checkpoint(path, model, optimizer, scheduler, generator, progress, identity):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), generator={k:v.get_state() for k,v in generator.items()},
                   progress=progress, identity=identity,
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, generator, identity):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['identity'] != identity:
        raise ValueError('Checkpoint configuration, dataset or normalization differs from this run.')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    for key, value in generator.items():
        value.set_state(checkpoint['generator'][key])
    random.setstate(checkpoint['python_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['progress']
