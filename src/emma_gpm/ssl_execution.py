"""Explicit, checksum-bound migration of runtime settings between SSL runs."""
import hashlib
import json
from pathlib import Path

from .training_checkpoint import restore_checkpoint

RUNTIME_KEYS = frozenset({'batch_size', 'loader_batch_size', 'gradient_accumulation',
                          'num_workers', 'sources', 'input_pipeline', 'cuda_runtime'})


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_migration(parent, current, checkpoint_digest, revision):
    if checkpoint_digest not in revision['parent_checkpoints']:
        raise ValueError('Checkpoint is absent from the explicit execution revision')
    if parent['sources'] != revision['parent_sources']:
        raise ValueError('Checkpoint source identity differs from the archived protocol')
    for name, checksum in current['sources'].items():
        if revision['sources'].get(name) != checksum:
            raise ValueError(f'Execution revision source differs: {name}')
    old_science = {key: value for key, value in parent.items() if key not in RUNTIME_KEYS}
    new_science = {key: value for key, value in current.items() if key not in RUNTIME_KEYS}
    if old_science != new_science:
        raise ValueError('Execution migration would change scientific configuration')
    old_batch = parent['batch_size'] * parent['gradient_accumulation']
    new_batch = current['batch_size'] * current['gradient_accumulation']
    if old_batch != new_batch:
        raise ValueError('Execution migration must preserve the effective batch size')
    for key in ('batch_size', 'loader_batch_size', 'gradient_accumulation', 'num_workers'):
        if current[key] != revision['runtime'][key]:
            raise ValueError(f'Execution settings differ from the revision: {key}')
    if current.get('cuda_runtime') != revision.get('cuda_runtime'):
        raise ValueError('CUDA execution settings differ from the revision')
    pipeline = current.get('input_pipeline', {})
    if pipeline.get('metadata_sha256') != revision['clip_metadata_sha256']:
        raise ValueError('Execution revision belongs to another clip-file collection')


def restore_with_revision(path, model, optimizer, scheduler, generators, identity, revision_path):
    import torch
    parent = torch.load(path, map_location='cpu', weights_only=False)['identity']
    if parent == identity:
        return restore_checkpoint(path, model, optimizer, scheduler, generators, identity)
    if revision_path is None:
        raise ValueError('Checkpoint identity differs; an explicit execution revision is required')
    revision = json.loads(Path(revision_path).read_text())
    checksum = file_digest(path)
    validate_migration(parent, identity, checksum, revision)
    progress = restore_checkpoint(path, model, optimizer, scheduler, generators, parent)
    progress.setdefault('execution_history', []).append(dict(
        revision_sha256=file_digest(revision_path), parent_checkpoint_sha256=checksum,
        after_epoch=progress['completed_epochs'],
        previous={key: parent.get(key) for key in sorted(RUNTIME_KEYS)},
        current={key: identity.get(key) for key in sorted(RUNTIME_KEYS)}))
    return progress
