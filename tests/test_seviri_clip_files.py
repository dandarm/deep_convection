import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from emma_gpm import seviri_clip_files as clips
from emma_gpm.seviri_clips import IncompleteSeviriClipError
from emma_gpm.ssl_execution import validate_migration
from scripts.pretrain_videomae import build_loader


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, 'SHAPE', (16, 7, 4, 4))
    manifest = tmp_path / 'train.csv'
    pd.DataFrame(dict(start_time=['2020-09-12T00:00:00Z'] * 11,
                      origin_y=range(11), origin_x=[0] * 11)).to_csv(manifest, index=False)
    names, records, arrays = [], {}, []
    for i in range(11):
        values = (np.arange(np.prod(clips.SHAPE)).reshape(clips.SHAPE) / 16 + 200 + i).astype(np.float16)
        encoded = clips.codec().encode(values)
        name = f'clip_{i:06}.blosc'
        (tmp_path / name).write_bytes(encoded)
        names.append(name)
        arrays.append(values)
        records[name] = dict(key=clips.clip_key('2020-09-12T00:00:00Z', i, 0),
                             bytes=len(encoded), sha256=clips.digest_bytes(encoded))
    metadata = dict(version=1, shape=list(clips.SHAPE), dtype='float16', units='K',
        channels=list(clips.VIDEO_MAE_CHANNELS), orientation='north-up west-left',
        cadence_minutes=5, lossless=True, codec=clips.codec().get_config(), zarr_root=str(tmp_path),
        manifests={clips.digest_bytes(manifest.read_bytes()): dict(files=names)}, clips=records)
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    return clips.ClipFileDataset(tmp_path, manifest, zarr_root=tmp_path), arrays


def test_lossless_decode_and_detect_same_size_corruption(collection):
    dataset, arrays = collection
    result = np.empty_like(arrays[0])
    for i, expected in enumerate(arrays):
        dataset.decode_into(i, result)
        np.testing.assert_array_equal(result.view('uint16'), expected.view('uint16'))
    path = dataset.root / dataset.records[0][0]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(IncompleteSeviriClipError, match='checksum'):
        dataset.decode_into(0, result)


def test_threaded_order_normalization_and_sampler_resume(collection):
    dataset, arrays = collection
    options = dict(batch_size=4, workers=2, prefetch=2, mean=251.371, std=17.231,
                   device='cpu', seed=42, shuffle=True)
    loader = clips.ClipBatchLoader(dataset, **options)
    args = SimpleNamespace(batch_size=2, num_workers=0, seed=42, pin_memory=False)
    reference = build_loader(range(len(dataset)), args, torch.device('cpu'))
    for epoch in range(2):
        order = list(reference.sampler)
        actual = torch.cat([batch['pixel_values'] for batch in loader]).numpy()
        expected = (np.stack([arrays[i] for i in order]).astype('float32') - np.float32(options['mean'])) / np.float32(options['std'])
        np.testing.assert_array_equal(actual, expected)
        if epoch == 0:
            sampler_state = loader.sampler.generator.get_state()
            generator_state = loader.generator.get_state()
    resumed = clips.ClipBatchLoader(dataset, **options)
    resumed.sampler.generator.set_state(sampler_state)
    resumed.generator.set_state(generator_state)
    restored = torch.cat([batch['pixel_values'] for batch in resumed]).numpy()
    np.testing.assert_array_equal(restored, actual)
    assert len(loader) == 3


def test_reordered_channel_metadata_is_rejected(collection):
    dataset, _ = collection
    path = dataset.root / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata['channels'].reverse()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='channels'):
        clips.ClipFileDataset(dataset.root, dataset.root / 'train.csv', zarr_root=dataset.root)


def test_revision_preserves_science_and_effective_batch():
    parent = dict(batch_size=16, loader_batch_size=2, gradient_accumulation=2, num_workers=8,
                  sources={'trainer': 'old'}, objective='mvm', train_sha256='train', learning_rate=.00015)
    current = {**parent, 'batch_size': 32, 'loader_batch_size': 32, 'gradient_accumulation': 1,
               'num_workers': 4, 'sources': {'trainer': 'new'}, 'input_pipeline': {'metadata_sha256': 'clips'}}
    revision = dict(parent_checkpoints={'checkpoint': 'path'}, parent_sources=parent['sources'],
                    sources=current['sources'], clip_metadata_sha256='clips', runtime={
                    key: current[key] for key in ('batch_size', 'loader_batch_size', 'gradient_accumulation', 'num_workers')})
    validate_migration(parent, current, 'checkpoint', revision)
    for key, value in [('objective', 'future'), ('train_sha256', 'other'), ('learning_rate', .1)]:
        with pytest.raises(ValueError, match='scientific'):
            validate_migration(parent, {**current, key: value}, 'checkpoint', revision)
    with pytest.raises(ValueError, match='effective batch'):
        validate_migration(parent, {**current, 'batch_size': 64}, 'checkpoint', revision)
    with pytest.raises(ValueError, match='absent'):
        validate_migration(parent, current, 'unapproved', revision)
    with pytest.raises(ValueError, match='source'):
        validate_migration(parent, {**current, 'sources': {'trainer': 'unexpected'}}, 'checkpoint', revision)
