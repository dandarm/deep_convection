from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from emma_gpm.seviri_clips import SeviriVideoMAEDataset, SeviriZarrClipReader
from test_seviri_clips import write_day


def test_spawned_loader_preserves_clip_values_across_epochs(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.pretrain_videomae import build_loader

    write_day(tmp_path, '20200101', pd.date_range('2020-01-01', periods=2, freq='5min'), spatial_pattern=True)
    manifest = pd.DataFrame([dict(start_time='2020-01-01', origin_y=0, origin_x=0)])
    reader = SeviriZarrClipReader(tmp_path, num_frames=2, crop_size=2)
    dataset = SeviriVideoMAEDataset(manifest, reader, global_mean=200.0, global_std=50.0)
    expected = dataset[0]['pixel_values']
    reader.close()
    args = Namespace(batch_size=1, num_workers=1, prefetch_factor=1, pin_memory=False, seed=42)
    loader = build_loader(dataset, args, torch.device('cpu'))
    for _ in range(2):
        actual = next(iter(loader))['pixel_values'][0]
        assert torch.equal(actual, expected)
        assert np.isfinite(actual.numpy()).all()
    reader.close()


def test_small_io_tasks_preserve_shuffled_batches_and_partial_tail():
    torch = pytest.importorskip('torch')
    from scripts.pretrain_videomae import build_loader, iter_training_batches

    dataset = [{'pixel_values': torch.full((2, 7, 2, 2), float(i))} for i in range(7)]
    args = Namespace(batch_size=4, num_workers=0, prefetch_factor=1,
                     pin_memory=False, seed=42, loader_batch_size=0)
    original = list(build_loader(dataset, args, torch.device('cpu')))
    args.loader_batch_size = 2
    loader = build_loader(dataset, args, torch.device('cpu'))
    assembled = list(iter_training_batches(loader, args.batch_size))
    assert [batch['pixel_values'].shape[0] for batch in assembled] == [4, 3]
    for before, after in zip(original, assembled, strict=True):
        assert torch.equal(before['pixel_values'], after['pixel_values'])
