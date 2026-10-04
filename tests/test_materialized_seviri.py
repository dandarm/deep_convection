import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from emma_gpm.seviri_clips import (
    MaterializedSeviriDataset, IncompleteSeviriClipError, VIDEO_MAE_CHANNELS,
    normalize_seviri_clip,
)


def test_materialized_kelvin_samples_preserve_global_transform_and_reject_corruption(tmp_path):
    pytest.importorskip('torch')
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([dict(start_time='2020-01-01', origin_y=0, origin_x=0)]).to_csv(manifest, index=False)
    sample = np.full((16, 7, 224, 224), 230., dtype=np.float16)
    sample[:, 1] = 240.
    path = tmp_path / 'sample_000000.npy'
    np.save(path, sample)
    metadata = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    channels=list(VIDEO_MAE_CHANNELS), shape=[16, 7, 224, 224], dtype='float16',
                    units='K', orientation='north-up west-left', cadence_minutes=5, lossless=True,
                    zarr_root=str(tmp_path), samples=[dict(file=path.name, bytes=path.stat().st_size)])
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    dataset = MaterializedSeviriDataset(tmp_path, manifest, zarr_root=tmp_path, global_mean=245., global_std=10.)
    actual = dataset[0]['pixel_values'].numpy()
    assert np.array_equal(actual, normalize_seviri_clip(sample.astype(np.float32), 245., 10.))
    assert np.all(actual[:, 1] - actual[:, 0] == 1.)
    sample[0, 0, 0, 0] = np.nan
    np.save(path, sample)
    with pytest.raises(IncompleteSeviriClipError, match='non-finite'):
        dataset[0]
    manifest.write_text(manifest.read_text() + '2020-01-02,0,0\n')
    with pytest.raises(ValueError, match='different manifest'):
        MaterializedSeviriDataset(tmp_path, manifest, zarr_root=tmp_path, global_mean=245., global_std=10.)
