import hashlib
import json
from pathlib import Path

import pytest

from emma_gpm.seviri import THERMAL_CHANNELS
from scripts.pretrain_videomae import load_training_stats


def test_persisted_stats_reject_a_different_manifest_and_vector_normalization(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('start_time,origin_y,origin_x\n2020-01-01,0,0\n')
    path = tmp_path / 'stats.json'
    stats = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 channels=list(THERMAL_CHANNELS), zarr_root=str(tmp_path),
                 normalization='single_global_affine_transform', global_mean_k=260., global_std_k=20.)
    path.write_text(json.dumps(stats))
    assert load_training_stats(path, manifest, tmp_path)['global_mean_k'] == 260.
    manifest.write_text(manifest.read_text() + '2020-01-02,0,0\n')
    with pytest.raises(ValueError, match='different manifest'):
        load_training_stats(path, manifest, tmp_path)
    stats['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    stats['global_mean_k'] = [260.] * 7
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='scalars'):
        load_training_stats(path, manifest, tmp_path)


def test_parallel_validation_matches_serial_moments_and_rejects_missing_frames(tmp_path):
    import numpy as np
    import pandas as pd
    from test_seviri_clips import write_day
    from scripts.build_random_seviri_clip_manifest import (
        initialize_validator, validate_candidate, parallel_candidates,
    )

    write_day(tmp_path, '20200101', pd.date_range('2020-01-01', periods=2, freq='5min'), spatial_pattern=True)
    candidates = [('2020-01-01T00:00:00Z', 0, 0), ('2020-01-01T00:00:00Z', 1, 1),
                  ('2020-01-02T00:00:00Z', 0, 0)]
    init_args = (tmp_path, 2, 5, 2, True)
    initialize_validator(*init_args)
    expected = [validate_candidate(candidate) for candidate in candidates]
    actual = list(parallel_candidates(iter(candidates), 2, init_args))
    assert actual == expected
    assert actual[-1] is None
    assert actual[0][1][2] == 2 * 7 * 2 * 2
    assert np.isfinite(actual[0][1][:2]).all()
