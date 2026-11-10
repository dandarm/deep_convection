import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from emma_gpm.seviri_clip_files import SHAPE, codec, clip_key
from emma_gpm.seviri_clips import VIDEO_MAE_CHANNELS
from emma_gpm.seviri_immutable_clips import ImmutableClipFileDataset
from emma_gpm.ssl_ram import cache_files, stage_frame_cache


def test_verified_clip_reads_decode_exactly_without_payload_checksums(tmp_path, monkeypatch):
    source = tmp_path/'clips'
    source.mkdir()
    manifest = tmp_path/'manifest.csv'
    pd.DataFrame([dict(start_time='2022-01-01T00:00Z', origin_y=3, origin_x=8)]).to_csv(manifest, index=False)
    raw = np.broadcast_to(np.arange(224, dtype=np.float16)[None, None, None, :]+200, SHAPE).copy()
    encoded = codec().encode(raw)
    (source/'one.blosc').write_bytes(encoded)
    metadata = dict(version=1, shape=list(SHAPE), dtype='float16', units='K',
        channels=list(VIDEO_MAE_CHANNELS), orientation='north-up west-left', cadence_minutes=5,
        lossless=True, codec=codec().get_config(), zarr_root='/canonical/sources',
        clips={'one.blosc':dict(key=clip_key('2022-01-01T00:00Z', 3, 8), bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest())},
        manifests={hashlib.sha256(manifest.read_bytes()).hexdigest():dict(files=['one.blosc'])})
    (source/'metadata.json').write_text(json.dumps(metadata))
    checksum = hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest()
    assert set(cache_files(source, checksum)) == {'metadata.json', 'one.blosc'}
    target = tmp_path/'ram'
    report = stage_frame_cache(source, target, checksum, reserve_bytes=0)
    assert report['integrity_audit'] is False
    mapping = tmp_path/'roots.json'
    mapping.write_text(json.dumps(dict(version=1, roots={'/canonical/clips':str(target)})))
    monkeypatch.setenv('SEVIRI_STORAGE_ROOTS', str(mapping))
    dataset = ImmutableClipFileDataset('/canonical/clips', manifest, zarr_root='/canonical/sources')
    monkeypatch.setattr('emma_gpm.seviri_clip_files.digest_bytes', lambda _: pytest.fail('Repeated payload checksum'))
    for _ in range(2):
        out = np.empty_like(raw)
        dataset.decode_into(0, out)
        np.testing.assert_array_equal(out.view('u2'), raw.view('u2'))
