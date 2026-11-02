import json
import hashlib
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_frame_cache import build_frame_cache, SeviriFrameCacheReader
from emma_gpm.seviri_frame_cache import source_stamp
from emma_gpm.storage_roots import storage_root, storage_mapping_digest


def test_relocated_native_cache_preserves_pixels_and_source_checks(tmp_path, monkeypatch):
    root, cache = tmp_path / 'original', tmp_path / 'cache'
    store = root / '2020/20200912.zarr'
    values = np.arange(2*7*4*5, dtype=np.float16).reshape(2,7,4,5) + 200
    xr.Dataset({'seviri': (('time','channel','y','x'), values)}, coords=dict(
        time=pd.date_range('2020-09-12', periods=2, freq='5min'),
        channel=list(THERMAL_CHANNELS), y=[0,1,2,3], x=[4,3,2,1,0])).to_zarr(store, zarr_format=2, consolidated=True)
    build_frame_cache(root, cache, ['20200912'])
    with SeviriFrameCacheReader(root, cache, num_frames=2, crop_size=3) as reader:
        expected = reader.load_native('2020-09-12', origin_y=1, origin_x=1)
    metadata = (cache/'metadata.json').read_bytes()
    new_root, new_cache = tmp_path/'relocated_source', tmp_path/'relocated_cache'
    root.rename(new_root); cache.rename(new_cache)
    mapping = tmp_path/'locations.json'
    mapping.write_text(json.dumps(dict(version=1, roots={str(root):str(new_root), str(cache):str(new_cache)})))
    monkeypatch.setenv('SEVIRI_STORAGE_ROOTS',str(mapping))
    assert storage_root(cache)==new_cache
    assert storage_root(cache/'unmapped_child')==cache/'unmapped_child'
    assert storage_mapping_digest()
    with SeviriFrameCacheReader(root, cache, num_frames=2, crop_size=3) as reader:
        np.testing.assert_array_equal(reader.load_native('2020-09-12', origin_y=1, origin_x=1), expected)
    assert (new_cache/'metadata.json').read_bytes()==metadata
    (new_root/'2020/20200912.zarr/unexpected').write_bytes(b'altered')
    with SeviriFrameCacheReader(root, cache, num_frames=2, crop_size=3) as reader:
        with pytest.raises(ValueError, match='source has changed'):
            reader.load_native('2020-09-12', origin_y=1, origin_x=1)


def test_storage_mapping_rejects_relative_paths(tmp_path, monkeypatch):
    mapping=tmp_path/'locations.json'
    mapping.write_text(json.dumps(dict(version=1, roots={'relative':'/absolute'})))
    monkeypatch.setenv('SEVIRI_STORAGE_ROOTS',str(mapping))
    with pytest.raises(ValueError, match='absolute'):
        storage_root('/original')


def test_coarse_timestamps_require_exact_source_content(tmp_path, monkeypatch):
    root=tmp_path/'sources'; store=root/'2020/20200912.zarr';store.mkdir(parents=True)
    chunk=store/'chunk';chunk.write_bytes(b'original')
    ns=1712345678123456789
    os.utime(chunk,ns=(ns,ns))
    original=source_stamp(store)
    snapshot=tmp_path/'snapshot.json'
    snapshot.write_text(json.dumps({'2020/20200912.zarr/chunk':hashlib.sha256(chunk.read_bytes()).hexdigest()}))
    times=tmp_path/'times.json'
    times.write_text(json.dumps({'sources/2020/20200912.zarr/chunk':[8,ns]}))
    os.utime(chunk,ns=(ns,ns//10**9*10**9))
    assert source_stamp(store)!=original
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps(dict(version=1,roots={'/original':str(root)},source_provenance=dict(
        root=str(root),snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        timestamps=str(times),timestamps_sha256=hashlib.sha256(times.read_bytes()).hexdigest()))))
    monkeypatch.setenv('SEVIRI_STORAGE_ROOTS',str(mapping))
    assert source_stamp(store)==original
    chunk.write_bytes(b'corrupt!')
    os.utime(chunk,ns=(ns,ns//10**9*10**9))
    with pytest.raises(ValueError,match='content differs'):
        source_stamp(store)
    snapshot.write_text('{}')
    with pytest.raises(ValueError,match='inventory checksum'):
        source_stamp(store)
