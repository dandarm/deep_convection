import json
from types import SimpleNamespace

import numpy as np
import pytest

from emma_gpm.ssl_ram import digest,stage_frame_cache


@pytest.fixture
def frame_cache(tmp_path):
    source=tmp_path/'source';source.mkdir()
    values=(220+np.arange(16*7*4*5).reshape(16,7,4,5)/128).astype(np.float16)
    np.save(source/'20200912.npy',values)
    file=source/'20200912.npy'
    metadata=dict(layout='native time,channel,y,x',lossless=True,
        stores={'2020/20200912.zarr':dict(file=file.name,sha256=digest(file),bytes=file.stat().st_size)})
    (source/'metadata.json').write_text(json.dumps(metadata))
    return source,tmp_path/'ram',digest(source/'metadata.json')


def test_staging_preserves_float16_bits_and_metadata_and_rechecks_reuse(frame_cache):
    source,target,checksum=frame_cache
    report=stage_frame_cache(source,target,checksum,reserve_bytes=0,verify=True)
    assert report['status']=='verified'
    assert (source/'metadata.json').read_bytes()==(target/'metadata.json').read_bytes()
    np.testing.assert_array_equal(np.load(source/'20200912.npy').view('u2'),np.load(target/'20200912.npy').view('u2'))
    assert stage_frame_cache(source,target,checksum,reserve_bytes=0,verify=True)['copied_bytes']==0
    with (target/'20200912.npy').open('r+b') as stream:
        stream.seek(-1,2);stream.write(b'\x00')
    with pytest.raises(ValueError,match='Existing staged file checksum'):
        stage_frame_cache(source,target,checksum,reserve_bytes=0,verify=True)


def test_staging_rejects_source_corruption_before_publishing_metadata(frame_cache):
    source,target,checksum=frame_cache
    with (source/'20200912.npy').open('r+b') as stream:
        stream.seek(-1,2);stream.write(b'\x00')
    with pytest.raises(ValueError,match='Copied frame checksum'):
        stage_frame_cache(source,target,checksum,reserve_bytes=0,verify=True)
    assert not (target/'metadata.json').exists()
    assert not (target/'20200912.npy').exists()


def test_staging_checks_space_before_copy(frame_cache,monkeypatch):
    source,target,checksum=frame_cache
    monkeypatch.setattr('emma_gpm.ssl_ram.shutil.disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='Insufficient space'):
        stage_frame_cache(source,target,checksum,reserve_bytes=0)
    assert not (target/'20200912.npy').exists()


def test_normal_ram_copy_and_reuse_do_not_compute_checksums(frame_cache,monkeypatch):
    source,target,checksum=frame_cache
    def forbidden(*args,**kwargs):
        raise AssertionError('Repeated copy verification is disabled')
    monkeypatch.setattr('emma_gpm.ssl_ram.digest',forbidden)
    monkeypatch.setattr('emma_gpm.ssl_ram.hashlib.sha256',forbidden)
    report=stage_frame_cache(source,target,checksum,reserve_bytes=0)
    assert report['status']=='ready' and not report['integrity_audit']
    np.testing.assert_array_equal(np.load(source/'20200912.npy').view('u2'),np.load(target/'20200912.npy').view('u2'))
    assert stage_frame_cache(source,target,checksum,reserve_bytes=0)['copied_bytes']==0
