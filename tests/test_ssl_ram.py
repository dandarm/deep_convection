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


def test_training_and_validation_share_one_ram_copy(frame_cache, tmp_path, monkeypatch):
    import uuid
    from scripts import stage_seviri_ssl_ram as staging
    source, _, checksum = frame_cache
    root = tmp_path / ('shared_' + uuid.uuid4().hex)
    root.mkdir()
    locations = {}
    for name in ('training', 'extended'):
        directory = root/name
        directory.mkdir()
        (directory/'frames').symlink_to(source, target_is_directory=True)
        (directory/'transfer_plan.json').write_text(json.dumps(dict(
            canonical_frame_root='/canonical/frames', frames={'metadata.json':dict(sha256=checksum)})))
        locations[name] = str(directory)
    (root/'data_locations.json').write_text(json.dumps(locations))
    (root/'storage_roots.json').write_text(json.dumps(dict(version=1, roots={'/canonical/frames':str(source)})))
    monkeypatch.setattr(staging.subprocess, 'check_output', lambda *a, **k: 'tmpfs\n')
    monkeypatch.setattr(staging.shutil, 'disk_usage', lambda _: SimpleNamespace(free=2**40))
    copies = []
    def copy(source, destination, checksum, **kwargs):
        copies.append((source, destination))
        return dict(status='ready')
    monkeypatch.setattr(staging, 'stage_frame_cache', copy)
    try:
        report = staging.stage(root)
        assert len(copies) == 1
        assert set(report['datasets']) == {'training'}
        mapping = json.loads((root/'ram_storage_roots.json').read_text())
        assert mapping['roots']['/canonical/frames'] == str(copies[0][1])
    finally:
        staging.cleanup(root)


def test_memfd_arrays_are_readable_by_child_processes_and_sealed_without_hashing(frame_cache, monkeypatch):
    import os
    import subprocess
    import sys
    from emma_gpm.ssl_ram import stage_memfd_cache, _MEMFD_FILES
    source, target, checksum = frame_cache
    monkeypatch.setattr('emma_gpm.ssl_ram.digest', lambda *_: pytest.fail('Unexpected byte audit'))
    monkeypatch.setattr('emma_gpm.ssl_ram.hashlib.sha256', lambda *_: pytest.fail('Unexpected byte audit'))
    try:
        report = stage_memfd_cache(source, target, checksum)
        assert not report['integrity_audit']
        np.testing.assert_array_equal(np.load(source/'20200912.npy'), np.load(target/'20200912.npy'))
        script = 'import numpy as np,sys; a=np.load(sys.argv[1],mmap_mode="r"); print(a.shape)'
        result = subprocess.check_output([sys.executable, '-c', script, str(target/'20200912.npy')], text=True)
        assert result.strip() == '(16, 7, 4, 5)'
        with pytest.raises(PermissionError):
            os.write(_MEMFD_FILES[str(target/'20200912.npy')], b'changed')
        assert stage_memfd_cache(source, target, checksum)['copied_bytes'] == 0
    finally:
        for path in list(_MEMFD_FILES):
            if path.startswith(str(target)):
                os.close(_MEMFD_FILES.pop(path))
