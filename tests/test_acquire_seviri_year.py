from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest

from scripts.acquire_seviri_year import (
    acquire_day,
    exclusive_file_lock,
    frames_per_second,
    product_slot,
)


class SatelliteProduct:
    def __init__(self,satellite,slot,suffix=''):
        self.sensing_start=slot+timedelta(seconds=12)
        self.name=f'{satellite}-SEVI-MSG15-0100-NA-{slot+timedelta(minutes=4):%Y%m%d%H%M%S}.000{suffix}Z-NA'

    def __str__(self):
        return self.name


def test_satellite_preference_resolves_only_overlap_and_keeps_single_available_satellite():
    from scripts.acquire_seviri_year import select_slot_products
    slot=datetime(2021,1,7,9)
    a,b=SatelliteProduct('MSG2',slot),SatelliteProduct('MSG3',slot)
    next_product=SatelliteProduct('MSG2',slot+timedelta(minutes=5))
    for order in [[a,b,next_product],[next_product,b,a]]:
        chosen,decisions=select_slot_products(order,'MSG3')
        assert chosen==[b,next_product]
        assert decisions[0]['selected_product']==str(b)
        assert decisions[0]['unselected_products']==[str(a)]


@pytest.mark.parametrize('satellites,preferred',[
    (['MSG2','MSG3'],None),(['MSG2','MSG3'],'MSG4'),
    (['MSG3','MSG3'],'MSG3'),(['MSG2','MSG2','MSG3'],'MSG3'),
    (['UNKNOWN','MSG3'],'MSG3')])
def test_ambiguous_provider_products_still_fail(satellites,preferred):
    from scripts.acquire_seviri_year import select_slot_products
    products=[SatelliteProduct(s,datetime(2021,1,7,9),suffix=str(i)) for i,s in enumerate(satellites)]
    with pytest.raises(ValueError,match='Multiple provider products'):
        select_slot_products(products,preferred)


def test_satellite_policy_is_persisted_and_cannot_change_on_resume(tmp_path):
    from scripts.acquire_seviri_year import pin_selection_policy
    pin_selection_policy(tmp_path,'MSG3')
    original=(tmp_path/'selection_policy.json').read_bytes()
    pin_selection_policy(tmp_path,'MSG3')
    assert (tmp_path/'selection_policy.json').read_bytes()==original
    with pytest.raises(ValueError,match='policy changed'):
        pin_selection_policy(tmp_path,'MSG2')


@pytest.mark.parametrize('conversion_workers',[0,2])
def test_overlap_never_converts_cached_unselected_satellite_and_writes_in_time_order(tmp_path,monkeypatch,conversion_workers):
    from scripts import acquire_seviri_year as job
    from emma_gpm.seviri import THERMAL_CHANNELS
    day=datetime(2021,1,7);slot=day+timedelta(hours=9)
    a,b=SatelliteProduct('MSG2',slot),SatelliteProduct('MSG3',slot)
    following=SatelliteProduct('MSG2',slot+timedelta(minutes=5))
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1,preferred_satellite='MSG3',conversion_workers=conversion_workers)
    work=tmp_path/'_acquisition/2021';buffer=work/'native_buffer/20210107';buffer.mkdir(parents=True)
    # Simulate a cached alternative from an interrupted earlier attempt.
    (buffer/(str(a)+'.nat')).write_bytes(b'unselected MSG2 payload')
    loaded,written=[],[]
    def download(collection,*args,**kwargs):
        selected=list(collection.search());assert selected==[b,following]
        for product in selected:(buffer/(str(product)+'.nat')).write_bytes(str(product).encode())
    def load(path,*args,**kwargs):
        loaded.append(path.stem)
        return SimpleNamespace(channels=THERMAL_CHANNELS,units='K',
            start_time=job.native_slot_start_from_name(path),data=SimpleNamespace(shape=(7,763,1762)))
    def append(path,crops,dtype):
        path.mkdir(exist_ok=True,parents=True);written.extend(c.start_time for c in crops)
    monkeypatch.setattr(job,'download_interval',download)
    monkeypatch.setattr(job,'load_native_crop',load)
    monkeypatch.setattr(job,'append_batch',append)
    monkeypatch.setattr(job,'verify_store',lambda path,times:len(times))
    if conversion_workers:
        from concurrent.futures import ThreadPoolExecutor
        from emma_gpm import seviri_conversion
        monkeypatch.setattr(seviri_conversion,'ProcessPoolExecutor',lambda **kw:ThreadPoolExecutor(max_workers=kw['max_workers']))
        monkeypatch.setattr(seviri_conversion,'_convert',lambda path,bounds:load(path,bounds))
    class Collection:
        def search(self,**kwargs):return [a,b,following]
    report=acquire_day(day,Collection(),args,work,'fake','fake')
    assert sorted(loaded)==sorted([str(b),str(following)])
    assert written==[slot,slot+timedelta(minutes=5)]
    assert report['provider_products']==3 and report['selected_provider_products']==2
    assert report['selected_satellites']=={'MSG2':1,'MSG3':1}
    retained=work/'unselected_native/20210107'/(str(a)+'.nat')
    assert retained.read_bytes()==b'unselected MSG2 payload'
    manifest=json.loads(Path(report['product_selection_report']).read_text())
    assert manifest['retained_unselected_native'][0]['sha256']==job.native_digest(retained)
    assert acquire_day(day,Collection(),args,work,'fake','fake')==report


def test_product_slot_is_utc_and_on_native_five_minute_grid():
    assert product_slot(SimpleNamespace(sensing_start=datetime(2020,1,1,0,4,30,tzinfo=timezone.utc)))==datetime(2020,1,1)


def test_conversion_rate_is_finite_before_monotonic_clock_advances():
    assert frames_per_second(0,0.0)==0.0
    assert frames_per_second(4,2.0)==2.0


def test_acquisition_lock_rejects_a_second_holder(tmp_path):
    lock_path=tmp_path/'acquisition.lock'
    with exclusive_file_lock(lock_path):
        with pytest.raises(RuntimeError,match='already held'):
            with exclusive_file_lock(lock_path):
                pass


def test_empty_provider_day_can_resume_without_a_nonexistent_store(tmp_path):
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1)
    work=tmp_path/'_acquisition'/'2020'
    class Collection:
        def search(self,**kwargs): return []
    report=acquire_day(datetime(2020,1,1),Collection(),args,work,'fake','fake')
    assert report['stored_frames']==0 and len(report['missing_nominal_slots'])==288
    again=acquire_day(datetime(2020,1,1),Collection(),args,work,'fake','fake')
    assert report==again


def test_failure_preserves_native_buffer_and_existing_zarr(tmp_path,monkeypatch):
    import scripts.acquire_seviri_year as job
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1)
    work=tmp_path/'_acquisition'/'2020'
    day=datetime(2020,1,1)
    store=tmp_path/'2020'/'20200101.zarr'
    store.mkdir(parents=True)
    (store/'sentinel').write_text('original')
    monkeypatch.setattr(job,'existing_times',lambda p:set())
    def download(*args,**kwargs):
        buffer=args[3]; buffer.mkdir(parents=True)
        (buffer/'test.nat').write_bytes(b'cached native')
        raise RuntimeError('network failed')
    monkeypatch.setattr(job,'download_interval',download)
    class Collection:
        def search(self,**kwargs):return [SimpleNamespace(sensing_start=day)]
    with pytest.raises(RuntimeError,match='network failed'):
        acquire_day(day,Collection(),args,work,'fake','fake')
    assert (store/'sentinel').read_text()=='original'
    assert (work/'native_buffer'/'20200101'/'test.nat').exists()
    assert not (work/'completed_days'/'20200101.json').exists()


def test_dotenv_quotes_and_existing_environment_are_preserved(tmp_path,monkeypatch):
    from scripts.download_msg_seviri import load_dotenv
    monkeypatch.delenv('UNIT_TEST_TOKEN',raising=False)
    monkeypatch.setenv('UNIT_TEST_EXISTING','already present')
    path=tmp_path/'env'
    path.write_text('export UNIT_TEST_TOKEN="quoted value"\nUNIT_TEST_EXISTING=override\n')
    load_dotenv(path)
    import os
    assert os.environ['UNIT_TEST_TOKEN']=='quoted value'
    assert os.environ['UNIT_TEST_EXISTING']=='already present'


def test_parallel_download_cache_and_truncated_payload(tmp_path,monkeypatch):
    import scripts.download_msg_seviri as download
    payload=download._NATIVE_MAGIC+b' valid test payload'
    metadata_calls=[]
    network_calls=[]
    class Ref:
        sensing_start=datetime(2020,1,1)
        def __str__(self):return 'SCENE'
        @property
        def entries(self):
            metadata_calls.append(True)
            return ['SCENE.nat']
    class Collection:
        def search(self,**kwargs):return [Ref()]
    datastore=SimpleNamespace(urls=SimpleNamespace(get=lambda *args,**kwargs:'https://unit.example/test'),token=SimpleNamespace(auth=None))
    product=SimpleNamespace(entries=['SCENE.nat'],datastore=datastore)
    monkeypatch.setattr(download,'thread_datastore',lambda *args:SimpleNamespace(get_product=lambda **kwargs:product))
    truncated=False
    class Response:
        @property
        def headers(self):return {'Content-Length':str(len(payload)+(1 if truncated else 0))}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,**kwargs):yield payload
    def request(*args,**kwargs):
        network_calls.append(True)
        return Response()
    monkeypatch.setattr(download.requests,'get',request)
    options=dict(collection_id='test',key='fake',secret='fake',workers=1,retries=0,read_timeout=10)
    download.download_interval(Collection(),datetime(2020,1,1),datetime(2020,1,2),tmp_path/'good',False,**options)
    assert not metadata_calls and len(network_calls)==1
    download.download_interval(Collection(),datetime(2020,1,1),datetime(2020,1,2),tmp_path/'good',False,**options)
    assert len(metadata_calls)==0 and len(network_calls)==1
    truncated=True
    with pytest.raises(RuntimeError,match='Incomplete download'):
        download.download_interval(Collection(),datetime(2020,1,1),datetime(2020,1,2),tmp_path/'bad',False,**options)
    assert not list((tmp_path/'bad').rglob('*.nat'))
    assert not list((tmp_path/'bad').rglob('*.part'))


def test_success_publishes_new_frames_preserves_old_frames_and_cleans_owned_buffer(tmp_path,monkeypatch):
    import numpy as np
    import xarray as xr
    from scripts import acquire_seviri_year as job
    from scripts.crop_msg_seviri_native import append_batch
    from emma_gpm.seviri import NativeSeviriCrop,THERMAL_CHANNELS
    day=datetime(2020,1,1)
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1)
    work=tmp_path/'_acquisition'/'2020'
    store=tmp_path/'2020'/'20200101.zarr'
    def crop(start,value):
        return NativeSeviriCrop(data=np.full((7,763,1762),value,dtype=np.float32),
            x=np.arange(1762)[::-1]*3000.,y=np.arange(763)*3000.,channels=THERMAL_CHANNELS,
            start_time=start,end_time=start,calibration='brightness_temperature',units='K',source_path=Path('fake.nat'))
    old_time=datetime(2020,1,1,0,5)
    append_batch(store,[crop(old_time,270.)],'float16')
    def download(*args,**kwargs):
        buffer=args[3];buffer.mkdir(parents=True)
        (buffer/'SCENE-20200101000400.000Z-NA.nat').write_bytes(b'fake payload')
    monkeypatch.setattr(job,'download_interval',download)
    monkeypatch.setattr(job,'load_native_crop',lambda *args,**kwargs:crop(day,260.))
    class Collection:
        def search(self,**kwargs):return [SimpleNamespace(sensing_start=day)]
    report=acquire_day(day,Collection(),args,work,'fake','fake')
    assert report['previous_frames']==1 and report['new_frames']==1 and report['stored_frames']==2
    with xr.open_zarr(store,consolidated=True,chunks=None) as ds:
        assert np.all(ds.seviri.sel(time=old_time).values==270.)
        assert np.all(ds.seviri.sel(time=day).values==260.)
    assert not (work/'native_buffer'/'20200101').exists()
    assert (work/'completed_days'/'20200101.json').exists()
    assert not list((work/'backups').glob('*'))


def test_inconsistent_native_timestamp_is_reported_as_gap_without_repair(tmp_path,monkeypatch):
    from scripts import acquire_seviri_year as job
    from emma_gpm.seviri import THERMAL_CHANNELS
    day=datetime(2020,1,1)
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1)
    work=tmp_path/'_acquisition'/'2020'
    def download(*args,**kwargs):
        buffer=args[3];buffer.mkdir(parents=True)
        (buffer/'SCENE-20200101000400.000Z-NA.nat').write_bytes(b'bad time')
    monkeypatch.setattr(job,'download_interval',download)
    monkeypatch.setattr(job,'load_native_crop',lambda *args,**kwargs:SimpleNamespace(
        channels=THERMAL_CHANNELS,units='K',start_time=datetime(2020,1,1,0,5),end_time=datetime(2020,1,1,0,5)))
    class Collection:
        def search(self,**kwargs):return [SimpleNamespace(sensing_start=day)]
    report=acquire_day(day,Collection(),args,work,'fake','fake')
    assert report['stored_frames']==0
    assert len(report['missing_nominal_slots'])==288
    assert len(report['rejected_products'])==1
    assert (work/'rejected_products'/'20200101.json').exists()
    assert not (tmp_path/'2020'/'20200101.zarr').exists()


def test_explicit_exclusion_retains_native_and_reports_gap(tmp_path,monkeypatch):
    import hashlib
    from scripts import acquire_seviri_year as job
    day=datetime(2020,1,1)
    name='SCENE-20200101000400.000Z-NA.nat'
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1,exclude_native=[name])
    work=tmp_path/'_acquisition'/'2020'
    payload=b'diagnosed unreadable native'
    def download(*args,**kwargs):
        buffer=args[3];buffer.mkdir(parents=True)
        (buffer/name).write_bytes(payload)
    monkeypatch.setattr(job,'download_interval',download)
    def read(*args,**kwargs):raise AssertionError('Excluded native must not be converted')
    monkeypatch.setattr(job,'load_native_crop',read)
    class Collection:
        def search(self,**kwargs):return [SimpleNamespace(sensing_start=day)]
    report=acquire_day(day,Collection(),args,work,'fake','fake')
    assert report['stored_frames']==0 and len(report['missing_nominal_slots'])==288
    rejected=report['rejected_products'][0]
    assert rejected['sha256']==hashlib.sha256(payload).hexdigest()
    assert Path(rejected['quarantine']).read_bytes()==payload
    assert not (work/'native_buffer'/'20200101').exists()
    assert not (tmp_path/'2020'/'20200101.zarr').exists()
    assert acquire_day(day,Collection(),args,work,'fake','fake')==report


def test_unlisted_native_read_failure_still_stops_acquisition(tmp_path,monkeypatch):
    from scripts import acquire_seviri_year as job
    day=datetime(2020,1,1)
    name='SCENE-20200101000400.000Z-NA.nat'
    args=SimpleNamespace(zarr_root=tmp_path,workers=1,crop_batch_size=1,exclude_native=[])
    work=tmp_path/'_acquisition'/'2020'
    def download(*args,**kwargs):
        buffer=args[3];buffer.mkdir(parents=True)
        (buffer/name).write_bytes(b'bad native')
    monkeypatch.setattr(job,'download_interval',download)
    def read(*args,**kwargs):raise ValueError('buffer is smaller than requested size')
    monkeypatch.setattr(job,'load_native_crop',read)
    class Collection:
        def search(self,**kwargs):return [SimpleNamespace(sensing_start=day)]
    with pytest.raises(RuntimeError,match=name):
        acquire_day(day,Collection(),args,work,'fake','fake')
    assert not (work/'completed_days'/'20200101.json').exists()
    assert (work/'native_buffer'/'20200101'/name).exists()


@pytest.mark.parametrize('fresh_payload,readable,outcome',[
    (b'bad',False,'excluded'),
    (b'fixed',True,'recovered'),
    (b'other bad',False,'blocked'),
])
def test_automatic_native_recovery_checks_redownload_and_preserves_original(tmp_path,monkeypatch,fresh_payload,readable,outcome):
    from scripts import acquire_seviri_year as job
    day=datetime(2020,1,1)
    native=tmp_path/'buffer'/'SCENE-20200101000400.000Z-NA.nat'
    native.parent.mkdir();native.write_bytes(b'bad')
    work=tmp_path/'work'
    args=SimpleNamespace(recover_native_read_errors=True)
    monkeypatch.setattr(job,'check_native_header',lambda p: False if p==native else readable)
    def download(*args,**kwargs):
        directory=args[3];directory.mkdir(parents=True,exist_ok=True)
        (directory/native.name).write_bytes(fresh_payload)
    monkeypatch.setattr(job,'download_interval',download)
    crop=object()
    monkeypatch.setattr(job,'load_native_crop',lambda *args,**kwargs:crop)
    call=lambda:job.recover_native(native,day,[SimpleNamespace(sensing_start=day)],None,args,work,'fake','fake',ValueError('buffer is smaller than requested size'))
    if outcome=='blocked':
        with pytest.raises(ValueError,match='refusing automatic exclusion'):call()
        assert native.read_bytes()==b'bad'
    else:
        result,rejection=call()
        if outcome=='excluded':
            assert result is None and rejection['sha256']==rejection['redownload_sha256']
            assert Path(rejection['quarantine']).read_bytes()==b'bad'
            assert native.read_bytes()==b'bad'
        else:
            assert result is crop and rejection is None and native.read_bytes()==b'fixed'
    assert (work/'recovery'/native.stem/'original.nat').read_bytes()==b'bad'
    assert (work/'recovery'/native.stem/'report.json').exists()


@pytest.mark.parametrize('error,independent_readable',[(ValueError('unexpected parser error'),False),(ValueError('buffer is smaller than requested size'),True)])
def test_automatic_recovery_does_not_hide_unrelated_errors(tmp_path,monkeypatch,error,independent_readable):
    from scripts import acquire_seviri_year as job
    monkeypatch.setattr(job,'check_native_header',lambda p:independent_readable)
    def download(*args,**kwargs):raise AssertionError('Unexpected redownload')
    monkeypatch.setattr(job,'download_interval',download)
    with pytest.raises(ValueError,match=str(error)):
        job.recover_native(tmp_path/'bad.nat',datetime(2020,1,1),[],None,SimpleNamespace(recover_native_read_errors=True),tmp_path,'fake','fake',error)
