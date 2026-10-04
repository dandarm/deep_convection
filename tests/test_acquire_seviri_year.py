from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest

from scripts.acquire_seviri_year import acquire_day,product_slot


def test_product_slot_is_utc_and_on_native_five_minute_grid():
    assert product_slot(SimpleNamespace(sensing_start=datetime(2020,1,1,0,4,30,tzinfo=timezone.utc)))==datetime(2020,1,1)


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
