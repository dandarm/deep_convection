"""Stopping diagnostic crops must preserve full-domain output and resumption."""
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from emma_gpm.seviri import THERMAL_CHANNELS


@pytest.mark.parametrize('pipelined',[False,True])
def test_full_domain_cli_without_catalog_processes_and_resumes(tmp_path,monkeypatch,pipelined):
    from scripts import build_rdt_pseudolabels as job
    root=tmp_path/'source';(root/'2020').mkdir(parents=True)
    times=pd.date_range('2020-01-01',periods=16,freq='5min')
    data=np.full((16,7,2,2),270.,np.float32)
    ds=xr.Dataset({'seviri':(('time','channel','y','x'),data)},coords={'time':times,'channel':list(THERMAL_CHANNELS),'x':[0.,3000.],'y':[0.,3000.]},attrs={'units':'K','calibration':'brightness_temperature','projection_wkt':'EPSG:3857'})
    ds.to_zarr(root/'2020/20200101.zarr',consolidated=True,zarr_format=2)
    def forbidden(*args,**kwargs):raise AssertionError('Crop catalog should be disabled')
    monkeypatch.setattr(job,'publish_crop_catalog',forbidden)
    output=tmp_path/'labels'
    command=['job','--zarr-root',str(root),'--start-date','2020-01-01','--output-dir',str(output),'--area-cache',str(tmp_path/'absent'),'--packed-fields','--no-joined-table','--no-crop-catalog']
    if pipelined:
        command += ['--detector-workers','2','--async-save']
    monkeypatch.setattr(sys,'argv',command)
    assert job.main()==0
    files=sorted((output/'frames').glob('*/record.json'))
    original={str(p):p.read_bytes() for p in files}
    assert len(files)==16 and len(list((output/'frames').glob('*/fields.zarr.zip')))==16
    assert not (output/'catalog.csv').exists()
    assert json.loads((output/'request.json').read_text())['crop_catalog_enabled'] is False
    assert job.main()==0
    assert original=={str(p):p.read_bytes() for p in files}
    assert json.loads((output/'progress.json').read_text())['processed']==16
