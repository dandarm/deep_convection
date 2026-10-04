import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_seviri_v2_experiments as queue


def test_temporal_window_spacing_has_no_shared_endpoint_frame():
    starts=pd.date_range('2020-01-01',periods=60,freq='5min')
    selected=queue.spaced_starts(starts)
    assert list(selected)==list(starts[::16])
    assert (selected[1:]-selected[:-1]>pd.Timedelta(minutes=75)).all()


def queue_fixture(tmp_path,monkeypatch):
    output=tmp_path/'experiments';output.mkdir()
    (output/'preflight.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['queue','--zarr-root',str(tmp_path),'--output-dir',str(output)])
    monkeypatch.setattr(queue,'prepare',lambda *args:None)
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(
        terminate=lambda:None,wait=lambda **kwargs:None))
    return output


def test_queue_runs_five_scratch_experiments_and_selects_common_validation(tmp_path,monkeypatch):
    output=queue_fixture(tmp_path,monkeypatch)
    calls=[];curves=[]
    def train(output,root,name,count,ratio,epochs):
        calls.append((name,count,ratio,epochs))
        destination=output/name;destination.mkdir()
        mse={.5:.3,.75:.2,.9:.4}[ratio]
        (destination/'best_validation.json').write_text(json.dumps(dict(epoch=100,mse=mse,rmse_k=mse**.5*20)))
    monkeypatch.setattr(queue,'train',train)
    monkeypatch.setattr(queue,'update_size_comparison',lambda output,results:curves.append(list(results)))
    queue.main()
    assert calls==[('mask50',2000,.5,150),('mask75',2000,.75,150),('mask90',2000,.9,150),
                   ('large4000',4000,.75,150),('large8000',8000,.75,150)]
    assert [row['samples'] for row in curves[-1]]==[2000,4000,8000]
    assert json.loads((output/'queue_status.json').read_text())['stage']=='completed'


def test_queue_failure_stops_later_training_and_records_error(tmp_path,monkeypatch):
    output=queue_fixture(tmp_path,monkeypatch)
    calls=[]
    def train(*args):
        calls.append(args[2]);raise RuntimeError('test interruption')
    monkeypatch.setattr(queue,'train',train)
    with pytest.raises(RuntimeError,match='test interruption'):
        queue.main()
    assert calls==['mask50']
    assert json.loads((output/'queue_status.json').read_text())['stage']=='failed'
