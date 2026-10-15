import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.rdt_domain_dashboard import DomainDayArchive


def write_archive(tmp_path):
    root, output = tmp_path/"source", tmp_path/"labels"
    x, y = np.arange(8)[::-1]*3000., np.arange(6)*3000.
    pattern = 240+np.arange(6)[:, None]*2+np.arange(8)[None, :]
    frame = np.stack([pattern+i for i in range(7)]).astype(np.float32)
    times = pd.DatetimeIndex(["2020-07-11T00:00", "2020-07-11T00:10"])
    dataset = xr.Dataset({"seviri": (("time", "channel", "y", "x"), np.stack([frame]*2))},
        coords={"time": times, "channel": list(THERMAL_CHANNELS), "x": x, "y": y},
        attrs={"projection_wkt": "EPSG:3857", "units": "K", "calibration": "brightness_temperature"})
    (root/"2020").mkdir(parents=True)
    dataset.to_zarr(root/"2020/20200711.zarr", consolidated=True, zarr_format=2)
    output.mkdir()
    geometry_hash = hashlib.sha256(x.tobytes()+y.tobytes()+b"EPSG:3857").hexdigest()
    (output/"protocol.json").write_text(json.dumps(dict(root=str(root.resolve()), orientation="north-up west-left",
        geometry_sha256=geometry_hash, shape=[6, 8], config_hash="synthetic")))
    (output/"request.json").write_text(json.dumps(dict(start="2020-07-11T00:00:00", end_exclusive="2020-07-12T00:00:00")))
    for stamp in times:
        directory = output/"frames"/f"{stamp:%Y%m%dT%H%M}"
        directory.mkdir(parents=True)
        (directory/"record.json").write_text('{}')
        group = zarr.open_group(str(directory/"fields.zarr"), mode="w", zarr_format=2)
        group.attrs.update(config_hash="synthetic", time=stamp.isoformat())
        group.create_array("semantic_label", data=np.zeros((6, 8), np.uint8))
        group.create_array("valid_observation", data=np.ones((6, 8), bool))
        group.create_array("thermal_depth", data=np.full((6, 8), 1.5, np.float32))
    # A staged but uncommitted record must never appear in the player.
    staged = output/"frames/20200711T0015.part"
    staged.mkdir(); (staged/"record.json").write_text('{}')
    pd.DataFrame(dict(time=times, convective_objects=[1, 3], ot_pixels=[0, 4])).to_csv(output/"frame_metrics.csv", index=False)
    return root, output, frame, times


def test_full_domain_read_orientation_lazy_fields_clock_and_cache(tmp_path):
    root, output, raw, times = write_archive(tmp_path)
    archive = DomainDayArchive(root, output, cache_frames=1)
    assert len(archive.committed) == 2
    assert len(archive.slots("2020-07-11")) == 3  # Missing 00:05 retained.
    first = archive.load(times[0])
    assert np.array_equal(first["ir"], raw[4, ::-1, ::-1])
    assert np.array_equal(first["wv"], raw[0, ::-1, ::-1])
    assert "thermal_depth" not in first["fields"]
    assert np.all(archive.load(times[0], "thermal_depth")["fields"]["thermal_depth"] == 1.5)
    with pytest.raises(FileNotFoundError, match="Slot non disponibile"):
        archive.load("2020-07-11T00:05")
    archive.load(times[1])
    assert len(archive.cache) == 1 and times[0] not in archive.cache
    archive.close()


def test_mismatched_dataset_geometry_or_frame_version_rejected(tmp_path):
    root, output, raw, times = write_archive(tmp_path)
    with pytest.raises(ValueError, match="different SEVIRI archive"):
        DomainDayArchive(tmp_path/"another_source", output)
    protocol = json.loads((output/"protocol.json").read_text())
    original_hash = protocol["geometry_sha256"]
    protocol["geometry_sha256"] = "different_geometry"
    (output/"protocol.json").write_text(json.dumps(protocol))
    archive = DomainDayArchive(root, output)
    with pytest.raises(ValueError, match="geometry differs"):
        archive.load(times[0])
    archive.close()
    protocol["geometry_sha256"] = original_hash
    (output/"protocol.json").write_text(json.dumps(protocol))
    group = zarr.open_group(str(output/"frames/20200711T0000/fields.zarr"), mode="a")
    group.attrs["config_hash"] = "another_teacher"
    archive = DomainDayArchive(root, output)
    with pytest.raises(ValueError, match="version/timestamp"):
        archive.load(times[0])
    archive.close()


def test_global_peak_changes_day_and_preserves_actual_clock(tmp_path,monkeypatch):
    from emma_gpm.rdt_domain_dashboard import RDTDomainDayViewer
    root,output,_,times=write_archive(tmp_path)
    extra=output/'frames/20200801T0010';extra.mkdir();(extra/'record.json').write_text('{}')
    request=json.loads((output/'request.json').read_text());request['end_exclusive']='2020-08-02'
    (output/'request.json').write_text(json.dumps(request))
    index=output/'exploration';index.mkdir()
    pd.DataFrame(dict(time=[times[0],times[1],pd.Timestamp('2020-08-01T00:10')],convective_percent=[1,2,10],ot_percent=[0,.1,0])).to_csv(index/'frame_index.csv',index=False)
    monkeypatch.setattr(RDTDomainDayViewer,'show_frame',lambda self,*args:None)
    monkeypatch.setattr(RDTDomainDayViewer,'show_activity',lambda self,*args:None)
    viewer=RDTDomainDayViewer(root,{'Annual':output})
    viewer.jump_to_peak('convective_objects')
    assert viewer.day.value=='2020-08-01' and viewer.frame.value==2
    viewer.scope.value='Giornata visualizzata'
    viewer.day.value='2020-07-11'
    viewer.jump_to_peak('ot_pixels')
    assert viewer.day.value=='2020-07-11' and viewer.frame.value==2
    viewer.close()


def test_completed_archive_index_cli_resume_is_idempotent(tmp_path):
    import os,subprocess,sys
    root,output,_,_=write_archive(tmp_path)
    protocol=json.loads((output/'protocol.json').read_text());protocol['config']={'min_proxy_confidence':.35}
    (output/'protocol.json').write_text(json.dumps(protocol));(output/'progress.json').write_text('{"status":"complete"}')
    for directory in (output/'frames').glob('*'):
        if not directory.name.endswith('.part'):pd.DataFrame().to_csv(directory/'objects.csv.gz',index=False)
    script=Path(__file__).parents[1]/'scripts/index_rdt_domain_events.py'
    command=[sys.executable,str(script),'--label-root',str(output),'--workers','1','--follow']
    for _ in range(2):subprocess.run(command,check=True,capture_output=True,timeout=60,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    index=pd.read_csv(output/'exploration/frame_index.csv')
    assert len(index)==2 and index.time.is_unique and (index.convective_percent==0).all()
    state=json.loads((output/'exploration/progress.json').read_text())
    assert state['status']=='complete' and state['indexed']==2
