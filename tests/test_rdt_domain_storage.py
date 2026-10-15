import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.rdt import RDTDomainTeacher, RDTTeacherConfig
from emma_gpm.rdt.storage import DomainFrameReader, atomic_json, restore_latest, save_frame, load_domain_crop, publish_domain_tables


def synthetic_frame():
    y, x = np.indices((40, 48))
    radius = np.hypot(y-20, x-24)
    ir = np.full(radius.shape, 290., np.float32)
    ir[radius <= 10] = 225
    ir[radius <= 2] = 205
    return np.stack([ir+2, ir-1, ir+.5, ir-3, ir, ir-.5, ir-4])


def write_source(root, times):
    frame = synthetic_frame()
    # Both native axes reversed relative to north-up/west-left.
    dataset = xr.Dataset({"seviri": (("time", "channel", "y", "x"), np.stack([frame[:, ::-1, ::-1]]*len(times)))},
        coords={"time": times, "channel": list(THERMAL_CHANNELS), "x": np.arange(48)[::-1]*3000., "y": np.arange(40)*3000.},
        attrs={"units": "K", "calibration": "brightness_temperature", "projection_wkt": "EPSG:3857"})
    path = root/"2020/20200711.zarr"
    path.parent.mkdir(parents=True)
    dataset.to_zarr(path, consolidated=True, zarr_format=2)
    return frame


def test_rectangular_reader_orientation_and_metadata(tmp_path):
    root = tmp_path/"source"
    times = pd.date_range("2020-07-11", periods=2, freq="5min")
    frame = write_source(root, times)
    reader = DomainFrameReader(root)
    assert np.array_equal(reader.load(times[0]), frame)
    with pytest.raises(ValueError, match="Missing RSS"):
        reader.load("2020-07-11T00:15")
    reader.close()


def test_atomic_records_resume_and_crop_equal_full_domain(tmp_path):
    root, output = tmp_path/"source", tmp_path/"labels"
    times = pd.date_range("2020-07-11", periods=16, freq="5min")
    frame = write_source(root, times)
    output.mkdir()
    area = np.full(frame.shape[1:], 9.)
    np.save(output/"pixel_area_north_up.npy", area)
    atomic_json(output/"protocol.json", dict(root=str(root.resolve()), projection_wkt="EPSG:3857"))
    teacher = RDTDomainTeacher(area, RDTTeacherConfig(smoothing_sigma_pixels=0))
    expected = []
    for i, stamp in enumerate(times):
        result = teacher.step(frame, stamp)
        if i < 8:
            result = replace(result,fields={key:value for key,value in result.fields.items() if key!='invalid_radiometry'})
        expected.append(result.fields["track_id"].copy())
        save_frame(output, result, teacher, dict(source_frame_sha256="synthetic"))
        if i == 4:
            restored, records = restore_latest(output, teacher.config.config_hash)
            assert restored.last_time == teacher.last_time and restored.next_track_id == teacher.next_track_id
            assert len(records) == 5
            teacher = restored
    assert len(list((output/"frames").glob("*/state.pkl.zlib"))) == 2
    # A native top-left crop corresponds to the north-up bottom-right crop.
    example = load_domain_crop(root, output, times[0], 0, 0, crop_size=24)
    assert np.array_equal(example["clip"][0], frame[:, 16:, 24:])
    assert np.array_equal(example["result"]["tracks"], np.stack(expected)[:, 16:, 24:])
    assert example['result']['invalid_radiometry'].shape==(16,24,24)
    assert not example['result']['invalid_radiometry'].any()
    assert example["result"]["objects"].track_id.nunique() >= 1
    publish_domain_tables(output)
    joined = pd.read_csv(output/"objects.csv.gz")
    assert joined.child_track_ids.iloc[0] != "[]"
    assert joined.children_observed_next_frame.iloc[0]
    assert not joined.children_observed_next_frame.iloc[-1]
    restored, _ = restore_latest(output, teacher.config.config_hash)
    # Changing a compressed chunk must fail resume, even with valid metadata.
    chunk = next(p for p in (output/"frames"/"20200711T0000"/"fields.zarr"/"track_id").iterdir() if not p.name.startswith('.'))
    chunk.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt published"):
        restore_latest(output, teacher.config.config_hash)
