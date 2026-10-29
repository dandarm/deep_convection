from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from scripts.watch_seviri_upload import ready_candidate


def test_watcher_requires_physical_chunks_and_records_stable_hashes(tmp_path):
    root = tmp_path / "archive"
    store = root / "2020/20200912.zarr"
    store.parent.mkdir(parents=True)
    values = np.full((16, 7, 224, 224), 270., dtype=np.float16)
    dataset = xr.Dataset({"seviri": (("time", "channel", "y", "x"), values)},
        coords=dict(time=pd.date_range("2020-09-12", periods=16, freq="5min"),
            channel=list(THERMAL_CHANNELS), y=np.arange(224), x=np.arange(224)))
    dataset.to_zarr(store, zarr_format=2, consolidated=True,
                    encoding={"seviri": {"chunks": (1, 7, 224, 224)}})
    missing = store / "seviri/15.0.0.0"
    content = missing.read_bytes()
    missing.unlink()
    assert ready_candidate(root, stable_seconds=0) is None
    missing.write_bytes(content)
    candidate = ready_candidate(root, stable_seconds=0)
    assert candidate["start_time"] == "2020-09-12T00:00:00Z"
    assert candidate["origin_y"] == candidate["origin_x"] == 0
    assert "2020/20200912.zarr/seviri/15.0.0.0" in candidate["source_sha256"]
    assert ready_candidate(root, stable_seconds=60) is None
    coordinate = store / "time/0"
    coordinate.unlink()
    assert ready_candidate(root, stable_seconds=0) is None


def test_watcher_maps_chronological_window_to_unsorted_native_chunk_indices(tmp_path):
    root = tmp_path / "archive"
    store = root / "2020/20200912.zarr"
    store.parent.mkdir(parents=True)
    times = pd.date_range("2020-09-12", periods=32, freq="5min")
    native_times = times[16:].append(times[:16])
    dataset = xr.Dataset({"seviri": (("time", "channel", "y", "x"), np.full((32, 7, 224, 224), 270., dtype=np.float16))},
        coords=dict(time=native_times, channel=list(THERMAL_CHANNELS), y=np.arange(224), x=np.arange(224)))
    dataset.to_zarr(store, zarr_format=2, consolidated=True,
                    encoding={"seviri": {"chunks": (1, 7, 224, 224)}})
    # Only the first chronological window has arrived, at native indices 16..31.
    for index in range(16):
        (store / f"seviri/{index}.0.0.0").unlink()
    candidate = ready_candidate(root, stable_seconds=0)
    assert candidate["start_time"] == "2020-09-12T00:00:00Z"
    assert "2020/20200912.zarr/seviri/16.0.0.0" in candidate["source_sha256"]
    assert "2020/20200912.zarr/seviri/0.0.0.0" not in candidate["source_sha256"]
