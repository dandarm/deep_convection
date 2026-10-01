from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from emma_gpm.seviri import NativeSeviriCrop, THERMAL_CHANNELS


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "crop_msg_seviri_native.py"
SPEC = importlib.util.spec_from_file_location("crop_msg_seviri_native", MODULE_PATH)
crop_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crop_script)


def fake_crop(hour: int, minute: int) -> NativeSeviriCrop:
    start = datetime(2020, 5, 30, hour, minute)
    return NativeSeviriCrop(
        data=np.zeros((len(THERMAL_CHANNELS), 224, 224), dtype=np.float32),
        x=np.arange(224),
        y=np.arange(224),
        channels=THERMAL_CHANNELS,
        start_time=start,
        end_time=start,
        calibration="brightness_temperature",
        units="K",
        source_path=Path("fake.nat"),
    )


def test_append_preserves_minute_resolution_in_time_coordinate(tmp_path):
    store = tmp_path / "20200530.zarr"
    crop_script.append_batch(store, [fake_crop(0, 0)], "float16")
    crop_script.append_batch(store, [fake_crop(1, 30)], "float16")
    dataset = xr.open_zarr(store, consolidated=True)
    assert list(pd.DatetimeIndex(dataset.time.values)) == [
        pd.Timestamp("2020-05-30T00:00:00"),
        pd.Timestamp("2020-05-30T01:30:00"),
    ]
