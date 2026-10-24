from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.download_opera_year import opera_key, validate_opera_file, year_slots


def test_opera_key_uses_historical_quarter_hour_product() -> None:
    timestamp = pd.Timestamp("2020-02-29T23:45:00")
    assert opera_key(timestamp) == (
        "2020/02/29/OPERA/COMP/"
        "OPERA@20200229T2345@0@DBZH_QIND.h5"
    )


def test_leap_year_has_all_quarter_hour_slots() -> None:
    slots = year_slots(2020)
    assert len(slots) == 366 * 96
    assert slots[0] == pd.Timestamp("2020-01-01T00:00:00")
    assert slots[-1] == pd.Timestamp("2020-12-31T23:45:00")


def test_validate_opera_file_accepts_expected_odim_structure(tmp_path: Path) -> None:
    path = tmp_path / "opera.h5"
    with h5py.File(path, "w") as handle:
        where = handle.create_group("where")
        where.attrs["xsize"] = 1900
        where.attrs["ysize"] = 2200
        for index, quantity in enumerate(("DBZH", "QIND"), start=1):
            dataset = handle.create_group(f"dataset{index}")
            what = dataset.create_group("what")
            what.attrs["quantity"] = np.bytes_(quantity)
    validate_opera_file(path)
