from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri import THERMAL_CHANNELS
from scripts.prepare_seviri_ssl import prepare
from scripts.pretrain_seviri_ssl import prepare_data


def test_preparation_and_training_load_real_format_zarr_with_train_only_stats(tmp_path):
    root = tmp_path / "archive"
    for day, offset in (("20200101", 0.), ("20200104", 10.)):
        values = (250. + offset + np.arange(16, dtype=np.float32)[:, None, None, None]
                  + np.arange(7, dtype=np.float32)[None, :, None, None]
                  + np.arange(226, dtype=np.float32)[None, None, None, :] / 10)
        values = np.broadcast_to(values, (16, 7, 224, 226)).copy()
        store = root / "2020" / f"{day}.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        data = xr.Dataset({"seviri": (("time", "channel", "y", "x"), values)},
            coords=dict(time=pd.date_range(pd.Timestamp(day), periods=16, freq="5min"),
                channel=list(THERMAL_CHANNELS), y=np.arange(224), x=np.arange(226)[::-1]))
        data.to_zarr(store, zarr_format=2, consolidated=True)
    train_path, val_path = tmp_path / "candidates.csv", tmp_path / "validation.csv"
    pd.DataFrame(dict(start_time=["2020-01-01"] * 3, origin_y=[0] * 3, origin_x=[0, 1, 2])).to_csv(train_path, index=False)
    pd.DataFrame(dict(start_time=["2020-01-04"], origin_y=[0], origin_x=[0])).to_csv(val_path, index=False)
    output = tmp_path / "prepared"
    output.mkdir()
    args = Namespace(manifest=train_path, validation_manifest=val_path, reserved_days=None,
        output_dir=output, zarr_root=root, count=None, cpu_threads=1, compensated=False,
        max_motion_pixels=12, uniform_fraction=.5, seed=1)
    prepare(args)
    stats = json.loads((output / "channel_stats.json").read_text())
    assert stats["samples"] == 3 and stats["pixel_count"] == 3 * 16 * 7 * 224 ** 2
    assert stats["global_mean_k"] == pytest.approx(271.75, abs=1e-4)
    args.manifest, args.validation_manifest = output / "train.csv", output / "validation.csv"
    args.synthetic_smoke, args.global_stats, args.resume = False, output / "channel_stats.json", None
    training, validation, loaded_stats = prepare_data(args)
    clip = training[0]["pixel_values"]
    history, target = ForecastWindow().split(clip[None])
    assert history.shape == (1, 10, 7, 224, 224) and target.shape == (1, 2, 7, 224, 224)
    # Native x decreases, so the loader must put column 223 on the west side.
    recovered = history * stats["global_std_k"] + stats["global_mean_k"]
    assert float(recovered[0, 0, 0, 0, 0]) == pytest.approx(272.3, abs=1e-4)
    assert float((validation[0]["pixel_values"] - clip).mean()) == pytest.approx(10. / stats["global_std_k"], abs=1e-5)
    assert loaded_stats == stats
    training.reader.close()
    validation.reader.close()
