from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.seviri_clips import (
    IncompleteSeviriClipError,
    SeviriZarrClipReader,
    normalize_seviri_clip,
)


def write_day(
    root: Path,
    day: str,
    times: pd.DatetimeIndex,
    *,
    channels: tuple[str, ...] = THERMAL_CHANNELS,
    spatial_pattern: bool = False,
    x_coordinates: np.ndarray | None = None,
    y_coordinates: np.ndarray | None = None,
) -> None:
    store = root / day[:4] / f"{day}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty((len(times), len(channels), 4, 4), dtype=np.float16)
    for time_index in range(len(times)):
        for channel_index in range(len(channels)):
            data[time_index, channel_index] = time_index + channel_index * 10
            if spatial_pattern:
                data[time_index, channel_index] += (
                    np.arange(4)[:, None] * 100 + np.arange(4)[None, :]
                )
    dataset = xr.Dataset(
        {"seviri": (("time", "channel", "y", "x"), data)},
        coords={
            "time": times,
            "channel": list(channels),
            "y": np.arange(4) if y_coordinates is None else y_coordinates,
            "x": np.arange(4) if x_coordinates is None else x_coordinates,
        },
    )
    dataset.to_zarr(store, mode="w", consolidated=True, zarr_format=2)


def test_reader_loads_seven_channels_across_midnight(tmp_path):
    write_day(
        tmp_path,
        "20200101",
        pd.date_range("2020-01-01T23:40:00", periods=4, freq="5min"),
    )
    write_day(
        tmp_path,
        "20200102",
        pd.date_range("2020-01-02T00:00:00", periods=4, freq="5min"),
    )

    with SeviriZarrClipReader(tmp_path, num_frames=8, crop_size=2) as reader:
        clip = reader.load("2020-01-01T23:40:00Z", origin_y=1, origin_x=1)

    assert clip.shape == (8, 7, 2, 2)
    assert tuple(THERMAL_CHANNELS)[-1] == "IR_134"
    assert np.all(clip[0, 6] == 60)
    assert np.all(clip[4, 6] == 60)


def test_reader_rejects_reordered_channels(tmp_path):
    channels = tuple(reversed(THERMAL_CHANNELS))
    write_day(
        tmp_path,
        "20200101",
        pd.date_range("2020-01-01T00:00:00", periods=2, freq="5min"),
        channels=channels,
    )
    with SeviriZarrClipReader(tmp_path, num_frames=2, crop_size=2) as reader:
        with pytest.raises(IncompleteSeviriClipError, match="Unexpected channel order"):
            reader.load("2020-01-01", origin_y=0, origin_x=0)


def test_reader_returns_north_up_and_west_left(tmp_path):
    write_day(
        tmp_path,
        "20200101",
        pd.date_range("2020-01-01T00:00:00", periods=1, freq="5min"),
        spatial_pattern=True,
        x_coordinates=np.array([3.0, 2.0, 1.0, 0.0]),
        y_coordinates=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    with SeviriZarrClipReader(tmp_path, num_frames=1, crop_size=4) as reader:
        clip = reader.load("2020-01-01", origin_y=0, origin_x=0)

    # Native row 3 is north and native column 3 is west.
    assert clip[0, 0, 0, 0] == 303
    assert clip[0, 0, -1, -1] == 0


def test_normalization_uses_one_global_transform_and_preserves_differences():
    clip = np.full((2, 7, 2, 2), 200.0, dtype=np.float32)
    clip[:, 1] = 210.0
    with pytest.raises(ValueError, match="must be scalars"):
        normalize_seviri_clip(clip, [200.0] * 7, [10.0] * 7)
    normalized = normalize_seviri_clip(clip, 250.0, 50.0)
    assert np.all(normalized[:, 0] == -1.0)
    assert np.allclose(normalized[:, 1] - normalized[:, 0], 0.2)


def test_small_config_and_reconstruction_width_when_training_stack_is_available():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from emma_gpm.videomae import (
        reconstruction_values_per_tubelet,
        videomae_small_7ch_config,
    )

    config = videomae_small_7ch_config()
    assert config.num_channels == 7
    assert config.norm_pix_loss is False
    assert reconstruction_values_per_tubelet(config) == 3584
