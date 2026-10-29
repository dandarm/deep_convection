import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emma_gpm.seviri_clips import SeviriZarrClipReader, VIDEO_MAE_CHANNELS, IncompleteSeviriClipError
from emma_gpm.seviri_frame_cache import build_frame_cache, SeviriFrameCacheReader, manifest_days


@pytest.fixture
def cached_frames(tmp_path):
    root, cache = tmp_path / "source", tmp_path / "cache"
    for offset, (day, start) in enumerate([("20200912", "2020-09-12T23:50"), ("20200913", "2020-09-13T00:00")]):
        values = (220 + offset * 20 + np.arange(2 * 7 * 4 * 5).reshape(2, 7, 4, 5) / 8).astype(np.float16)
        ds = xr.Dataset({"seviri": (("time", "channel", "y", "x"), values)},
                        coords=dict(time=pd.date_range(start, periods=2, freq="5min"),
                                    channel=list(VIDEO_MAE_CHANNELS), y=[0, 1, 2, 3], x=[4, 3, 2, 1, 0]))
        ds.to_zarr(root / "2020" / f"{day}.zarr", mode="w", consolidated=True, zarr_format=2)
    build_frame_cache(root, cache, ["20200912", "20200913"], frames_per_read=1)
    return root, cache


def test_cached_and_direct_pinned_paths_preserve_values_orientation_and_midnight(cached_frames):
    root, cache = cached_frames
    with SeviriZarrClipReader(root, num_frames=3, crop_size=3) as original, SeviriFrameCacheReader(root, cache, num_frames=3, crop_size=3) as cached:
        args = dict(start_time="2020-09-12T23:55Z", origin_y=1, origin_x=1)
        expected = original.load(**args)
        np.testing.assert_array_equal(cached.load(**args), expected)
        np.testing.assert_array_equal(cached.load_native(**args).astype(np.float32), expected)
        destination = np.empty((3, 7, 3, 3), dtype=np.float16)
        assert cached.load_native_into(**args, out=destination) is destination
        np.testing.assert_array_equal(destination.astype(np.float32), expected)
        np.testing.assert_array_equal(cached.load(**args, global_mean=245., global_std=21.),
                                      original.load(**args, global_mean=245., global_std=21.))


def test_cache_refuses_source_changes(cached_frames):
    root, cache = cached_frames
    (root / "2020/20200912.zarr/new_chunk").write_bytes(b"changed")
    with SeviriFrameCacheReader(root, cache, num_frames=1, crop_size=3) as reader:
        with pytest.raises(ValueError, match="source has changed"):
            reader.load("2020-09-12T23:55Z", origin_y=0, origin_x=0)


def test_cache_refuses_truncation_and_channel_permutation(cached_frames):
    root, cache = cached_frames
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["stores"]["2020/20200912.zarr"]["channel"].reverse()
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with SeviriFrameCacheReader(root, cache, num_frames=1, crop_size=3) as reader:
        with pytest.raises(IncompleteSeviriClipError, match="channel order"):
            reader.load_native("2020-09-12T23:55Z", origin_y=0, origin_x=0)
    metadata["stores"]["2020/20200912.zarr"]["channel"].reverse()
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with (cache / "20200912.npy").open("r+b") as stream:
        stream.truncate(128)
    with SeviriFrameCacheReader(root, cache, num_frames=1, crop_size=3) as reader:
        with pytest.raises(IncompleteSeviriClipError, match="Truncated"):
            reader.load_native("2020-09-12T23:55Z", origin_y=0, origin_x=0)


def test_native_cache_refuses_missing_times_and_nonfinite_values(cached_frames):
    root, cache = cached_frames
    with SeviriFrameCacheReader(root, cache, num_frames=3, crop_size=3) as reader:
        with pytest.raises(IncompleteSeviriClipError, match="Missing exact"):
            reader.load_native("2020-09-13T00:00Z", origin_y=0, origin_x=0)
    data = np.load(cache / "20200912.npy", mmap_mode="r+")
    data[1, 0, 1, 1] = np.nan
    data.flush()
    del data
    with SeviriFrameCacheReader(root, cache, num_frames=1, crop_size=3) as reader:
        with pytest.raises(IncompleteSeviriClipError, match="NaN"):
            reader.load_native("2020-09-12T23:55Z", origin_y=0, origin_x=0)


def test_manifest_days_include_cross_midnight_observations():
    frame = pd.DataFrame({"start_time": ["2020-09-12T23:55Z"]})
    assert manifest_days(frame) == ["20200912", "20200913"]


def test_deferred_finite_validation_rejects_clip_before_forecasting(tmp_path):
    import torch
    from emma_gpm.forecasting import ForecastWindow
    root, cache = tmp_path / "source", tmp_path / "cache"
    values = np.full((16, 7, 4, 5), 260., dtype=np.float16)
    values[0, 0, 1, 1] = np.nan
    xr.Dataset({"seviri": (("time", "channel", "y", "x"), values)},
               coords=dict(time=pd.date_range("2020-09-12", periods=16, freq="5min"),
                           channel=list(VIDEO_MAE_CHANNELS), y=np.arange(4), x=np.arange(5))).to_zarr(
        root / "2020/20200912.zarr", mode="w", consolidated=True, zarr_format=2)
    build_frame_cache(root, cache, ["20200912"])
    with SeviriFrameCacheReader(root, cache, crop_size=3, require_finite=False) as reader:
        raw = reader.load_native("2020-09-12", origin_y=0, origin_x=0)
    normalized = (torch.from_numpy(raw).unsqueeze(0).float() - 245.) / 21.
    with pytest.raises(ValueError, match="Nonfinite observations"):
        ForecastWindow().split(normalized)
