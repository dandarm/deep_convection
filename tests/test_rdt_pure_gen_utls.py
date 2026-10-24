"""Contract tests for the improved SEVIRI-only RDT/UTLS teacher."""

from __future__ import annotations

import numpy as np
import pytest

from emma_gpm.rdt.cloud_type_ir import SeviriIrCloudTypeTracker
from emma_gpm.rdt.pure_gen_utls import (
    RDT_ORIGINAL_CLASS_NAMES,
    PureGenUtlsTracker,
    run_pure_gen_utls_sequence,
)
from scripts.render_rdt_pure_gen_utls_week import CLASS_COLORS, _radiance_composite
from scripts.search_rdt_pure_gen_utls_events_2022 import _ordered_inventory, _select_events


def _synthetic_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    area = np.full((height, width), 9.0, dtype=np.float32)
    frames = []
    for index in range(6):
        radius = np.hypot(yy - 48.0, xx - (40.0 + index))
        ir108 = np.full((height, width), 290.0, dtype=np.float32)
        cloud = radius <= 24.0
        ir108[cloud] = 222.0 + 0.08 * radius[cloud]
        ir108[radius <= 7.0] = 213.0 + 0.45 * radius[radius <= 7.0]
        if index >= 4:
            ir108[radius <= 1.5] = 204.0

        btd_wv_ir = np.full((height, width), -10.0, dtype=np.float32)
        btd_wv_ir[cloud] = 0.5
        btd_wv_ir[radius <= 2.0] = 3.0
        wv062 = ir108 + btd_wv_ir
        wv073 = wv062 + 8.0
        ir087 = ir108 + 0.5
        ir097 = ir108 + 7.0
        ir120 = ir108 - 0.5
        ir134 = ir108 + 10.0
        frames.append(np.stack([wv062, wv073, ir087, ir097, ir108, ir120, ir134]))
    times = np.asarray(
        [
            np.datetime64("2020-01-01T00:00") + np.timedelta64(5 * index, "m")
            for index in range(len(frames))
        ]
    )
    return np.stack(frames).astype(np.float32), times, area


def test_pure_gen_utls_keeps_original_rdt_classes_and_proxy() -> None:
    clip, times, area = _synthetic_sequence()
    output = run_pure_gen_utls_sequence(
        clip,
        times,
        area,
        lat_deg=45.0,
        lon_deg=10.0,
    )

    labels = output["semantic_label"]
    assert labels.shape == (6, 96, 96)
    assert set(np.unique(labels)).issubset(set(RDT_ORIGINAL_CLASS_NAMES))
    assert np.all(labels[output["ot_mask"]] == 6)
    assert int(output["ot_mask"].sum()) > 0
    assert np.isfinite(output["t_utls_proxy_k"]).any()
    assert float(output["proxy_confidence"].max()) >= 0.35
    assert np.isfinite(output["thermal_depth"]).any()
    assert output["cloud_type"].shape == labels.shape
    assert np.all(labels[~output["opaque_cloud_mask"]] == 0)


def test_ir_cloud_type_rejects_cold_land_and_keeps_deep_cloud() -> None:
    height = width = 64
    ir108 = np.full((height, width), 276.5, dtype=np.float32)
    frame = np.stack(
        [
            ir108 - 29.0,
            ir108 - 12.0,
            ir108 - 4.5,
            ir108 - 21.0,
            ir108,
            ir108 + 1.25,
            ir108 - 15.25,
        ]
    )
    cloud = np.zeros((height, width), dtype=bool)
    cloud[20:44, 20:44] = True
    frame[4, cloud] = 222.0
    frame[0, cloud] = 216.0
    frame[1, cloud] = 219.5
    frame[2, cloud] = 222.5
    frame[3, cloud] = 221.5
    frame[5, cloud] = 221.0
    frame[6, cloud] = 218.0

    result = SeviriIrCloudTypeTracker().step(frame, np.datetime64("2022-01-01T00:00"))
    assert not result.opaque_cloud_mask[5, 5]
    assert result.surface_like_mask[5, 5]
    assert result.opaque_cloud_mask[32, 32]
    assert result.ct_class[32, 32] == 3


def test_ir_cloud_type_uses_causal_surface_reference_for_warm_cloud() -> None:
    height = width = 32
    clear_ir = np.full((height, width), 290.0, dtype=np.float32)
    clear = np.stack(
        [
            clear_ir - 30.0,
            clear_ir - 12.0,
            clear_ir - 4.0,
            clear_ir - 20.0,
            clear_ir,
            clear_ir + 1.0,
            clear_ir - 15.0,
        ]
    )
    tracker = SeviriIrCloudTypeTracker()
    tracker.step(clear, np.datetime64("2022-01-01T00:00"))

    warm_cloud = clear.copy()
    patch = np.zeros((height, width), dtype=bool)
    patch[10:22, 10:22] = True
    warm_cloud[4, patch] = 270.0
    warm_cloud[0, patch] = 255.0
    warm_cloud[1, patch] = 263.0
    warm_cloud[2, patch] = 269.0
    warm_cloud[3, patch] = 258.0
    warm_cloud[5, patch] = 268.0
    warm_cloud[6, patch] = 260.0
    result = tracker.step(warm_cloud, np.datetime64("2022-01-01T00:05"))
    assert result.surface_depression_k[15, 15] >= 19.0
    assert result.opaque_cloud_mask[15, 15]
    assert result.ct_class[15, 15] == 1


def test_pure_gen_utls_requires_exact_five_minute_cadence() -> None:
    clip, times, area = _synthetic_sequence()
    tracker = PureGenUtlsTracker()
    tracker.step_frame(clip[0], times[0], area, lat_deg=45.0, lon_deg=10.0)
    with pytest.raises(ValueError, match="exact, strictly increasing 5-minute cadence"):
        tracker.step_frame(
            clip[1],
            times[0] + np.timedelta64(10, "m"),
            area,
            lat_deg=45.0,
            lon_deg=10.0,
        )


def test_pure_gen_utls_uses_native_grid_without_parallax() -> None:
    clip, times, area = _synthetic_sequence()
    output = run_pure_gen_utls_sequence(
        clip[:2],
        times[:2],
        area,
        lat_deg=65.0,
        lon_deg=35.0,
    )
    # The moving synthetic core remains centred on its apparent satellite
    # position; this branch never remaps labels toward a ground coordinate.
    first_track = output["track_id_map"][0]
    yy, xx = np.where(first_track > 0)
    assert 37.0 <= float(xx.mean()) <= 43.0
    assert 45.0 <= float(yy.mean()) <= 51.0


def test_pure_gen_utls_history_does_not_retain_old_full_domain_hierarchies() -> None:
    clip, times, area = _synthetic_sequence()
    tracker = PureGenUtlsTracker()
    for index in range(3):
        tracker.step_frame(
            clip[index],
            times[index],
            area,
            lat_deg=45.0,
            lon_deg=10.0,
        )

    snapshots = [
        snapshot
        for history in tracker.track_histories.values()
        for snapshot in history
    ]
    assert snapshots
    assert all(type(snapshot["vertical_profile"]) is dict for snapshot in snapshots)
    assert all(not hasattr(snapshot["vertical_profile"], "_hierarchy") for snapshot in snapshots)


def test_rdt_original_palette_and_exact_ot_pixels() -> None:
    assert CLASS_COLORS[1].tolist() == [59.0, 130.0, 246.0]
    assert CLASS_COLORS[6].tolist() == [245.0, 105.0, 255.0]
    ir108 = np.full((5, 5), 245.0, dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[2, 2] = 6
    plain = _radiance_composite(ir108, np.zeros_like(labels), np.ones_like(labels, dtype=bool))
    rendered = _radiance_composite(ir108, labels, np.ones_like(labels, dtype=bool))
    changed = np.any(rendered != plain, axis=-1)
    assert int(changed.sum()) == 1
    assert changed[2, 2]


def test_event_selection_keeps_independent_temporal_samples() -> None:
    summary = {
        "candidate_tracks": [
            {"day": "20220101", "local_frame": 24, "time": "2022-01-01T02:00:00", "score": 30.0},
            {"day": "20220101", "local_frame": 30, "time": "2022-01-01T02:30:00", "score": 20.0},
            {"day": "20220101", "local_frame": 60, "time": "2022-01-01T05:00:00", "score": 10.0},
        ]
    }
    selected = _select_events(
        [summary],
        target_events=3,
        clip_frames=16,
        min_separation_minutes=90.0,
    )
    assert [event["peak_time"] for event in selected] == [
        "2022-01-01T02:00:00",
        "2022-01-01T05:00:00",
    ]
    assert all(event["clip_stop_frame_exclusive"] - event["clip_start_frame"] == 16 for event in selected)


def test_daily_inventory_sorts_storage_order_without_changing_physical_time() -> None:
    values = np.arange(288, dtype=np.int64) * 300 + 1_640_995_200
    raw_order = np.r_[np.arange(180, 288), np.arange(180)]

    class _Array:
        def __getitem__(self, key):
            return values[raw_order][key]

    times, indices = _ordered_inventory({"time": _Array()})
    assert np.all(np.diff(times.asi8) == 300 * 1_000_000_000)
    assert np.array_equal(values[raw_order][indices], values)
