from __future__ import annotations

import numpy as np
import pytest

from emma_gpm.seviri import (
    THERMAL_CHANNELS,
    geographic_bbox_to_geos_bounds,
    native_slot_start_from_name,
)


def test_thermal_channel_order_is_stable() -> None:
    assert THERMAL_CHANNELS == (
        "WV_062",
        "WV_073",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
    )


def test_projected_emma_domain_has_finite_bounds() -> None:
    bounds = geographic_bbox_to_geos_bounds(-20.0, 40.0, 30.0, 60.0)
    values = np.asarray(bounds.xy_bbox)
    assert np.isfinite(values).all()
    assert bounds.x_min < 0 < bounds.x_max
    assert 3_000_000 < bounds.y_min < bounds.y_max < 5_100_000


def test_invalid_geographic_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        geographic_bbox_to_geos_bounds(40.0, -20.0, 30.0, 60.0)


def test_native_product_end_time_maps_to_five_minute_slot() -> None:
    name = "MSG3-SEVI-MSG15-0100-NA-20200501000415.534000000Z-NA.nat"
    assert native_slot_start_from_name(name).isoformat() == "2020-05-01T00:00:00"
