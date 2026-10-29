from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import CRS

from emma_gpm.nwcsaf_rdt_archive import (
    GifGeoReference,
    archive_candidate_urls,
    archive_slots,
    binary_spatial_metrics,
    differential_view_parallax,
    extract_reference_phases,
)
from scripts.render_nwcsaf_rdt_comparison_video import _stack_panels


ROOT = Path(__file__).resolve().parents[1]


def _small_georef() -> GifGeoReference:
    return GifGeoReference(
        archive="test",
        image_width=80,
        image_height=60,
        map_bbox_xyxy=(5, 5, 70, 50),
        geos_crs=CRS.from_epsg(3857),
        geos_to_pixel=np.eye(3),
        calibration={},
    )


def _rectangle(image: np.ndarray, bounds: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    image[y0:y1 + 1, x0] = color
    image[y0:y1 + 1, x1] = color
    image[y0, x0:x1 + 1] = color
    image[y1, x0:x1 + 1] = color


def test_archive_urls_use_year_day_and_satellite_fallback_order() -> None:
    urls = archive_candidate_urls("2024-02-29T12:15:00", archive="v2025_msg")
    assert "/2024060/" in urls[0]
    assert "MSG4_REFSYS-VISIR_20240229T121500Z.rdt.gif" in urls[0]
    assert [url.split("RDT-CW_")[1].split("_")[0] for url in urls] == [
        "MSG4",
        "MSG3",
        "MSG2",
        "MSG1",
    ]
    assert len(archive_slots("2025-01-01T00:00", "2025-01-01T01:00")) == 4


def test_phase_extraction_keeps_four_classes_and_excludes_split_and_legend() -> None:
    image = np.full((60, 80, 3), 100, dtype=np.uint8)
    _rectangle(image, (10, 10, 20, 20), (255, 255, 0))
    _rectangle(image, (25, 10, 35, 20), (255, 0, 0))
    _rectangle(image, (10, 27, 20, 37), (128, 0, 128))
    _rectangle(image, (25, 27, 35, 37), (0, 0, 255))
    _rectangle(image, (45, 10, 55, 20), (255, 165, 0))
    # A coloured legend-like rectangle outside the calibrated map is ignored.
    _rectangle(image, (10, 53, 20, 58), (255, 0, 0))

    result = extract_reference_phases(
        image,
        _small_georef(),
        color_tolerance=0,
        close_radius=1,
        min_line_pixels=4,
    )
    assert result.labels[15, 15] == 2
    assert result.labels[15, 30] == 3
    assert result.labels[32, 15] == 4
    assert result.labels[32, 30] == 5
    assert result.excluded[15, 50]
    assert result.labels[55, 15] == 0
    assert result.excluded[55, 15]


def test_zero_height_parallax_is_identity_and_cloud_height_is_bounded() -> None:
    longitude = np.asarray([0.0, 10.0, 25.0])
    latitude = np.asarray([35.0, 50.0, 65.0])
    zero_lon, zero_lat, zero_valid = differential_view_parallax(
        longitude, latitude, cloud_height_km=0.0
    )
    assert zero_valid.all()
    assert np.array_equal(zero_lon, longitude)
    assert np.array_equal(zero_lat, latitude)

    cloud_lon, cloud_lat, valid = differential_view_parallax(
        longitude, latitude, cloud_height_km=12.0
    )
    assert valid.all()
    displacement = np.hypot(cloud_lon - longitude, cloud_lat - latitude)
    assert np.all(displacement > 0.0)
    assert np.all(displacement < 0.25)


def test_calibrated_projection_roundtrip_is_subpixel() -> None:
    x = np.asarray([-1_000_000.0, 0.0, 1_500_000.0])
    y = np.asarray([3_200_000.0, 4_000_000.0, 5_000_000.0])
    for filename in ("nwcsaf_rdt_gif_v2021.json", "nwcsaf_rdt_gif_v2025_msg.json"):
        georef = GifGeoReference.from_json(ROOT / "configs" / filename)
        row, column = georef.projected_to_pixel(x, y)
        recovered_x, recovered_y = georef.pixel_to_projected(row, column)
        assert np.allclose(recovered_x, x, atol=1e-6)
        assert np.allclose(recovered_y, y, atol=1e-6)


def test_buffered_metrics_reward_nearby_but_not_exact_objects() -> None:
    reference = np.zeros((20, 20), dtype=bool)
    prediction = np.zeros_like(reference)
    reference[5:10, 5:10] = True
    prediction[5:10, 7:12] = True
    metrics = binary_spatial_metrics(
        reference,
        prediction,
        np.ones_like(reference),
        np.ones_like(reference, dtype=np.float32),
        spacing_km=(3.0, 3.0),
        buffer_km=(5.0, 10.0),
    )
    assert 0.0 < metrics["iou"] < 1.0
    assert metrics["buffered_iou_10km"] > metrics["iou"]
    assert metrics["f1_10km"] == 1.0


def test_video_stack_pads_one_pixel_archive_canvas_difference() -> None:
    top = Image.new("RGB", (10, 8), (255, 0, 0))
    bottom = Image.new("RGB", (10, 7), (0, 0, 255))

    result = _stack_panels(top, bottom, output_width=10)

    assert result.size == (10, 118)


def test_video_stack_crops_two_terminal_archive_rows() -> None:
    top = Image.new("RGB", (10, 8), (255, 0, 0))
    bottom = Image.new("RGB", (10, 6), (0, 0, 255))

    result = _stack_panels(top, bottom, output_width=10)

    assert result.size == (10, 116)
