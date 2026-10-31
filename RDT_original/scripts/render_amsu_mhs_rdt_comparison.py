#!/usr/bin/env python3
"""Render geo-aligned static comparisons between AMSU/MHS PMW Medicane snapshots and Pure SEVIRI RDT.

For each geolocated AMSU/MHS snapshot in ``E:\\Datasets\\Medicanes-AMSU-MHS\\`` that has
pre-computed Pure RDT semantic masks (``_cache_YYYYMMDD_pure.npz``) and daily SEVIRI RSS
Zarr archives (``E:\\Datasets\\MSG-SEVIRI\\YYYY\\YYYYMMDD.zarr``), this script:
1. Calibrates the exact 2D pixel-to-(lat, lon) transformation of the AMSU/MHS map box
   from its axis tick positions and internal 0.5-deg dashed graticule.
2. Selects the closest 5-minute SEVIRI RSS + Pure RDT frame (|dt| <= 2.5 min) relative
   to the overpass timestamp printed in the AMSU/MHS figure title.
3. Reprojects the parallax-corrected SEVIRI IR 10.8 background and Pure RDT 6-class
   convective lifecycle mask onto the exact pixel grid of the AMSU/MHS map box.
4. Produces a 3-panel high-resolution comparison figure:
   - Panel 1: Original AMSU/MHS TB (183+-7 GHz) with PMW convective 'x' markers & ellipse.
   - Panel 2: Pure SEVIRI RDT on IR 10.8 reprojected onto the identical geographic frame.
   - Panel 3: Direct Geographic Overlay of Pure RDT convective cells + contours on AMSU/MHS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from scipy import ndimage
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[2]
RDT_ORIG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RDT_ORIG_ROOT))

from rdt_nwcsaf.ancillary import (
    apply_parallax_correction_2d,
    compute_hrw_amv_guess_field,
    compute_nwcsaf_cloud_type,
)
from rdt_nwcsaf.config import NwcSafRdtConfig
from rdt_nwcsaf.detection import detect_rdt_cells
from rdt_nwcsaf.discrimination import discriminate_and_classify_cell
from rdt_nwcsaf.otd import detect_cell_overshooting_tops
from rdt_nwcsaf.tracking import track_rdt_cells
from scripts.process_medicane_windows import _make_snapshot, fill_internal_mask_holes


@dataclass(frozen=True)
class AmsuSnapshotSpec:
    filename: str
    medicane: str
    event_folder: str
    day_str: str
    file_time_utc: str
    overpass_time_utc: str  # Exact timestamp printed in the figure title (UTC)
    lat_max: float          # Topmost labeled latitude tick on left axis (deg N)
    lat_min: float          # Bottommost labeled latitude tick on left axis (deg N)
    lon_min: float          # Leftmost labeled longitude tick on bottom axis (deg E)
    lon_max: float          # Rightmost labeled longitude tick on bottom axis (deg E)


# Complete verified catalog of the 32 AMSU/MHS snapshots with cached Pure RDT data
AMSU_SNAPSHOTS: tuple[AmsuSnapshotSpec, ...] = (
    # Rolf (Nov 2011)
    AmsuSnapshotSpec("AMSU_20111106_084201_rolf.jpg", "Rolf", "Rolf_7001328_20111105_20111108", "20111106", "08:42", "2011-11-06 09:19", 43.50, 38.50, 1.25, 6.75),
    AmsuSnapshotSpec("AMSU_20111108_094200_rolf.jpg", "Rolf", "Rolf_7001328_20111105_20111108", "20111108", "09:42", "2011-11-08 10:18", 44.25, 38.75, 4.00, 9.00),
    AmsuSnapshotSpec("AMSU_20111108_194800_rolf.jpg", "Rolf", "Rolf_7001328_20111105_20111108", "20111108", "19:48", "2011-11-08 19:58", 44.75, 39.75, 3.50, 9.00),
    # Qendresa (Nov 2014)
    AmsuSnapshotSpec("AMSU_20141107_0757_qendresa.jpg", "Qendresa", "Qendresa_7001461_20141107_20141108", "20141107", "07:57", "2014-11-07 08:36", 38.75, 33.25, 9.50, 15.00),
    AmsuSnapshotSpec("AMSU_20141108_2015_qendresa.jpg", "Qendresa", "Qendresa_7001461_20141107_20141108", "20141108", "20:15", "2014-11-08 20:24", 37.75, 32.75, 15.50, 21.00),
    # Trixie / Medicane_1542 (Oct 2016)
    AmsuSnapshotSpec("AMSU_20161028_0742_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161028", "07:42", "2016-10-28 08:20", 39.75, 34.25, 14.25, 19.75),
    AmsuSnapshotSpec("AMSU_20161028_1930_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161028", "19:30", "2016-10-28 19:40", 38.00, 32.50, 13.50, 19.00),
    AmsuSnapshotSpec("AMSU_20161028_2018_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161028", "20:18", "2016-10-28 20:26", 38.00, 32.50, 13.50, 19.00),
    AmsuSnapshotSpec("AMSU_20161030_0745_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161030", "07:45", "2016-10-30 08:25", 38.00, 32.50, 14.50, 20.00),
    AmsuSnapshotSpec("AMSU_20161030_0842_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161030", "08:42", "2016-10-30 09:19", 38.00, 32.50, 14.50, 20.00),
    AmsuSnapshotSpec("AMSU_20161031_0821_trixie.jpg", "Trixie", "Medicane_1542_7001542_20161028_20161031", "20161031", "08:21", "2016-10-31 08:58", 37.75, 32.25, 20.25, 25.25),
    # Numa (Nov 2017)
    AmsuSnapshotSpec("AMSU_20171116_0739_numa.jpg", "Numa", "Numa_7001575_20171116_20171118", "20171116", "07:39", "2017-11-16 08:18", 37.75, 32.75, 12.75, 18.25),
    AmsuSnapshotSpec("AMSU_20171116_0836_numa.jpg", "Numa", "Numa_7001575_20171116_20171118", "20171116", "08:36", "2017-11-16 09:14", 37.75, 32.75, 13.00, 18.50),
    AmsuSnapshotSpec("AMSU_20171117_1909_numa.jpg", "Numa", "Numa_7001575_20171116_20171118", "20171117", "19:09", "2017-11-17 19:18", 41.75, 36.25, 15.25, 20.75),
    AmsuSnapshotSpec("AMSU_20171117_2006_numa.jpg", "Numa", "Numa_7001575_20171116_20171118", "20171117", "20:06", "2017-11-17 20:13", 41.50, 36.50, 15.25, 20.75),
    AmsuSnapshotSpec("AMSU_20171118_0839_numa.jpg", "Numa", "Numa_7001575_20171116_20171118", "20171118", "08:39", "2017-11-18 09:17", 41.75, 36.25, 15.75, 21.25),
    # Ianos (Sep 2020)
    AmsuSnapshotSpec("AMSU_20200916_0733_ianos.jpg", "Ianos", "Ianos_7001702_20200914_20200918", "20200916", "07:33", "2020-09-16 08:13", 37.25, 31.75, 13.75, 19.25),
    AmsuSnapshotSpec("AMSU_20200916_2012_ianos.jpg", "Ianos", "Ianos_7001702_20200914_20200918", "20200916", "20:12", "2020-09-16 20:20", 39.25, 33.75, 14.50, 20.00),
    AmsuSnapshotSpec("AMSU_20200917_1903_ianos.jpg", "Ianos", "Ianos_7001702_20200914_20200918", "20200917", "19:03", "2020-09-17 19:12", 40.00, 34.50, 16.75, 22.25),
    # Apollo (Oct 2021)
    AmsuSnapshotSpec("AMSU_20211028_1833_apollo.jpg", "Apollo", "Apollo_39299_20211028_20211030", "20211028", "18:33", "2021-10-28 18:43", 37.75, 32.75, 13.00, 18.50),
    AmsuSnapshotSpec("AMSU_20211029_0742_apollo.jpg", "Apollo", "Apollo_39299_20211028_20211030", "20211029", "07:42", "2021-10-29 08:19", 39.00, 34.00, 13.25, 18.75),
    AmsuSnapshotSpec("AMSU_20211029_1930_apollo.jpg", "Apollo", "Apollo_39299_20211028_20211030", "20211029", "19:30", "2021-10-29 19:39", 39.00, 34.00, 13.25, 18.75),
    AmsuSnapshotSpec("AMSU_20211029_2024_apollo.jpg", "Apollo", "Apollo_39299_20211028_20211030", "20211029", "20:24", "2021-10-29 20:31", 39.00, 34.00, 13.25, 18.75),
    AmsuSnapshotSpec("AMSU_20211030_1909_apollo.jpg", "Apollo", "Apollo_39299_20211028_20211030", "20211030", "19:09", "2021-10-30 19:19", 36.75, 31.75, 15.75, 21.25),
    # Juliette (Feb-Mar 2023)
    AmsuSnapshotSpec("AMSU_20230228_1954_juliette.jpg", "Juliette", "Juliette_83565_20230227_20230302", "20230228", "19:54", "2023-02-28 20:03", 41.00, 35.50, 4.50, 9.50),
    AmsuSnapshotSpec("AMSU_20230228_2048_juliette.jpg", "Juliette", "Juliette_83565_20230227_20230302", "20230228", "20:48", "2023-02-28 20:56", 41.00, 35.50, 4.75, 9.75),
    AmsuSnapshotSpec("AMSU_20230301_1933_juliette.jpg", "Juliette", "Juliette_83565_20230227_20230302", "20230301", "19:33", "2023-03-01 19:42", 42.75, 37.75, 7.75, 13.25),
    AmsuSnapshotSpec("AMSU_20230302_0815_juliette.jpg", "Juliette", "Juliette_83565_20230227_20230302", "20230302", "08:15", "2023-03-02 08:53", 43.25, 38.25, 8.25, 13.75),
    AmsuSnapshotSpec("AMSU_20230302_0903_juliette.jpg", "Juliette", "Juliette_83565_20230227_20230302", "20230302", "09:03", "2023-03-02 09:42", 43.25, 38.25, 8.25, 13.75),
    # Daniel (Sep 2023)
    AmsuSnapshotSpec("AMSU_20230906_192100_daniel.jpg", "Daniel", "Daniel_46218_20230905_20230910", "20230906", "19:21", "2023-09-06 19:30", 37.25, 31.75, 14.50, 20.00),
    AmsuSnapshotSpec("AMSU_20230909_1912_daniel.jpg", "Daniel", "Daniel_46218_20230905_20230910", "20230909", "19:12", "2023-09-09 19:21", 35.00, 30.00, 16.50, 22.00),
    AmsuSnapshotSpec("AMSU_20230909_200000_daniel.jpg", "Daniel", "Daniel_46218_20230905_20230910", "20230909", "20:00", "2023-09-09 20:08", 35.00, 30.00, 16.50, 22.00),
)

CLASS_COLORS: dict[int, tuple[int, int, int, float]] = {
    1: (59, 130, 246, 0.50),    # Non-convective cold cloud -> Steel Blue
    2: (250, 204, 21, 0.82),    # Triggering -> Bright Yellow
    3: (249, 115, 22, 0.82),    # Growing -> Vibrant Orange
    4: (220, 38, 38, 0.82),     # Mature -> Vivid Crimson Red
    5: (153, 27, 27, 0.82),     # Decaying -> Dark Brick Red
    6: (245, 105, 255, 0.96),   # Overshooting Top (OT) -> Bright Magenta
}


def _orient_north_up_west_left(
    data: np.ndarray, y_raw: np.ndarray, x_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = data
    y_out = y_raw.copy()
    x_out = x_raw.copy()
    if y_out[0] < y_out[-1]:
        out = out[..., ::-1, :]
        y_out = y_out[::-1]
    if x_out[0] > x_out[-1]:
        out = out[..., :, ::-1]
        x_out = x_out[::-1]
    return np.ascontiguousarray(out), y_out, x_out


def load_seviri_geometry(zarr_path: Path) -> dict[str, Any]:
    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        x_raw = ds.x.values.copy()
        y_raw = ds.y.values.copy()
        proj_wkt = ds.attrs["projection_wkt"]

    dummy = np.zeros((1, len(y_raw), len(x_raw)), dtype=np.float32)
    _, y_m, x_m = _orient_north_up_west_left(dummy, y_raw, x_raw)

    crs = CRS.from_wkt(proj_wkt)
    xx, yy = np.meshgrid(x_m, y_m)
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_deg, lat_deg = to_ll.transform(xx, yy)
    from_ll = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    lat_deg_f = np.where(np.isfinite(lat_deg), lat_deg, np.nan).astype(np.float32)
    lon_deg_f = np.where(np.isfinite(lon_deg), lon_deg, np.nan).astype(np.float32)
    lat_rad = np.radians(lat_deg_f)
    lon_rad = np.radians(lon_deg_f)
    dlat_dr, dlat_dc = np.gradient(lat_rad)
    dlon_dr, dlon_dc = np.gradient(lon_rad)
    a_wgs = 6378.137
    e2_wgs = 0.00669437999014
    sin_lat = np.sin(lat_rad)
    denom = np.sqrt(np.maximum(1.0 - e2_wgs * sin_lat * sin_lat, 1e-6))
    r_merid = a_wgs * (1.0 - e2_wgs) / (denom ** 3)
    r_prime = a_wgs / denom
    jac_rad2 = np.abs(dlat_dr * dlon_dc - dlat_dc * dlon_dr)
    area_km2 = (r_merid * r_prime * np.cos(lat_rad) * jac_rad2).astype(np.float32)
    area_km2 = np.where(np.isfinite(area_km2) & (area_km2 > 0.0) & (area_km2 < 250.0), area_km2, 12.0).astype(np.float32)

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_m": x_m,
        "y_m": y_m,
        "crs": crs,
        "from_ll": from_ll,
        "area_km2": area_km2,
        "lat_deg": lat_deg_f,
        "lon_deg": lon_deg_f,
    }


def _compute_pure_rdt_slot_from_zarr(
    g: Any,
    chrono_idx: np.ndarray,
    step_idx: int,
    geom: dict[str, Any],
    warmup_frames: int = 6,
) -> np.ndarray:
    """Compute causal Pure RDT semantic mask for a single slot using a short warmup window."""
    pure_cfg = NwcSafRdtConfig(enforce_stretched_mask=False, nwp_conv_filter=False, discri_scheme="GEN")
    area_km2 = geom["area_km2"]
    lat_deg = geom["lat_deg"]
    lon_deg = geom["lon_deg"]
    h, w = area_km2.shape

    start_i = max(0, step_idx - warmup_frames)
    prev_f: np.ndarray | None = None
    prev_labels: np.ndarray | None = None
    prev_hierarchy = None
    prev_cells_pure: dict[int, dict[str, Any]] = {}
    hist_pure: dict[int, list[dict[str, Any]]] = {}
    sem_pure = np.zeros((h, w), dtype=np.uint8)

    for idx_cur in range(start_i, step_idx + 1):
        raw_i = int(chrono_idx[idx_cur])
        raw_curr = np.asarray(g["seviri"][raw_i], dtype=np.float32)
        f_curr, _, _ = _orient_north_up_west_left(raw_curr, geom["y_raw"], geom["x_raw"])

        cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
        t_tr_c = (
            float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5))
            if int(np.count_nonzero(cold_c)) >= 50
            else 215.0
        )
        pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)
        pure_tropo_c_scalar = float(np.clip(t_tr_c, 205.0, 226.0))

        ct_curr = compute_nwcsaf_cloud_type(
            f_curr[4], ir120=f_curr[5], ir087=f_curr[2], wv062=f_curr[0], wv073=f_curr[1],
            t_tropo_k=pure_tropo_c, t_surf_k=292.0,
        )
        if prev_f is not None:
            dy_px, dx_px = compute_hrw_amv_guess_field(
                prev_f[4], f_curr[4], prev_wv062=prev_f[0], curr_wv062=f_curr[0], dt_seconds=300.0
            )
            guess_motion_yx = np.stack([dy_px, dx_px], axis=0)
        else:
            guess_motion_yx = None

        curr_labels, curr_cells_base, curr_hierarchy, _ = detect_rdt_cells(
            f_curr, area_km2, pure_cfg, ct_cloudy_mask=ct_curr.opaque_cloud_mask, tropopause_temp_k=pure_tropo_c
        )
        if prev_labels is not None and prev_hierarchy is not None and prev_f is not None:
            parents, children, edges, primary_parent, events, _ = track_rdt_cells(
                prev_cells_pure, prev_labels, prev_hierarchy, prev_f[4],
                curr_cells_base, curr_labels, curr_hierarchy, f_curr[4],
                area_km2, pure_cfg, guess_motion_yx=guess_motion_yx,
            )
        else:
            parents, children, edges, primary_parent = {}, {}, {}, {}
            events = {cid: "new" for cid in curr_cells_base}

        curr_cells_pure: dict[int, dict[str, Any]] = {}
        new_hist_pure: dict[int, list[dict[str, Any]]] = {}
        sem_pure = np.zeros((h, w), dtype=np.uint8)

        for cid, base_c in curr_cells_base.items():
            ev = events.get(cid, "new")
            p_main = primary_parent.get(cid)
            p_list = parents.get(cid, [])
            cy_i = int(np.clip(round(float(base_c["weight_centroid_yx"][0])), 0, h - 1))
            cx_i = int(np.clip(round(float(base_c["weight_centroid_yx"][1])), 0, w - 1))
            c_lat = float(lat_deg[cy_i, cx_i])
            c_lon = float(lon_deg[cy_i, cx_i])
            base_c["lat_deg"] = c_lat
            base_c["lon_deg"] = c_lon
            base_c["event"] = ev

            ot_c = detect_cell_overshooting_tops(base_c, f_curr, area_km2, pure_cfg, cell_lat_deg=c_lat)
            base_c["ot_confirmed_count"] = int(ot_c["confirmed_count"])
            base_c["ot_confirmed_pixels"] = int(ot_c["confirmed_pixels"])
            bbox = base_c["bbox"]
            mask = base_c["mask"]
            ot_conf_loc = ot_c["confirmed_mask"] & mask

            cp = dict(base_c)
            prev_p_c = prev_cells_pure.get(p_main) if p_main is not None else None
            conv_p_p = prev_p_c if (prev_p_c is not None and prev_p_c.get("is_convective")) else None
            if conv_p_p is None:
                for pid in sorted(p_list):
                    if prev_cells_pure.get(pid, {}).get("is_convective"):
                        conv_p_p = prev_cells_pure[pid]
                        break

            hp = (
                hist_pure.get(p_main, [])
                if (p_main is not None and ev in ("continuation", "merge_main", "split_main"))
                else []
            )
            dp = discriminate_and_classify_cell(
                cp, prev_p_c, conv_p_p, hp, prev_hierarchy, curr_hierarchy, ot_c,
                5.0, pure_cfg, nwp_mask_value=2, lightning_strokes=0,
                cell_lat_deg=c_lat, calibrated_model=None, cell_tropo_temp_k=pure_tropo_c_scalar,
            )
            cp.update(dp)
            curr_cells_pure[cid] = cp
            new_hist_pure[cid] = (hp + [_make_snapshot(cp, cid, ev, idx_cur)])[-12:]
            sem_pure[bbox][mask] = int(dp["class_id"])
            if dp["is_convective"] and np.any(ot_conf_loc):
                sem_pure[bbox][ot_conf_loc] = 6

        prev_f = f_curr
        prev_labels = curr_labels
        prev_hierarchy = curr_hierarchy
        prev_cells_pure = curr_cells_pure
        hist_pure = new_hist_pure

    sem_pure = apply_parallax_correction_2d(
        sem_pure, f_curr[4], lat_deg, lon_deg, t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=0
    )
    return fill_internal_mask_holes(sem_pure, max_hole_size=16)


def calibrate_amsu_image_geometry(
    img_path: Path, spec: AmsuSnapshotSpec
) -> dict[str, Any]:
    """Load AMSU JPEG (normalized to 1200x1200) and calibrate pixel <-> (lat, lon) mapping."""
    im_pil = Image.open(img_path)
    raw_arr = np.asarray(im_pil)
    if im_pil.size != (1200, 1200):
        im_pil = im_pil.resize((1200, 1200), Image.Resampling.LANCZOS)
    im_rgb = np.asarray(im_pil, dtype=np.uint8)
    gray = im_rgb.mean(axis=2)

    # For 6000x6000 images (Rolf), min-pool preserves crisp 1-px dotted gridlines for slant measurement
    if raw_arr.shape[0] == 6000:
        gray_grid = raw_arr.reshape(1200, 5, 1200, 5, 3).min(axis=(1, 3)).mean(axis=2)
    else:
        gray_grid = gray

    colored = (im_rgb.max(axis=2).astype(int) - im_rgb.min(axis=2).astype(int)) > 25
    in_map = colored[:, :1010] | (gray[:, :1010] < 120)
    sub = in_map[170:1030, 110:1005]
    ys = np.where(sub.mean(axis=1) > 0.5)[0] + 170
    xs = np.where(sub.mean(axis=0) > 0.5)[0] + 110
    y0, y1 = int(ys[0]), int(ys[-1])
    x0, x1 = int(xs[0]), int(xs[-1])

    # Detect left-axis latitude tick centers (at x = x0)
    left_strip = (gray[y0 - 8 : y1 + 8, 15 : x0 - 6] < 180).any(axis=1)
    lbl_y, n_y = ndimage.label(left_strip)
    y_ticks = np.array(
        [
            float(np.mean(np.where(lbl_y == i)[0])) + (y0 - 8)
            for i in range(1, n_y + 1)
            if np.sum(lbl_y == i) >= 5
        ],
        dtype=np.float64,
    )

    # Detect bottom-axis longitude tick centers (at y = y1)
    bot_strip = (gray[y1 + 6 : y1 + 110, x0 - 8 : x1 + 8] < 180).any(axis=0)
    lbl_x, n_x = ndimage.label(bot_strip)
    x_ticks = np.array(
        [
            float(np.mean(np.where(lbl_x == i)[0])) + (x0 - 8)
            for i in range(1, n_x + 1)
            if np.sum(lbl_x == i) >= 5
        ],
        dtype=np.float64,
    )

    lat_vals = np.linspace(spec.lat_max, spec.lat_min, len(y_ticks))
    lon_vals = np.linspace(spec.lon_min, spec.lon_max, len(x_ticks))

    # Snap interior y_ticks and x_ticks to exact dotted gridline peaks (within +-5 px)
    h_diff = np.minimum(gray_grid[:, 2:-2] - gray_grid[:, :-4], gray_grid[:, 2:-2] - gray_grid[:, 4:])
    dark_v = np.maximum(0.0, -h_diff)
    v_diff = np.minimum(gray_grid[2:-2, :] - gray_grid[:-4, :], gray_grid[2:-2, :] - gray_grid[4:, :])
    dark_h = np.maximum(0.0, -v_diff)

    dark_h_prof = dark_h[:, x0 + 15 : x0 + 350].mean(axis=1)
    for idx_t in range(len(y_ticks)):
        yt = y_ticks[idx_t]
        if y0 + 10 <= yt <= y1 - 10:
            rc = int(round(yt)) - 2
            r_win = np.arange(max(0, rc - 5), min(len(dark_h_prof), rc + 6))
            if len(r_win) > 0 and np.max(dark_h_prof[r_win]) > 1.5:
                y_ticks[idx_t] = float(r_win[int(np.argmax(dark_h_prof[r_win]))] + 2)

    dark_v_bot = dark_v[max(y0 + 20, y1 - 240) : y1 - 8, :].mean(axis=0)
    for idx_t in range(len(x_ticks)):
        xt = x_ticks[idx_t]
        if x0 + 10 <= xt <= x1 - 10:
            cc = int(round(xt)) - 2
            c_win = np.arange(max(0, cc - 6), min(len(dark_v_bot), cc + 7))
            if len(c_win) > 0 and np.max(dark_v_bot[c_win]) > 1.2:
                x_ticks[idx_t] = float(c_win[int(np.argmax(dark_v_bot[c_win]))] + 2)

    yy_s = np.arange(max(y0 + 30, 300), min(y1 - 30, 900))
    xx_s = np.arange(max(x0 + 80, 250), min(x1 - 80, 860))
    slopes = np.linspace(-0.26, 0.26, 261)

    dy_vec = (yy_s - 600)[:, None]
    best_sv, best_val_v = 0.0, -1.0
    for s in slopes:
        cols = np.clip(np.round(xx_s[None, :] + s * dy_vec).astype(int), 0, dark_v.shape[1] - 1)
        prof = dark_v[yy_s[:, None], cols].mean(axis=0)
        val = float(np.percentile(prof, 96) - np.percentile(prof, 50))
        if val > best_val_v:
            best_val_v, best_sv = val, float(s)

    dx_vec = (xx_s - 560)[None, :]
    best_sh, best_val_h = 0.0, -1.0
    for s in np.linspace(-0.08, 0.08, 161):
        rows = np.clip(np.round(yy_s[:, None] + s * dx_vec).astype(int), 0, dark_h.shape[0] - 1)
        prof = dark_h[rows, xx_s[None, :]].mean(axis=1)
        val = float(np.percentile(prof, 96) - np.percentile(prof, 50))
        if val > best_val_h:
            best_val_h, best_sh = val, float(s)

    # Also account for meridian convergence across the box: d(sv)/d(lon) ~ +0.016 / deg
    lon_center = 0.5 * (spec.lon_min + spec.lon_max)

    # Build 2D (lat, lon) arrays for every pixel (y, x) inside the map box [y0:y1, x0:x1]
    box_h = y1 - y0
    box_w = x1 - x0
    yy_box, xx_box = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )

    # 1. Parallel equation: y_px(lat, x) = y_left(lat) + best_sh * (x_px - x0)
    #    => y_left_equiv = y_px - best_sh * (x_px - x0)
    y_left_equiv = yy_box - best_sh * (xx_box - x0)
    # Linear fit / piecewise interpolation with extrapolation at box top/bottom edges
    lat_poly = np.polyfit(y_ticks, lat_vals, deg=2)
    lat_box = np.polyval(lat_poly, y_left_equiv)

    # 2. Meridian equation: x_px(y, lon) = x_bot(lon) + sv(lon) * (y_px - y1)
    #    First estimate lon0 from mean slant best_sv:
    x_bot_approx = xx_box - best_sv * (yy_box - y1)
    lon_poly = np.polyfit(x_ticks, lon_vals, deg=1)
    lon_approx = np.polyval(lon_poly, x_bot_approx)
    # Refine with local meridian convergence around lon_center:
    sv_local = best_sv + 0.0160 * (lon_approx - lon_center)
    x_bot_equiv = xx_box - sv_local * (yy_box - y1)
    lon_box = np.polyval(lon_poly, x_bot_equiv)

    return {
        "im_rgb": im_rgb,
        "box": (x0, y0, x1, y1),
        "box_h": box_h,
        "box_w": box_w,
        "y_ticks": y_ticks,
        "x_ticks": x_ticks,
        "lat_vals": lat_vals,
        "lon_vals": lon_vals,
        "best_sv": best_sv,
        "best_sh": best_sh,
        "lat_box": lat_box.astype(np.float32),
        "lon_box": lon_box.astype(np.float32),
    }


def _ir108_to_rgb_base(ir108_k: np.ndarray) -> np.ndarray:
    norm = np.clip((300.0 - np.nan_to_num(ir108_k, nan=295.0)) / (300.0 - 202.0), 0.0, 1.0)
    gray = (30.0 + 205.0 * norm).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _render_rdt_on_ir(base_rgb: np.ndarray, sem_mask: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    for cls_id, (r, g, b, alpha) in CLASS_COLORS.items():
        m = sem_mask == cls_id
        if np.any(m):
            if cls_id == 6:
                m = ndimage.binary_dilation(m, iterations=2)
            out[m, 0] = (1.0 - alpha) * out[m, 0] + alpha * r
            out[m, 1] = (1.0 - alpha) * out[m, 1] + alpha * g
            out[m, 2] = (1.0 - alpha) * out[m, 2] + alpha * b
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_rdt_overlay_on_amsu(amsu_box_rgb: np.ndarray, sem_mask: np.ndarray) -> np.ndarray:
    """Overlay Pure RDT semantic classes (semi-transparent fill + crisp outlines) onto the AMSU map."""
    out = amsu_box_rgb.astype(np.float32).copy()

    # Preserve original dark features (coastlines, cyclone ellipse, 'x' markers, dashed grid)
    gray_amsu = amsu_box_rgb.mean(axis=2)
    is_dark_annotation = gray_amsu < 75.0

    # 1. Non-convective cold cloud (Class 1): light dashed/thin contour + subtle tint
    m_cold = sem_mask == 1
    if np.any(m_cold):
        alpha_fill = 0.22
        out[m_cold, 0] = (1.0 - alpha_fill) * out[m_cold, 0] + alpha_fill * 59.0
        out[m_cold, 1] = (1.0 - alpha_fill) * out[m_cold, 1] + alpha_fill * 130.0
        out[m_cold, 2] = (1.0 - alpha_fill) * out[m_cold, 2] + alpha_fill * 246.0
        edge_cold = m_cold ^ ndimage.binary_erosion(m_cold, iterations=2)
        out[edge_cold] = [30.0, 90.0, 230.0]

    # 2. Convective classes (2..5): semi-transparent fill (0.38) + bold 3-px contour + white outer halo
    conv_any = (sem_mask >= 2) & (sem_mask <= 6)
    if np.any(conv_any):
        # White outer halo around convective cells so they pop against orange/green/blue AMSU TB
        halo = ndimage.binary_dilation(conv_any, iterations=4) & (~ndimage.binary_dilation(conv_any, iterations=2))
        out[halo] = 0.25 * out[halo] + 0.75 * np.array([255.0, 255.0, 255.0], dtype=np.float32)

    for cls_id in (2, 3, 4, 5):
        m = sem_mask == cls_id
        if not np.any(m):
            continue
        r, g, b, _ = CLASS_COLORS[cls_id]
        alpha_fill = 0.36
        out[m, 0] = (1.0 - alpha_fill) * out[m, 0] + alpha_fill * r
        out[m, 1] = (1.0 - alpha_fill) * out[m, 1] + alpha_fill * g
        out[m, 2] = (1.0 - alpha_fill) * out[m, 2] + alpha_fill * b

        edge = ndimage.binary_dilation(m, iterations=1) ^ ndimage.binary_erosion(m, iterations=2)
        out[edge] = [r, g, b]

    # 3. Overshooting Top (Class 6): vivid magenta fill + white ring
    m_ot = sem_mask == 6
    if np.any(m_ot):
        ot_dil = ndimage.binary_dilation(m_ot, iterations=3)
        ot_ring = ndimage.binary_dilation(ot_dil, iterations=2) & (~ot_dil)
        out[ot_ring] = [255.0, 255.0, 255.0]
        out[ot_dil] = [245.0, 60.0, 255.0]

    # Restore AMSU black annotations ('x' convective markers, ellipse, coastlines) on top!
    out[is_dark_annotation] = amsu_box_rgb[is_dark_annotation].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_graticule_on_box(
    box_rgb: np.ndarray,
    lat_box: np.ndarray,
    lon_box: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    amsu_box_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Transfer the exact AMSU graticule and coastline/ellipse/'x' annotations onto Panel 2."""
    out = box_rgb.copy()
    if amsu_box_rgb is not None:
        gray_a = amsu_box_rgb.mean(axis=2)
        # Local darkness relative to immediate neighborhood (detects dotted gridlines cleanly)
        med_bg = ndimage.maximum_filter(gray_a, size=5)
        local_drop = med_bg - gray_a
        is_grid_dot = (local_drop > 28.0) & (gray_a >= 50.0)
        out[is_grid_dot] = (0.40 * out[is_grid_dot] + 0.60 * np.array([225, 235, 250])).astype(np.uint8)

        # Solid dark pixels (coastline + ellipse + 'x' convective markers)
        very_dark = gray_a < 55.0
        out[very_dark] = [0, 255, 255]
    return out


def render_comparison_for_snapshot(
    spec: AmsuSnapshotSpec,
    amsu_dir: Path,
    medicanes_res_dir: Path,
    zarr_root: Path,
    out_dir: Path,
    geom_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    img_path = amsu_dir / spec.filename
    cache_path = medicanes_res_dir / spec.event_folder / f"_cache_{spec.day_str}_pure.npz"
    zarr_path = zarr_root / spec.day_str[:4] / f"{spec.day_str}.zarr"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing AMSU image: {img_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing Pure RDT cache: {cache_path}")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing SEVIRI Zarr: {zarr_path}")

    # 1. Load SEVIRI geometry (cached by exact (H, W) shape of the Zarr store)
    g = zarr.open(str(zarr_path), mode="r")
    shape_key = f"{g['seviri'].shape[2]}x{g['seviri'].shape[3]}"
    if shape_key not in geom_cache:
        geom_cache[shape_key] = load_seviri_geometry(zarr_path)
    geom = geom_cache[shape_key]

    # 2. Find closest 5-minute SEVIRI RSS slot to overpass_time_utc
    target_ts = pd.Timestamp(spec.overpass_time_utc, tz="UTC")
    if "time" in g:
        raw_unix = np.asarray(g["time"][:], dtype=np.int64)
        chrono_idx = np.argsort(raw_unix)
        times_utc = pd.to_datetime(raw_unix[chrono_idx], unit="s", utc=True)
    else:
        chrono_idx = np.arange(len(g["seviri"]))
        times_utc = pd.date_range(spec.day_str, periods=len(g["seviri"]), freq="5min", tz="UTC")

    dt_sec = np.abs((times_utc - target_ts).total_seconds().to_numpy())
    step_idx = int(np.argmin(dt_sec))
    raw_idx = int(chrono_idx[step_idx])
    matched_ts = times_utc[step_idx]
    signed_dt_min = float((matched_ts - target_ts).total_seconds() / 60.0)

    # 3. Load Pure RDT mask (from _cache_{day}_pure.npz or _cache_{day}_extra_slots.npz) and SEVIRI IR 10.8 frame
    sem_pure_full: np.ndarray | None = None
    with np.load(cache_path) as d:
        sem_arr = d["semantic_pure"]
        c_idx: int | None = None
        if "timestamps_unix" in d:
            cache_ts = pd.to_datetime(np.asarray(d["timestamps_unix"], dtype=np.int64), unit="s", utc=True)
            dt_cache_sec = np.abs((cache_ts - matched_ts).total_seconds().to_numpy())
            if float(np.min(dt_cache_sec)) <= 60.0:
                c_idx = int(np.argmin(dt_cache_sec))
        elif step_idx < sem_arr.shape[0]:
            c_idx = step_idx

        if c_idx is not None:
            sem_pure_full = np.asarray(sem_arr[c_idx], dtype=np.uint8).copy()
            # Bridge 1-frame split/merge dropouts (Class 2..6 -> Class 1 -> Class 2..6) as in SmoothSemanticRenderer
            if 0 < c_idx < (sem_arr.shape[0] - 1):
                sem_prev = np.asarray(sem_arr[c_idx - 1], dtype=np.uint8)
                sem_next = np.asarray(sem_arr[c_idx + 1], dtype=np.uint8)
                one_frame_glitch = (sem_pure_full == 1) & (sem_prev >= 2) & (sem_next >= 2)
                if np.any(one_frame_glitch):
                    sem_pure_full[one_frame_glitch] = sem_prev[one_frame_glitch]

    if sem_pure_full is None:
        extra_cache_path = cache_path.with_name(f"_cache_{spec.day_str}_extra_slots.npz")
        extra_dict: dict[str, np.ndarray] = {}
        if extra_cache_path.exists():
            with np.load(extra_cache_path) as ed:
                extra_dict = {k: np.asarray(ed[k]) for k in ed.keys()}
        slot_key = f"slot_{step_idx:03d}"
        if slot_key in extra_dict:
            sem_pure_full = np.asarray(extra_dict[slot_key], dtype=np.uint8)
        else:
            print(f"   [Warmup RDT] Computing out-of-window slot {matched_ts:%Y-%m-%d %H:%M} UTC from Zarr...", flush=True)
            sem_pure_full = _compute_pure_rdt_slot_from_zarr(g, chrono_idx, step_idx, geom, warmup_frames=6)
            extra_dict[slot_key] = sem_pure_full
            np.savez_compressed(extra_cache_path, **extra_dict)

    raw_frame = np.asarray(g["seviri"][raw_idx], dtype=np.float32)
    f_curr, _, _ = _orient_north_up_west_left(raw_frame, geom["y_raw"], geom["x_raw"])

    cold_c = np.isfinite(f_curr[4]) & (f_curr[4] < 235.0)
    t_tr_c = (
        float(0.6 * np.nanpercentile(f_curr[0][cold_c], 10) + 0.4 * np.nanpercentile(f_curr[4][cold_c], 5))
        if int(np.count_nonzero(cold_c)) >= 50
        else 215.0
    )
    pure_tropo_c = np.full(f_curr[4].shape, float(np.clip(t_tr_c, 205.0, 226.0)), dtype=np.float32)
    ir108_ground = apply_parallax_correction_2d(
        f_curr[4], f_curr[4], geom["lat_deg"], geom["lon_deg"],
        t_surf_k=292.0, t_tropo_k=pure_tropo_c, fill_value=290.0
    )

    # 4. Calibrate AMSU map box & project every AMSU map pixel to SEVIRI (row, col)
    cal = calibrate_amsu_image_geometry(img_path, spec)
    x0, y0, x1, y1 = cal["box"]
    lat_box = cal["lat_box"]
    lon_box = cal["lon_box"]

    x_geos, y_geos = geom["from_ll"].transform(lon_box, lat_box)
    x_m = geom["x_m"]
    y_m = geom["y_m"]

    # x_m is ascending (west-left), y_m is descending (north-up)
    col_sev = (x_geos - float(x_m[0])) / float(x_m[1] - x_m[0])
    row_sev = (y_geos - float(y_m[0])) / float(y_m[1] - y_m[0])

    valid_proj = (
        np.isfinite(col_sev)
        & np.isfinite(row_sev)
        & (row_sev >= 0)
        & (row_sev <= len(y_m) - 1)
        & (col_sev >= 0)
        & (col_sev <= len(x_m) - 1)
    )

    coords_yx = np.array([
        np.where(valid_proj, row_sev, 0.0),
        np.where(valid_proj, col_sev, 0.0),
    ], dtype=np.float32)

    ir108_reproj = ndimage.map_coordinates(ir108_ground, coords_yx, order=1, mode="nearest")
    ir108_reproj = np.where(valid_proj, ir108_reproj, np.nan)

    row_nn = np.clip(np.round(row_sev).astype(int), 0, len(y_m) - 1)
    col_nn = np.clip(np.round(col_sev).astype(int), 0, len(x_m) - 1)
    sem_reproj = np.where(valid_proj, sem_pure_full[row_nn, col_nn], 0).astype(np.uint8)

    # 5. Construct the 3 synchronized 1200x1200 templates (keeping original axes & colorbar!)
    panel1_full = cal["im_rgb"].copy()
    amsu_box_rgb = panel1_full[y0:y1, x0:x1].copy()

    # Panel 2: Replace the map box [y0:y1, x0:x1] with reprojected SEVIRI IR 10.8 + Pure RDT
    # (removing the AMSU TB colorbar on the right of Panel 2 so it's clean)
    panel2_full = cal["im_rgb"].copy()
    panel2_full[:, x1 + 4 :] = 255  # Clear right-hand AMSU colorbar on Panel 2
    panel2_full[: y0 - 4, :] = 255  # Clear top title so we draw clean title
    ir_rgb_box = _ir108_to_rgb_base(ir108_reproj)
    rdt_ir_box = _render_rdt_on_ir(ir_rgb_box, sem_reproj)
    rdt_ir_box = _draw_graticule_on_box(
        rdt_ir_box, lat_box, lon_box, cal["lat_vals"], cal["lon_vals"], amsu_box_rgb=amsu_box_rgb
    )
    panel2_full[y0:y1, x0:x1] = rdt_ir_box
    # Re-draw 2px black border around box
    panel2_full[y0 - 2 : y0, x0 - 2 : x1 + 2] = 0
    panel2_full[y1 : y1 + 2, x0 - 2 : x1 + 2] = 0
    panel2_full[y0 - 2 : y1 + 2, x0 - 2 : x0] = 0
    panel2_full[y0 - 2 : y1 + 2, x1 : x1 + 2] = 0

    # Panel 3: Replace the map box [y0:y1, x0:x1] with the Direct Overlay on AMSU
    panel3_full = cal["im_rgb"].copy()
    panel3_full[: y0 - 4, : x1 + 12] = 255
    overlay_box = _render_rdt_overlay_on_amsu(amsu_box_rgb, sem_reproj)
    panel3_full[y0:y1, x0:x1] = overlay_box

    # Crop vertical margins tightly [135:1115] and horizontal margins [10:1190] (or [10:1030] for Panel 2)
    crop_y0, crop_y1 = 135, 1115
    p1_crop = panel1_full[crop_y0:crop_y1, 10:1185]
    p2_crop = panel2_full[crop_y0:crop_y1, 10:1025]
    p3_crop = panel3_full[crop_y0:crop_y1, 10:1185]

    # Compute convective diagnostics inside the AMSU geographic box
    conv_mask = (sem_reproj >= 2) & (sem_reproj <= 6)
    ot_mask = sem_reproj == 6
    cold_mask = sem_reproj == 1
    conv_frac_pct = 100.0 * float(np.count_nonzero(conv_mask)) / float(max(sem_reproj.size, 1))
    ot_px_count = int(np.count_nonzero(ot_mask))
    ir_min_k = float(np.nanmin(ir108_reproj)) if np.any(np.isfinite(ir108_reproj)) else float("nan")

    # Detect cool/depressed PMW TB pixels in AMSU map box (green/cyan/blue pixels where G > R - 15 and B > 110 or R < 210)
    r_a = amsu_box_rgb[:, :, 0].astype(np.float32)
    g_a = amsu_box_rgb[:, :, 1].astype(np.float32)
    b_a = amsu_box_rgb[:, :, 2].astype(np.float32)
    not_dark = (r_a + g_a + b_a) > 220.0
    pmw_cold_depression = not_dark & (g_a > r_a - 25.0) & (r_a < 225.0)
    # Distance from any RDT convective pixel (in AMSU box pixels; 1 deg ~ 145 px -> 25 km ~ 32 px)
    if np.any(conv_mask) and np.any(pmw_cold_depression):
        dist_px = ndimage.distance_transform_edt(~conv_mask)
        pmw_overlap_25km_pct = 100.0 * float(np.mean(dist_px[pmw_cold_depression] <= 32.0))
    else:
        pmw_overlap_25km_pct = 0.0

    # Assemble 3-panel canvas
    pad = 16
    header_h = 84
    footer_h = 58
    total_w = pad * 4 + p1_crop.shape[1] + p2_crop.shape[1] + p3_crop.shape[1]
    total_h = header_h + p1_crop.shape[0] + footer_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    try:
        font_h1 = ImageFont.truetype("arialbd.ttf", 24)
        font_h2 = ImageFont.truetype("arial.ttf", 17)
        font_p_title = ImageFont.truetype("arialbd.ttf", 20)
        font_leg = ImageFont.truetype("arialbd.ttf", 15)
    except Exception:
        font_h1 = font_h2 = font_p_title = font_leg = ImageFont.load_default()

    # Header
    title_main = (
        f"MEDICANE {spec.medicane.upper()} — Confronto Georeferenziato AMSU/MHS PMW vs SEVIRI RDT Puro"
    )
    subtitle = (
        f"Passaggio PMW (titolo): {spec.overpass_time_utc} UTC (file: {spec.file_time_utc} UTC)  |  "
        f"Slot SEVIRI RSS + RDT Puro: {matched_ts:%Y-%m-%d %H:%M} UTC (Δt = {signed_dt_min:+.0f} min)  |  "
        f"Copertura convettiva RDT nel box: {conv_frac_pct:.2f}% (IR10.8 min: {ir_min_k:.1f} K, OT px: {ot_px_count})"
    )
    draw.text((pad + 4, 14), title_main, fill=(248, 250, 252), font=font_h1)
    draw.text((pad + 4, 48), subtitle, fill=(186, 214, 248), font=font_h2)

    # Paste the 3 panels and draw panel titles inside their white top bands
    x_cur = pad
    panel_Configs = [
        (p1_crop, f"1. Originale AMSU/MHS ({spec.overpass_time_utc} UTC)"),
        (p2_crop, f"2. RDT Puro su SEVIRI IR 10.8 ({matched_ts:%H:%M} UTC) + target PMW (ciano)"),
        (p3_crop, f"3. Sovrapposizione Geografica: RDT Puro su AMSU/MHS"),
    ]
    for p_arr, p_label in panel_Configs:
        p_pil = Image.fromarray(p_arr)
        p_draw = ImageDraw.Draw(p_pil)
        if p_arr is not p1_crop:
            p_draw.text((115, 22), p_label, fill=(15, 23, 42), font=font_p_title)
        canvas.paste(p_pil, (x_cur, header_h))
        draw.rectangle(
            [x_cur - 1, header_h - 1, x_cur + p_arr.shape[1], header_h + p_arr.shape[0]],
            outline=(71, 85, 105),
            width=2,
        )
        x_cur += p_arr.shape[1] + pad

    # Footer legend
    fy = header_h + p1_crop.shape[0] + 18
    lx = pad + 8
    legend_items = [
        ((59, 130, 246), "Nube fredda non convettiva"),
        ((250, 204, 21), "Triggering (Innesco)"),
        ((249, 115, 22), "Growing (Sviluppo)"),
        ((220, 38, 38), "Mature (Maturità)"),
        ((153, 27, 27), "Decaying (Decadimento)"),
        ((245, 105, 255), "Overshooting Top (OT)"),
        ((0, 255, 255), "Annotazioni PMW ('x', ellisse, coste su Pannello 2)"),
    ]
    for color, label_txt in legend_items:
        draw.rectangle([lx, fy, lx + 22, fy + 20], fill=color, outline=(255, 255, 255))
        draw.text((lx + 28, fy + 1), label_txt, fill=(235, 240, 248), font=font_leg)
        lx += 34 + int(len(label_txt) * 8.2) + 22

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{Path(spec.filename).stem}_rdt_pure.png"
    canvas.save(out_path, optimize=True)

    return {
        "filename": spec.filename,
        "medicane": spec.medicane,
        "day": spec.day_str,
        "file_time_utc": spec.file_time_utc,
        "overpass_time_utc": spec.overpass_time_utc,
        "seviri_slot_utc": matched_ts.strftime("%Y-%m-%d %H:%M"),
        "dt_minutes": signed_dt_min,
        "convective_area_pct": round(conv_frac_pct, 3),
        "cold_cloud_area_pct": round(100.0 * float(np.count_nonzero(cold_mask)) / float(sem_reproj.size), 3),
        "ot_pixels": ot_px_count,
        "ir108_min_k": round(ir_min_k, 2),
        "pmw_cold_overlap_25km_pct": round(pmw_overlap_25km_pct, 1),
        "output_png": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amsu-dir",
        type=Path,
        default=Path(r"E:\Datasets\Medicanes-AMSU-MHS"),
    )
    parser.add_argument(
        "--medicanes-res-dir",
        type=Path,
        default=RDT_ORIG_ROOT / "results" / "medicanes",
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(r"E:\Datasets\MSG-SEVIRI"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RDT_ORIG_ROOT / "results" / "medicanes_amsu_mhs_comparison",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional substring filter on filename or medicane name",
    )
    args = parser.parse_args()

    geom_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    specs = [
        s
        for s in AMSU_SNAPSHOTS
        if args.filter is None
        or args.filter.lower() in s.filename.lower()
        or args.filter.lower() in s.medicane.lower()
    ]
    print(f"Rendering {len(specs)} AMSU/MHS vs Pure RDT geo-aligned comparisons...")
    for idx, spec in enumerate(specs, start=1):
        rec = render_comparison_for_snapshot(
            spec,
            args.amsu_dir,
            args.medicanes_res_dir,
            args.zarr_root,
            args.out_dir,
            geom_cache,
        )
        records.append(rec)
        print(
            f"[{idx:02d}/{len(specs):02d}] {spec.filename:36s} -> {rec['seviri_slot_utc']} UTC "
            f"(dt={rec['dt_minutes']:+.0f}m) | conv={rec['convective_area_pct']:5.2f}% | "
            f"PMW<=25km={rec['pmw_cold_overlap_25km_pct']:5.1f}%",
            flush=True,
        )

    if records and args.filter is None:
        df = pd.DataFrame(records)
        csv_path = args.out_dir / "amsu_mhs_rdt_pure_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary table to {csv_path}")


if __name__ == "__main__":
    main()
