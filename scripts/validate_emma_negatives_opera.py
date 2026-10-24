#!/usr/bin/env python3
"""Cross-check EMMA-negative SEVIRI crops with OPERA DBZH and QIND.

The output classes are deliberately provisional:

* N0_candidate: warm/easy background with good radar coverage and no echo;
* N1_candidate: cold cloud with good radar coverage but no convective echo;
* N2_candidate: no EMMA MCS but a spatially collocated convective radar echo;
* NU: missing/poor coverage or an intermediate, ambiguous signal.

Historical openradar files bundle DBZH and QIND in ``DBZH_QIND.h5``.  Metrics
are computed only on the largest IR10.8 < 241 K component in each crop and on
five OPERA slots from -30 to +30 minutes around the SEVIRI image.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_emma_heatmap_examples import load_ir108, utc_naive  # noqa: E402
from emma_gpm.seviri import RSS_CRS  # noqa: E402


CROP = 224
COLD_THRESHOLD_K = 241.0
OPERA_SHAPE = (2200, 1900)
OPERA_PIXEL_M = 2000.0
OPERA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 "
    "+datum=WGS84 +units=m +no_defs"
)
OPERA_ENDPOINT = "https://s3.waw3-1.cloudferro.com/openradar-archive"
OFFSETS_MINUTES = (-30, -15, 0, 15, 30)


def opera_key(timestamp: pd.Timestamp) -> str:
    return (
        f"{timestamp:%Y/%m/%d}/OPERA/COMP/"
        f"OPERA@{timestamp:%Y%m%dT%H%M}@0@DBZH_QIND.h5"
    )


def download_opera(timestamp: pd.Timestamp, cache_root: Path) -> Path:
    key = opera_key(timestamp)
    path = cache_root / key
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    url = f"{OPERA_ENDPOINT}/{key}"
    try:
        with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(part, path)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return path


def read_quantity(handle: h5py.File, quantity: str) -> np.ndarray:
    for name in handle:
        if not name.startswith("dataset"):
            continue
        group = handle[name]
        if "what" not in group or "data1" not in group:
            continue
        value = group["what"].attrs.get("quantity", b"")
        if isinstance(value, bytes):
            value = value.decode("ascii")
        if value == quantity:
            data = np.asarray(group["data1/data"], dtype=np.float32)
            if data.shape != OPERA_SHAPE:
                raise ValueError(f"Unexpected OPERA shape {data.shape}")
            nodata = float(group["what"].attrs.get("nodata", -9999000.0))
            undetect = float(group["what"].attrs.get("undetect", -8888000.0))
            data[data == nodata] = np.nan
            if quantity == "DBZH":
                # ODIM distinguishes pixels outside/missing from pixels where
                # the radar made a valid observation but detected no echo.
                # Preserve the latter as a finite floor value.
                data[data == undetect] = -32.0
            else:
                data[data == undetect] = np.nan
            return data
    raise KeyError(f"Quantity {quantity} not found")


def read_opera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        dbzh = read_quantity(handle, "DBZH")
        qind = read_quantity(handle, "QIND")
    return dbzh, qind


def largest_cold_component(image: np.ndarray) -> np.ndarray:
    cold = np.isfinite(image) & (image < COLD_THRESHOLD_K)
    components, count = label(cold, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(cold)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def opera_indices_for_crop(
    x: np.ndarray,
    y: np.ndarray,
    origin_x: int,
    origin_y: int,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop_x = np.asarray(x[origin_x : origin_x + CROP])
    crop_y = np.asarray(y[origin_y : origin_y + CROP])
    grid_x, grid_y = np.meshgrid(crop_x, crop_y)
    opera_x, opera_y = transformer.transform(grid_x, grid_y)
    finite = np.isfinite(opera_x) & np.isfinite(opera_y)
    col = np.full(opera_x.shape, -1, dtype=np.int32)
    row = np.full(opera_y.shape, -1, dtype=np.int32)
    col[finite] = np.floor(opera_x[finite] / OPERA_PIXEL_M).astype(np.int32)
    row[finite] = np.floor(-opera_y[finite] / OPERA_PIXEL_M).astype(np.int32)
    inside = (
        finite
        & (row >= 0)
        & (row < OPERA_SHAPE[0])
        & (col >= 0)
        & (col < OPERA_SHAPE[1])
    )
    return row, col, inside


def mapped_field(
    field: np.ndarray, row: np.ndarray, col: np.ndarray, inside: np.ndarray
) -> np.ndarray:
    result = np.full(row.shape, np.nan, dtype=np.float32)
    result[inside] = field[row[inside], col[inside]]
    return result


def component_metrics(
    component: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    inside: np.ndarray,
    opera_frames: list[tuple[np.ndarray, np.ndarray]],
    qind_threshold: float,
) -> dict[str, float | int]:
    component_size = int(component.sum())
    if component_size == 0:
        component = np.ones_like(component, dtype=bool)
        component_size = int(component.sum())
    mapped_component = component & inside
    inside_fraction = float(mapped_component.sum() / component_size)
    slot_rows = []
    for dbzh, qind in opera_frames:
        q = mapped_field(qind, row, col, inside)
        z = mapped_field(dbzh, row, col, inside)
        # DBZH nodata means no radar observation.  DBZH=-32 is ODIM
        # ``undetect``: a valid no-echo observation for which QIND is absent.
        # QIND is populated only where an echo exists in these historical
        # OPERA files, so requiring finite QIND would incorrectly reject true
        # quiet pixels as missing coverage.
        valid = mapped_component & np.isfinite(z)
        no_echo = valid & np.isclose(z, -32.0)
        quality_echo = valid & np.isfinite(q) & (q >= qind_threshold)
        good = no_echo | quality_echo
        good_count = int(good.sum())
        if good_count:
            max_dbzh = float(np.nanmax(z[good]))
            q_values = q[quality_echo]
            mean_qind = float(np.nanmean(q_values)) if len(q_values) else np.nan
        else:
            max_dbzh = np.nan
            mean_qind = np.nan
        # Count unique radar pixels, otherwise the 3-km SEVIRI grid would
        # duplicate 2-km OPERA pixels and inflate radar area.
        unique_pixels = np.unique(
            np.column_stack((row[good], col[good])), axis=0
        ) if good_count else np.empty((0, 2), dtype=np.int32)
        if len(unique_pixels):
            values = dbzh[unique_pixels[:, 0], unique_pixels[:, 1]]
            area35 = float(np.count_nonzero(values >= 35.0) * 4.0)
            area40 = float(np.count_nonzero(values >= 40.0) * 4.0)
        else:
            area35 = 0.0
            area40 = 0.0
        slot_rows.append(
            {
                "valid_fraction": float(valid.sum() / component_size),
                "good_fraction": float(good_count / component_size),
                "mean_qind": mean_qind,
                "max_dbzh": max_dbzh,
                "area35_km2": area35,
                "area40_km2": area40,
            }
        )
    frame = pd.DataFrame(slot_rows)
    finite_mean_qind = frame.mean_qind[np.isfinite(frame.mean_qind)]
    usable_fraction_median = float(frame.good_fraction.median())
    return {
        "cold_component_pixels": component_size,
        "opera_inside_fraction": inside_fraction,
        "opera_valid_fraction_median": float(frame.valid_fraction.median()),
        # ``good`` is kept as a compatibility alias for earlier catalogues;
        # after the ODIM fix this is the usable fraction: valid no-echo pixels
        # plus echo pixels whose QIND passes the threshold.
        "opera_good_fraction_median": usable_fraction_median,
        "opera_usable_fraction_median": usable_fraction_median,
        "opera_mean_qind_median": (
            float(np.median(finite_mean_qind)) if len(finite_mean_qind) else np.nan
        ),
        "opera_max_dbzh_5slots": float(frame.max_dbzh.max(skipna=True)),
        "opera_max_area35_km2_5slots": float(frame.area35_km2.max()),
        "opera_max_area40_km2_5slots": float(frame.area40_km2.max()),
        "opera_slots_area35_ge_16km2": int((frame.area35_km2 >= 16.0).sum()),
    }


def provisional_class(row: pd.Series) -> str:
    enough_coverage = (
        row.opera_inside_fraction >= 0.80
        and row.opera_valid_fraction_median >= 0.70
        # QIND is continuous (0 poorest, 1 best) and historical ODYSSEY
        # composites often have only a minority of pixels above 0.2.  Requiring
        # half of the component above 0.2 would reject otherwise well-covered
        # scenes.  Keep the valid-coverage requirement strict and only require
        # a non-trivial high-quality subset for the reflectivity test.
        and row.opera_good_fraction_median >= 0.05
    )
    if not enough_coverage:
        return "NU"
    radar_convective = (
        row.opera_max_area40_km2_5slots >= 16.0
        and row.opera_slots_area35_ge_16km2 >= 2
    )
    if radar_convective:
        return "N2_candidate"
    radar_quiet = row.opera_max_dbzh_5slots < 30.0
    if row.cold_fraction_lt241k < 0.02 and radar_quiet:
        return "N0_candidate"
    if row.cold_fraction_lt241k >= 0.02 and radar_quiet:
        return "N1_candidate"
    return "NU"


def load_manifests(cold_path: Path, background_path: Path) -> pd.DataFrame:
    cold = pd.read_csv(cold_path)
    cold["candidate_source"] = "cold_cloud_no_emma"
    background = pd.read_csv(background_path)
    background["candidate_source"] = background["kind"].astype(str)
    columns = [
        "target_time",
        "seviri_time",
        "origin_y",
        "origin_x",
        "candidate_source",
    ]
    combined = pd.concat([cold[columns], background[columns]], ignore_index=True)
    combined["target_time"] = pd.to_datetime(combined.target_time, utc=True).dt.tz_localize(None)
    combined["seviri_time"] = pd.to_datetime(combined.seviri_time, utc=True).dt.tz_localize(None)
    return combined.drop_duplicates(["seviri_time", "origin_y", "origin_x"]).sort_values(
        ["seviri_time", "candidate_source", "origin_y", "origin_x"]
    ).reset_index(drop=True)


def select_overview(frame: pd.DataFrame, count: int = 30) -> pd.DataFrame:
    quotas = {"N0_candidate": 8, "N1_candidate": 8, "N2_candidate": 8, "NU": 6}
    selections = []
    sort_columns = {
        "N0_candidate": ["cold_fraction_lt241k", "opera_max_dbzh_5slots"],
        "N1_candidate": ["cold_fraction_lt241k", "opera_good_fraction_median"],
        "N2_candidate": ["opera_max_area40_km2_5slots", "cold_fraction_lt241k"],
        "NU": ["cold_fraction_lt241k", "opera_good_fraction_median"],
    }
    ascending = {
        "N0_candidate": [True, True],
        "N1_candidate": [False, False],
        "N2_candidate": [False, False],
        "NU": [False, False],
    }
    used = set()
    for category, quota in quotas.items():
        subset = frame[frame.provisional_class == category].sort_values(
            sort_columns[category], ascending=ascending[category]
        )
        # Prefer different times before taking multiple spatial candidates from
        # the same scene.
        first = subset.drop_duplicates("seviri_time").head(quota)
        if len(first) < quota:
            first = pd.concat(
                [first, subset.loc[~subset.index.isin(first.index)].head(quota - len(first))]
            )
        selections.append(first)
        used.update(first.index)
    result = pd.concat(selections) if selections else frame.iloc[:0]
    if len(result) < count:
        remainder = frame.loc[~frame.index.isin(used)].copy()
        # When a requested class is absent, spend the unused slots on
        # radar-confirmed N2 candidates before filling with unknowns.  This
        # makes the audit sheet expose label conflict instead of being
        # dominated by out-of-coverage examples.
        remainder["_class_priority"] = remainder.provisional_class.map(
            {"N1_candidate": 0, "N2_candidate": 1, "N0_candidate": 2, "NU": 3}
        ).fillna(4)
        remainder = remainder.sort_values(
            ["_class_priority", "opera_max_area40_km2_5slots", "cold_fraction_lt241k"],
            ascending=[True, False, False],
        )
        result = pd.concat([result, remainder.head(count - len(result))])
    return result.drop(columns=["_class_priority"], errors="ignore").head(count).reset_index(drop=True)


def plot_overview(
    selected: pd.DataFrame,
    seviri_root: Path,
    opera_cache: Path,
    out: Path,
    qind_threshold: float,
) -> None:
    colors = {
        "N0_candidate": "#16a34a",
        "N1_candidate": "#2563eb",
        "N2_candidate": "#dc2626",
        "NU": "#a855f7",
    }
    figure, axes = plt.subplots(6, 5, figsize=(17, 20), constrained_layout=True)
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    for axis, item in zip(axes.flat, selected.itertuples(index=False)):
        timestamp = utc_naive(item.seviri_time)
        image, x, y = load_ir108(seviri_root, timestamp, datasets)
        y0, x0 = int(item.origin_y), int(item.origin_x)
        crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
        row, col, inside = opera_indices_for_crop(x, y, x0, y0, transformer)
        dbzh, qind = read_opera(download_opera(timestamp, opera_cache))
        z = mapped_field(dbzh, row, col, inside)
        q = mapped_field(qind, row, col, inside)
        axis.imshow(crop, cmap="gray_r", vmin=205, vmax=295)
        good_z = np.where(q >= qind_threshold, z, np.nan)
        finite = good_z[np.isfinite(good_z)]
        if finite.size and np.nanmax(finite) >= 20:
            levels = [level for level in (20.0, 35.0, 45.0) if np.nanmin(finite) <= level <= np.nanmax(finite)]
            if levels:
                contour = axis.contour(
                    good_z,
                    levels=levels,
                    colors=[{20.0: "#06b6d4", 35.0: "#f59e0b", 45.0: "#ef4444"}[v] for v in levels],
                    linewidths=0.8,
                )
                axis.clabel(contour, inline=True, fontsize=5, fmt="%d")
        component = largest_cold_component(crop)
        if component.any():
            axis.contour(component, levels=[0.5], colors="white", linewidths=0.65)
        category = item.provisional_class
        axis.set_title(
            f"{category} · {timestamp:%m-%d %H:%M}Z\n"
            f"max {item.opera_max_dbzh_5slots:.0f} dBZ · A40 {item.opera_max_area40_km2_5slots:.0f} km² · Q {item.opera_good_fraction_median:.2f}",
            fontsize=7.5,
            color=colors[category],
            weight="bold",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(colors[category])
            spine.set_linewidth(2.0)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    for dataset in datasets.values():
        dataset.close()
    figure.suptitle(
        "EMMA-negative candidates cross-checked with OPERA\n"
        f"IR10.8 grayscale · largest <241 K component in white · DBZH contours 20/35/45 dBZ · QIND ≥ {qind_threshold:.1f}",
        fontsize=14,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-manifest", type=Path, required=True)
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--opera-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qind-threshold", type=float, default=0.2)
    parser.add_argument("--overview-count", type=int, default=30)
    args = parser.parse_args()

    candidates = load_manifests(args.cold_manifest, args.background_manifest)
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    datasets = {}
    rows = []
    grouped = candidates.groupby("seviri_time", sort=True)
    for group_number, (time_value, group) in enumerate(grouped, start=1):
        timestamp = utc_naive(time_value)
        image, x, y = load_ir108(args.seviri_root, timestamp, datasets)
        opera_frames = []
        for offset in OFFSETS_MINUTES:
            radar_time = pd.Timestamp(timestamp) + pd.Timedelta(int(offset), unit="min")
            opera_frames.append(read_opera(download_opera(radar_time, args.opera_cache)))
        for item in group.itertuples(index=False):
            y0, x0 = int(item.origin_y), int(item.origin_x)
            crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
            component = largest_cold_component(crop)
            row, col, inside = opera_indices_for_crop(x, y, x0, y0, transformer)
            metrics = component_metrics(
                component, row, col, inside, opera_frames, args.qind_threshold
            )
            centre_opera_x, centre_opera_y = transformer.transform(
                float(x[x0 + CROP // 2]), float(y[y0 + CROP // 2])
            )
            centre_lon, centre_lat = Transformer.from_crs(
                OPERA_CRS, "EPSG:4326", always_xy=True
            ).transform(centre_opera_x, centre_opera_y)
            rows.append(
                {
                    **item._asdict(),
                    "centre_lon": float(centre_lon),
                    "centre_lat": float(centre_lat),
                    "ir108_min_k": float(np.nanmin(crop)),
                    "ir108_p10_k": float(np.nanquantile(crop, 0.10)),
                    "cold_fraction_lt241k": float(np.nanmean(crop < COLD_THRESHOLD_K)),
                    **metrics,
                }
            )
        print(
            f"[{group_number:02d}/{grouped.ngroups:02d}] {timestamp} "
            f"candidates={len(group)}"
        )
    for dataset in datasets.values():
        dataset.close()

    result = pd.DataFrame(rows)
    result["provisional_class"] = result.apply(provisional_class, axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "opera_negative_validation_all.csv", index=False)
    selected = select_overview(result, args.overview_count)
    selected.to_csv(args.out_dir / "opera_negative_overview_30.csv", index=False)
    plot_overview(
        selected,
        args.seviri_root,
        args.opera_cache,
        args.out_dir / "opera_negative_overview_30.png",
        args.qind_threshold,
    )
    print(result.provisional_class.value_counts(dropna=False).to_string())
    print(f"all={len(result)} selected={len(selected)} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
