#!/usr/bin/env python3
"""Track cold cloud objects and compare temporal/multispectral diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy import ndimage
from scipy.stats import theilslopes
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from plot_emma_heatmap_examples import centered_crop, load_case_mask, utc_naive  # noqa: E402
from emma_gpm.seviri import RSS_CRS  # noqa: E402


CROP = 224
TRACK_THRESHOLD_K = 241.0
GEOD = Geod(ellps="WGS84")


def open_day(root: Path, timestamp: pd.Timestamp) -> xr.Dataset:
    return xr.open_zarr(
        root / f"{timestamp:%Y}" / f"{timestamp:%Y%m%d}.zarr", consolidated=True
    ).sortby("time")


def components(field: np.ndarray, threshold: float = TRACK_THRESHOLD_K):
    valid = np.isfinite(field)
    labels, count = ndimage.label(valid & (field < threshold), structure=np.ones((3, 3)))
    masks = []
    for label_id in range(1, count + 1):
        mask = labels == label_id
        if mask.sum() >= 6:
            masks.append(mask)
    return masks


def centroid(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    return points.mean(axis=0) if len(points) else np.array([np.nan, np.nan])


def select_reference_component(
    field: np.ndarray, reference_mask: np.ndarray | None
) -> np.ndarray:
    candidates = components(field)
    if not candidates:
        raise RuntimeError("No cold-cloud component at the reference frame")
    if reference_mask is not None:
        return max(
            candidates,
            key=lambda mask: (
                np.count_nonzero(mask & reference_mask),
                -np.nanpercentile(field[mask], 10),
            ),
        )
    centre = np.array([(field.shape[0] - 1) / 2, (field.shape[1] - 1) / 2])
    return max(
        candidates,
        key=lambda mask: (
            -np.linalg.norm(centroid(mask) - centre),
            -np.nanpercentile(field[mask], 10),
        ),
    )


def match_component(field: np.ndarray, previous: np.ndarray) -> np.ndarray:
    candidates = components(field)
    if not candidates:
        return np.zeros_like(previous)
    expanded = ndimage.binary_dilation(previous, iterations=10)
    previous_centroid = centroid(previous)

    def score(mask: np.ndarray) -> float:
        overlap = np.count_nonzero(mask & expanded) / max(1, min(mask.sum(), expanded.sum()))
        distance = np.linalg.norm(centroid(mask) - previous_centroid)
        coldness = max(0.0, TRACK_THRESHOLD_K - np.nanpercentile(field[mask], 10))
        return 4.0 * overlap - 0.025 * distance + 0.01 * coldness

    best = max(candidates, key=score)
    return best


def track_object(
    ir108: np.ndarray, reference_index: int, reference_mask: np.ndarray | None
) -> list[np.ndarray]:
    tracked = [np.zeros(ir108.shape[1:], dtype=bool) for _ in range(len(ir108))]
    tracked[reference_index] = select_reference_component(
        ir108[reference_index], reference_mask
    )
    for index in range(reference_index + 1, len(ir108)):
        tracked[index] = match_component(ir108[index], tracked[index - 1])
    for index in range(reference_index - 1, -1, -1):
        tracked[index] = match_component(ir108[index], tracked[index + 1])
    return tracked


def pixel_area_km2(x: np.ndarray, y: np.ndarray, centre_y: int, centre_x: int) -> float:
    to_geo = Transformer.from_crs(RSS_CRS, "EPSG:4326", always_xy=True)
    lon0, lat0 = to_geo.transform(float(x[centre_x]), float(y[centre_y]))
    lonx, latx = to_geo.transform(float(x[centre_x + 1]), float(y[centre_y]))
    lony, laty = to_geo.transform(float(x[centre_x]), float(y[centre_y + 1]))
    _, _, dx = GEOD.inv(lon0, lat0, lonx, latx)
    _, _, dy = GEOD.inv(lon0, lat0, lony, laty)
    return abs(dx * dy) / 1_000_000.0


def shape_features(mask: np.ndarray) -> dict[str, float]:
    area = float(mask.sum())
    if area == 0:
        return {"perimeter_px": np.nan, "compactness": np.nan, "eccentricity": np.nan}
    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.count_nonzero(mask & ~eroded))
    points = np.argwhere(mask)
    if len(points) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[-1], 1e-9))))
    else:
        eccentricity = np.nan
    return {
        "perimeter_px": perimeter,
        "compactness": float(4 * np.pi * area / max(perimeter**2, 1.0)),
        "eccentricity": eccentricity,
    }


def ot_like_features(ir108: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    # Bedka-style proxy: a very cold local minimum substantially colder than
    # its surrounding anvil. This is diagnostic, not an operational OT label.
    inner_mean = ndimage.uniform_filter(ir108, size=5, mode="nearest")
    outer_mean = ndimage.uniform_filter(ir108, size=17, mode="nearest")
    ring_mean = (17**2 * outer_mean - 5**2 * inner_mean) / (17**2 - 5**2)
    anomaly = ring_mean - ir108
    local_minimum = ir108 <= ndimage.minimum_filter(ir108, size=5, mode="nearest")
    candidates = mask & local_minimum & (ir108 <= 225.0) & (anomaly >= 6.5)
    return int(candidates.sum()), float(np.nanmax(np.where(mask, anomaly, np.nan)))


def extract_features(
    case: str,
    cube: np.ndarray,
    times: pd.DatetimeIndex,
    channels: list[str],
    masks: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    origin_y: int,
    origin_x: int,
    reference_time: pd.Timestamp,
) -> pd.DataFrame:
    channel = {name: index for index, name in enumerate(channels)}
    rows = []
    area_per_pixel = pixel_area_km2(x, y, origin_y + CROP // 2, origin_x + CROP // 2)
    to_geo = Transformer.from_crs(RSS_CRS, "EPSG:4326", always_xy=True)
    for frame_index, (timestamp, mask) in enumerate(zip(times, masks)):
        ir = cube[frame_index, channel["IR_108"]]
        values = ir[mask]
        cy, cx = centroid(mask)
        lon, lat = to_geo.transform(
            float(x[origin_x] + cx * (x[origin_x + 1] - x[origin_x])),
            float(y[origin_y] + cy * (y[origin_y + 1] - y[origin_y])),
        )
        gy, gx = np.gradient(ir)
        gradient = np.hypot(gx, gy)
        laplacian = ndimage.laplace(ir)
        ot_count, ot_anomaly = ot_like_features(ir, mask)
        shape = shape_features(mask)
        outer = ndimage.binary_dilation(mask, iterations=25)
        inner = ndimage.binary_dilation(mask, iterations=8)
        annulus = outer & ~inner & np.isfinite(ir)
        cold_labels, cold_count = ndimage.label(ir < TRACK_THRESHOLD_K)
        nearby_components = 0
        for label_id in range(1, cold_count + 1):
            component = cold_labels == label_id
            if component.sum() >= 6 and np.any(component & outer):
                nearby_components += 1
        row = {
            "case": case,
            "time": timestamp,
            "minutes_from_reference": (timestamp - reference_time).total_seconds() / 60,
            "object_pixels_lt241k": int(mask.sum()),
            "object_area_km2_approx": float(mask.sum() * area_per_pixel),
            "centroid_y": cy,
            "centroid_x": cx,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "ir108_min_k": float(np.nanmin(values)),
            "ir108_p05_k": float(np.nanpercentile(values, 5)),
            "ir108_p10_k": float(np.nanpercentile(values, 10)),
            "ir108_p25_k": float(np.nanpercentile(values, 25)),
            "ir108_median_k": float(np.nanmedian(values)),
            "ir108_iqr_k": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
            "object_annulus_contrast_k": float(
                np.nanmedian(ir[annulus]) - np.nanmedian(values)
            ) if annulus.any() else np.nan,
            "core_area_lt241k_km2": float(np.count_nonzero(mask & (ir < 241)) * area_per_pixel),
            "core_area_lt235k_km2": float(np.count_nonzero(mask & (ir < 235)) * area_per_pixel),
            "core_area_lt225k_km2": float(np.count_nonzero(mask & (ir < 225)) * area_per_pixel),
            "core_fraction_lt235k": float(np.mean(values < 235)),
            "core_fraction_lt225k": float(np.mean(values < 225)),
            "nearby_cold_component_count": nearby_components,
            "gradient_p90_k_per_pixel": float(np.nanpercentile(gradient[mask], 90)),
            "laplacian_std_k": float(np.nanstd(laplacian[mask])),
            "ot_like_candidate_pixels": ot_count,
            "max_local_cold_anomaly_k": ot_anomaly,
            **shape,
        }
        pairs = {
            "btd_wv062_wv073_k": ("WV_062", "WV_073"),
            "btd_wv062_ir108_k": ("WV_062", "IR_108"),
            "btd_ir108_ir087_k": ("IR_108", "IR_087"),
            "btd_ir120_ir108_k": ("IR_120", "IR_108"),
            "btd_ir134_ir108_k": ("IR_134", "IR_108"),
        }
        for name, (left, right) in pairs.items():
            difference = cube[frame_index, channel[left]] - cube[frame_index, channel[right]]
            row[name] = float(np.nanmedian(difference[mask]))
            row[name.replace("_k", "_p10_k")] = float(np.nanpercentile(difference[mask], 10))
            row[name.replace("_k", "_p90_k")] = float(np.nanpercentile(difference[mask], 90))
        wv_ir = cube[frame_index, channel["WV_062"]] - ir
        row["btd_wv062_ir108_positive_fraction"] = float(np.mean(wv_ir[mask] > 0))
        rows.append(row)

    result = pd.DataFrame(rows)
    result["speed_kmh"] = np.nan
    result["motion_compensated_cooling_k_per_5min"] = np.nan
    result["area_growth_km2_per_5min"] = result.object_area_km2_approx.diff()
    trend_columns = [
        "ir108_p10_k",
        "ir108_p25_k",
        "object_area_km2_approx",
        "core_area_lt235k_km2",
        "btd_wv062_wv073_k",
        "btd_wv062_ir108_k",
        "btd_ir108_ir087_k",
        "btd_ir120_ir108_k",
        "btd_ir134_ir108_k",
    ]
    for column in trend_columns:
        for frames, minutes in ((3, 15), (6, 30), (12, 60)):
            result[f"{column}_change_{minutes}min"] = result[column] - result[column].shift(frames)
    for index in range(1, len(result)):
        _, _, distance_m = GEOD.inv(
            result.loc[index - 1, "centroid_lon"], result.loc[index - 1, "centroid_lat"],
            result.loc[index, "centroid_lon"], result.loc[index, "centroid_lat"],
        )
        result.loc[index, "speed_kmh"] = distance_m / 1000 / (5 / 60)
        previous_centre = centroid(masks[index - 1])
        current_centre = centroid(masks[index])
        delta = current_centre - previous_centre
        previous_ir = ndimage.shift(
            cube[index - 1, channel["IR_108"]], shift=delta, order=1, mode="nearest"
        )
        previous_mask = ndimage.shift(
            masks[index - 1].astype(float), shift=delta, order=0, mode="constant"
        ) > 0.5
        overlap = masks[index] & previous_mask
        if overlap.sum() >= 6:
            current_ir = cube[index, channel["IR_108"]]
            result.loc[index, "motion_compensated_cooling_k_per_5min"] = float(
                np.nanmedian(current_ir[overlap] - previous_ir[overlap])
            )
    return result


def robust_slope(group: pd.DataFrame, column: str) -> float:
    valid = group[["minutes_from_reference", column]].dropna()
    if len(valid) < 3:
        return np.nan
    return float(theilslopes(valid[column], valid.minutes_from_reference).slope * 15)


def plot_comparison(features: pd.DataFrame, output: Path) -> None:
    colors = {"positive_emma": "#16a34a", "no_emma_ambiguous": "#dc2626"}
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 12), constrained_layout=True)
    panels = [
        ("ir108_p10_k", "IR10.8 p10 oggetto", "K"),
        ("object_area_km2_approx", "Core tracciato IR10.8 < 241 K", "km2 circa"),
        ("core_area_lt235k_km2", "Core IR10.8 < 235 K", "km2 circa"),
        ("motion_compensated_cooling_k_per_5min", "Raffreddamento compensato per il moto", "K / 5 min"),
        ("area_growth_km2_per_5min", "Espansione dell'inviluppo", "km2 / 5 min"),
        ("speed_kmh", "Velocita del centroide", "km h-1"),
        ("btd_wv062_ir108_k", "WV6.2 - IR10.8", "K"),
        ("btd_ir120_ir108_k", "IR12.0 - IR10.8", "K"),
        ("max_local_cold_anomaly_k", "Massima anomalia fredda locale (OT proxy)", "K"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), panels):
        for case, group in features.groupby("case"):
            axis.plot(
                group.minutes_from_reference, group[column], marker="o", ms=3,
                color=colors[case], label=case.replace("_", " ")
            )
        axis.axvline(0, color="black", lw=0.8, alpha=0.65)
        if column == "motion_compensated_cooling_k_per_5min":
            axis.axhline(0, color="#64748b", lw=0.8)
        if column == "max_local_cold_anomaly_k":
            axis.axhline(6.5, color="#7c3aed", lw=0.9, ls="--", label="proxy OT 6.5 K")
        axis.set_title(title)
        axis.set_xlabel("minuti dal frame selezionato")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8, frameon=True)
    figure.suptitle(
        "Confronto object-based dei video borderline\n"
        "Tracking del core freddo IR10.8 < 241 K; grandezze radar non ancora disponibili",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_supplementary(features: pd.DataFrame, output: Path) -> None:
    colors = {"positive_emma": "#16a34a", "no_emma_ambiguous": "#dc2626"}
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 12), constrained_layout=True)
    panels = [
        ("object_annulus_contrast_k", "Contrasto core-anello circostante", "K"),
        ("ir108_iqr_k", "Variabilita termica interna (IQR)", "K"),
        ("nearby_cold_component_count", "Core freddi vicini", "numero"),
        ("btd_wv062_wv073_k", "WV6.2 - WV7.3", "K"),
        ("btd_ir108_ir087_k", "IR10.8 - IR8.7 (glaciazione proxy)", "K"),
        ("btd_ir134_ir108_k", "IR13.4 - IR10.8 (altezza proxy)", "K"),
        ("gradient_p90_k_per_pixel", "Gradiente termico p90", "K / pixel"),
        ("compactness", "Compattezza del core", "4 pi A / P2"),
        ("ot_like_candidate_pixels", "Minimi freddi OT-like", "pixel"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), panels):
        for case, group in features.groupby("case"):
            axis.plot(
                group.minutes_from_reference, group[column], marker="o", ms=3,
                color=colors[case], label=case.replace("_", " ")
            )
        axis.axvline(0, color="black", lw=0.8, alpha=0.65)
        axis.set_title(title)
        axis.set_xlabel("minuti dal frame selezionato")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8, frameon=True)
    figure.suptitle(
        "Diagnostiche multispettrali, morfologiche e di contesto\n"
        "Sono proxy osservativi: nessuna singola curva definisce una cella convettiva",
        fontsize=15,
    )
    figure.savefig(output, dpi=190)
    plt.close(figure)


def render_tracking_video(
    cases: list[dict], features: pd.DataFrame, output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9), constrained_layout=True)
    image_axes = axes[0]
    curve_axes = axes[1]
    image_artists = []
    contour_artists: list[list] = [[], []]
    time_markers = []
    colors = ["#16a34a", "#dc2626"]
    for case_index, case in enumerate(cases):
        ir = case["cube"][:, case["channels"].index("IR_108")]
        image_artists.append(
            image_axes[case_index].imshow(ir[0], cmap="turbo_r", vmin=205, vmax=285)
        )
        image_axes[case_index].set_title(case["label"])
        image_axes[case_index].set_xticks([])
        image_axes[case_index].set_yticks([])
        group = features[features.case == case["case"]]
        curve_axes[case_index].plot(
            group.minutes_from_reference, group.ir108_p10_k,
            color="#2563eb", marker="o", ms=3, label="IR10.8 p10"
        )
        second = curve_axes[case_index].twinx()
        second.plot(
            group.minutes_from_reference, group.object_area_km2_approx,
            color=colors[case_index], marker="o", ms=3, label="area <241 K"
        )
        curve_axes[case_index].set_xlabel("minuti dal frame selezionato")
        curve_axes[case_index].set_ylabel("IR10.8 p10 [K]", color="#2563eb")
        second.set_ylabel("area approssimata [km2]", color=colors[case_index])
        curve_axes[case_index].grid(alpha=0.25)
        time_markers.append(curve_axes[case_index].axvline(-45, color="black", lw=1.2))
    title = figure.suptitle("")
    writer = FFMpegWriter(fps=2, codec="libx264", bitrate=3500)
    with writer.saving(figure, output, dpi=125):
        for frame_index in range(len(cases[0]["times"])):
            for case_index, case in enumerate(cases):
                ir = case["cube"][:, case["channels"].index("IR_108")]
                image_artists[case_index].set_data(ir[frame_index])
                for artist in contour_artists[case_index]:
                    artist.remove()
                contour_artists[case_index] = list(
                    image_axes[case_index].contour(
                        case["masks"][frame_index], levels=[0.5], colors=[colors[case_index]],
                        linewidths=1.4
                    ).get_children()
                )
                minute = (case["times"][frame_index] - case["reference_time"]).total_seconds() / 60
                time_markers[case_index].set_xdata([minute, minute])
            title.set_text(
                f"Tracking object-based IR10.8 < 241 K | "
                f"tempo relativo {minute:+g} min"
            )
            writer.grab_frame()
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seviri-root", type=Path, required=True)
    parser.add_argument("--emma-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected).iloc[0]
    positive_target = utc_naive(selected.positive_target_time)
    positive_reference = utc_naive(selected.positive_seviri_time)
    negative_reference = utc_naive(selected.negative_seviri_time)
    positive_day = open_day(args.seviri_root, positive_reference)
    negative_day = open_day(args.seviri_root, negative_reference)
    try:
        x = np.asarray(positive_day.x.values)
        y = np.asarray(positive_day.y.values)
        full_mask = load_case_mask(
            args.emma_root, positive_target, int(selected.positive_mcs_id), x, y, {}
        )
        positive_ys, positive_xs, _ = centered_crop(full_mask)
        negative_y0, negative_x0 = int(selected.negative_origin_y), int(selected.negative_origin_x)
        channels = [str(value) for value in positive_day.channel.values]
        positive_cube = np.asarray(
            positive_day.seviri.isel(y=positive_ys, x=positive_xs).values, dtype=np.float32
        )
        negative_cube = np.asarray(
            negative_day.seviri.isel(
                y=slice(negative_y0, negative_y0 + CROP),
                x=slice(negative_x0, negative_x0 + CROP),
            ).values,
            dtype=np.float32,
        )
        positive_times = pd.DatetimeIndex(positive_day.time.values)
        negative_times = pd.DatetimeIndex(negative_day.time.values)
        positive_reference_index = int(np.argmin(abs(positive_times - positive_reference)))
        negative_reference_index = int(np.argmin(abs(negative_times - negative_reference)))
        positive_local_mask = full_mask[positive_ys, positive_xs]
        positive_masks = track_object(
            positive_cube[:, channels.index("IR_108")],
            positive_reference_index,
            positive_local_mask,
        )
        negative_masks = track_object(
            negative_cube[:, channels.index("IR_108")], negative_reference_index, None
        )
        positive_features = extract_features(
            "positive_emma", positive_cube, positive_times, channels, positive_masks,
            x, y, positive_ys.start, positive_xs.start, positive_reference
        )
        negative_features = extract_features(
            "no_emma_ambiguous", negative_cube, negative_times, channels, negative_masks,
            x, y, negative_y0, negative_x0, negative_reference
        )
        features = pd.concat([positive_features, negative_features], ignore_index=True)
        features.to_csv(args.out_dir / "borderline_pair_object_features.csv", index=False)
        summary_rows = []
        for case, group in features.groupby("case"):
            summary_rows.append(
                {
                    "case": case,
                    "ir108_p10_theilsen_k_per_15min": robust_slope(group, "ir108_p10_k"),
                    "area_theilsen_km2_per_15min": robust_slope(group, "object_area_km2_approx"),
                    "core235_theilsen_km2_per_15min": robust_slope(group, "core_area_lt235k_km2"),
                    "median_speed_kmh": float(np.nanmedian(group.speed_kmh)),
                    "median_motion_compensated_cooling_k_per_5min": float(
                        np.nanmedian(group.motion_compensated_cooling_k_per_5min)
                    ),
                    "max_ot_like_candidate_pixels": int(group.ot_like_candidate_pixels.max()),
                    "max_local_cold_anomaly_k": float(group.max_local_cold_anomaly_k.max()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            args.out_dir / "borderline_pair_object_summary.csv", index=False
        )
        plot_comparison(features, args.out_dir / "borderline_pair_object_comparison.png")
        plot_supplementary(
            features, args.out_dir / "borderline_pair_multispectral_shape_comparison.png"
        )
        cases = [
            {
                "case": "positive_emma", "label": "Positivo EMMA", "cube": positive_cube,
                "channels": channels, "times": positive_times, "reference_time": positive_reference,
                "masks": positive_masks,
            },
            {
                "case": "no_emma_ambiguous", "label": "Senza EMMA - ambiguo", "cube": negative_cube,
                "channels": channels, "times": negative_times, "reference_time": negative_reference,
                "masks": negative_masks,
            },
        ]
        render_tracking_video(
            cases, features, args.out_dir / "borderline_pair_tracked_ir108_comparison.mp4"
        )
    finally:
        positive_day.close()
        negative_day.close()
    print(f"saved diagnostics in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
