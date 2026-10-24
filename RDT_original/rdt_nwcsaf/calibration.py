"""OPERA Radar Ground-Truth Bridge, NWC SAF Statistical Calibration (CAL), and SEVIRI-Only Distillation.

Implements:
1. Reading and reprojecting EUMETNET OPERA ``DBZH_QIND`` radar composites onto the SEVIRI grid
   and labeling detected RDT cloud cells as Verified Convective (``y=1``), Verified Non-Convective
   (``y=0``), or Uncertain/Uncovered (``y=-1``).
2. Météo-France NWC SAF ``CAL`` (Calibrated) statistical discrimination training:
   Stratified MAP / L2-regularized Logistic Regression across the 6 vertical categories
   (``0..5``) and history regimes (instantaneous vs. tracked), plus ROC-optimal probability
   thresholds per category.
3. Distillation of ERA5 WMO Tropopause Temperature (``T_tropo``) and Convective Instability
   (``NWPCONV``) into a purely geostationary 7-channel SEVIRI surrogate model so that
   ``RDT_seviri_only`` can operate without external NWP or radar inputs at inference time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from emma_gpm.seviri import RSS_CRS
from typing import Any
from .config import classify_vertical_category


OPERA_SHAPE = (2200, 1900)
OPERA_PIXEL_M = 2000.0
OPERA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 "
    "+datum=WGS84 +units=m +no_defs"
)

FEATURE_NAMES = (
    "bias",
    "tropo_excess_10k",
    "contrast_10k",
    "tower_dt_20k",
    "grad_p95_2k",
    "btd_62_108_norm",
    "wbtd_62_73_norm",
    "btd_108_120_opaque",
    "btd_87_108_ice",
    "cooling_rate_norm",
    "expansion_rate_norm",
    "wv_trend_norm",
)


@dataclass(frozen=True)
class OperaCellVerification:
    """Radar verification metrics for a single RDT cloud cell."""

    inside_fraction: float
    valid_fraction: float
    max_dbzh: float
    p95_dbzh: float
    area35_km2: float
    area40_km2: float
    area45_km2: float
    label: int  # 1 = convective, 0 = non-convective, -1 = uncertain/uncovered


@dataclass
class CalibratedRdtModel:
    """Calibrated NWC SAF RDT-CW statistical discrimination weights and SEVIRI-only surrogates."""

    # Logistic weights per key "{vertical_cat}_{has_history}" -> list of 12 floats
    weights: dict[str, list[float]]
    # Probability thresholds per vertical category "0".."5"
    thresholds: dict[str, float]
    # SEVIRI-only tropopause regression weights (predicts ERA5 T_tropo from 7-ch SEVIRI + lat)
    seviri_tropo_weights: list[float]
    # SEVIRI-only instability logistic weights (predicts ERA5 NWPCONV from 7-ch SEVIRI)
    seviri_nwpconv_weights: list[float]
    training_summary: dict[str, object]

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibratedRdtModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=data["weights"],
            thresholds=data["thresholds"],
            seviri_tropo_weights=data["seviri_tropo_weights"],
            seviri_nwpconv_weights=data["seviri_nwpconv_weights"],
            training_summary=data.get("training_summary", {}),
        )


def opera_file_path(opera_root: str | Path, timestamp: str | pd.Timestamp) -> Path:
    """Return the local path to an OPERA ``DBZH_QIND.h5`` composite file."""
    ts = pd.Timestamp(timestamp).tz_localize(None).round("15min")
    return (
        Path(opera_root)
        / f"{ts:%Y}"
        / f"{ts:%m}"
        / f"{ts:%d}"
        / "OPERA"
        / "COMP"
        / f"OPERA@{ts:%Y%m%dT%H%M}@0@DBZH_QIND.h5"
    )


def read_opera_h5(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``DBZH`` and ``QIND`` from an ODIM HDF5 OPERA file."""
    dbzh = None
    qind = None
    with h5py.File(path, "r") as handle:
        for name in handle:
            if not name.startswith("dataset"):
                continue
            group = handle[name]
            if "what" not in group or "data1" not in group:
                continue
            qty = group["what"].attrs.get("quantity", b"")
            if isinstance(qty, bytes):
                qty = qty.decode("ascii")
            if qty in ("DBZH", "QIND"):
                arr = np.asarray(group["data1/data"], dtype=np.float32)
                nodata = float(group["what"].attrs.get("nodata", -9999000.0))
                undetect = float(group["what"].attrs.get("undetect", -8888000.0))
                arr[arr == nodata] = np.nan
                if qty == "DBZH":
                    arr[arr == undetect] = -32.0
                    dbzh = arr
                else:
                    arr[arr == undetect] = np.nan
                    qind = arr
    if dbzh is None or qind is None:
        raise KeyError(f"DBZH or QIND missing in {path}")
    return dbzh, qind


def build_seviri_to_opera_lookup(
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 1D SEVIRI geostationary coordinates ``(x_m, y_m)`` to OPERA ``(row, col, inside)``."""
    transformer = Transformer.from_crs(RSS_CRS, OPERA_CRS, always_xy=True)
    gx, gy = np.meshgrid(np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64))
    ox, oy = transformer.transform(gx, gy)
    finite = np.isfinite(ox) & np.isfinite(oy)
    col = np.full(ox.shape, -1, dtype=np.int32)
    row = np.full(oy.shape, -1, dtype=np.int32)
    col[finite] = np.floor(ox[finite] / OPERA_PIXEL_M).astype(np.int32)
    row[finite] = np.floor(-oy[finite] / OPERA_PIXEL_M).astype(np.int32)
    inside = (
        finite
        & (row >= 0)
        & (row < OPERA_SHAPE[0])
        & (col >= 0)
        & (col < OPERA_SHAPE[1])
    )
    return row, col, inside


def verify_cell_with_opera(
    cell_mask: np.ndarray,
    opera_row: np.ndarray,
    opera_col: np.ndarray,
    opera_inside: np.ndarray,
    dbzh: np.ndarray,
    qind: np.ndarray,
    *,
    qind_threshold: float = 0.20,
) -> OperaCellVerification:
    """Evaluate OPERA radar reflectivity inside an RDT cloud cell contour."""
    n_pix = int(np.count_nonzero(cell_mask))
    if n_pix == 0:
        return OperaCellVerification(0.0, 0.0, np.nan, np.nan, 0.0, 0.0, 0.0, -1)

    in_mask = cell_mask & opera_inside
    inside_fraction = float(np.count_nonzero(in_mask) / n_pix)
    if inside_fraction < 0.70:
        return OperaCellVerification(inside_fraction, 0.0, np.nan, np.nan, 0.0, 0.0, 0.0, -1)

    r_idx = opera_row[in_mask]
    c_idx = opera_col[in_mask]
    z_vals = dbzh[r_idx, c_idx]
    q_vals = qind[r_idx, c_idx]

    valid = np.isfinite(z_vals)
    no_echo = valid & np.isclose(z_vals, -32.0)
    quality_echo = valid & np.isfinite(q_vals) & (q_vals >= qind_threshold)
    usable = no_echo | quality_echo
    valid_fraction = float(np.count_nonzero(usable) / n_pix)

    if valid_fraction < 0.55:
        return OperaCellVerification(inside_fraction, valid_fraction, np.nan, np.nan, 0.0, 0.0, 0.0, -1)

    z_usable = z_vals[usable]
    max_dbzh = float(np.nanmax(z_usable)) if len(z_usable) else -32.0
    p95_dbzh = float(np.nanpercentile(z_usable, 95)) if len(z_usable) else -32.0

    # Unique 2-km OPERA pixels (4 km^2 each)
    uniq_rc = np.unique(np.column_stack((r_idx[usable], c_idx[usable])), axis=0)
    uniq_z = dbzh[uniq_rc[:, 0], uniq_rc[:, 1]]
    area35 = float(np.count_nonzero(uniq_z >= 35.0) * 4.0)
    area40 = float(np.count_nonzero(uniq_z >= 40.0) * 4.0)
    area45 = float(np.count_nonzero(uniq_z >= 45.0) * 4.0)

    if area35 >= 16.0 or area40 >= 8.0 or (max_dbzh >= 38.0 and p95_dbzh >= 28.0):
        label = 1
    elif max_dbzh < 28.0 and area35 == 0.0:
        label = 0
    else:
        label = -1  # Intermediate stratiform rain (28..34 dBZ) kept out of binary training

    return OperaCellVerification(
        inside_fraction=inside_fraction,
        valid_fraction=valid_fraction,
        max_dbzh=max_dbzh,
        p95_dbzh=p95_dbzh,
        area35_km2=area35,
        area40_km2=area40,
        area45_km2=area45,
        label=label,
    )


def build_feature_vector(
    cell: dict[str, Any],
    trends: dict[str, Any],
    depth_regime: str,
    t_tropo_k: float = 215.0,
) -> np.ndarray:
    """Build the 12-element standardized NWC SAF predictor vector x for a cell."""
    has_hist = depth_regime != "0"

    t_min = float(cell["t_min_k"])
    tropo_excess = np.clip((float(t_tropo_k) + 15.0 - t_min) / 12.0, -3.0, 3.0)
    contrast_norm = np.clip((float(cell["contrast_k"]) - 8.0) / 10.0, -1.5, 3.5)
    tower_dt = np.clip((float(cell["tower_depth_k"]) - 10.0) / 20.0, -0.5, 3.0)
    grad_p = np.clip((float(cell["periph_grad_p95_k_km"]) - 1.2) / 2.0, -1.0, 3.5)

    btd_62_108 = np.clip((float(cell["btd_wv62_ir108_p90_k"]) + 6.0) / 6.0, -2.5, 2.5)
    wbtd_62_73 = np.clip((float(cell["wbtd_wv62_wv73_p90_k"]) + 12.0) / 8.0, -2.5, 2.5)
    btd_108_120 = np.clip((1.8 - float(cell["btd_ir108_ir120_median_k"])) / 2.0, -2.5, 2.5)
    btd_87_108 = np.clip((float(cell["btd_ir87_ir108_p90_k"]) + 1.5) / 2.5, -2.5, 2.5)

    cool_15 = trends.get("cooling_rate_15m_k_per_h", np.nan)
    cool_inst = trends.get("cooling_rate_k_per_h", np.nan)
    cool_val = float(cool_15) if np.isfinite(cool_15) else (float(cool_inst) if np.isfinite(cool_inst) else 0.0)
    cooling_rate = np.clip((cool_val - 4.0) / 10.0, -2.0, 4.0) if has_hist else 0.0

    exp_15 = trends.get("rel_expansion_15m_per_h", np.nan)
    exp_inst = trends.get("rel_expansion_per_h", np.nan)
    exp_val = float(exp_15) if np.isfinite(exp_15) else (float(exp_inst) if np.isfinite(exp_inst) else 0.0)
    expansion_rate = np.clip(exp_val, -1.5, 3.5) if has_hist else 0.0

    btd_tr = trends.get("btd_change_15m_k", np.nan)
    wv_trend = np.clip((float(btd_tr) * 4.0) / 8.0, -2.0, 3.5) if (has_hist and np.isfinite(btd_tr)) else 0.0

    return np.array(
        [
            1.0,
            tropo_excess,
            contrast_norm,
            tower_dt,
            grad_p,
            btd_62_108,
            wbtd_62_73,
            btd_108_120,
            btd_87_108,
            cooling_rate,
            expansion_rate,
            wv_trend,
        ],
        dtype=np.float64,
    )


def default_prior_weights(vertical_cat: int, has_history: bool) -> np.ndarray:
    """Return physical NWC SAF prior weights beta_0 for MAP logistic regression.

    Vertical categories in NWC SAF RDT-CW:
    0 = Mature (coldest, T_min <= -40 C for >= 30 min)
    1 = Mature Transition (crossing -40 C or T_BT <= -35 C)
    2 = Cold Transition (-40 C < T_min <= -35 C or T_BT <= -25 C)
    3 = Warm1 Transition (-35 C < T_min <= -25 C)
    4 = Warm2 Transition (-25 C < T_min <= -15 C)
    5 = Warm (T_min > -15 C)
    """
    base_bias = {0: 0.45, 1: -0.15, 2: -0.65, 3: -1.15, 4: -1.55, 5: -2.10}.get(vertical_cat, -0.8)
    return np.array(
        [
            base_bias,
            0.70,  # tropo_excess_10k (colder relative to ERA5 WMO tropopause)
            0.85,  # contrast_10k
            0.65,  # tower_dt_20k
            0.75,  # grad_p95_2k
            0.90,  # btd_62_108_norm
            0.55,  # wbtd_62_73_norm
            0.60,  # btd_108_120_opaque (penalizes thin cirrus, rewards thick cumulonimbus)
            0.35,  # btd_87_108_ice
            0.95 if has_history else 0.0,  # cooling_rate_norm
            0.60 if has_history else 0.0,  # expansion_rate_norm
            0.45 if has_history else 0.0,  # wv_trend_norm
        ],
        dtype=np.float64,
    )


def _fit_map_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    beta_prior: np.ndarray,
    l2_lambda: float = 4.0,
    max_iter: int = 25,
) -> np.ndarray:
    """Fit L2-regularized MAP logistic regression around ``beta_prior`` with physical monotonicity bounds."""
    if len(y) == 0:
        return beta_prior.copy()

    X_clean = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X_clean.shape

    beta = beta_prior.copy()
    # Scale regularization with sample count so large strata do not overwhelm the physical prior
    eff_lambda = max(float(l2_lambda), 0.20 * float(n_samples))
    reg_mat = eff_lambda * np.eye(n_features, dtype=np.float64)
    reg_mat[0, 0] = 0.35 * eff_lambda

    # Moderate class-imbalance weighting
    pos_frac = float(np.mean(y_clean == 1))
    if 0.01 < pos_frac < 0.99:
        w_pos = np.clip(0.35 / pos_frac, 0.5, 4.0)
        w_neg = np.clip(0.65 / (1.0 - pos_frac), 0.5, 2.0)
        sample_w = np.where(y_clean == 1, w_pos, w_neg)
    else:
        sample_w = np.ones_like(y_clean, dtype=np.float64)

    # Physical box constraints around beta_prior:
    # Intercept can shift by +/- 1.25; physical slopes stay positive within [0.3 * prior, 2.2 * prior]
    lower_bounds = np.where(beta_prior >= 0, 0.30 * beta_prior, 2.20 * beta_prior)
    upper_bounds = np.where(beta_prior >= 0, 2.20 * beta_prior, 0.30 * beta_prior)
    lower_bounds[0] = beta_prior[0] - 1.25
    upper_bounds[0] = beta_prior[0] + 1.25

    for _ in range(max_iter):
        logits = np.clip(X_clean @ beta, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = X_clean.T @ (sample_w * (probs - y_clean)) + reg_mat @ (beta - beta_prior)
        w_diag = sample_w * np.maximum(probs * (1.0 - probs), 1e-3)
        hess = (X_clean.T * w_diag) @ X_clean + reg_mat
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -0.5, 0.5)
        beta = np.clip(beta - 0.5 * step, lower_bounds, upper_bounds)
        if float(np.max(np.abs(step))) < 1e-4:
            break

    return beta


def fit_calibrated_rdt_model(
    records: list[dict[str, object]],
    tropo_samples: list[tuple[np.ndarray, np.ndarray]],
    nwpconv_samples: list[tuple[np.ndarray, np.ndarray]],
) -> CalibratedRdtModel:
    """Fit the stratified NWC SAF ``CAL`` logistic regression model and SEVIRI-only surrogates."""
    weights_dict: dict[str, list[float]] = {}
    thresholds_dict: dict[str, float] = {}
    strata_stats: dict[str, object] = {}

    for vcat in range(6):
        vcat_probs = []
        vcat_labels = []
        for has_hist_flag in (0, 1):
            key = f"{vcat}_{has_hist_flag}"
            prior = default_prior_weights(vcat, bool(has_hist_flag))
            sub = [
                r
                for r in records
                if int(r["vertical_cat"]) == vcat
                and int(r["has_history"]) == has_hist_flag
                and int(r["label"]) in (0, 1)
            ]
            # Pool adjacent vertical category if stratum has < 15 samples
            if len(sub) < 15:
                sub = [
                    r
                    for r in records
                    if abs(int(r["vertical_cat"]) - vcat) <= 1
                    and int(r["has_history"]) == has_hist_flag
                    and int(r["label"]) in (0, 1)
                ]
            if len(sub) >= 6:
                X = np.vstack([np.asarray(r["features"], dtype=np.float64) for r in sub])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y = np.array([int(r["label"]) for r in sub], dtype=np.float64)
                # Zero out temporal features if instantaneous stratum
                if has_hist_flag == 0:
                    X[:, 9:12] = 0.0
                beta = _fit_map_logistic_regression(X, y, prior, l2_lambda=6.0)
                if has_hist_flag == 0:
                    beta[9:12] = 0.0
                probs = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -20.0, 20.0)))
                vcat_probs.extend(probs.tolist())
                vcat_labels.extend(y.tolist())
            else:
                beta = prior

            weights_dict[key] = [round(float(x), 5) for x in beta]
            strata_stats[key] = {
                "n_samples": len(sub),
                "n_pos": int(sum(int(r["label"]) == 1 for r in sub)),
            }

        # Optimize CSI / F1 probability threshold for vertical category vcat
        default_thresh = {0: 0.45, 1: 0.48, 2: 0.50, 3: 0.54, 4: 0.58, 5: 0.62}[vcat]
        if len(vcat_labels) >= 12 and 0 < sum(vcat_labels) < len(vcat_labels):
            p_arr = np.asarray(vcat_probs, dtype=np.float64)
            y_arr = np.asarray(vcat_labels, dtype=np.int32)
            best_t = default_thresh
            best_csi = -1.0
            for cand_t in np.linspace(0.38, 0.65, 28):
                pred = p_arr >= cand_t
                tp = int(np.sum(pred & (y_arr == 1)))
                fp = int(np.sum(pred & (y_arr == 0)))
                fn = int(np.sum((~pred) & (y_arr == 1)))
                csi = tp / float(max(tp + fp + fn, 1))
                if csi > best_csi:
                    best_csi = csi
                    best_t = float(cand_t)
            thresholds_dict[str(vcat)] = round(0.5 * best_t + 0.5 * default_thresh, 4)
        else:
            thresholds_dict[str(vcat)] = default_thresh

    # Fit SEVIRI-only Tropopause regression anchored to physical UTLS channel slopes:
    prior_t = np.array([20.0, 0.35, 0.15, 0.20, 0.20, -0.05, -6.0], dtype=np.float64)
    if tropo_samples:
        X_t = np.vstack([s[0] for s in tropo_samples])
        y_t = np.concatenate([s[1] for s in tropo_samples])
        valid_t = np.all(np.isfinite(X_t), axis=1) & np.isfinite(y_t)
        if int(np.sum(valid_t)) >= 20:
            X_tv = X_t[valid_t]
            y_tv = y_t[valid_t]
            w_tropo = prior_t.copy()
            w_tropo[0] = float(np.median(y_tv - (X_tv[:, 1:] @ prior_t[1:])))
        else:
            w_tropo = prior_t
    else:
        w_tropo = prior_t

    # Fit SEVIRI-only NWPCONV instability logistic model:
    prior_c = np.array([0.3, 0.8, 1.1, 0.9, 0.7], dtype=np.float64)
    if nwpconv_samples:
        X_c = np.vstack([s[0] for s in nwpconv_samples])
        y_c = np.concatenate([s[1] for s in nwpconv_samples])
        valid_c = np.all(np.isfinite(X_c), axis=1) & np.isfinite(y_c)
        if int(np.sum(valid_c)) >= 20:
            w_nwpconv = _fit_map_logistic_regression(
                X_c[valid_c], y_c[valid_c], prior_c, l2_lambda=25.0
            )
        else:
            w_nwpconv = prior_c
    else:
        w_nwpconv = prior_c

    return CalibratedRdtModel(
        weights=weights_dict,
        thresholds=thresholds_dict,
        seviri_tropo_weights=[round(float(x), 6) for x in w_tropo],
        seviri_nwpconv_weights=[round(float(x), 6) for x in w_nwpconv],
        training_summary={
            "total_labeled_cells": len(records),
            "total_convective_cells": int(sum(int(r["label"]) == 1 for r in records)),
            "total_non_convective_cells": int(sum(int(r["label"]) == 0 for r in records)),
            "strata": strata_stats,
        },
    )


def predict_seviri_only_nwp_surrogates(
    model: CalibratedRdtModel,
    channels_7ch: dict[str, np.ndarray],
    lat_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (T_tropo_k, NWPCONV_mask) purely from the 7 geostationary SEVIRI channels.

    Used by ``RDT_seviri_only`` so it requires zero external NWP data at runtime while
    closely matching the ERA5 WMO tropopause and convective airmass mask.
    """
    ir108 = np.asarray(channels_7ch["IR_108"], dtype=np.float64)
    wv062 = np.asarray(channels_7ch["WV_062"], dtype=np.float64)
    wv073 = np.asarray(channels_7ch["WV_073"], dtype=np.float64)
    ir097 = np.asarray(channels_7ch["IR_097"], dtype=np.float64)
    ir120 = np.asarray(channels_7ch["IR_120"], dtype=np.float64)
    ir134 = np.asarray(channels_7ch["IR_134"], dtype=np.float64)

    H, W = ir108.shape
    if lat_deg is None:
        lat_norm = np.full((H, W), 45.0 / 45.0, dtype=np.float64)
    else:
        lat_norm = np.abs(np.broadcast_to(np.asarray(lat_deg, dtype=np.float64), (H, W))) / 45.0

    # Use regional upper-tropospheric clear/ambient percentiles for smooth synoptic tropopause
    w_t = np.asarray(model.seviri_tropo_weights, dtype=np.float64)
    wv62_amb = float(np.nanpercentile(wv062, 25))
    wv73_amb = float(np.nanpercentile(wv073, 25))
    ir97_amb = float(np.nanpercentile(ir097, 25))
    ir134_amb = float(np.nanpercentile(ir134, 25))
    ir108_amb = float(np.nanpercentile(ir108, 75))

    t_tropo_est = (
        w_t[0]
        + w_t[1] * (0.5 * wv062 + 0.5 * wv62_amb)
        + w_t[2] * (0.5 * wv073 + 0.5 * wv73_amb)
        + w_t[3] * (0.5 * ir097 + 0.5 * ir97_amb)
        + w_t[4] * (0.5 * ir134 + 0.5 * ir134_amb)
        + w_t[5] * ir108_amb
        + w_t[6] * lat_norm
    )
    t_tropo_k = np.clip(t_tropo_est, 198.0, 228.0).astype(np.float32)

    # Predict NWPCONV instability mask from low-level warmth, lapse rate (IR10.8 - IR13.4),
    # split-window boundary-layer water vapor (IR10.8 - IR12.0), and mid-level moisture
    w_c = np.asarray(model.seviri_nwpconv_weights, dtype=np.float64)
    f1 = (np.maximum(ir108, ir108_amb) - 273.15) / 20.0
    f2 = (ir108 - ir134) / 25.0
    f3 = np.clip(ir108 - ir120, -1.0, 4.0) / 2.0
    f4 = (wv073 - wv062) / 15.0

    logit_c = w_c[0] + w_c[1] * f1 + w_c[2] * f2 + w_c[3] * f3 + w_c[4] * f4
    prob_c = 1.0 / (1.0 + np.exp(-np.clip(logit_c, -20.0, 20.0)))
    # Also always allow deep cold clouds (IR10.8 < T_tropo + 25 K) to be evaluated
    nwpconv_est = ((prob_c >= 0.35) | (ir108 <= t_tropo_k + 25.0)).astype(np.uint8)

    return t_tropo_k, nwpconv_est
