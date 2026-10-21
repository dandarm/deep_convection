"""NWC SAF GEO-RDT-CW Convective Discrimination, Phase of Development, Severity & Forecasting.

Implements:
1. ATBD Section 3.2.1.2.3 (pp. 34-44) & Figure 20 (p. 35):
   - NWP instability mask filter (LI, SHW, KI) when NWP inputs are provided.
   - 6 vertical categories (0: Mature, 1: Mature Transition, 2: Cold Transition,
     3: Warm1 Transition, 4: Warm2 Transition, 5: Warm) and 5 history-depth regimes
     (0: 0 min, C: 15 min, M: 30 min, L: 60 min, VL: 90 min).
   - Convective inheritance from primary or secondary convective parent (ch5 = 4..7).
   - Empirical declassification rules (30 min hold in Cat 5, 60 min hold in Cat 1-4 & Cat 0,
     Mature warming > -40 C or BTD(WV6.2-IR10.8) < -1 C with negative 60-min trend,
     and father Severity >= 1 protection).
   - 5-channel IRxWVx (IR10.8, WV6.2, WV7.3, IR8.7, IR12.0) logistic regression
     statistical discrimination on cooling/ascending or mature cells.
   - Mandatory Overshooting Top (OTD) forced convective override (ch5 = 11).
2. ATBD Section 3.2.1.2.5.9 (pp. 50-51, Table 8):
   - Phase of development (0: Triggering, 1: Growing, 2: Mature, 3: Decaying, 4: Split).
3. ATBD Section 3.2.1.2.5.10 (pp. 51-52):
   - Severity index (0: NIL, 1: Low, 2: Moderate, 3: High, 4: Very High) & Hazard code.
4. ATBD Section 3.2.1.2.4 (pp. 44-45):
   - Lagrangian forecast (+15, +30, +45, +60 min) with optional dilation & smoothing.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .config import NwcSafRdtConfig, classify_vertical_category
from .detection import ThresholdSliceHierarchy
from .tracking import common_threshold_area


def select_history_depth_regime(
    vertical_cat: int,
    age_total_min: float,
    age_in_cat_min: float,
    scheme: str = "CAL",
) -> tuple[str, float]:
    """Return (regime_code, window_minutes) according to ATBD Sec 3.2.1.2.3.3.1 (p. 40) & v1.1 Fig. 23.

    CAL depth regimes:
    - '0': 0 min (triggering without history, Transition categories 1-4 only)
    - 'C': 15 min (short history, Transition categories 1-4)
    - 'M': 30 min (medium history)
    - 'L': 60 min (long history)
    - 'VL': 90 min (very long history, Mature category 0 only)
    """
    if vertical_cat == 0:
        if age_total_min >= 75.0:
            return ("VL", 90.0) if scheme == "CAL" else ("DT60", 60.0)
        if age_total_min >= 45.0:
            return ("L", 60.0) if scheme == "CAL" else ("DT45", 45.0)
        return ("M", 30.0) if scheme == "CAL" else ("DT30", 30.0)

    if vertical_cat in (1, 2, 3, 4):
        if age_total_min < 5.0:
            return ("0", 0.0)
        if age_total_min <= 20.0:
            return ("C", 15.0) if scheme == "CAL" else ("DT15", 15.0)
        if age_total_min <= 40.0:
            return ("M", 30.0) if scheme == "CAL" else ("DT30", 30.0)
        return ("L", 60.0) if scheme == "CAL" else ("DT60", 60.0)

    # Category 5 (Warm)
    if age_total_min <= 20.0:
        return ("C", 15.0)
    if age_total_min <= 40.0:
        return ("M", 30.0)
    return ("L", 60.0)


def compute_temporal_predictors(
    cell: dict[str, Any],
    history: list[dict[str, Any]],
    prev_cell: dict[str, Any] | None,
    prev_hierarchy: ThresholdSliceHierarchy | None,
    curr_hierarchy: ThresholdSliceHierarchy | None,
    dt_minutes: float,
    cfg: NwcSafRdtConfig,
) -> dict[str, float]:
    """Compute instantaneous and multi-step history trends along the main father trajectory.

    Includes:
    - Instantaneous cooling rate (K/h, positive when cooling) and 15-min cooling (K/15min)
    - Common-threshold surface expansion rate (%/h and km2/15min, ATBD Line H / Sec 3.2.1.2.3.4)
    - 15-min, 30-min, and 60-min trends and extrema of T_min, contrast, and 5-channel BTDs.
    """
    dt_hours = max(dt_minutes / 60.0, 1e-6)
    can_compare_prev = (
        prev_cell is not None
        and not cell["border"]
        and not prev_cell.get("border", False)
    )

    if can_compare_prev and prev_cell is not None:
        dt_min_inst = float(cell["t_min_k"] - prev_cell["t_min_k"])  # negative when cooling
        cooling_rate_k_per_h = float(-dt_min_inst / dt_hours)  # positive when cooling (ATBD Line S ch26)
        # Common threshold T_common = min(T_BT(t), T_BT(t-dt)) (or max if min is below T_min)
        t_comm_cold = min(cell["t_bt_k"], prev_cell["t_bt_k"])
        t_comm_warm = max(cell["t_bt_k"], prev_cell["t_bt_k"])
        s_curr_cold = common_threshold_area(cell, t_comm_cold, curr_hierarchy, cfg)
        s_prev_cold = common_threshold_area(prev_cell, t_comm_cold, prev_hierarchy, cfg)
        if s_curr_cold >= cfg.smin_km2 and s_prev_cold >= cfg.smin_km2:
            t_common = t_comm_cold
            s_curr_comm, s_prev_comm = s_curr_cold, s_prev_cold
        else:
            t_common = t_comm_warm
            s_curr_comm = common_threshold_area(cell, t_comm_warm, curr_hierarchy, cfg)
            s_prev_comm = common_threshold_area(prev_cell, t_comm_warm, prev_hierarchy, cfg)

        if s_prev_comm > 0.0 and cell.get("event") == "continuation":
            rel_exp_per_h = float(((s_curr_comm - s_prev_comm) / s_prev_comm) / dt_hours)
            d_area_inst_km2 = float(s_curr_comm - s_prev_comm)
        else:
            rel_exp_per_h = np.nan
            d_area_inst_km2 = np.nan
        btd_trend_k_per_h = float(
            (cell["btd_wv62_ir108_max_k"] - prev_cell["btd_wv62_ir108_max_k"]) / dt_hours
        )
    else:
        dt_min_inst = np.nan
        cooling_rate_k_per_h = np.nan
        t_common = np.nan
        rel_exp_per_h = np.nan
        d_area_inst_km2 = np.nan
        btd_trend_k_per_h = np.nan

    # Actual 15-minute tendencies (3 intervals of 5 min, or matching 15-min lookback)
    steps_15m = max(1, int(round(15.0 / max(dt_minutes, 1.0))))
    if (
        len(history) >= steps_15m
        and not cell["border"]
        and not any(h.get("border", False) for h in history[-steps_15m:])
    ):
        ref_15 = history[-steps_15m]
        cooling_15m_k = float(cell["t_min_k"] - ref_15["t_min_k"])  # negative = cooling
        cooling_rate_15m_k_per_h = float(-cooling_15m_k * (60.0 / 15.0))
        cont_15m = cell.get("event") == "continuation" and all(
            h.get("event") == "continuation" for h in history[-(steps_15m - 1) :]
        ) if steps_15m > 1 else (cell.get("event") == "continuation")
        if cont_15m:
            # Evaluate area change at common threshold between t-15m and t when profile available
            t_comm_15 = max(cell["t_bt_k"], ref_15["t_bt_k"])
            s_now_15 = common_threshold_area(cell, t_comm_15, curr_hierarchy, cfg)
            s_ref_15 = common_threshold_area(ref_15, t_comm_15, None, cfg)
            area_change_15m_km2 = float(s_now_15 - s_ref_15)
            rel_exp_15m_per_h = (
                float((area_change_15m_km2 / max(s_ref_15, 1.0)) * 4.0)
                if s_ref_15 > 0.0
                else np.nan
            )
        else:
            area_change_15m_km2 = np.nan
            rel_exp_15m_per_h = np.nan
        btd_change_15m_k = float(
            cell["btd_wv62_ir108_max_k"] - ref_15["btd_wv62_ir108_max_k"]
        )
    else:
        cooling_15m_k = np.nan
        cooling_rate_15m_k_per_h = cooling_rate_k_per_h
        area_change_15m_km2 = np.nan
        rel_exp_15m_per_h = rel_exp_per_h
        btd_change_15m_k = np.nan

    # 60-minute BTD trend for Mature declassification rule (ATBD Sec 3.2.1.2.3.5, p. 44)
    steps_60m = max(1, int(round(60.0 / max(dt_minutes, 1.0))))
    if len(history) >= steps_60m:
        ref_60 = history[-steps_60m]
        btd_trend_60m_k = float(
            cell["btd_wv62_ir108_max_k"] - ref_60["btd_wv62_ir108_max_k"]
        )
    elif history:
        btd_trend_60m_k = float(
            cell["btd_wv62_ir108_max_k"] - history[0]["btd_wv62_ir108_max_k"]
        )
    else:
        btd_trend_60m_k = np.nan

    # Extrema over recent history window (up to 60 min)
    recent_window = [*history[-steps_60m:], cell]
    max_contrast_hist = float(max(r["contrast_k"] for r in recent_window))
    max_btd_hist = float(max(r["btd_wv62_ir108_max_k"] for r in recent_window))
    max_cooling_rate_hist = float(
        np.nanmax(
            [
                r.get("cooling_rate_k_per_h", np.nan)
                for r in recent_window[:-1]
                if np.isfinite(r.get("cooling_rate_k_per_h", np.nan))
            ]
            + ([cooling_rate_k_per_h] if np.isfinite(cooling_rate_k_per_h) else [0.0])
        )
    )

    return {
        "dt_min_inst_k": dt_min_inst,
        "cooling_rate_k_per_h": cooling_rate_k_per_h,
        "cooling_rate_15m_k_per_h": cooling_rate_15m_k_per_h,
        "cooling_15m_k": cooling_15m_k,
        "common_threshold_k": t_common,
        "rel_expansion_per_h": rel_exp_per_h,
        "rel_expansion_15m_per_h": rel_exp_15m_per_h,
        "area_change_inst_km2": d_area_inst_km2,
        "area_change_15m_km2": area_change_15m_km2,
        "btd_trend_k_per_h": btd_trend_k_per_h,
        "btd_change_15m_k": btd_change_15m_k,
        "btd_trend_60m_k": btd_trend_60m_k,
        "max_contrast_hist_k": max_contrast_hist,
        "max_btd_wv62_ir108_hist_k": max_btd_hist,
        "max_cooling_rate_hist_k_per_h": max_cooling_rate_hist,
    }


def evaluate_logistic_discrimination(
    cell: dict[str, Any],
    trends: dict[str, Any],
    vertical_cat: int,
    depth_regime: str,
    cfg: NwcSafRdtConfig,
    *,
    calibrated_model: Any | None = None,
    cell_tropo_temp_k: float = 215.0,
) -> tuple[bool, float, float]:
    """Evaluate NWC SAF 5-channel (IR10.8, WV6.2, WV7.3, IR8.7, IR12.0) logistic regression model.

    Supports both:
    - ``calibrated_model`` (Météo-France empirical ``CAL`` scheme trained against OPERA radar / lightning)
    - Default physical NWC SAF category-specific logistic regression weights.
    """
    if calibrated_model is not None:
        from .calibration import build_feature_vector

        has_hist_flag = 1 if depth_regime != "0" else 0
        key = f"{int(vertical_cat)}_{has_hist_flag}"
        if key in calibrated_model.weights:
            beta = np.asarray(calibrated_model.weights[key], dtype=np.float64)
            x_vec = build_feature_vector(cell, trends, depth_regime, t_tropo_k=cell_tropo_temp_k)
            z_cal = float(np.clip(np.dot(beta, x_vec), -30.0, 30.0))
            prob_cal = float(1.0 / (1.0 + np.exp(-z_cal)))
            p_thresh_cal = float(calibrated_model.thresholds.get(str(int(vertical_cat)), 0.50))
            if depth_regime == "0":
                p_thresh_cal = min(0.75, p_thresh_cal + 0.05)
            return bool(prob_cal >= p_thresh_cal), prob_cal, float(100.0 * (1.0 - prob_cal))

    contrast = float(cell["contrast_k"])
    tower_depth = float(cell["tower_depth_k"])
    grad_p95 = float(cell["periph_grad_p95_k_km"])
    grad_mean = float(cell["periph_grad_mean_k_km"])

    # Cooling rate in K/h (positive when cooling)
    cool_15 = trends.get("cooling_rate_15m_k_per_h", np.nan)
    cool_inst = trends.get("cooling_rate_k_per_h", np.nan)
    cooling_k_h = (
        float(cool_15)
        if np.isfinite(cool_15)
        else (float(cool_inst) if np.isfinite(cool_inst) else 0.0)
    )

    # Relative expansion rate (1/h)
    exp_15 = trends.get("rel_expansion_15m_per_h", np.nan)
    exp_inst = trends.get("rel_expansion_per_h", np.nan)
    rel_exp_h = (
        float(exp_15)
        if np.isfinite(exp_15)
        else (float(exp_inst) if np.isfinite(exp_inst) else 0.0)
    )
    rel_exp_h = float(np.clip(rel_exp_h, -2.0, 5.0))

    # 5-channel multi-spectral predictors
    btd_p90 = float(cell["btd_wv62_ir108_p90_k"])
    btd_max = float(cell["btd_wv62_ir108_max_k"])
    btd_sig = float(cell["btd_wv62_ir108_sig"])
    wbtd_p90 = float(cell["wbtd_wv62_wv73_p90_k"])
    glac_p90 = float(cell["btd_ir87_ir108_p90_k"])
    split_win = float(cell["btd_ir108_ir120_median_k"])  # near 0 for thick opaque cumulonimbus, large > 2.5 K for thin cirrus

    btd_trend = trends.get("btd_change_15m_k", np.nan)
    btd_rate_k_h = float(btd_trend * 4.0) if np.isfinite(btd_trend) else 0.0

    # Category-specific logistic regression intercepts & predictor weights
    # (calibrated to the NWC SAF ATBD vertical category physics:
    #  - Categories 0 & 1 [Mature / Mature Transition, T_min <= -40 C]: driven by strong BTD(WV6.2-IR10.8),
    #    steep peripheral gradient Grad_P, deep contrast, opaque split-window, and cooling/history;
    #  - Categories 2, 3, 4 [Cold/Warm1/Warm2 Transitions]: driven by rapid cooling rate -dT_min/dt,
    #    positive expansion rate dS/dt, rising BTD(WV6.2-IR10.8)/WBTD(WV6.2-WV7.3), and sharp peripheral gradient;
    #  - Category 5 [Warm, T_min > -15 C]: requires very strong cooling rate, expansion, and compact gradient).
    thin_cirrus_penalty = max(0.0, split_win - 2.0) * 0.85

    if vertical_cat == 0:
        # Mature category (T_min <= -40 C for >= 30 min)
        z = (
            -2.10
            + 0.14 * (contrast - 8.0)
            + 0.06 * (tower_depth - 12.0)
            + 0.35 * (grad_p95 - 1.2)
            + 0.25 * (grad_mean - 0.6)
            + 0.22 * (btd_p90 + 3.5)
            + 0.15 * (btd_max + 1.0)
            + 0.80 * btd_sig
            + 0.08 * (wbtd_p90 + 12.0)
            + 0.15 * (glac_p90 + 1.5)
            + 0.05 * max(cooling_k_h, -5.0)
            + 0.04 * trends.get("max_cooling_rate_hist_k_per_h", 0.0)
            - thin_cirrus_penalty
        )
        p_thresh = 0.52
    elif vertical_cat == 1:
        # Mature Transition (crossing -40 C or T_BT <= -35 C)
        z = (
            -2.25
            + 0.15 * (contrast - 9.0)
            + 0.06 * (tower_depth - 10.0)
            + 0.38 * (grad_p95 - 1.2)
            + 0.10 * (cooling_k_h - 6.0)
            + 0.45 * rel_exp_h
            + 0.20 * (btd_p90 + 4.5)
            + 0.14 * (btd_max + 1.5)
            + 0.70 * btd_sig
            + 0.08 * (wbtd_p90 + 14.0)
            + 0.06 * btd_rate_k_h
            - thin_cirrus_penalty
        )
        p_thresh = 0.54 if depth_regime != "0" else 0.62
    elif vertical_cat == 2:
        # Cold Transition (-40 C < T_min <= -35 C or T_BT <= -25 C)
        z = (
            -2.40
            + 0.15 * (contrast - 9.5)
            + 0.40 * (grad_p95 - 1.3)
            + 0.14 * (cooling_k_h - 10.0)
            + 0.60 * rel_exp_h
            + 0.18 * (btd_p90 + 6.5)
            + 0.10 * (wbtd_p90 + 16.0)
            + 0.08 * btd_rate_k_h
            - thin_cirrus_penalty
        )
        p_thresh = 0.55 if depth_regime != "0" else 0.65
    elif vertical_cat in (3, 4):
        # Warm1 (-35..-25 C) & Warm2 (-25..-15 C) Transitions
        z = (
            -2.65
            + 0.16 * (contrast - 10.0)
            + 0.45 * (grad_p95 - 1.5)
            + 0.18 * (cooling_k_h - 12.0)
            + 0.75 * rel_exp_h
            + 0.14 * (btd_p90 + 10.0)
            + 0.10 * (wbtd_p90 + 18.0)
            + 0.10 * btd_rate_k_h
            - thin_cirrus_penalty
        )
        p_thresh = 0.58 if depth_regime != "0" else 0.68
    else:
        # Category 5: Warm (T_min > -15 C)
        z = (
            -3.10
            + 0.18 * (contrast - 10.0)
            + 0.50 * (grad_p95 - 1.8)
            + 0.22 * (cooling_k_h - 16.0)
            + 0.90 * rel_exp_h
            + 0.12 * (wbtd_p90 + 20.0)
            + 0.12 * btd_rate_k_h
            - thin_cirrus_penalty
        )
        p_thresh = 0.62

    z_clamped = float(np.clip(z, -30.0, 30.0))
    prob_conv = float(1.0 / (1.0 + np.exp(-z_clamped)))
    error_likelihood_pct = float(100.0 * (1.0 - prob_conv))
    is_conv = bool(prob_conv >= p_thresh)
    return is_conv, prob_conv, error_likelihood_pct


def compute_phase_of_development(
    event: str,
    vertical_cat: int,
    t_min_k: float,
    trends: dict[str, float],
    is_convective: bool,
    ot_confirmed_count: int,
    lightning_strokes: int,
    cfg: NwcSafRdtConfig,
    mother_phase: int | None = None,
) -> int:
    """Compute NWC SAF Phase of Development (0..3) using ATBD Table 8 (p. 51).

    Encoded Phase:
    0 = Triggering (birth / new trajectory)
    1 = Growing
    2 = Mature
    3 = Decaying
    Note: For 'split_new' child cells, we inherit the mother cell's phase instead of
    introducing an artificial non-physical 5th phase.
    """
    if event in ("new", "complex"):
        return 0  # Triggering
    if event == "split_new":
        if mother_phase is not None and mother_phase in (0, 1, 2, 3):
            return int(mother_phase)
        return 1  # Default to Growing for split progeny

    # Continued trajectory ('continuation', 'merge_main', 'split_main')
    cool_15 = trends.get("cooling_rate_15m_k_per_h", np.nan)
    cool_inst = trends.get("cooling_rate_k_per_h", np.nan)
    cooling_rate = (
        float(cool_15)
        if np.isfinite(cool_15)
        else (float(cool_inst) if np.isfinite(cool_inst) else 0.0)
    )
    # Note: cooling_rate > 0 means dT_min < 0 (cooling); cooling_rate < 0 means dT_min > 0 (warming)
    d_area_15 = trends.get("area_change_15m_km2", np.nan)
    d_area_inst = trends.get("area_change_inst_km2", np.nan)
    d_area = (
        float(d_area_15)
        if np.isfinite(d_area_15)
        else (float(d_area_inst) if np.isfinite(d_area_inst) else 0.0)
    )

    is_mature_cat = (vertical_cat == 0) or (t_min_k <= cfg.cat_mature_tmin_k)
    has_lgh = lightning_strokes > 0
    has_ot = ot_confirmed_count > 0

    # Cooling: dT_min / dt < -2 K/h (cooling_rate > +2.0 K/h)
    if cooling_rate > 2.0:
        if not is_mature_cat:
            return 1  # Growing
        return 1 if d_area > 0.0 else 2  # Growing if expanding, else Mature

    # Approximately same temperature: |cooling_rate| <= 2.0 K/h
    if abs(cooling_rate) <= 2.0:
        if is_mature_cat:
            return 2  # Mature
        if d_area > 0.0 or is_convective or has_lgh:
            return 1  # Growing
        return 3  # Decaying

    # Warming: dT_min / dt > 0 (cooling_rate < -2.0 K/h)
    warming_rate_k_h = -cooling_rate
    low_warming = warming_rate_k_h <= cfg.low_warming_max_k_per_h
    if not is_mature_cat:
        if has_lgh or (is_convective and low_warming):
            return 1  # Growing
        return 3  # Decaying
    else:
        if has_lgh or (is_convective and low_warming) or has_ot:
            return 2  # Mature
        return 3  # Decaying


def compute_severity_and_hazard(
    is_convective: bool,
    ot_confirmed_count: int,
    trends: dict[str, float],
    lightning_strokes_15m: int = 0,
    lightning_jump_active: bool = False,
    crr_p99_mm_h: float = 0.0,
    high_alt_icing_hazard: bool = False,
) -> tuple[int, int]:
    """Compute NWC SAF Severity (0..4) and Primary Hazard Type (0..7) (ATBD Sec 3.2.1.2.5.10, pp. 51-52).

    Severity scale:
    0 = NIL, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High

    Primary Hazard Type:
    0 = NIL, 1 = turbulence (OTD), 2 = lightning, 3 = icing,
    4 = high ice water content (HIcgHzd), 5 = hail (Lightning Jump), 6 = heavy rain rate
    """
    if not is_convective:
        return 0, 0

    score = 1  # Base convective severity = 1 (Low)
    cool_15 = trends.get("cooling_rate_15m_k_per_h", np.nan)
    cool_inst = trends.get("cooling_rate_k_per_h", np.nan)
    cooling_rate = (
        float(cool_15)
        if np.isfinite(cool_15)
        else (float(cool_inst) if np.isfinite(cool_inst) else 0.0)
    )
    rel_exp = trends.get("rel_expansion_per_h", np.nan)
    if not np.isfinite(rel_exp):
        rel_exp = trends.get("rel_expansion_15m_per_h", 0.0)

    # 1. Overshooting top presence
    if ot_confirmed_count >= 1:
        score += 1
    # 2. High cooling rate thresholds: 20, 50, 100 degC/h (ATBD p. 51)
    if cooling_rate >= 100.0:
        score += 3
    elif cooling_rate >= 50.0:
        score += 2
    elif cooling_rate >= 20.0:
        score += 1
    # 3. Strong horizontal expansion rate: +100% / 5 min = +12.0 / h (ATBD p. 51)
    if np.isfinite(rel_exp) and rel_exp >= 12.0:
        score += 1
    # 4. High lightning activity: 20 and 50 strokes / 15 min
    if lightning_strokes_15m >= 50:
        score += 2
    elif lightning_strokes_15m >= 20:
        score += 1
    # 5. Lightning jump
    if lightning_jump_active:
        score += 2
    # 6. High convective rain rate (>= 30 mm/h)
    if crr_p99_mm_h >= 30.0:
        score += 1
    # 7. High altitude ice crystals hazard
    if high_alt_icing_hazard:
        score += 1

    severity = int(min(4, max(1, score)))

    # Primary hazard code (ATBD Annex Line I ch20, p. 58)
    if lightning_jump_active:
        hazard = 5  # hail
    elif ot_confirmed_count >= 1:
        hazard = 1  # turbulence
    elif crr_p99_mm_h >= 30.0:
        hazard = 6  # heavy rain rate
    elif high_alt_icing_hazard:
        hazard = 4  # high ice water content
    elif lightning_strokes_15m > 0:
        hazard = 2  # lightning
    else:
        hazard = 0

    return severity, hazard


def discriminate_and_classify_cell(
    cell: dict[str, Any],
    prev_cell: dict[str, Any] | None,
    conv_parent_cell: dict[str, Any] | None,
    history: list[dict[str, Any]],
    prev_hierarchy: ThresholdSliceHierarchy | None,
    curr_hierarchy: ThresholdSliceHierarchy | None,
    ot_result: dict[str, Any],
    dt_minutes: float,
    cfg: NwcSafRdtConfig,
    *,
    nwp_mask_value: int = 2,
    lightning_strokes: int = 0,
    cell_lat_deg: float = 45.0,
    calibrated_model: Any | None = None,
    cell_tropo_temp_k: float = 215.0,
) -> dict[str, Any]:
    """Run complete NWC SAF RDT-CW discrimination, declassification, overrides, phase & severity.

    Returns dictionary of discrimination & synthesis attributes, including the 5-class
    semantic segmentation label:
        0: background
        1: non-convective cloud cell
        2: convective cell in triggering / growing phase (Phase 0, 1, 4)
        3: convective cell in mature / decaying phase (Phase 2, 3)
        4: confirmed Overshooting Top (pixel-level override inside convective cell)
    """
    event = str(cell.get("event", "new"))
    trends = compute_temporal_predictors(
        cell, history, prev_cell, prev_hierarchy, curr_hierarchy, dt_minutes, cfg
    )

    # Track age along primary trajectory and age with T_min <= -40 C
    age_total_min = float(len(history) * dt_minutes)
    if cell["t_min_k"] <= cfg.cat_mature_tmin_k:
        prev_age_m40 = (
            float(prev_cell.get("age_below_minus40_min", 0.0))
            if prev_cell is not None
            else 0.0
        )
        age_below_minus40_min = prev_age_m40 + (dt_minutes if prev_cell is not None else 0.0)
    else:
        age_below_minus40_min = 0.0

    vertical_cat = classify_vertical_category(
        cell["t_min_k"], cell["t_bt_k"], age_below_minus40_min, cfg
    )
    prev_vertical_cat = (
        int(prev_cell["vertical_cat"]) if prev_cell is not None and "vertical_cat" in prev_cell else vertical_cat
    )

    if prev_cell is not None and vertical_cat == prev_vertical_cat:
        age_in_cat_min = float(prev_cell.get("age_in_cat_min", 0.0)) + dt_minutes
    else:
        age_in_cat_min = 0.0

    depth_regime, depth_window_min = select_history_depth_regime(
        vertical_cat, age_total_min, age_in_cat_min, scheme=cfg.discri_scheme
    )

    ot_confirmed_count = int(ot_result["confirmed_count"])
    is_convective = False
    conv_diagnosis_code = 0  # Line I ch5 (0: non-conv, 1: stat, 3: declassified, 4: inherited primary, 6: inherited secondary, 11: forced OTD)
    prob_conv = 0.0
    error_likelihood_pct = 100.0
    convective_time_in_cat_min = 0.0
    discri_reason = "non_convective_init"

    # Step 1: Check Convective Inheritance (ATBD Fig. 20, p. 35 & Sec 3.2.1.2.3.4, p. 43)
    if conv_parent_cell is not None and conv_parent_cell.get("is_convective", False):
        is_primary_conv = (
            prev_cell is not None
            and conv_parent_cell.get("cell_id") == prev_cell.get("cell_id")
        )
        # Update convective time counter in vertical category (ATBD p. 43)
        parent_cat = int(conv_parent_cell.get("vertical_cat", vertical_cat))
        if vertical_cat < parent_cat:
            # Ascended into a colder vertical category -> reset convective counter to 0
            convective_time_in_cat_min = 0.0
        else:
            convective_time_in_cat_min = float(
                conv_parent_cell.get("convective_time_in_cat_min", 0.0)
            ) + dt_minutes

        # Step 2: Empirical Declassification Rules (ATBD Sec 3.2.1.2.3.5, pp. 43-44)
        father_severity = int(conv_parent_cell.get("severity", 0))
        cool_rate = trends.get("cooling_rate_15m_k_per_h", np.nan)
        if not np.isfinite(cool_rate):
            cool_rate = trends.get("cooling_rate_k_per_h", 0.0)
        warming_rate_k_h = -float(cool_rate) if np.isfinite(cool_rate) else 0.0

        declassify = False
        # Exception: If father cell had positive severity (Severity >= 1) and OTD/activity persists,
        # ATBD p. 44 protects against premature declassification unless strong warming occurs.
        severity_protected = father_severity >= 2 or ot_confirmed_count > 0

        if not severity_protected:
            if vertical_cat == 5:
                # Warmest category (Cat 5): 30 min hold or immediate on strong warming
                if (
                    warming_rate_k_h >= cfg.declass_strong_warming_k_per_h
                    or convective_time_in_cat_min > cfg.declass_warm_hold_min
                ):
                    declassify = True
            elif vertical_cat in (1, 2, 3, 4):
                # Transition categories (Cat 1-4): 60 min hold if staying in same or warmer category
                if (
                    convective_time_in_cat_min > cfg.declass_transition_hold_min
                    and vertical_cat >= parent_cat
                ) or warming_rate_k_h >= 1.5 * cfg.declass_strong_warming_k_per_h:
                    declassify = True
            elif vertical_cat == 0:
                # Mature category (Cat 0): after 60 min, declassify if T_min > -40 C OR
                # (BTD_max(WV6.2-IR10.8) < -1.0 C and negative 60-min BTD trend)
                if convective_time_in_cat_min > cfg.declass_transition_hold_min:
                    btd_60_trend = trends.get("btd_trend_60m_k", 0.0)
                    cond_warm = cell["t_min_k"] > cfg.cat_mature_tmin_k
                    cond_btd_decay = (
                        cell["btd_wv62_ir108_max_k"] < cfg.declass_mature_btd_wv_ir_k
                        and np.isfinite(btd_60_trend)
                        and btd_60_trend < 0.0
                    )
                    if cond_warm or cond_btd_decay:
                        declassify = True

        if declassify:
            is_convective = False
            conv_diagnosis_code = 3  # Previously convective and declassified
            discri_reason = "declassified_empirical_rule"
        else:
            is_convective = True
            conv_diagnosis_code = 4 if is_primary_conv else 6
            prob_conv = float(conv_parent_cell.get("prob_conv", 0.85))
            error_likelihood_pct = float(100.0 * (1.0 - prob_conv))
            discri_reason = (
                "inherited_primary_father" if is_primary_conv else "inherited_secondary_father"
            )

    # Step 3: Statistical Discrimination if not already convective (ATBD Fig. 20, p. 35)
    if not is_convective and conv_diagnosis_code != 3:
        if cfg.nwp_conv_filter and nwp_mask_value == 0:
            discri_reason = "filtered_nwp_stable"
        else:
            # Ascending/cooling pre-condition (ATBD Fig. 20, p. 35):
            # Must be in Mature category (0 or 1) OR cooling (dT_min < 0) OR ascending to colder category
            dt_inst = trends.get("dt_min_inst_k", np.nan)
            dt_15 = trends.get("cooling_15m_k", np.nan)
            is_cooling = (
                (np.isfinite(dt_15) and dt_15 < 0.0)
                or (np.isfinite(dt_inst) and dt_inst < 0.0)
            )
            is_ascending = prev_cell is not None and vertical_cat < prev_vertical_cat

            if vertical_cat in (0, 1) or is_cooling or is_ascending:
                stat_conv, prob_conv, error_likelihood_pct = evaluate_logistic_discrimination(
                    cell,
                    trends,
                    vertical_cat,
                    depth_regime,
                    cfg,
                    calibrated_model=calibrated_model,
                    cell_tropo_temp_k=cell_tropo_temp_k,
                )
                if stat_conv:
                    is_convective = True
                    conv_diagnosis_code = 1
                    convective_time_in_cat_min = 0.0
                    discri_reason = f"statistical_{cfg.discri_scheme}_cat{vertical_cat}_{depth_regime}"
                else:
                    discri_reason = f"below_stat_threshold_cat{vertical_cat}_{depth_regime}"
            else:
                discri_reason = "not_cooling_or_ascending"

    # Step 4: Forced Convective Diagnosis Overrides ("No" -> "Yes" only, ATBD Sec 3.2.1.2.3.6-8, p. 44)
    if not is_convective:
        if cfg.force_conv_on_otd and ot_confirmed_count >= 1:
            is_convective = True
            conv_diagnosis_code = 11  # Forced by OTD
            prob_conv = max(prob_conv, 0.95)
            error_likelihood_pct = float(100.0 * (1.0 - prob_conv))
            discri_reason = "forced_by_overshooting_top"
        elif lightning_strokes >= cfg.lgh_forced_min_strokes and cfg.lgh_forced_min_strokes > 0:
            is_convective = True
            conv_diagnosis_code = 12  # Forced by Lightning
            prob_conv = max(prob_conv, 0.95)
            error_likelihood_pct = float(100.0 * (1.0 - prob_conv))
            discri_reason = "forced_by_lightning"
        elif (
            cfg.tropical_discri_temp_k is not None
            and abs(float(cell_lat_deg)) <= 30.0
            and cell["t_min_k"] < cfg.tropical_discri_temp_k
        ):
            is_convective = True
            conv_diagnosis_code = 14  # Forced by Tropical T_min
            prob_conv = max(prob_conv, 0.90)
            error_likelihood_pct = float(100.0 * (1.0 - prob_conv))
            discri_reason = "forced_by_tropical_tmin"

    # Step 5: Phase of Development (0..3, ATBD Table 8, p. 51) & Severity (0..4, ATBD pp. 51-52)
    m_phase = int(prev_cell.get('phase_code', 1)) if prev_cell is not None and 'phase_code' in prev_cell else None
    phase_code = compute_phase_of_development(
        event,
        vertical_cat,
        cell["t_min_k"],
        trends,
        is_convective,
        ot_confirmed_count,
        lightning_strokes,
        cfg,
        mother_phase=m_phase,
    )
    severity_code, hazard_code = compute_severity_and_hazard(
        is_convective,
        ot_confirmed_count,
        trends,
        lightning_strokes_15m=lightning_strokes,
    )

    # Map to the full 4-phase official RDT convective lifecycle schema:
    # 0: Background
    # 1: Non-convective cold cloud cell
    # 2: Convective cell — Triggering (Phase 0)
    # 3: Convective cell — Growing (Phase 1)
    # 4: Convective cell — Mature (Phase 2)
    # 5: Convective cell — Decaying (Phase 3)
    # 6: Confirmed Overshooting Top (pixel-level override inside convective cell)
    if not is_convective:
        semantic_class_id = 1
    elif phase_code == 0:
        semantic_class_id = 2  # Triggering
    elif phase_code == 1:
        semantic_class_id = 3  # Growing
    elif phase_code == 2:
        semantic_class_id = 4  # Mature
    else:
        semantic_class_id = 5  # Decaying

    temporal_valid = bool(
        prev_cell is not None
        and not cell["border"]
        and not prev_cell.get("border", False)
    )
    growth_valid = bool(temporal_valid and event == "continuation")

    return {
        "vertical_cat": int(vertical_cat),
        "depth_regime": depth_regime,
        "depth_window_min": float(depth_window_min),
        "age_total_min": age_total_min,
        "age_in_cat_min": age_in_cat_min,
        "age_below_minus40_min": age_below_minus40_min,
        "convective_time_in_cat_min": convective_time_in_cat_min,
        "is_convective": bool(is_convective),
        "conv_diagnosis_code": int(conv_diagnosis_code),
        "prob_conv": float(prob_conv),
        "error_likelihood_pct": float(error_likelihood_pct),
        "phase_code": int(phase_code),
        "severity": int(severity_code),
        "hazard_code": int(hazard_code),
        "class_id": int(semantic_class_id),
        "reason": discri_reason,
        "temporal_valid": temporal_valid,
        "growth_valid": growth_valid,
        "phase_valid": bool(temporal_valid or is_convective),
        **trends,
    }


def forecast_convective_cells(
    cells: dict[int, dict[str, Any]],
    dt_slot_minutes: float,
    cfg: NwcSafRdtConfig,
) -> list[dict[str, Any]]:
    """Lagrangian forecast of RDT-CW cells up to +60 min (ATBD Sec 3.2.1.2.4, pp. 44-45).

    Advects each tracked cell's gravity centre and contour by V(C) * dt_f for
    lead times dt_f in {15, 30, 45, 60} min, with optional radial expansion (DILAT).
    """
    if cfg.fcst_max_min <= 0 or cfg.fcst_step_min <= 0:
        return []
    lead_times_min = list(range(cfg.fcst_step_min, cfg.fcst_max_min + 1, cfg.fcst_step_min))
    forecasts: list[dict[str, Any]] = []
    for cid, cell in cells.items():
        if not cell.get("is_convective", False):
            continue
        vel_per_slot = np.asarray(cell.get("motion_yx", np.zeros(2)), dtype=float)
        vel_per_min = vel_per_slot / max(dt_slot_minutes, 1.0)
        g_yx = np.asarray(cell["weight_centroid_yx"], dtype=float)
        rel_exp_h = float(cell.get("rel_expansion_per_h", 0.0))
        if not np.isfinite(rel_exp_h):
            rel_exp_h = 0.0

        for lead_min in lead_times_min:
            adv_g_yx = g_yx + vel_per_min * lead_min
            scale_area = (
                max(0.25, 1.0 + rel_exp_h * (lead_min / 60.0)) if cfg.dilat else 1.0
            )
            forecasts.append(
                {
                    "cell_id": int(cid),
                    "track_id": int(cell["track_id"]),
                    "lead_time_min": int(lead_min),
                    "forecast_centroid_y": float(adv_g_yx[0]),
                    "forecast_centroid_x": float(adv_g_yx[1]),
                    "forecast_area_km2": float(cell["area_km2"] * scale_area),
                    "phase_code": int(cell["phase_code"]),
                    "severity": int(cell["severity"]),
                }
            )
    return forecasts
