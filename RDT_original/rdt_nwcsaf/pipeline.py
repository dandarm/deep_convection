"""Complete NWC SAF GEO-RDT-CW Causal Pipeline & Sequence Runner.

Connects all 6 stages of the official EUMETSAT NWC SAF RDT-CW algorithm:
1. Pre-processing & stretched-pixel filtering
2. Adaptive multi-threshold detection & single/multi-tower contour refinement (Lines S, S2, H, X)
3. 4-step tracking with forward advection, local IR10.8 cross-correlation, backward advection,
   small-cell 1-pixel enlargement, and common-threshold main father selection (Line T)
4. Two-step Overshooting Top Detection (OTD, Line O)
5. 6-category vertical stratification, convective inheritance, empirical declassification,
   5-channel IRxWVx logistic regression discrimination, Phase of Development (0..4),
   and Severity (0..4) (Line I)
6. Lagrangian cell advection forecasting (+15, +30, +45, +60 min)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .config import (
    NwcSafRdtConfig,
    QUALITY_GROWTH_VALID,
    QUALITY_OBSERVATION_VALID,
    QUALITY_PHASE_VALID,
    QUALITY_TEMPORAL_VALID,
    SEVIRI_7CH_NAMES,
)
from .detection import ThresholdSliceHierarchy, detect_rdt_cells
from .discrimination import discriminate_and_classify_cell, forecast_convective_cells
from .otd import detect_cell_overshooting_tops
from .tracking import track_rdt_cells


def _to_timestamp_minutes(ts: Any, index: int, default_cadence_min: float) -> float:
    """Convert numpy datetime64 / pandas Timestamp / ISO string into elapsed minutes."""
    if ts is None:
        return float(index * default_cadence_min)
    try:
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp() / 60.0)
        s = str(ts).rstrip("Z")
        ts64 = np.datetime64(s, "ns")
        if np.isnat(ts64):
            return float(index * default_cadence_min)
        return float(ts64.astype("int64") / (60.0 * 1e9))
    except Exception:
        return float(index * default_cadence_min)


class NwcSafRdtTracker:
    """Stateful causal tracker implementing the full NWC SAF GEO-RDT-CW v5.0 algorithm."""

    def __init__(
        self,
        cfg: NwcSafRdtConfig | None = None,
        *,
        calibrated_model: Any | None = None,
        use_ancillary_ct: bool = False,
        use_hrw_amv: bool = False,
        use_parallax_correction: bool = False,
    ) -> None:
        self.cfg = cfg or NwcSafRdtConfig()
        self.calibrated_model = calibrated_model
        self.use_ancillary_ct = use_ancillary_ct
        self.use_hrw_amv = use_hrw_amv
        self.use_parallax_correction = use_parallax_correction
        self.reset()

    def reset(self) -> None:
        """Reset all tracking state and trajectory history buffers."""
        self.prev_labels: np.ndarray | None = None
        self.prev_cells: dict[int, dict[str, Any]] = {}
        self.prev_hierarchy: ThresholdSliceHierarchy | None = None
        self.prev_ir: np.ndarray | None = None
        self.prev_wv062: np.ndarray | None = None
        self.prev_time_min: float | None = None
        self.track_histories: dict[int, list[dict[str, Any]]] = {}
        self.next_track_id: int = 1
        self.frame_index: int = 0

    def step_frame(
        self,
        frame_7ch: np.ndarray,
        timestamp: Any = None,
        area_km2: np.ndarray | None = None,
        *,
        lat_deg: np.ndarray | float | None = None,
        lon_deg: np.ndarray | float | None = None,
        tropopause_temp_k: np.ndarray | float | None = None,
        cloud_mask: np.ndarray | None = None,
        nwp_instability_mask: np.ndarray | None = None,
        vis_refl_2d: np.ndarray | None = None,
        guess_motion_yx: np.ndarray | None = None,
        lightning_counts: dict[int, int] | None = None,
        u700_ms: np.ndarray | float | None = None,
        v700_ms: np.ndarray | float | None = None,
        t_surf_k: np.ndarray | float | None = None,
    ) -> dict[str, Any]:
        """Process a single 7-channel SEVIRI slot (7, H, W) in Kelvin."""
        frame = np.asarray(frame_7ch, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[0] != len(SEVIRI_7CH_NAMES):
            raise ValueError(
                f"Expected frame_7ch shape (7, H, W), got {frame.shape}"
            )
        _, h, w = frame.shape
        area = (
            np.asarray(area_km2, dtype=np.float32)
            if area_km2 is not None
            else np.full((h, w), 25.0, dtype=np.float32)
        )

        curr_time_min = _to_timestamp_minutes(
            timestamp, self.frame_index, self.cfg.expected_cadence_minutes
        )
        if self.prev_time_min is None:
            dt_minutes = float(self.cfg.expected_cadence_minutes)
            has_valid_prev = False
        else:
            dt_minutes = float(curr_time_min - self.prev_time_min)
            # Reset tracking continuity if gap is invalid or exceeds 3 slots (15 min)
            has_valid_prev = 0.5 <= dt_minutes <= (3.0 * self.cfg.expected_cadence_minutes + 1.0)
            if not has_valid_prev:
                self.prev_labels = None
                self.prev_cells = {}
                self.prev_hierarchy = None
                self.prev_ir = None
                self.prev_wv062 = None
                self.track_histories.clear()
                dt_minutes = float(self.cfg.expected_cadence_minutes)

        # Optional NWC SAF GEO-CT radiative cloud mask & thin cirrus filtering
        if self.use_ancillary_ct and cloud_mask is None:
            from .ancillary import compute_nwcsaf_cloud_type

            ct_res = compute_nwcsaf_cloud_type(
                frame[4],
                ir120=frame[5],
                ir087=frame[2],
                wv062=frame[0],
                wv073=frame[1],
                t_tropo_k=tropopause_temp_k if tropopause_temp_k is not None else 215.0,
                t_surf_k=t_surf_k if t_surf_k is not None else 292.0,
            )
            cloud_mask = ct_res.opaque_cloud_mask

        # Optional NWC SAF GEO-HRW + NWP 700 hPa first-guess motion field
        if self.use_hrw_amv and guess_motion_yx is None:
            from .ancillary import compute_hrw_amv_guess_field

            dy_px, dx_px = compute_hrw_amv_guess_field(
                self.prev_ir if has_valid_prev else None,
                frame[4],
                prev_wv062=self.prev_wv062 if has_valid_prev else None,
                curr_wv062=frame[0],
                u700_ms=u700_ms,
                v700_ms=v700_ms,
                dt_seconds=dt_minutes * 60.0,
            )
            guess_motion_yx = np.stack([dy_px, dx_px], axis=0)

        # Stage 1 & 2: Detection of cloud towers and single/multi-tower contour refinement
        curr_labels, curr_cells, curr_hierarchy, valid_obs_mask = detect_rdt_cells(
            frame,
            area,
            self.cfg,
            ct_cloudy_mask=cloud_mask,
            tropopause_temp_k=tropopause_temp_k,
        )

        # Stage 3: 4-step tracking & trajectory linking
        parents, children, edges, primary_parent, events, convective_parent = track_rdt_cells(
            self.prev_cells if has_valid_prev else {},
            self.prev_labels if has_valid_prev else None,
            self.prev_hierarchy if has_valid_prev else None,
            self.prev_ir if has_valid_prev else None,
            curr_cells,
            curr_labels,
            curr_hierarchy,
            frame[4],
            area,
            self.cfg,
            guess_motion_yx=guess_motion_yx,
        )

        # Prepare output 2D maps
        semantic_label = np.zeros((h, w), dtype=np.uint8)
        track_id_map = np.zeros((h, w), dtype=np.int32)
        phase_map = np.full((h, w), -1, dtype=np.int8)
        severity_map = np.zeros((h, w), dtype=np.uint8)
        vertical_cat_map = np.full((h, w), -1, dtype=np.int8)
        base_threshold_map_k = np.full((h, w), np.nan, dtype=np.float32)
        top_threshold_map_k = np.full((h, w), np.nan, dtype=np.float32)
        ot_candidate_mask = np.zeros((h, w), dtype=bool)
        ot_confirmed_mask = np.zeros((h, w), dtype=bool)
        quality_mask = np.where(
            valid_obs_mask, QUALITY_OBSERVATION_VALID, 0
        ).astype(np.uint8)

        new_track_histories: dict[int, list[dict[str, Any]]] = {}

        # Stage 4 & 5: Per-cell OTD, Convective Discrimination, Phase of Development & Severity
        for cid, cell in curr_cells.items():
            event = events.get(cid, "new")
            cell["event"] = event
            cell["parents"] = parents.get(cid, [])
            p_main_id = primary_parent.get(cid)
            p_conv_id = convective_parent.get(cid)

            prev_cell = (
                self.prev_cells.get(p_main_id) if p_main_id is not None else None
            )
            conv_parent_cell = (
                self.prev_cells.get(p_conv_id) if p_conv_id is not None else None
            )

            if prev_cell is not None and event in ("continuation", "merge_main", "split_main"):
                tid = int(prev_cell["track_id"])
                history = list(self.track_histories.get(tid, []))
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                history = []

            cell["track_id"] = tid
            cell["primary_parent_id"] = p_main_id
            cell["convective_parent_id"] = p_conv_id

            # Resolve cell latitude / longitude if coordinate grids were provided
            cy_f, cx_f = float(cell["weight_centroid_yx"][0]), float(cell["weight_centroid_yx"][1])
            cy_i = int(np.clip(round(cy_f), 0, h - 1))
            cx_i = int(np.clip(round(cx_f), 0, w - 1))
            if isinstance(lat_deg, np.ndarray):
                cell_lat = float(lat_deg[cy_i, cx_i]) if lat_deg.ndim == 2 else float(lat_deg[cy_i])
            elif lat_deg is not None:
                cell_lat = float(lat_deg)
            else:
                cell_lat = 45.0

            if isinstance(lon_deg, np.ndarray):
                cell_lon = float(lon_deg[cy_i, cx_i]) if lon_deg.ndim == 2 else float(lon_deg[cx_i])
            elif lon_deg is not None:
                cell_lon = float(lon_deg)
            else:
                cell_lon = 10.0

            cell["lat_deg"] = cell_lat
            cell["lon_deg"] = cell_lon

            # Stage 4: Two-step Overshooting Top Detection (OTD)
            ot_res = detect_cell_overshooting_tops(
                cell,
                frame,
                area,
                self.cfg,
                cell_lat_deg=cell_lat,
                vis_refl_2d=vis_refl_2d,
            )
            cell["ot_confirmed_count"] = int(ot_res["confirmed_count"])
            cell["ot_confirmed_pixels"] = int(ot_res["confirmed_pixels"])
            cell["ot_candidates"] = ot_res["candidates"]

            # NWP instability mask value over the cell (0=stable, 1=unstable, 2=missing)
            if nwp_instability_mask is not None:
                loc_nwp = nwp_instability_mask[cell["bbox"]][cell["mask"]]
                nwp_val = int(np.max(loc_nwp)) if len(loc_nwp) > 0 else 2
            else:
                nwp_val = 2

            # Local tropopause temperature at cell centroid/core
            if isinstance(tropopause_temp_k, np.ndarray):
                loc_tropo = tropopause_temp_k[cell["bbox"]][cell["mask"]]
                cell_tropo = float(np.nanmedian(loc_tropo)) if len(loc_tropo) > 0 else 215.0
            elif tropopause_temp_k is not None:
                cell_tropo = float(tropopause_temp_k)
            else:
                cell_tropo = 215.0
            cell["tropopause_temp_k"] = cell_tropo

            lgh_strokes = int((lightning_counts or {}).get(cid, 0))

            # Stage 5: Convective Discrimination, Declassification, Overrides, Phase & Severity
            discri = discriminate_and_classify_cell(
                cell,
                prev_cell,
                conv_parent_cell,
                history,
                self.prev_hierarchy if has_valid_prev else None,
                curr_hierarchy,
                ot_res,
                dt_minutes,
                self.cfg,
                nwp_mask_value=nwp_val,
                lightning_strokes=lgh_strokes,
                cell_lat_deg=cell_lat,
                calibrated_model=self.calibrated_model,
                cell_tropo_temp_k=cell_tropo,
            )
            cell.update(discri)

            # Update trajectory history buffer (keep up to cfg.history_slots frames = 90 min)
            snapshot = {
                "frame_index": self.frame_index,
                "timestamp": timestamp,
                "cell_id": cid,
                "track_id": tid,
                "event": event,
                "border": cell["border"],
                "seed_yx": cell["seed_yx"],
                "t_bt_k": cell["t_bt_k"],
                "t_st_k": cell["t_st_k"],
                "t_min_k": cell["t_min_k"],
                "t_mean_k": cell["t_mean_k"],
                "contrast_k": cell["contrast_k"],
                "area_km2": cell["area_km2"],
                "vertical_profile": cell["vertical_profile"],
                "btd_wv62_ir108_max_k": cell["btd_wv62_ir108_max_k"],
                "btd_wv62_ir108_p90_k": cell["btd_wv62_ir108_p90_k"],
                "cooling_rate_k_per_h": cell["cooling_rate_k_per_h"],
                "is_convective": cell["is_convective"],
                "vertical_cat": cell["vertical_cat"],
                "phase_code": cell["phase_code"],
                "severity": cell["severity"],
            }
            new_hist = [*history, snapshot][-self.cfg.history_slots :]
            new_track_histories[tid] = new_hist

            # Paint 2D output rasters
            bbox = cell["bbox"]
            mask = cell["mask"]
            track_id_map[bbox][mask] = tid
            phase_map[bbox][mask] = int(cell["phase_code"])
            severity_map[bbox][mask] = int(cell["severity"])
            vertical_cat_map[bbox][mask] = int(cell["vertical_cat"])
            base_threshold_map_k[bbox][mask] = float(cell["t_bt_k"])
            top_threshold_map_k[bbox][mask] = float(cell["t_st_k"])
            semantic_label[bbox][mask] = int(cell["class_id"])

            ot_cand_local = ot_res["candidate_mask"]
            ot_conf_local = ot_res["confirmed_mask"]
            ot_candidate_mask[bbox][ot_cand_local] = True
            ot_confirmed_mask[bbox][ot_conf_local] = True

            # Confirmed Overshooting Top pixels inside convective cells receive semantic class 4
            if cell["is_convective"] and ot_conf_local.any():
                semantic_label[bbox][ot_conf_local] = 4

            q_bits = QUALITY_OBSERVATION_VALID
            if cell["temporal_valid"]:
                q_bits |= QUALITY_TEMPORAL_VALID
            if cell["growth_valid"]:
                q_bits |= QUALITY_GROWTH_VALID
            if cell["phase_valid"]:
                q_bits |= QUALITY_PHASE_VALID
            quality_mask[bbox][mask] |= np.uint8(q_bits)

        # Stage 6: Lagrangian forecast (+15, +30, +45, +60 min)
        forecasts = forecast_convective_cells(curr_cells, dt_minutes, self.cfg)

        # Optional 3D Geostationary Parallax Correction to ground truth coordinates
        if self.use_parallax_correction and isinstance(lat_deg, np.ndarray) and isinstance(lon_deg, np.ndarray):
            from .ancillary import apply_parallax_correction_2d

            tsurf_val = t_surf_k if t_surf_k is not None else 292.0
            ttropo_val = tropopause_temp_k if tropopause_temp_k is not None else 215.0
            semantic_label = apply_parallax_correction_2d(
                semantic_label, frame[4], lat_deg, lon_deg,
                t_surf_k=tsurf_val, t_tropo_k=ttropo_val, fill_value=0,
            )
            track_id_map = apply_parallax_correction_2d(
                track_id_map, frame[4], lat_deg, lon_deg,
                t_surf_k=tsurf_val, t_tropo_k=ttropo_val, fill_value=0,
            )
            phase_map = apply_parallax_correction_2d(
                phase_map, frame[4], lat_deg, lon_deg,
                t_surf_k=tsurf_val, t_tropo_k=ttropo_val, fill_value=-1,
            )
            severity_map = apply_parallax_correction_2d(
                severity_map, frame[4], lat_deg, lon_deg,
                t_surf_k=tsurf_val, t_tropo_k=ttropo_val, fill_value=0,
            )
            ot_confirmed_mask = apply_parallax_correction_2d(
                ot_confirmed_mask.astype(np.uint8), frame[4], lat_deg, lon_deg,
                t_surf_k=tsurf_val, t_tropo_k=ttropo_val, fill_value=0,
            ).astype(bool)

        # Advance state to next slot
        self.prev_labels = curr_labels
        self.prev_cells = curr_cells
        self.prev_hierarchy = curr_hierarchy
        self.prev_ir = frame[4].copy()
        self.prev_wv062 = frame[0].copy()
        self.prev_time_min = curr_time_min
        self.track_histories = new_track_histories
        self.frame_index += 1

        return {
            "semantic_label": semantic_label,
            "cell_id_map": curr_labels,
            "track_id_map": track_id_map,
            "phase_map": phase_map,
            "severity_map": severity_map,
            "vertical_cat_map": vertical_cat_map,
            "base_threshold_map_k": base_threshold_map_k,
            "top_threshold_map_k": top_threshold_map_k,
            "ot_candidate_mask": ot_candidate_mask,
            "ot_confirmed_mask": ot_confirmed_mask,
            "quality_mask": quality_mask,
            "cells": curr_cells,
            "forecasts": forecasts,
            "tracking_edges": edges,
        }


def run_nwcsaf_rdt_sequence(
    clip_tchw: np.ndarray,
    times: np.ndarray | list[Any] | None = None,
    area_km2: np.ndarray | None = None,
    cfg: NwcSafRdtConfig | None = None,
    *,
    lat_deg: np.ndarray | float | None = None,
    lon_deg: np.ndarray | float | None = None,
    tropopause_temp_k: np.ndarray | float | None = None,
    cloud_mask: np.ndarray | None = None,
    nwp_instability_mask: np.ndarray | None = None,
    vis_refl_thw: np.ndarray | None = None,
    calibrated_model: Any | None = None,
    use_ancillary_ct: bool = False,
    use_hrw_amv: bool = False,
    use_parallax_correction: bool = False,
    u700_ms: np.ndarray | float | None = None,
    v700_ms: np.ndarray | float | None = None,
    t_surf_k: np.ndarray | float | None = None,
) -> dict[str, Any]:
    """Run the complete NWC SAF GEO-RDT-CW algorithm over a (T, 7, H, W) SEVIRI sequence.

    Returns stacked 3D (T, H, W) arrays and per-slot object & forecast tables.
    """
    clip = np.asarray(clip_tchw, dtype=np.float32)
    if clip.ndim != 4 or clip.shape[1] != len(SEVIRI_7CH_NAMES):
        raise ValueError(
            f"Expected clip_tchw shape (T, 7, H, W), got {clip.shape}"
        )
    num_t, _, h, w = clip.shape
    tracker = NwcSafRdtTracker(
        cfg,
        calibrated_model=calibrated_model,
        use_ancillary_ct=use_ancillary_ct,
        use_hrw_amv=use_hrw_amv,
        use_parallax_correction=use_parallax_correction,
    )

    semantic_labels = np.zeros((num_t, h, w), dtype=np.uint8)
    cell_id_maps = np.zeros((num_t, h, w), dtype=np.int32)
    track_id_maps = np.zeros((num_t, h, w), dtype=np.int32)
    phase_maps = np.full((num_t, h, w), -1, dtype=np.int8)
    severity_maps = np.zeros((num_t, h, w), dtype=np.uint8)
    vertical_cat_maps = np.full((num_t, h, w), -1, dtype=np.int8)
    base_threshold_maps = np.full((num_t, h, w), np.nan, dtype=np.float32)
    top_threshold_maps = np.full((num_t, h, w), np.nan, dtype=np.float32)
    ot_candidate_masks = np.zeros((num_t, h, w), dtype=bool)
    ot_confirmed_masks = np.zeros((num_t, h, w), dtype=bool)
    quality_masks = np.zeros((num_t, h, w), dtype=np.uint8)

    frame_records: list[dict[str, Any]] = []
    cell_table: list[dict[str, Any]] = []
    forecast_table: list[dict[str, Any]] = []

    for t_idx in range(num_t):
        ts = times[t_idx] if times is not None else None
        tropo_t = (
            tropopause_temp_k[t_idx]
            if isinstance(tropopause_temp_k, np.ndarray) and tropopause_temp_k.ndim == 3
            else tropopause_temp_k
        )
        cmask_t = (
            cloud_mask[t_idx]
            if isinstance(cloud_mask, np.ndarray) and cloud_mask.ndim == 3
            else cloud_mask
        )
        nwp_t = (
            nwp_instability_mask[t_idx]
            if isinstance(nwp_instability_mask, np.ndarray) and nwp_instability_mask.ndim == 3
            else nwp_instability_mask
        )
        vis_t = (
            vis_refl_thw[t_idx]
            if isinstance(vis_refl_thw, np.ndarray) and vis_refl_thw.ndim == 3
            else None
        )
        u700_t = (
            u700_ms[t_idx]
            if isinstance(u700_ms, np.ndarray) and u700_ms.ndim == 3
            else u700_ms
        )
        v700_t = (
            v700_ms[t_idx]
            if isinstance(v700_ms, np.ndarray) and v700_ms.ndim == 3
            else v700_ms
        )
        tsurf_t = (
            t_surf_k[t_idx]
            if isinstance(t_surf_k, np.ndarray) and t_surf_k.ndim == 3
            else t_surf_k
        )

        step_out = tracker.step_frame(
            clip[t_idx],
            ts,
            area_km2,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tropopause_temp_k=tropo_t,
            cloud_mask=cmask_t,
            nwp_instability_mask=nwp_t,
            vis_refl_2d=vis_t,
            u700_ms=u700_t,
            v700_ms=v700_t,
            t_surf_k=tsurf_t,
        )

        semantic_labels[t_idx] = step_out["semantic_label"]
        cell_id_maps[t_idx] = step_out["cell_id_map"]
        track_id_maps[t_idx] = step_out["track_id_map"]
        phase_maps[t_idx] = step_out["phase_map"]
        severity_maps[t_idx] = step_out["severity_map"]
        vertical_cat_maps[t_idx] = step_out["vertical_cat_map"]
        base_threshold_maps[t_idx] = step_out["base_threshold_map_k"]
        top_threshold_maps[t_idx] = step_out["top_threshold_map_k"]
        ot_candidate_masks[t_idx] = step_out["ot_candidate_mask"]
        ot_confirmed_masks[t_idx] = step_out["ot_confirmed_mask"]
        quality_masks[t_idx] = step_out["quality_mask"]

        frame_records.append(step_out)
        for cid, cdict in step_out["cells"].items():
            row = {
                k: v
                for k, v in cdict.items()
                if k not in ("mask", "st_mask", "bbox", "vertical_profile", "ot_candidates")
            }
            row["frame_index"] = t_idx
            row["timestamp"] = str(ts) if ts is not None else ""
            cell_table.append(row)

        for fcst in step_out["forecasts"]:
            frow = dict(fcst)
            frow["frame_index"] = t_idx
            frow["timestamp"] = str(ts) if ts is not None else ""
            forecast_table.append(frow)

    return {
        "semantic_label": semantic_labels,
        "cell_id_map": cell_id_maps,
        "track_id_map": track_id_maps,
        "phase_map": phase_maps,
        "severity_map": severity_maps,
        "vertical_cat_map": vertical_cat_maps,
        "base_threshold_map_k": base_threshold_maps,
        "top_threshold_map_k": top_threshold_maps,
        "ot_candidate_mask": ot_candidate_masks,
        "ot_confirmed_mask": ot_confirmed_masks,
        "quality_mask": quality_masks,
        "frame_records": frame_records,
        "cell_table": cell_table,
        "forecast_table": forecast_table,
    }
