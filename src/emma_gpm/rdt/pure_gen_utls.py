"""SEVIRI-only RDT-CW-inspired teacher with an object-local UTLS proxy.

This module treats :mod:`RDT_original.rdt_nwcsaf` as a read-only algorithmic
reference.  It composes its adaptive thermal hierarchy, four-stage tracking,
GEN discrimination and radial OTD with the project's causal multispectral
UTLS/anvil proxy.  Before thermal segmentation it supplies the Cloud Type
pre-filter expected by RDT through a causal seven-channel IR/WV proxy.  No
ERA5, OPERA, CAL coefficients, lightning, visible channels or parallax
remapping are used.

The semantic output deliberately keeps the detailed RDT lifecycle requested
for the probing dataset::

    0 background
    1 non-convective cold cloud
    2 triggering
    3 growing
    4 mature
    5 decaying
    6 persistent overshooting top

The proxy is a radiometric anvil/UTLS reference, not a WMO tropopause
retrieval.  It is used only after thermal cells have been detected: detection
therefore uses the fixed ATBD BTLIMIT, while proxy confidence gates the hybrid
OT branch and the relative-depth diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from RDT_original.rdt_nwcsaf.config import (
    NwcSafRdtConfig,
    QUALITY_GROWTH_VALID,
    QUALITY_OBSERVATION_VALID,
    QUALITY_PHASE_VALID,
    QUALITY_TEMPORAL_VALID,
)
from RDT_original.rdt_nwcsaf.detection import ThresholdSliceHierarchy, detect_rdt_cells
from RDT_original.rdt_nwcsaf.discrimination import discriminate_and_classify_cell
from RDT_original.rdt_nwcsaf.otd import detect_cell_overshooting_tops
from RDT_original.rdt_nwcsaf.tracking import track_rdt_cells

from .config import RDTTeacherConfig
from .cloud_type_ir import IrCloudTypeConfig, SeviriIrCloudTypeTracker
from .features import FEATURE_NAMES, spectral_features
from .lifecycle import ot_candidates
from .utls_proxy import estimate_proxy, thermal_depth


RDT_ORIGINAL_CLASS_NAMES = {
    0: "background",
    1: "non_convective_cold_cloud",
    2: "triggering",
    3: "growing",
    4: "mature",
    5: "decaying",
    6: "overshooting_top",
}


def default_pure_gen_rdt_config() -> NwcSafRdtConfig:
    """Return the explicit GEN configuration used by the hybrid teacher.

    The v2.1 Case-B ratio of three is selected.  GEN mature-category age is
    45 minutes.  Detection does not receive a scene-global tropopause field:
    its cold limit is the documented fixed BTLIMIT, and the object-local proxy
    is introduced only after segmentation.
    """

    return NwcSafRdtConfig(
        version="rdt_cw_inspired_pure_gen_utls_v1",
        discri_scheme="GEN",
        nwp_conv_filter=False,
        enforce_stretched_mask=True,
        limittropo=False,
        max_base_top_ratio=3.0,
        cat_mature_min_age_min=45.0,
        parallax=False,
    )


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A finite timestamp is required")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _empty_radial_ot(mask: np.ndarray, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    empty = np.zeros(mask.shape, dtype=bool)
    return {
        "candidate_mask": empty,
        "confirmed_mask": empty.copy(),
        "candidates": [] if candidates is None else candidates,
        "confirmed_count": 0,
        "confirmed_pixels": 0,
    }


def _advect_persistence_counts(
    candidate: np.ndarray,
    cell: dict[str, Any],
    previous: dict[str, Any] | None,
    event: str,
    required_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(candidate.shape, dtype=np.uint16)
    if previous is not None and event == "continuation" and "hybrid_ot_count" in previous:
        displacement = np.rint(cell["weight_centroid_yx"] - previous["weight_centroid_yx"]).astype(int)
        old = np.asarray(previous["hybrid_ot_count"], dtype=np.uint16)
        old_bbox = previous["bbox"]
        bbox = cell["bbox"]
        yy, xx = np.where(old > 0)
        ry = yy + old_bbox[0].start + displacement[0] - bbox[0].start
        rx = xx + old_bbox[1].start + displacement[1] - bbox[1].start
        inside = (ry >= 0) & (rx >= 0) & (ry < candidate.shape[0]) & (rx < candidate.shape[1])
        counts[ry[inside], rx[inside]] = old[yy[inside], xx[inside]]
        counts = ndimage.maximum_filter(counts, size=3, mode="constant")
    counts = np.where(
        candidate,
        np.minimum(counts.astype(np.uint32) + 1, required_frames),
        0,
    ).astype(np.uint16)
    return counts, candidate & (counts >= required_frames)


class PureGenUtlsTracker:
    """Causal full-domain tracker for the improved SEVIRI-only teacher."""

    def __init__(
        self,
        *,
        rdt_config: NwcSafRdtConfig | None = None,
        proxy_config: RDTTeacherConfig | None = None,
        cloud_type_config: IrCloudTypeConfig | None = None,
    ) -> None:
        self.rdt_config = rdt_config or default_pure_gen_rdt_config()
        if self.rdt_config.discri_scheme != "GEN":
            raise ValueError("PureGenUtlsTracker requires discri_scheme='GEN'")
        if self.rdt_config.nwp_conv_filter:
            raise ValueError("PureGenUtlsTracker forbids the NWP convective filter")
        if self.rdt_config.limittropo:
            raise ValueError("Detection must not use a scene-global tropopause field")
        self.proxy_config = proxy_config or RDTTeacherConfig()
        self.cloud_type_config = cloud_type_config or IrCloudTypeConfig()
        self.cloud_type_tracker = SeviriIrCloudTypeTracker(self.cloud_type_config)
        self.version = "rdt_cw_inspired_pure_gen_utls_v2"
        payload = {
            "version": self.version,
            "rdt": asdict(self.rdt_config),
            "proxy": asdict(self.proxy_config),
            "cloud_type": asdict(self.cloud_type_config),
            "semantic_classes": RDT_ORIGINAL_CLASS_NAMES,
            "radial_relative_ot_intersection": True,
            "native_satellite_grid": True,
        }
        self.config_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.reset()

    def reset(self) -> None:
        self.previous_labels: np.ndarray | None = None
        self.previous_cells: dict[int, dict[str, Any]] = {}
        self.previous_hierarchy: ThresholdSliceHierarchy | None = None
        self.previous_ir: np.ndarray | None = None
        self.previous_time: pd.Timestamp | None = None
        self.track_histories: dict[int, list[dict[str, Any]]] = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.cloud_type_tracker.reset()

    def _validate_time(self, timestamp: Any) -> pd.Timestamp:
        stamp = _normalise_timestamp(timestamp)
        if self.previous_time is not None:
            delta = stamp - self.previous_time
            expected = pd.to_timedelta(
                int(round(self.rdt_config.expected_cadence_minutes * 60.0)),
                unit="s",
            )
            if delta != expected:
                raise ValueError(
                    "Pure GEN UTLS requires an exact, strictly increasing 5-minute cadence; "
                    f"received {self.previous_time.isoformat()} -> {stamp.isoformat()} ({delta})"
                )
        return stamp

    @staticmethod
    def _cell_coordinate(
        coordinate: np.ndarray | float | None,
        row: int,
        column: int,
        default: float,
        *,
        latitude: bool,
    ) -> float:
        if isinstance(coordinate, np.ndarray):
            if coordinate.ndim == 2:
                value = coordinate[row, column]
            elif latitude:
                value = coordinate[row]
            else:
                value = coordinate[column]
            return float(value) if np.isfinite(value) else default
        if coordinate is None:
            return default
        return float(coordinate)

    def step_frame(
        self,
        frame_7ch: np.ndarray,
        timestamp: Any,
        area_km2: np.ndarray,
        *,
        lat_deg: np.ndarray | float | None = None,
        lon_deg: np.ndarray | float | None = None,
    ) -> dict[str, Any]:
        """Process one north-up/west-left seven-channel Kelvin observation."""

        stamp = self._validate_time(timestamp)
        frame = np.asarray(frame_7ch, dtype=np.float32)
        area = np.asarray(area_km2, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[0] != 7:
            raise ValueError(f"Expected frame shape (7,H,W), got {frame.shape}")
        if area.shape != frame.shape[-2:]:
            raise ValueError(f"Area shape {area.shape} does not match frame {frame.shape[-2:]}")
        height, width = area.shape
        features = spectral_features(frame)
        cloud_type = self.cloud_type_tracker.step(frame, stamp)

        observation_valid = (
            np.isfinite(frame).all(axis=0)
            & np.isfinite(area)
            & (area > 0.0)
            & ((frame >= 100.0) & (frame <= 400.0)).all(axis=0)
        )
        if self.rdt_config.enforce_stretched_mask:
            observation_valid &= area <= (
                self.rdt_config.stretch_area_factor
                * self.rdt_config.nominal_pixel_area_km2
            )

        current_labels, current_cells, current_hierarchy, _detection_valid = detect_rdt_cells(
            frame,
            area,
            self.rdt_config,
            ct_cloudy_mask=cloud_type.opaque_cloud_mask,
            tropopause_temp_k=None,
        )
        parents, _children, edges, primary_parent, events, convective_parent = track_rdt_cells(
            self.previous_cells,
            self.previous_labels,
            self.previous_hierarchy,
            self.previous_ir,
            current_cells,
            current_labels,
            current_hierarchy,
            frame[4],
            area,
            self.rdt_config,
            guess_motion_yx=None,
        )

        fields: dict[str, Any] = {
            "semantic_label": np.zeros((height, width), dtype=np.uint8),
            "cell_id_map": current_labels,
            "track_id_map": np.zeros((height, width), dtype=np.int32),
            "phase_map": np.full((height, width), -1, dtype=np.int8),
            "vertical_cat_map": np.full((height, width), -1, dtype=np.int8),
            "base_threshold_map_k": np.full((height, width), np.nan, dtype=np.float32),
            "top_threshold_map_k": np.full((height, width), np.nan, dtype=np.float32),
            "t_utls_instant_k": np.full((height, width), np.nan, dtype=np.float32),
            "t_utls_proxy_k": np.full((height, width), np.nan, dtype=np.float32),
            "proxy_confidence": np.zeros((height, width), dtype=np.float32),
            "thermal_depth": np.full((height, width), np.nan, dtype=np.float32),
            "plateau_mask": np.zeros((height, width), dtype=bool),
            "radial_ot_mask": np.zeros((height, width), dtype=bool),
            "provisional_ot_mask": np.zeros((height, width), dtype=bool),
            "ot_mask": np.zeros((height, width), dtype=bool),
            "cloud_type": cloud_type.ct_class,
            "opaque_cloud_mask": cloud_type.opaque_cloud_mask,
            "thin_cirrus_mask": cloud_type.thin_cirrus_mask,
            "surface_like_mask": cloud_type.surface_like_mask,
            "surface_reference_k": cloud_type.surface_reference_k,
            "surface_depression_k": cloud_type.surface_depression_k,
            "spectral_cloud_votes": cloud_type.spectral_cloud_votes,
            "valid_observation": observation_valid,
            "quality_mask": np.where(
                observation_valid, QUALITY_OBSERVATION_VALID, 0
            ).astype(np.uint8),
        }
        for name, values in zip(FEATURE_NAMES, features):
            fields[name] = values.astype(np.float32, copy=False)

        new_track_histories: dict[int, list[dict[str, Any]]] = {}
        object_rows: list[dict[str, Any]] = []
        for cell_id, cell in current_cells.items():
            event = events.get(cell_id, "new")
            cell["event"] = event
            cell["parents"] = parents.get(cell_id, [])
            main_parent_id = primary_parent.get(cell_id)
            conv_parent_id = convective_parent.get(cell_id)
            previous = self.previous_cells.get(main_parent_id) if main_parent_id is not None else None
            conv_parent = self.previous_cells.get(conv_parent_id) if conv_parent_id is not None else None

            if previous is not None and event in ("continuation", "merge_main", "split_main"):
                track_id = int(previous["track_id"])
                history = list(self.track_histories.get(track_id, []))
                proxy_history = list(previous.get("proxy_history", []))
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                history = []
                proxy_history = []
            cell["track_id"] = track_id
            cell["primary_parent_id"] = main_parent_id
            cell["convective_parent_id"] = conv_parent_id

            centre_y = int(np.clip(round(float(cell["weight_centroid_yx"][0])), 0, height - 1))
            centre_x = int(np.clip(round(float(cell["weight_centroid_yx"][1])), 0, width - 1))
            cell_lat = self._cell_coordinate(lat_deg, centre_y, centre_x, 45.0, latitude=True)
            cell_lon = self._cell_coordinate(lon_deg, centre_y, centre_x, 10.0, latitude=False)
            cell["lat_deg"] = cell_lat
            cell["lon_deg"] = cell_lon

            bbox = cell["bbox"]
            mask = cell["mask"]
            local_ir = frame[4][bbox]
            local_features = features[:, bbox[0], bbox[1]]
            local_area = area[bbox]
            tendency = (
                local_ir - self.previous_ir[bbox]
                if self.previous_ir is not None
                else None
            )
            support_confidence = 0.5 if cell["border"] else 1.0
            proxy = estimate_proxy(
                local_ir,
                local_features,
                mask,
                local_area,
                float(cell["t_bt_k"]),
                self.proxy_config,
                history=proxy_history,
                tendency=tendency,
                support_confidence=support_confidence,
            )
            proxy_valid = bool(
                proxy.reason == "valid"
                and np.isfinite(proxy.filtered_k)
                and proxy.confidence >= self.proxy_config.min_proxy_confidence
            )
            cell["t_utls_instant_k"] = float(proxy.instant_k)
            cell["t_utls_proxy_k"] = float(proxy.filtered_k)
            cell["proxy_confidence"] = float(proxy.confidence)
            cell["proxy_reason"] = proxy.reason
            cell["proxy_valid"] = proxy_valid
            cell["tropopause_temp_k"] = float(proxy.filtered_k) if proxy_valid else np.nan
            depth = thermal_depth(
                local_ir,
                mask,
                float(cell["t_bt_k"]),
                float(proxy.filtered_k),
                self.proxy_config,
            )
            cell["max_relative_depth"] = (
                float(np.nanmax(depth[mask])) if np.isfinite(depth[mask]).any() else np.nan
            )

            radial = detect_cell_overshooting_tops(
                cell,
                frame,
                area,
                self.rdt_config,
                cell_lat_deg=cell_lat,
                vis_refl_2d=None,
            )
            if not proxy_valid:
                radial_for_discrimination = _empty_radial_ot(mask, radial["candidates"])
            else:
                radial_for_discrimination = radial

            discrimination = discriminate_and_classify_cell(
                cell,
                previous,
                conv_parent,
                history,
                self.previous_hierarchy,
                current_hierarchy,
                radial_for_discrimination,
                self.rdt_config.expected_cadence_minutes,
                self.rdt_config,
                nwp_mask_value=2,
                lightning_strokes=0,
                cell_lat_deg=cell_lat,
                calibrated_model=None,
                cell_tropo_temp_k=float(proxy.filtered_k) if proxy_valid else np.nan,
            )
            cell.update(discrimination)

            relative_candidate = ot_candidates(
                local_ir,
                local_features,
                mask,
                local_area,
                depth,
                proxy,
                self.proxy_config,
                convective=bool(cell["is_convective"]),
            )
            radial_support = ndimage.binary_dilation(
                radial_for_discrimination["confirmed_mask"],
                structure=np.ones((3, 3), dtype=bool),
            )
            hybrid_candidate = relative_candidate & radial_support
            persistence, persistent_ot = _advect_persistence_counts(
                hybrid_candidate,
                cell,
                previous,
                event,
                self.proxy_config.ot_persistence_frames,
            )
            cell["hybrid_ot_count"] = persistence
            cell["ot_confirmed_count"] = int(persistent_ot.any())
            cell["ot_confirmed_pixels"] = int(persistent_ot.sum())
            cell["radial_ot_pixels"] = int(radial_for_discrimination["confirmed_mask"].sum())

            fields["semantic_label"][bbox][mask] = np.uint8(cell["class_id"])
            fields["semantic_label"][bbox][persistent_ot] = np.uint8(6)
            fields["track_id_map"][bbox][mask] = track_id
            fields["phase_map"][bbox][mask] = np.int8(cell["phase_code"])
            fields["vertical_cat_map"][bbox][mask] = np.int8(cell["vertical_cat"])
            fields["base_threshold_map_k"][bbox][mask] = float(cell["t_bt_k"])
            fields["top_threshold_map_k"][bbox][mask] = float(cell["t_st_k"])
            fields["t_utls_instant_k"][bbox][mask] = float(proxy.instant_k)
            fields["t_utls_proxy_k"][bbox][mask] = float(proxy.filtered_k)
            fields["proxy_confidence"][bbox][mask] = float(proxy.confidence)
            fields["thermal_depth"][bbox][mask] = depth[mask]
            fields["plateau_mask"][bbox] |= proxy.plateau
            fields["radial_ot_mask"][bbox] |= radial_for_discrimination["confirmed_mask"]
            fields["provisional_ot_mask"][bbox] |= hybrid_candidate & ~persistent_ot
            fields["ot_mask"][bbox] |= persistent_ot

            quality = QUALITY_OBSERVATION_VALID
            if cell["temporal_valid"]:
                quality |= QUALITY_TEMPORAL_VALID
            if cell["growth_valid"]:
                quality |= QUALITY_GROWTH_VALID
            if cell["phase_valid"]:
                quality |= QUALITY_PHASE_VALID
            fields["quality_mask"][bbox][mask] |= np.uint8(quality)

            snapshot = {
                "frame_index": self.frame_index,
                "timestamp": stamp.isoformat(),
                "cell_id": cell_id,
                "track_id": track_id,
                "event": event,
                "border": cell["border"],
                "seed_yx": cell["seed_yx"],
                "t_bt_k": cell["t_bt_k"],
                "t_st_k": cell["t_st_k"],
                "t_min_k": cell["t_min_k"],
                "t_mean_k": cell["t_mean_k"],
                "contrast_k": cell["contrast_k"],
                "area_km2": cell["area_km2"],
                # The detector exposes a lazy mapping backed by the complete
                # threshold hierarchy.  Histories need only the levels already
                # queried by discrimination; materialising those values avoids
                # retaining one full-domain hierarchy for every past frame.
                "vertical_profile": dict(cell["vertical_profile"]),
                "btd_wv62_ir108_max_k": cell["btd_wv62_ir108_max_k"],
                "btd_wv62_ir108_p90_k": cell["btd_wv62_ir108_p90_k"],
                "cooling_rate_k_per_h": cell["cooling_rate_k_per_h"],
                "is_convective": cell["is_convective"],
                "vertical_cat": cell["vertical_cat"],
                "phase_code": cell["phase_code"],
                "severity": cell["severity"],
            }
            new_track_histories[track_id] = [*history, snapshot][
                -self.rdt_config.history_slots :
            ]
            cell["proxy_history"] = (
                [*proxy_history, float(proxy.instant_k)][-self.proxy_config.proxy_window_frames :]
                if np.isfinite(proxy.instant_k)
                else []
            )
            object_rows.append(
                {
                    "frame": self.frame_index,
                    "time": stamp.isoformat(),
                    "cell_id": cell_id,
                    "track_id": track_id,
                    "event": event,
                    "class_id": int(cell["class_id"]),
                    "is_convective": bool(cell["is_convective"]),
                    "phase_code": int(cell["phase_code"]),
                    "area_km2": float(cell["area_km2"]),
                    "t_min_k": float(cell["t_min_k"]),
                    "t_edge_k": float(cell["t_bt_k"]),
                    "t_utls_proxy_k": float(proxy.filtered_k),
                    "proxy_confidence": float(proxy.confidence),
                    "proxy_reason": proxy.reason,
                    "max_relative_depth": float(cell["max_relative_depth"]),
                    "radial_ot_pixels": int(cell["radial_ot_pixels"]),
                    "provisional_ot_pixels": int((hybrid_candidate & ~persistent_ot).sum()),
                    "ot_pixels": int(persistent_ot.sum()),
                    "reason": cell["reason"],
                    "config_hash": self.config_hash,
                }
            )

        self.previous_labels = current_labels
        self.previous_cells = current_cells
        self.previous_hierarchy = current_hierarchy
        self.previous_ir = frame[4].copy()
        self.previous_time = stamp
        self.track_histories = new_track_histories
        self.frame_index += 1

        fields.update(
            {
                "cells": current_cells,
                "objects": object_rows,
                "tracking_edges": edges,
                "time": stamp,
                "version": self.version,
                "config_hash": self.config_hash,
            }
        )
        return fields


def run_pure_gen_utls_sequence(
    clip_tchw: np.ndarray,
    times: list[Any] | np.ndarray,
    area_km2: np.ndarray,
    *,
    lat_deg: np.ndarray | float | None = None,
    lon_deg: np.ndarray | float | None = None,
    rdt_config: NwcSafRdtConfig | None = None,
    proxy_config: RDTTeacherConfig | None = None,
    cloud_type_config: IrCloudTypeConfig | None = None,
) -> dict[str, Any]:
    """Run the improved teacher over a small in-memory sequence."""

    clip = np.asarray(clip_tchw, dtype=np.float32)
    if clip.ndim != 4 or clip.shape[1] != 7:
        raise ValueError(f"Expected (T,7,H,W), got {clip.shape}")
    if len(times) != len(clip):
        raise ValueError("times and clip must have the same number of frames")
    tracker = PureGenUtlsTracker(
        rdt_config=rdt_config,
        proxy_config=proxy_config,
        cloud_type_config=cloud_type_config,
    )
    frames = [
        tracker.step_frame(
            clip[index],
            times[index],
            area_km2,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )
        for index in range(len(clip))
    ]
    stack_names = (
        "semantic_label",
        "cell_id_map",
        "track_id_map",
        "phase_map",
        "vertical_cat_map",
        "base_threshold_map_k",
        "top_threshold_map_k",
        "t_utls_instant_k",
        "t_utls_proxy_k",
        "proxy_confidence",
        "thermal_depth",
        "plateau_mask",
        "radial_ot_mask",
        "provisional_ot_mask",
        "ot_mask",
        "cloud_type",
        "opaque_cloud_mask",
        "thin_cirrus_mask",
        "surface_like_mask",
        "surface_reference_k",
        "surface_depression_k",
        "spectral_cloud_votes",
        "valid_observation",
        "quality_mask",
    )
    return {
        **{name: np.stack([frame[name] for frame in frames]) for name in stack_names},
        "frame_records": frames,
        "object_table": [row for frame in frames for row in frame["objects"]],
        "version": tracker.version,
        "config_hash": tracker.config_hash,
        "semantic_class_names": RDT_ORIGINAL_CLASS_NAMES,
    }
