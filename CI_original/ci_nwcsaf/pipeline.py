"""Stateful pipeline processor for NWC SAF GEO-CI (Convection Initiation).

References:
- ATBD v2.1 (Issue 2.1)
- User Manual v1.0 (Issue 1.0)
- NWC SAF GEO v2025 CI Product Description
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import NwcSafCiConfig, CHANNEL_INDICES, CiProbabilityClass
from .candidate_selection import select_eligible_ci_pixels, compute_pure_ir_cloud_mask
from .detection_tracking import detect_warm_cloud_cells, WarmCellTracker, CiCloudCell
from .interest_fields import compute_ci_interest_fields, count_relevant_parameters, CiInterestFields
from .discrimination import evaluate_ci_diagnosis, enforce_lead_time_monotonicity
from .forward_scheme import generate_ci_forecast_horizons
from .ancillary import compute_pure_ir_amv_motion


@dataclass
class NwcSafCiStepResult:
    """Output containers and intermediate diagnostics from a single CI step."""
    ci_prob30: np.ndarray             # uint8 array of classes 0..4 (0-30 min)
    ci_prob60: np.ndarray             # uint8 array of classes 0..4 (0-60 min)
    ci_prob90: np.ndarray             # uint8 array of classes 0..4 (0-90 min)
    ci_status_flag: int               # Bitmask of used inputs (Table 7)
    pre_ci_mask: np.ndarray           # bool array of pre-CI pixels
    eligible_mask: np.ndarray         # bool array of eligible-CI pixels
    detected_cells: List[CiCloudCell] # List of detected warm cells
    motion_u: np.ndarray              # 2D zonal velocity (pixels/minute)
    motion_v: np.ndarray              # 2D meridional velocity (pixels/minute)
    n_growth: np.ndarray              # uint8 array of growth criteria (0..3)
    n_glaciation: np.ndarray          # uint8 array of glaciation criteria (0..3)
    n_height: np.ndarray              # uint8 array of height criteria (0..4)
    base_diagnosis: np.ndarray        # un-advected base probability classes


class NwcSafCiTracker:
    """Stateful sequential processor for EUMETSAT NWC SAF GEO-CI."""

    def __init__(self, config: Optional[NwcSafCiConfig] = None):
        self.config = config or NwcSafCiConfig()
        self.cell_tracker = WarmCellTracker(self.config)
        self.previous_cube: Optional[np.ndarray] = None
        self.previous_time: Optional[pd.Timestamp] = None
        self.last_motion_u: Optional[np.ndarray] = None
        self.last_motion_v: Optional[np.ndarray] = None
        self.last_amv_u: Optional[np.ndarray] = None
        self.last_amv_v: Optional[np.ndarray] = None
        self.last_amv_time: Optional[pd.Timestamp] = None

    def step(
        self,
        seviri_cube: np.ndarray,
        timestamp: Optional[Union[pd.Timestamp, str]] = None,
        cloud_mask: Optional[np.ndarray] = None,
        nwp_mask: Optional[np.ndarray] = None,
        delta_t_min_override: Optional[float] = None,
    ) -> NwcSafCiStepResult:
        """Processes one temporal slot of 7-channel SEVIRI observation.

        Parameters
        ----------
        seviri_cube : np.ndarray
            Array of shape (7, H, W) in Kelvin.
        timestamp : Optional[Union[pd.Timestamp, str]]
            Observation timestamp for time delta estimation.
        cloud_mask : Optional[np.ndarray]
            Cloud mask (H, W). If None, calculated via pure-IR split-window.
        nwp_mask : Optional[np.ndarray]
            NWP convective mask (0: stable, 1: unclear, 2: unstable).
        delta_t_min_override : Optional[float]
            Explicit time step in minutes (default 15.0 if cannot be inferred).

        Returns
        -------
        NwcSafCiStepResult containing ci_prob30, ci_prob60, ci_prob90, and diagnostics.
        """
        if seviri_cube.ndim != 3 or seviri_cube.shape[0] != 7:
            raise ValueError(f"Expected seviri_cube of shape (7, H, W), got {seviri_cube.shape}")

        _, h, w = seviri_cube.shape

        # 1. Determine elapsed time delta
        if delta_t_min_override is not None:
            delta_t_min = float(delta_t_min_override)
        elif timestamp is not None and self.previous_time is not None:
            t_curr = pd.Timestamp(timestamp)
            delta_t_min = float((t_curr - self.previous_time).total_seconds() / 60.0)
            if delta_t_min <= 0.0:
                delta_t_min = 15.0
        else:
            delta_t_min = 15.0

        # 2. Stage 1: Candidate Selection & Eligible-CI Filtering
        eligible_mask, resolved_cloud_mask = select_eligible_ci_pixels(
            seviri_cube,
            config=self.config,
            cloud_mask=cloud_mask,
            nwp_mask=nwp_mask,
        )

        # 3. Stage 2: Guess Motion Field
        time_since_amv = (
            (pd.Timestamp(timestamp) - self.last_amv_time).total_seconds() / 60.0
            if (timestamp is not None and self.last_amv_time is not None)
            else 15.0
        )
        if self.previous_cube is not None and (self.last_amv_u is None or time_since_amv >= 14.0):
            # Autonomous pure-IR AMVs via cross-correlation (updated every ~15 min)
            amv_u, amv_v = compute_pure_ir_amv_motion(
                seviri_cube[CHANNEL_INDICES["IR_108"]],
                self.previous_cube[CHANNEL_INDICES["IR_108"]],
                delta_t_min=delta_t_min,
            )
            self.last_amv_u = amv_u
            self.last_amv_v = amv_v
            if timestamp is not None:
                self.last_amv_time = pd.Timestamp(timestamp)
        elif self.last_amv_u is not None and self.last_amv_v is not None:
            amv_u, amv_v = self.last_amv_u, self.last_amv_v
        elif self.last_motion_u is not None and self.last_motion_v is not None:
            amv_u, amv_v = self.last_motion_u, self.last_motion_v
        else:
            amv_u = np.full((h, w), self.config.default_u_mps, dtype=np.float32)
            amv_v = np.full((h, w), self.config.default_v_mps, dtype=np.float32)

        # 4. Stage 3: Warm Cloud Cell Detection & Tracking
        detected_cells = detect_warm_cloud_cells(
            seviri_cube[CHANNEL_INDICES["IR_108"]],
            config=self.config,
            eligible_mask=eligible_mask,
        )
        motion_u, motion_v = self.cell_tracker.track_and_update_motion(
            detected_cells,
            shape=(h, w),
            delta_t_min=delta_t_min,
            guess_u=amv_u,
            guess_v=amv_v,
        )

        # 5. Stage 4: Interest Fields (BTDs and Lagrangian Trends)
        fields = compute_ci_interest_fields(
            current_cube=seviri_cube,
            past_cube=self.previous_cube,
            motion_u_px_per_min=motion_u,
            motion_v_px_per_min=motion_v,
            delta_t_min=delta_t_min,
            config=self.config,
        )

        # 6. Stage 5: Criteria Counting & Pre-CI Mask
        n_growth, n_glaciation, n_height, pre_ci_mask = count_relevant_parameters(
            fields=fields,
            eligible_mask=eligible_mask,
            config=self.config,
        )

        # 7. Stage 6: Empirical CI Diagnosis (Table 6)
        base_diagnosis = evaluate_ci_diagnosis(n_growth, n_glaciation, n_height)

        # 8. Stage 7: Forward Scheme (0-30', 0-60', 0-90' horizons + Monotonicity)
        forecast_horizons = generate_ci_forecast_horizons(
            base_diagnosis=base_diagnosis,
            motion_u_px_per_min=motion_u,
            motion_v_px_per_min=motion_v,
            config=self.config,
        )

        # Construct status bitmask (Table 7)
        # Bit 0: high-res sat data used (1)
        # Bit 1: visible data used (0 in pure IR)
        # Bit 2: IR3.9 data used (0)
        # Bit 3: cloud type / mask used (1)
        # Bit 4: CMIC used (1 if day_with_cmic else 0)
        # Bit 5: NWP data used (1 if nwp_mask is not None else 0)
        status_flag = (1 << 0) | (1 << 3)
        if self.config.mode == "day_with_cmic":
            status_flag |= (1 << 4)
        if nwp_mask is not None:
            status_flag |= (1 << 5)

        # Update tracker state
        self.previous_cube = seviri_cube.copy()
        if timestamp is not None:
            self.previous_time = pd.Timestamp(timestamp)
        self.last_motion_u = motion_u
        self.last_motion_v = motion_v

        return NwcSafCiStepResult(
            ci_prob30=forecast_horizons["ci_prob30"],
            ci_prob60=forecast_horizons["ci_prob60"],
            ci_prob90=forecast_horizons["ci_prob90"],
            ci_status_flag=status_flag,
            pre_ci_mask=pre_ci_mask,
            eligible_mask=eligible_mask,
            detected_cells=detected_cells,
            motion_u=motion_u,
            motion_v=motion_v,
            n_growth=n_growth,
            n_glaciation=n_glaciation,
            n_height=n_height,
            base_diagnosis=base_diagnosis,
        )


def run_nwcsaf_ci_sequence(
    seviri_cubes: Union[np.ndarray, List[np.ndarray]],
    timestamps: Optional[List[Union[pd.Timestamp, str]]] = None,
    config: Optional[NwcSafCiConfig] = None,
) -> List[NwcSafCiStepResult]:
    """Runs NwcSafCiTracker over an entire sequence of temporal SEVIRI slots.

    Parameters
    ----------
    seviri_cubes : Union[np.ndarray, List[np.ndarray]]
        Array of shape (T, 7, H, W) or list of T arrays of shape (7, H, W).
    timestamps : Optional[List[Union[pd.Timestamp, str]]]
        List of T timestamps.
    config : Optional[NwcSafCiConfig]
        Configuration object.

    Returns
    -------
    List[NwcSafCiStepResult] of length T.
    """
    tracker = NwcSafCiTracker(config=config)
    results: List[NwcSafCiStepResult] = []

    cubes_list = list(seviri_cubes)
    num_frames = len(cubes_list)

    for i, cube in enumerate(cubes_list):
        ts = timestamps[i] if timestamps is not None else None
        res = tracker.step(cube, timestamp=ts)
        results.append(res)

    return results
