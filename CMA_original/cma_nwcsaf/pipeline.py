"""End-to-end stateful pipeline for NWC SAF GEO-CMA v2025 (v6.0) on SEVIRI 7-channel clips.

Provides:
- `NwcSafCmaTracker`: causal frame-by-frame tracker maintaining the 15-minute and 60-minute
  temporal buffers (`IR_108(t - 15 min)`, `IR_108(t - 60 min)`, `CMA(t - 60 min)`) and the
  causal clear-sky surface temperature exponential moving average (`T_surf`).
- `run_nwcsaf_cma_sequence`: batch runner over `(T, 7, H, W)` SEVIRI brightness temperature
  arrays or dicts of `(T, H, W)` channel arrays.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .ancillary import AncillaryData, estimate_autonomous_ir_ancillary
from .config import NwcSafCmaConfig
from .detector import CmaOutput, run_cma_frame

SEVIRI_7CH_NAMES: tuple[str, ...] = (
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)


@dataclass
class CmaSequenceResult:
    """Stacked `(T, H, W)` outputs for a sequence of `T` SEVIRI frames."""

    cma: np.ndarray               # (T, H, W) uint8: 0 = Cloud-free, 1 = Cloudy, 255 = Fill
    cma_cover: np.ndarray         # (T, H, W) uint8: 0 = Cloud-free, 1 = Cloud contaminated, 2 = Cloud filled, 3 = Snow/Ice
    cma_cloudsnow: np.ndarray     # (T, H, W) uint8: 0 = Cloud-free, 1 = Cloud, 2 = Thin ice over snow, 3 = Snow/Ice
    cma_dust: np.ndarray          # (T, H, W) uint8: 0 = No dust, 1 = Dust, 2 = Undefined
    cma_volcanic: np.ndarray      # (T, H, W) uint8: 0 = No volcanic plume, 1 = Volcanic plume, 2 = Undefined
    cma_smoke: np.ndarray         # (T, H, W) uint8: 0 = No smoke, 1 = Smoke, 2 = Undefined
    cma_testlist: np.ndarray      # (T, H, W) uint32: 28-bit triggered test mask
    cma_status_flag: np.ndarray   # (T, H, W) uint16: 11-bit status flag
    cma_quality: np.ndarray       # (T, H, W) uint8: quality flag
    t_surf_k: np.ndarray          # (T, H, W) float32: resolved clear-sky surface temperature (K)


def unpack_7ch_frame(frame: np.ndarray | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert a `(7, H, W)` array or channel dictionary into a validated 7-channel dict."""
    if isinstance(frame, dict):
        missing = [ch for ch in SEVIRI_7CH_NAMES if ch not in frame]
        if missing:
            raise ValueError(f"Missing SEVIRI channels: {missing}")
        return {ch: np.asarray(frame[ch], dtype=np.float32) for ch in SEVIRI_7CH_NAMES}

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 7:
        raise ValueError(f"Expected frame of shape (7, H, W), got {arr.shape}")
    return {ch: arr[idx] for idx, ch in enumerate(SEVIRI_7CH_NAMES)}


class NwcSafCmaTracker:
    """Causal multi-temporal tracker for NWC SAF GEO-CMA v2025 (v6.0)."""

    def __init__(
        self,
        cfg: NwcSafCmaConfig | None = None,
        *,
        cadence_minutes: float = 5.0,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
        is_land: np.ndarray | None = None,
        elevation_m: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg or NwcSafCmaConfig()
        self.cadence_minutes = float(cadence_minutes)
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.is_land = is_land
        self.elevation_m = elevation_m

        self._steps_15m = max(1, int(round(15.0 / self.cadence_minutes)))
        self._steps_60m = max(1, int(round(60.0 / self.cadence_minutes)))
        self._history_ir108: deque[np.ndarray] = deque(maxlen=self._steps_60m + 1)
        self._history_cma: deque[np.ndarray] = deque(maxlen=self._steps_60m + 1)
        self._prev_t_surf_k: np.ndarray | None = None

    def reset(self) -> None:
        """Clear temporal buffers."""
        self._history_ir108.clear()
        self._history_cma.clear()
        self._prev_t_surf_k = None

    def step(
        self,
        frame_7ch: np.ndarray | dict[str, np.ndarray],
        *,
        timestamp: datetime | None = None,
        snow_ice_mask: np.ndarray | None = None,
        t_surf_k: np.ndarray | None = None,
        t_950hpa_k: np.ndarray | None = None,
        t_850hpa_k: np.ndarray | None = None,
        t_700hpa_k: np.ndarray | None = None,
        t_500hpa_k: np.ndarray | None = None,
        t_tropo_k: np.ndarray | None = None,
        solar_zenith_deg: np.ndarray | None = None,
        sunglint_mask: np.ndarray | None = None,
    ) -> tuple[CmaOutput, AncillaryData]:
        """Process a single 7-channel SEVIRI frame causally."""
        _ = timestamp
        bt_dict = unpack_7ch_frame(frame_7ch)

        anc = estimate_autonomous_ir_ancillary(
            bt_dict,
            self.cfg,
            lat_deg=self.lat_deg,
            lon_deg=self.lon_deg,
            is_land=self.is_land,
            elevation_m=self.elevation_m,
            snow_ice_mask=snow_ice_mask,
            t_surf_k=t_surf_k,
            t_950hpa_k=t_950hpa_k,
            t_850hpa_k=t_850hpa_k,
            t_700hpa_k=t_700hpa_k,
            t_500hpa_k=t_500hpa_k,
            t_tropo_k=t_tropo_k,
            prev_t_surf_k=self._prev_t_surf_k,
            solar_zenith_deg=solar_zenith_deg,
            sunglint_mask=sunglint_mask,
        )
        self._prev_t_surf_k = anc.t_surf_k.copy()

        bt_15m = (
            self._history_ir108[-self._steps_15m]
            if len(self._history_ir108) >= self._steps_15m
            else None
        )
        bt_60m = (
            self._history_ir108[-self._steps_60m]
            if len(self._history_ir108) >= self._steps_60m
            else None
        )
        cma_60m = (
            self._history_cma[-self._steps_60m]
            if len(self._history_cma) >= self._steps_60m
            else None
        )

        cma_out = run_cma_frame(
            bt_dict,
            anc,
            self.cfg,
            bt_108_t_minus_15m=bt_15m,
            bt_108_t_minus_60m=bt_60m,
            cma_t_minus_60m=cma_60m,
        )

        self._history_ir108.append(bt_dict["IR_108"].copy())
        self._history_cma.append(cma_out.cma.copy())

        return cma_out, anc


def run_nwcsaf_cma_sequence(
    clip_7ch: np.ndarray,
    cfg: NwcSafCmaConfig | None = None,
    *,
    cadence_minutes: float = 5.0,
    lat_deg: np.ndarray | None = None,
    lon_deg: np.ndarray | None = None,
    is_land: np.ndarray | None = None,
    elevation_m: np.ndarray | None = None,
    snow_ice_mask: np.ndarray | None = None,
    t_surf_k: np.ndarray | None = None,
) -> CmaSequenceResult:
    """Run NWC SAF GEO-CMA v2025 causally across a `(T, 7, H, W)` SEVIRI sequence."""
    arr = np.asarray(clip_7ch, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1] != 7:
        raise ValueError(f"Expected clip_7ch of shape (T, 7, H, W), got {arr.shape}")

    t_steps, _, h, w = arr.shape
    tracker = NwcSafCmaTracker(
        cfg=cfg,
        cadence_minutes=cadence_minutes,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        is_land=is_land,
        elevation_m=elevation_m,
    )

    cma_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_cov_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_cs_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_dust_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_volc_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_smoke_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    cma_test_seq = np.empty((t_steps, h, w), dtype=np.uint32)
    cma_stat_seq = np.empty((t_steps, h, w), dtype=np.uint16)
    cma_qual_seq = np.empty((t_steps, h, w), dtype=np.uint8)
    tsurf_seq = np.empty((t_steps, h, w), dtype=np.float32)

    for t in range(t_steps):
        ts_frame = None
        if t_surf_k is not None:
            ts_frame = t_surf_k[t] if t_surf_k.ndim == 3 else t_surf_k
        snow_frame = None
        if snow_ice_mask is not None:
            snow_frame = snow_ice_mask[t] if snow_ice_mask.ndim == 3 else snow_ice_mask

        cma_out, anc = tracker.step(
            arr[t],
            snow_ice_mask=snow_frame,
            t_surf_k=ts_frame,
        )
        cma_seq[t] = cma_out.cma
        cma_cov_seq[t] = cma_out.cma_cover
        cma_cs_seq[t] = cma_out.cma_cloudsnow
        cma_dust_seq[t] = cma_out.cma_dust
        cma_volc_seq[t] = cma_out.cma_volcanic
        cma_smoke_seq[t] = cma_out.cma_smoke
        cma_test_seq[t] = cma_out.cma_testlist
        cma_stat_seq[t] = cma_out.cma_status_flag
        cma_qual_seq[t] = cma_out.cma_quality
        tsurf_seq[t] = anc.t_surf_k

    return CmaSequenceResult(
        cma=cma_seq,
        cma_cover=cma_cov_seq,
        cma_cloudsnow=cma_cs_seq,
        cma_dust=cma_dust_seq,
        cma_volcanic=cma_volc_seq,
        cma_smoke=cma_smoke_seq,
        cma_testlist=cma_test_seq,
        cma_status_flag=cma_stat_seq,
        cma_quality=cma_qual_seq,
        t_surf_k=tsurf_seq,
    )
