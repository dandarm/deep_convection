"""End-to-end stateful pipeline for the NWC SAF GEO-CMA v6.0 & GEO-CT v5.0 processor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from .ancillary import AtmosphericProfileFields, build_atmospheric_profile_fields, extract_seviri_7ch
from .cma import CmaResult, compute_nwcsaf_cma
from .config import NwcSafCmaCtConfig
from .ct import CtResult, compute_nwcsaf_ct


@dataclass
class NwcSafCmaCtStepResult:
    """Combined single-frame output of NWC SAF GEO-CMA and GEO-CT."""

    cma_result: CmaResult
    ct_result: CtResult
    profile: AtmosphericProfileFields

    @property
    def cma(self) -> np.ndarray:
        return self.cma_result.cma

    @property
    def cma_cloudsnow(self) -> np.ndarray:
        return self.cma_result.cma_cloudsnow

    @property
    def cma_dust(self) -> np.ndarray:
        return self.cma_result.cma_dust

    @property
    def cma_volcanic(self) -> np.ndarray:
        return self.cma_result.cma_volcanic

    @property
    def cma_testlist(self) -> np.ndarray:
        return self.cma_result.cma_testlist

    @property
    def cma_quality(self) -> np.ndarray:
        return self.cma_result.cma_quality

    @property
    def ct(self) -> np.ndarray:
        return self.ct_result.ct

    @property
    def ct_cumuliform(self) -> np.ndarray:
        return self.ct_result.ct_cumuliform

    @property
    def ct_multilayer(self) -> np.ndarray:
        return self.ct_result.ct_multilayer

    @property
    def ct_quality(self) -> np.ndarray:
        return self.ct_result.ct_quality

    @property
    def rdt_bridge_class(self) -> np.ndarray:
        return self.ct_result.rdt_bridge_class

    @property
    def opaque_cloud_mask(self) -> np.ndarray:
        return self.ct_result.opaque_cloud_mask

    @property
    def semitransparent_mask(self) -> np.ndarray:
        return self.ct_result.semitransparent_mask


@dataclass
class _HistoryEntry:
    cube_7ch: np.ndarray
    cma: np.ndarray
    ct: np.ndarray
    cma_quality: np.ndarray


class NwcSafCmaCtTracker:
    """Stateful NWC SAF GEO-CMA + GEO-CT processor across consecutive SEVIRI frames.

    Maintains:
    - Causal clear-sky surface temperature EMA reference `running_t_surf_k`
    - 15-minute lookback buffer for fast-moving cloud temporal differencing (ATBD 2.2.1.2.4)
    - 60-minute lookback buffer for stationary low/mid cloud temporal coherence and
      region-growing restoration (ATBD 2.2.1.2.5)
    """

    def __init__(self, cfg: NwcSafCmaCtConfig | None = None) -> None:
        self.cfg = cfg or NwcSafCmaCtConfig()
        self.running_t_surf_k: np.ndarray | None = None
        self._steps_15m = max(
            1, int(round(self.cfg.cma_temporal_window_minutes / max(self.cfg.cadence_minutes, 1.0)))
        )
        self._steps_1h = max(
            self._steps_15m + 1,
            int(round(self.cfg.cma_stationary_window_minutes / max(self.cfg.cadence_minutes, 1.0))),
        )
        self._history: deque[_HistoryEntry] = deque(maxlen=self._steps_1h + 1)

    def reset(self) -> None:
        """Reset all temporal state."""
        self.running_t_surf_k = None
        self._history.clear()

    def step(
        self,
        cube_7ch: np.ndarray,
        *,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
        land_mask: np.ndarray | None = None,
        snow_ice_mask: np.ndarray | None = None,
        nwp_fields: dict[str, np.ndarray | float] | None = None,
    ) -> NwcSafCmaCtStepResult:
        """Process a single `(7, H, W)` SEVIRI frame in Kelvin and update temporal buffers."""
        cube = np.asarray(cube_7ch, dtype=np.float32)
        _ = extract_seviri_7ch(cube)

        profile, self.running_t_surf_k = build_atmospheric_profile_fields(
            cube,
            self.cfg,
            running_t_surf_k=self.running_t_surf_k,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            land_mask=land_mask,
            snow_ice_mask=snow_ice_mask,
            nwp_fields=nwp_fields,
        )

        prev_15m_ir108: np.ndarray | None = None
        if len(self._history) >= self._steps_15m:
            entry_15m = self._history[-self._steps_15m]
            prev_15m_ir108 = entry_15m.cube_7ch[4]  # IR_108 is index 4

        prev_1h_cube: np.ndarray | None = None
        prev_1h_cma: np.ndarray | None = None
        prev_1h_ct: np.ndarray | None = None
        prev_1h_quality: np.ndarray | None = None
        if len(self._history) >= self._steps_1h:
            entry_1h = self._history[-self._steps_1h]
            prev_1h_cube = entry_1h.cube_7ch
            prev_1h_cma = entry_1h.cma
            prev_1h_ct = entry_1h.ct
            prev_1h_quality = entry_1h.cma_quality

        cma_res = compute_nwcsaf_cma(
            cube,
            profile,
            self.cfg,
            prev_15m_ir108=prev_15m_ir108,
            prev_1h_cube_7ch=prev_1h_cube,
            prev_1h_cma=prev_1h_cma,
            prev_1h_ct=prev_1h_ct,
            prev_1h_quality=prev_1h_quality,
        )

        ct_res = compute_nwcsaf_ct(cube, profile, cma_res, self.cfg)

        self._history.append(
            _HistoryEntry(
                cube_7ch=cube.copy(),
                cma=cma_res.cma.copy(),
                ct=ct_res.ct.copy(),
                cma_quality=cma_res.cma_quality.copy(),
            )
        )

        return NwcSafCmaCtStepResult(
            cma_result=cma_res,
            ct_result=ct_res,
            profile=profile,
        )


def run_nwcsaf_cma_ct_sequence(
    sequence_7ch: np.ndarray,
    cfg: NwcSafCmaCtConfig | None = None,
    *,
    lat_deg: np.ndarray | None = None,
    lon_deg: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    snow_ice_mask: np.ndarray | None = None,
    nwp_fields: dict[str, np.ndarray | float] | None = None,
) -> list[NwcSafCmaCtStepResult]:
    """Run the stateful NWC SAF GEO-CMA & GEO-CT pipeline over a `(T, 7, H, W)` SEVIRI sequence."""
    seq = np.asarray(sequence_7ch, dtype=np.float32)
    if seq.ndim != 4 or seq.shape[1] != 7:
        raise ValueError(
            f"Expected SEVIRI sequence of shape (T, 7, H, W), got {seq.shape}."
        )
    tracker = NwcSafCmaCtTracker(cfg)
    results: list[NwcSafCmaCtStepResult] = []
    for t_idx in range(seq.shape[0]):
        res = tracker.step(
            seq[t_idx],
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            land_mask=land_mask,
            snow_ice_mask=snow_ice_mask,
            nwp_fields=nwp_fields,
        )
        results.append(res)
    return results
