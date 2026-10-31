"""Standalone NWC SAF GEO-CMA v2025 (v6.0, NWC-004) Cloud Mask package.

Provides both Full-NWP and Autonomous Pure-SEVIRI 7-Channel IR execution of the
EUMETSAT NWC SAF Geostationary Cloud Mask (`GEO-CMA`), exposing:
- Binary Cloud Mask (`cma`: 0 = Cloud-free, 1 = Cloudy)
- 4-Class Cloud Cover Detector (`cma_cover`: 0 = Cloud-free [negative class] +
  3 positive classes: 1 = Cloud contaminated, 2 = Cloud filled, 3 = Snow/Ice)
- 4-Class v2025 Cloud & Snow Mask (`cma_cloudsnow`: 0 = Cloud-free [negative class] +
  3 positive classes: 1 = Cloud, 2 = Thin ice cloud over snow/ice, 3 = Snow/Ice)
- Aerosol Plume Masks (`cma_dust`, `cma_volcanic`, `cma_smoke`)
- 28-bit `cma_testlist`, 11-bit `cma_status_flag`, and `cma_quality`.
"""

from .ancillary import (
    AncillaryData,
    compute_satellite_zenith_secant,
    estimate_autonomous_ir_ancillary,
)
from .config import (
    CmaBinaryClass,
    CmaCloudSnowClass,
    CmaCoverClass,
    CmaDustClass,
    CmaQuality,
    CmaSmokeClass,
    CmaStatusBit,
    CmaTestBit,
    CmaVolcanicClass,
    NwcSafCmaConfig,
)
from .detector import CmaOutput, compute_local_3x3_stats, run_cma_frame
from .pipeline import (
    SEVIRI_7CH_NAMES,
    CmaSequenceResult,
    NwcSafCmaTracker,
    run_nwcsaf_cma_sequence,
    unpack_7ch_frame,
)

__all__ = [
    "AncillaryData",
    "CmaBinaryClass",
    "CmaCloudSnowClass",
    "CmaCoverClass",
    "CmaDustClass",
    "CmaOutput",
    "CmaQuality",
    "CmaSequenceResult",
    "CmaSmokeClass",
    "CmaStatusBit",
    "CmaTestBit",
    "CmaVolcanicClass",
    "NwcSafCmaConfig",
    "NwcSafCmaTracker",
    "SEVIRI_7CH_NAMES",
    "compute_local_3x3_stats",
    "compute_satellite_zenith_secant",
    "estimate_autonomous_ir_ancillary",
    "run_cma_frame",
    "run_nwcsaf_cma_sequence",
    "unpack_7ch_frame",
]
