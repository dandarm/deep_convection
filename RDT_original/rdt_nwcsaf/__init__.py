"""Self-contained implementation of the official EUMETSAT NWC SAF GEO-RDT-CW algorithm.

References:
- NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection v2.1 (15 Oct 2019) & v1.1 (01 Feb 2016)
- NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection v1.0 (01 Dec 2021)
"""

from .config import (
    NwcSafRdtConfig,
    PHASE_NAMES,
    QUALITY_GROWTH_VALID,
    QUALITY_OBSERVATION_VALID,
    QUALITY_PHASE_VALID,
    QUALITY_TEMPORAL_VALID,
    SEMANTIC_CLASS_NAMES,
    SEVERITY_NAMES,
    SEVIRI_7CH_NAMES,
    VERTICAL_CATEGORY_NAMES,
    classify_vertical_category,
)
from .detection import ThresholdSliceHierarchy, detect_rdt_cells
from .discrimination import (
    compute_phase_of_development,
    compute_severity_and_hazard,
    compute_temporal_predictors,
    discriminate_and_classify_cell,
    evaluate_logistic_discrimination,
    forecast_convective_cells,
    select_history_depth_regime,
)
from .otd import detect_cell_overshooting_tops
from .pipeline import NwcSafRdtTracker, run_nwcsaf_rdt_sequence
from .tracking import (
    common_threshold_area,
    estimate_local_correlation_motion,
    track_rdt_cells,
)

__all__ = [
    "NwcSafRdtConfig",
    "NwcSafRdtTracker",
    "PHASE_NAMES",
    "QUALITY_GROWTH_VALID",
    "QUALITY_OBSERVATION_VALID",
    "QUALITY_PHASE_VALID",
    "QUALITY_TEMPORAL_VALID",
    "SEMANTIC_CLASS_NAMES",
    "SEVERITY_NAMES",
    "SEVIRI_7CH_NAMES",
    "ThresholdSliceHierarchy",
    "VERTICAL_CATEGORY_NAMES",
    "classify_vertical_category",
    "common_threshold_area",
    "compute_phase_of_development",
    "compute_severity_and_hazard",
    "compute_temporal_predictors",
    "detect_cell_overshooting_tops",
    "detect_rdt_cells",
    "discriminate_and_classify_cell",
    "estimate_local_correlation_motion",
    "evaluate_logistic_discrimination",
    "forecast_convective_cells",
    "run_nwcsaf_rdt_sequence",
    "select_history_depth_regime",
    "track_rdt_cells",
]
