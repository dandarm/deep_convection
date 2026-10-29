"""Official EUMETSAT NWC SAF GEO-CI (Convection Initiation) Python Implementation.

Submodules:
- `config`: Operational thresholds, lead times, probability classes.
- `candidate_selection`: Area of interest, cloud masking, eligible-CI candidate selection.
- `detection_tracking`: Warm cloud cell multi-thresholding detection & 2D movement tracking.
- `interest_fields`: Static BTDs and dynamic Lagrangian cooling rate calculation.
- `discrimination`: Table 6 empirical rules and CI diagnosis classification.
- `forward_scheme`: Forward trajectory advection and probability spreading.
- `ancillary`: Pure-IR motion estimation (AMV) and cloud masking surrogates.
- `pipeline`: Stateful `NwcSafCiTracker` and `run_nwcsaf_ci_sequence`.
"""

from __future__ import annotations

from .config import (
    NwcSafCiConfig,
    CiProbabilityClass,
    SEVIRI_CHANNELS,
    CHANNEL_INDICES,
)
from .candidate_selection import (
    select_eligible_ci_pixels,
    compute_pure_ir_cloud_mask,
    compute_nwp_convective_mask,
)
from .detection_tracking import (
    CiCloudCell,
    detect_warm_cloud_cells,
    WarmCellTracker,
)
from .interest_fields import (
    CiInterestFields,
    compute_static_btds,
    compute_ci_interest_fields,
    count_relevant_parameters,
)
from .discrimination import (
    diagnose_ci_probability_pixel,
    evaluate_ci_diagnosis,
    enforce_lead_time_monotonicity,
)
from .forward_scheme import (
    apply_forward_scheme,
    generate_ci_forecast_horizons,
)
from .ancillary import (
    compute_pure_ir_amv_motion,
)
from .pipeline import (
    NwcSafCiTracker,
    NwcSafCiStepResult,
    run_nwcsaf_ci_sequence,
)

__all__ = [
    "NwcSafCiConfig",
    "CiProbabilityClass",
    "SEVIRI_CHANNELS",
    "CHANNEL_INDICES",
    "select_eligible_ci_pixels",
    "compute_pure_ir_cloud_mask",
    "compute_nwp_convective_mask",
    "CiCloudCell",
    "detect_warm_cloud_cells",
    "WarmCellTracker",
    "CiInterestFields",
    "compute_static_btds",
    "compute_ci_interest_fields",
    "count_relevant_parameters",
    "diagnose_ci_probability_pixel",
    "evaluate_ci_diagnosis",
    "enforce_lead_time_monotonicity",
    "apply_forward_scheme",
    "generate_ci_forecast_horizons",
    "compute_pure_ir_amv_motion",
    "NwcSafCiTracker",
    "NwcSafCiStepResult",
    "run_nwcsaf_ci_sequence",
]
