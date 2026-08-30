from __future__ import annotations

import pandas as pd


def build_emma_to_gpm(lifecycle: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    result = lifecycle.copy()
    positive = matches[matches.mcs_id.fillna(0).gt(0)].copy()
    if "is_mature_phase_reconstructed" in positive:
        positive["hit_robust"] = positive.is_mature_phase_reconstructed.eq(True)
    else:
        positive["hit_robust"] = positive.robust_mcs_id.fillna(0).gt(0)
    positive["hit_nonrobust"] = ~positive.hit_robust
    for sensor in ("GMI", "DPR"):
        subset = positive[positive.sensor.eq(sensor)]
        agg = subset.groupby("track_uid").agg(
            **{
                f"{sensor.lower()}_pf_feature_count": ("feature_uid", "nunique"),
                f"{sensor.lower()}_orbit_count": ("orbit", "nunique"),
                f"{sensor.lower()}_robust_feature_count": ("hit_robust", "sum"),
                f"{sensor.lower()}_nonrobust_feature_count": ("hit_nonrobust", "sum"),
            }
        )
        result = result.merge(agg, how="left", left_on="track_uid", right_index=True)
        count_cols = [column for column in agg.columns]
        result[count_cols] = result[count_cols].fillna(0).astype(int)
        result[f"observed_by_{sensor.lower()}_pf"] = result[
            f"{sensor.lower()}_pf_feature_count"
        ].gt(0)
        result[f"observed_by_{sensor.lower()}_pf_in_robust_phase"] = result[
            f"{sensor.lower()}_robust_feature_count"
        ].gt(0)
        result[f"observed_by_{sensor.lower()}_pf_in_nonrobust_phase"] = result[
            f"{sensor.lower()}_nonrobust_feature_count"
        ].gt(0)
    result["coverage_interpretation"] = (
        "PF-feature co-occurrence lower bound; absence is not proof of no sensor overpass"
    )
    return result
