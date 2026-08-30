#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.catalogs import build_emma_to_gpm
from emma_gpm.config import ensure_output_dirs, load_config, resolve_project_path
from emma_gpm.emma import emma_files, parse_emma_time, read_grid
from emma_gpm.matching import match_pf_to_emma


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    ensure_output_dirs(config, ROOT)
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    catalogs = resolve_project_path(ROOT, config["outputs"]["catalogs"])
    paths = emma_files(resolve_project_path(ROOT, config["emma"]["extracted_root"]))
    by_time = {parse_emma_time(path): path for path in paths}
    grid = read_grid(paths[0], config["domain"])

    matched: list[pd.DataFrame] = []
    for filename in ("gmi_pf_domain_2020.parquet", "dpr_pf_domain_2020.parquet"):
        path = interim / filename
        if path.exists():
            features = pd.read_parquet(path)
            matched.append(
                match_pf_to_emma(
                    features, by_time, grid, config["time_tolerance_minutes"]
                )
            )
    if not matched:
        raise FileNotFoundError("Run prepare_gpm_pf.py before build_catalogs.py")
    gpm_to_emma = pd.concat(matched, ignore_index=True, sort=False)
    # DPRrpf exposes feature-level counts of stratiform and convective pixels,
    # not the native per-pixel 8-digit typePrecip field.  Retain both counts and
    # add an explicitly derived dominant class without inventing an "other"
    # count that the monthly PF table does not provide.
    gpm_to_emma["dpr_dominant_class_pf"] = pd.NA
    dpr = gpm_to_emma.sensor.eq("DPR")
    classified = dpr & gpm_to_emma.nstrat_dpr.fillna(0).add(
        gpm_to_emma.nconv_dpr.fillna(0)
    ).gt(0)
    gpm_to_emma.loc[
        classified & gpm_to_emma.nconv_dpr.gt(gpm_to_emma.nstrat_dpr),
        "dpr_dominant_class_pf",
    ] = "convective"
    gpm_to_emma.loc[
        classified & gpm_to_emma.nstrat_dpr.gt(gpm_to_emma.nconv_dpr),
        "dpr_dominant_class_pf",
    ] = "stratiform"
    gpm_to_emma.loc[
        classified & gpm_to_emma.nstrat_dpr.eq(gpm_to_emma.nconv_dpr),
        "dpr_dominant_class_pf",
    ] = "tie"
    denominator = gpm_to_emma.nstrat_dpr.add(gpm_to_emma.nconv_dpr)
    gpm_to_emma["dpr_convective_fraction_of_classified_pixels"] = (
        gpm_to_emma.nconv_dpr / denominator.where(denominator.gt(0))
    )
    gpm_to_emma["dpr_type_class_source"] = pd.NA
    gpm_to_emma.loc[dpr, "dpr_type_class_source"] = (
        "PF aggregate NCONV_DPR/NSTRAT_DPR; native per-pixel typePrecip not in this catalog"
    )
    objects = pd.read_parquet(interim / "emma_hourly_objects_2020.parquet")
    phase = objects[[
        "mcs_id", "time", "full_track_area_km2", "is_mature_phase_reconstructed",
        "official_robust_flag",
    ]].rename(columns={"time": "emma_time"})
    gpm_to_emma = gpm_to_emma.merge(
        phase, how="left", on=["mcs_id", "emma_time"], validate="many_to_one"
    )
    gpm_to_emma["emma_class_reconstructed"] = gpm_to_emma["emma_class"]
    inside = gpm_to_emma.mcs_id.fillna(0).gt(0)
    mature = gpm_to_emma.is_mature_phase_reconstructed.eq(True)
    gpm_to_emma.loc[
        inside & mature,
        "emma_class_reconstructed",
    ] = "mature_mcs_reconstructed"
    gpm_to_emma.loc[
        inside & ~mature,
        "emma_class_reconstructed",
    ] = "development_or_decay_reconstructed"
    gpm_to_emma.to_parquet(catalogs / "gpm_to_emma_pf_2020.parquet", index=False)
    gpm_to_emma.to_csv(catalogs / "gpm_to_emma_pf_2020.csv", index=False)

    lifecycle = pd.read_parquet(interim / "emma_lifecycle_2020.parquet")
    emma_to_gpm = build_emma_to_gpm(lifecycle, gpm_to_emma)
    emma_to_gpm.to_parquet(catalogs / "emma_to_gpm_pf_2020.parquet", index=False)
    emma_to_gpm.to_csv(catalogs / "emma_to_gpm_pf_2020.csv", index=False)
    print(f"GPM -> EMMA rows: {len(gpm_to_emma)}")
    print(f"EMMA -> GPM rows: {len(emma_to_gpm)}")


if __name__ == "__main__":
    main()
