#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import ensure_output_dirs, load_config, resolve_project_path
from emma_gpm.gpm_pf import read_dpr_pf, read_gmi_pf


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    ensure_output_dirs(config, ROOT)
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    gmi = read_gmi_pf(
        resolve_project_path(ROOT, config["gpm_pf"]["gmi_root"]).glob("GPM.2020*.HDF5"),
        config["domain"],
    )
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    gmi = gmi[gmi.observation_time.between(start, end)].reset_index(drop=True)
    gmi.to_parquet(interim / "gmi_pf_domain_2020.parquet", index=False)
    print(f"GMI precipitation features in domain: {len(gmi)}")

    dpr_files = list(
        resolve_project_path(ROOT, config["gpm_pf"]["dpr_root"]).glob(
            "pf_2020*_level2.HDF"
        )
    )
    if dpr_files:
        dpr = read_dpr_pf(dpr_files, config["domain"])
        dpr = dpr[dpr.observation_time.between(start, end)].reset_index(drop=True)
        dpr.to_parquet(interim / "dpr_pf_domain_2020.parquet", index=False)
        print(f"DPR precipitation features in domain: {len(dpr)}")
    else:
        print("No DPR-PF files found; GMI output was still produced")


if __name__ == "__main__":
    main()
