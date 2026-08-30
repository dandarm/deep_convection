#!/usr/bin/env python3
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import ensure_output_dirs, load_config, resolve_project_path
from emma_gpm.download import cmr_granules, normalize_cmr_entries


COLLECTIONS_V07 = {
    "DPR_2A": "C2179081499-GES_DISC",
    "GMI_1C": "C2259345484-GES_DISC",
    "CMB_2B": "C2179081553-GES_DISC",
}


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    ensure_output_dirs(config, ROOT)
    domain = config["domain"]
    bbox = (domain["lon_min"], domain["lat_min"], domain["lon_max"], domain["lat_max"])
    rows = []
    raw = {}
    for name, concept_id in COLLECTIONS_V07.items():
        entries, hits = cmr_granules(concept_id, config["start"], config["end"], bbox)
        raw[name] = {"concept_id": concept_id, "cmr_hits": hits, "entries": entries}
        rows.extend(normalize_cmr_entries(entries, name))
        print(f"{name}: CMR hits={hits}, returned={len(entries)}")
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    with (interim / "cmr_gpm_v07_mjjas2020.json").open("w", encoding="utf-8") as stream:
        json.dump(raw, stream)
    frame = pd.DataFrame(rows)
    frame.to_parquet(interim / "cmr_gpm_v07_mjjas2020.parquet", index=False)
    frame.drop(columns="polygons_json").to_csv(
        resolve_project_path(ROOT, config["outputs"]["catalogs"])
        / "gpm_granule_inventory_cmr_v07_2020.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
