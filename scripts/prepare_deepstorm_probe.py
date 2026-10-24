#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.deepstorm import read_csatgpm_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unique-footprint GMI/CloudSat targets from 2B-CSATGPM V04/V05."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/deepstorm_probe.json")
    parser.add_argument("--input-root", type=Path, help="Override the configured raw-data root")
    parser.add_argument("--output", type=Path, help="Override the configured Parquet output")
    parser.add_argument("--limit", type=int, help="Read only the first N files for a smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_root = args.input_root or resolve_project_path(ROOT, config["source"]["root"])
    output = args.output or resolve_project_path(ROOT, config["outputs"]["catalog"])
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(source_root.glob(config["source"]["glob"]))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"No {config['source']['version']} files found below {source_root}. "
            "Download a PPS/STORM 2B-CSATGPM order first; see DEEPSTORM_DATASET.md."
        )

    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        try:
            frame = read_csatgpm_file(
                path,
                domain=config["domain"],
                deep_convection_code=config["targets"]["deep_convection_code"],
                iwp_threshold_kg_m2=config["targets"]["iwp_threshold_kg_m2"],
            )
            frames.append(frame)
            inventory.append({"source_file": path.name, "status": "ok", "samples": len(frame), "error": ""})
        except Exception as exc:
            inventory.append({"source_file": path.name, "status": "error", "samples": 0, "error": str(exc)})
        print(f"[{index}/{len(files)}] {path.name}: {inventory[-1]['status']} ({inventory[-1]['samples']} samples)")

    inventory_path = resolve_project_path(ROOT, config["outputs"]["inventory"])
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    failed = [row for row in inventory if row["status"] == "error"]
    if failed:
        raise SystemExit(
            f"{len(failed)} file(s) could not be read. Inspect {inventory_path}; no partial catalog was written."
        )

    catalog = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    if not catalog.empty and catalog.observation_time.notna().any():
        catalog = catalog[
            catalog.observation_time.isna()
            | ((catalog.observation_time >= start) & (catalog.observation_time < end))
        ].copy()
    if catalog.sample_uid.duplicated().any():
        raise SystemExit("Duplicate sample_uid values found; refusing to write a leakage-prone catalog")
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output, index=False)
    print(f"Wrote {len(catalog)} unique GMI footprints to {output}")


if __name__ == "__main__":
    main()
