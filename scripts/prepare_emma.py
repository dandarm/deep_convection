#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import ensure_output_dirs, load_config, resolve_project_path
from emma_gpm.emma import prepare_emma_objects


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    ensure_output_dirs(config, ROOT)
    objects, lifecycle, grid = prepare_emma_objects(
        resolve_project_path(ROOT, config["emma"]["extracted_root"]), config["domain"]
    )
    interim = resolve_project_path(ROOT, config["outputs"]["interim"])
    objects.to_parquet(interim / "emma_hourly_objects_2020.parquet", index=False)
    lifecycle.to_parquet(interim / "emma_lifecycle_2020.parquet", index=False)
    print(f"hourly object rows: {len(objects)}")
    print(f"MCS tracks in domain: {len(lifecycle)}")
    print(
        "effective EMMA grid coverage: "
        f"lon={grid.pilot_lon.min():.2f}..{grid.pilot_lon.max():.2f}, "
        f"lat={grid.pilot_lat.min():.2f}..{grid.pilot_lat.max():.2f}"
    )


if __name__ == "__main__":
    main()
