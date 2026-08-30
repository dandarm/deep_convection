#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.download import fetch_emma_year


def main() -> None:
    config = load_config(ROOT / "config/pilot_2020.json")
    archive = fetch_emma_year(
        config["emma"]["record_id"],
        config["year"],
        resolve_project_path(ROOT, "data/raw/emma"),
        resolve_project_path(ROOT, "data/interim/emma"),
    )
    print(f"Downloaded, checksum-verified and extracted {archive}")


if __name__ == "__main__":
    main()
