#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.config import load_config, resolve_project_path
from emma_gpm.download import fetch_emma_year


FIRST_EMMA_YEAR = 1998
LAST_EMMA_YEAR = 2024


def parse_years(values: list[str] | None, default_year: int) -> list[int]:
    if not values:
        return [default_year]
    if values == ["all"]:
        return list(range(FIRST_EMMA_YEAR, LAST_EMMA_YEAR + 1))
    years = sorted({int(value) for value in values})
    invalid = [year for year in years if not FIRST_EMMA_YEAR <= year <= LAST_EMMA_YEAR]
    if invalid:
        raise SystemExit(
            f"EMMA v1 years are {FIRST_EMMA_YEAR}-{LAST_EMMA_YEAR}; invalid: {invalid}"
        )
    return years


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, checksum and extract one or more EMMA v1 yearly archives."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        metavar="YEAR",
        help="Years to fetch, or the single value 'all' for 1998-2024 (default: pilot year).",
    )
    args = parser.parse_args()
    config = load_config(ROOT / "config/pilot_2020.json")
    raw_dir = resolve_project_path(ROOT, "data/raw/emma")
    interim_dir = resolve_project_path(ROOT, "data/interim/emma")
    years = parse_years(args.years, config["year"])
    for index, year in enumerate(years, start=1):
        print(f"[{index}/{len(years)}] EMMA {year}", flush=True)
        archive = fetch_emma_year(
            config["emma"]["record_id"],
            year,
            raw_dir,
            interim_dir,
        )
        print(f"checksum verified and extracted: {archive}", flush=True)


if __name__ == "__main__":
    main()
