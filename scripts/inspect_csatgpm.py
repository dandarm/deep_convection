#!/usr/bin/env python3
from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.deepstorm import dataset_inventory


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_csatgpm.py FILE.nc4")
    path = Path(sys.argv[1])
    with h5py.File(path) as handle:
        print(dataset_inventory(handle).to_string(index=False))


if __name__ == "__main__":
    main()
