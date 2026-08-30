#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.download import stream_download
from emma_gpm.config import resolve_project_path


MONTHS = range(5, 10)
GMI = "https://atmos.tamucc.edu/trmm/data/gpm/1C_PF/GPM.{year}{month:02d}.HDF5"
DPR = "https://atmos.tamucc.edu/trmm/data/gpm/level_2/dprrpf/pf_{year}{month:02d}_level2.HDF"


def main(year: int = 2020) -> None:
    for month in MONTHS:
        gmi_name = f"GPM.{year}{month:02d}.HDF5"
        dpr_name = f"pf_{year}{month:02d}_level2.HDF"
        print(stream_download(GMI.format(year=year, month=month), resolve_project_path(ROOT, "data/raw/gpm_pf_1c") / gmi_name))
        print(stream_download(DPR.format(year=year, month=month), resolve_project_path(ROOT, "data/raw/gpm_pf_dpr") / dpr_name))


if __name__ == "__main__":
    main()
