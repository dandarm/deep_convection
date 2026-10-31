"""Benchmark NWC SAF GEO-CMA v2025 (v6.0) on real 2020 7-channel SEVIRI data."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

_CMA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CMA_ROOT.parent
if str(_CMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CMA_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cma_nwcsaf import (
    CmaBinaryClass,
    CmaCloudSnowClass,
    CmaCoverClass,
    CmaDustClass,
    CmaQuality,
    CmaTestBit,
    CmaVolcanicClass,
    NwcSafCmaConfig,
    run_nwcsaf_cma_sequence,
)
from emma_gpm.seviri_clips import SeviriZarrClipReader


def load_real_seviri_window(
    zarr_root: Path,
    start_time: str,
    origin_y: int,
    origin_x: int,
    num_frames: int = 16,
    crop_size: int = 224,
) -> np.ndarray:
    """Load a `(T, 7, H, W)` SEVIRI brightness temperature cube in Kelvin from the Zarr archive."""
    reader = SeviriZarrClipReader(
        zarr_root,
        num_frames=num_frames,
        cadence_minutes=5,
        crop_size=crop_size,
        require_finite=False,
    )
    cube_raw = reader.load(
        pd.Timestamp(start_time),
        origin_y=origin_y,
        origin_x=origin_x,
        global_mean=None,
        global_std=None,
    )
    return np.asarray(cube_raw, dtype=np.float32)


def main() -> None:
    manifest_path = _REPO_ROOT / "catalogs" / "seviri_pretraining_random_100.csv"
    default_zarr = Path(
        r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"
    )
    out_dir = _CMA_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cma_seviri_benchmark_report.json"

    df = pd.read_csv(manifest_path).iloc[:4]
    cfg = NwcSafCmaConfig()
    clip_reports: list[dict] = []

    total_px = 0
    bin_counts = {k.name: 0 for k in CmaBinaryClass if k != CmaBinaryClass.FILL_VALUE}
    cov_counts = {k.name: 0 for k in CmaCoverClass if k != CmaCoverClass.FILL_VALUE}
    cs_counts = {k.name: 0 for k in CmaCloudSnowClass if k != CmaCloudSnowClass.FILL_VALUE}
    dust_counts = {k.name: 0 for k in CmaDustClass if k != CmaDustClass.FILL_VALUE}
    volc_counts = {k.name: 0 for k in CmaVolcanicClass if k != CmaVolcanicClass.FILL_VALUE}
    qual_counts = {k.name: 0 for k in CmaQuality}
    bit_counts = {bit.name: 0 for bit in CmaTestBit}

    t0_all = time.perf_counter()
    for idx, row_s in df.iterrows():
        row = row_s.to_dict()
        zarr_root = Path(str(row.get("zarr_path", default_zarr)))
        if not zarr_root.exists():
            zarr_root = default_zarr
        bt_clip = load_real_seviri_window(
            zarr_root,
            start_time=str(row["start_time"]),
            origin_y=int(row["origin_y"]),
            origin_x=int(row["origin_x"]),
        )
        t_start = time.perf_counter()
        res = run_nwcsaf_cma_sequence(bt_clip, cfg=cfg)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        n_px = int(res.cma.size)
        total_px += n_px

        for k in CmaBinaryClass:
            if k != CmaBinaryClass.FILL_VALUE:
                bin_counts[k.name] += int(np.sum(res.cma == int(k)))
        for k in CmaCoverClass:
            if k != CmaCoverClass.FILL_VALUE:
                cov_counts[k.name] += int(np.sum(res.cma_cover == int(k)))
        for k in CmaCloudSnowClass:
            if k != CmaCloudSnowClass.FILL_VALUE:
                cs_counts[k.name] += int(np.sum(res.cma_cloudsnow == int(k)))
        for k in CmaDustClass:
            if k != CmaDustClass.FILL_VALUE:
                dust_counts[k.name] += int(np.sum(res.cma_dust == int(k)))
        for k in CmaVolcanicClass:
            if k != CmaVolcanicClass.FILL_VALUE:
                volc_counts[k.name] += int(np.sum(res.cma_volcanic == int(k)))
        for k in CmaQuality:
            qual_counts[k.name] += int(np.sum(res.cma_quality == int(k)))
        for bit in CmaTestBit:
            bit_counts[bit.name] += int(np.sum((res.cma_testlist & (1 << int(bit))) != 0))

        clip_reports.append(
            {
                "clip_index": int(idx),
                "clip_id": str(row.get("clip_id", f"clip_{idx}")),
                "start_time": str(row.get("start_time", "")),
                "shape": list(bt_clip.shape),
                "runtime_ms": round(elapsed_ms, 2),
                "ms_per_frame": round(elapsed_ms / bt_clip.shape[0], 2),
                "cma_binary_pct": {
                    k.name: round(100.0 * float(np.mean(res.cma == int(k))), 2)
                    for k in CmaBinaryClass
                    if k != CmaBinaryClass.FILL_VALUE
                },
                "cma_cover_pct": {
                    k.name: round(100.0 * float(np.mean(res.cma_cover == int(k))), 2)
                    for k in CmaCoverClass
                    if k != CmaCoverClass.FILL_VALUE
                },
                "cma_cloudsnow_pct": {
                    k.name: round(100.0 * float(np.mean(res.cma_cloudsnow == int(k))), 2)
                    for k in CmaCloudSnowClass
                    if k != CmaCloudSnowClass.FILL_VALUE
                },
            }
        )

    total_ms = (time.perf_counter() - t0_all) * 1000.0
    summary = {
        "n_clips": len(df),
        "total_frames": len(df) * 16,
        "total_pixels": total_px,
        "total_runtime_ms": round(total_ms, 2),
        "cma_binary_pct": {k: round(100.0 * v / total_px, 2) for k, v in bin_counts.items()},
        "cma_cover_pct": {k: round(100.0 * v / total_px, 2) for k, v in cov_counts.items()},
        "cma_cloudsnow_pct": {k: round(100.0 * v / total_px, 2) for k, v in cs_counts.items()},
        "cma_dust_pct": {k: round(100.0 * v / total_px, 2) for k, v in dust_counts.items()},
        "cma_volcanic_pct": {k: round(100.0 * v / total_px, 2) for k, v in volc_counts.items()},
        "cma_quality_pct": {k: round(100.0 * v / total_px, 2) for k, v in qual_counts.items()},
        "cma_testlist_trigger_pct": {
            k: round(100.0 * v / total_px, 3) for k, v in bit_counts.items() if v > 0
        },
        "clips": clip_reports,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
