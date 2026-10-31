"""Empirical benchmark of the NWC SAF GEO-CMA v6.0 & GEO-CT v5.0 (v2025) processor on real SEVIRI data.

Evaluates the 15-class Pure-SEVIRI 7-channel IR `GEO-CMA`/`GEO-CT` processor on real 16-frame
SEVIRI convective windows from `E:\\Datasets\\Deep_convection\\processed\\msg_seviri\\rss_7ch_bt_emma_30N70N`
and compares it against:
1. `SeviriIrCloudTypeTracker` (`src/emma_gpm/rdt/cloud_type_ir.py`, 6-class IR proxy)
2. `compute_nwcsaf_cloud_type` (`RDT_original/rdt_nwcsaf/ancillary.py`, 6-class RDT ancillary)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "CT_original"))
sys.path.insert(0, str(REPO_ROOT / "RDT_original"))

from ct_nwcsaf import (
    CmaClass,
    CmaQuality,
    CtClass,
    CtCumuliformClass,
    CtMultilayerClass,
    NwcSafCmaCtConfig,
    RdtBridgeCtClass,
    run_nwcsaf_cma_ct_sequence,
)
from emma_gpm.rdt.cloud_type_ir import IrCloudTypeConfig, SeviriIrCloudTypeTracker
from emma_gpm.seviri_clips import SeviriZarrClipReader
from rdt_nwcsaf.ancillary import compute_nwcsaf_cloud_type
from rdt_nwcsaf.config import NwcSafRdtConfig


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
    start_ts = pd.Timestamp(start_time)
    cube_raw = reader.load(
        start_ts,
        origin_y=origin_y,
        origin_x=origin_x,
        global_mean=None,
        global_std=None,
    )
    return np.asarray(cube_raw, dtype=np.float32)


def run_benchmark(
    zarr_root: Path,
    manifest_csv: Path,
    num_windows: int = 4,
) -> dict[str, object]:
    df = pd.read_csv(manifest_csv)
    indices = list(range(min(num_windows, len(df))))

    cfg = NwcSafCmaCtConfig()
    ir_proxy_cfg = IrCloudTypeConfig()
    rdt_anc_cfg = NwcSafRdtConfig()

    window_reports: list[dict[str, object]] = []
    total_frames = 0
    total_runtime_s = 0.0

    agg_ct15_counts = {c.name: 0 for c in CtClass if c.value <= 15}
    agg_bridge6_counts = {c.name: 0 for c in RdtBridgeCtClass}
    agg_cumuliform_counts = {c.name: 0 for c in CtCumuliformClass if c.value <= 5}
    agg_multilayer_counts = {c.name: 0 for c in CtMultilayerClass if c.value <= 3}
    agg_cma_quality_counts = {c.name: 0 for c in CmaQuality}
    agg_ct_quality_counts = {c.name: 0 for c in CmaQuality}

    binary_cloud_agreements_vs_proxy: list[float] = []
    six_class_agreements_vs_proxy: list[float] = []
    opaque_agreements_vs_rdt_anc: list[float] = []

    for row_idx in indices:
        row = df.iloc[row_idx].to_dict()
        clip_id = str(row.get("clip_id", f"clip_{row_idx}"))
        start_time = str(row["start_time"])
        origin_y = int(row["origin_y"])
        origin_x = int(row["origin_x"])

        seq_7ch = load_real_seviri_window(
            zarr_root,
            start_time=start_time,
            origin_y=origin_y,
            origin_x=origin_x,
            num_frames=16,
            crop_size=224,
        )
        t_len, _, h_dim, w_dim = seq_7ch.shape

        t0 = time.perf_counter()
        cma_ct_steps = run_nwcsaf_cma_ct_sequence(seq_7ch, cfg)
        elapsed_s = time.perf_counter() - t0

        total_frames += t_len
        total_runtime_s += elapsed_s

        # Run lightweight 6-class IR proxy from src/emma_gpm/rdt/cloud_type_ir.py
        # and 6-class RDT ancillary from RDT_original/rdt_nwcsaf/ancillary.py
        proxy_tracker = SeviriIrCloudTypeTracker(ir_proxy_cfg)
        proxy_classes = []
        rdt_anc_opaque = []
        times = pd.date_range(pd.Timestamp(start_time), periods=t_len, freq="5min")
        for t_i in range(t_len):
            p_res = proxy_tracker.step(seq_7ch[t_i], times[t_i])
            proxy_classes.append(p_res.ct_class)

            anc_res = compute_nwcsaf_cloud_type(
                ir108=seq_7ch[t_i, 4],
                ir120=seq_7ch[t_i, 5],
                ir087=seq_7ch[t_i, 2],
                wv062=seq_7ch[t_i, 0],
                wv073=seq_7ch[t_i, 1],
            )
            rdt_anc_opaque.append(anc_res.opaque_cloud_mask)

        win_ct15 = np.stack([s.ct for s in cma_ct_steps], axis=0)
        win_bridge6 = np.stack([s.rdt_bridge_class for s in cma_ct_steps], axis=0)
        win_cma = np.stack([s.cma for s in cma_ct_steps], axis=0)
        win_opaque = np.stack([s.opaque_cloud_mask for s in cma_ct_steps], axis=0)
        win_proxy6 = np.stack(proxy_classes, axis=0)
        win_anc_opaque = np.stack(rdt_anc_opaque, axis=0)

        bin_agree = float(np.mean((win_cma == int(CmaClass.CLOUDY)) == (win_proxy6 > 0)))
        six_agree = float(np.mean(win_bridge6 == win_proxy6))
        opa_agree = float(np.mean(win_opaque == win_anc_opaque))

        binary_cloud_agreements_vs_proxy.append(bin_agree)
        six_class_agreements_vs_proxy.append(six_agree)
        opaque_agreements_vs_rdt_anc.append(opa_agree)

        for s in cma_ct_steps:
            for c in CtClass:
                if c.value <= 15:
                    agg_ct15_counts[c.name] += int(np.sum(s.ct == c.value))
            for c in RdtBridgeCtClass:
                agg_bridge6_counts[c.name] += int(np.sum(s.rdt_bridge_class == c.value))
            for c in CtCumuliformClass:
                if c.value <= 5:
                    agg_cumuliform_counts[c.name] += int(np.sum(s.ct_cumuliform == c.value))
            for c in CtMultilayerClass:
                if c.value <= 3:
                    agg_multilayer_counts[c.name] += int(np.sum(s.ct_multilayer == c.value))
            for c in CmaQuality:
                agg_cma_quality_counts[c.name] += int(np.sum(s.cma_quality == c.value))
                agg_ct_quality_counts[c.name] += int(np.sum(s.ct_quality == c.value))

        window_reports.append(
            {
                "window_index": int(row_idx),
                "clip_id": clip_id,
                "start_time": start_time,
                "origin_y": origin_y,
                "origin_x": origin_x,
                "shape": [int(t_len), 7, int(h_dim), int(w_dim)],
                "runtime_s": round(elapsed_s, 4),
                "ms_per_frame": round(1000.0 * elapsed_s / t_len, 2),
                "cloudy_fraction_cma": round(
                    float(np.mean(win_cma == int(CmaClass.CLOUDY))), 4
                ),
                "opaque_cloud_fraction_ct": round(float(np.mean(win_opaque)), 4),
                "semitransparent_fraction_ct": round(
                    float(np.mean([s.semitransparent_mask for s in cma_ct_steps])), 4
                ),
                "multilayer_fraction_ct": round(
                    float(
                        np.mean(
                            win_ct15
                            == int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM)
                        )
                    ),
                    4,
                ),
                "fractional_cloud_fraction_ct": round(
                    float(np.mean(win_ct15 == int(CtClass.FRACTIONAL_CLOUDS))), 4
                ),
                "binary_cloud_agreement_vs_ir_proxy": round(bin_agree, 4),
                "six_class_bridge_agreement_vs_ir_proxy": round(six_agree, 4),
                "opaque_mask_agreement_vs_rdt_ancillary": round(opa_agree, 4),
            }
        )

    total_pixels = sum(agg_ct15_counts.values())
    ct15_distribution_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_ct15_counts.items()
    }
    bridge6_distribution_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_bridge6_counts.items()
    }
    cumuliform_distribution_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_cumuliform_counts.items()
    }
    multilayer_distribution_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_multilayer_counts.items()
    }
    cma_quality_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_cma_quality_counts.items()
    }
    ct_quality_pct = {
        k: round(100.0 * v / max(total_pixels, 1), 3) for k, v in agg_ct_quality_counts.items()
    }

    summary = {
        "total_windows": len(window_reports),
        "total_frames": int(total_frames),
        "total_pixels": int(total_pixels),
        "total_runtime_s": round(total_runtime_s, 4),
        "mean_ms_per_frame": round(1000.0 * total_runtime_s / max(total_frames, 1), 2),
        "mean_binary_cloud_agreement_vs_ir_proxy": round(
            float(np.mean(binary_cloud_agreements_vs_proxy)), 4
        ),
        "mean_six_class_bridge_agreement_vs_ir_proxy": round(
            float(np.mean(six_class_agreements_vs_proxy)), 4
        ),
        "mean_opaque_mask_agreement_vs_rdt_ancillary": round(
            float(np.mean(opaque_agreements_vs_rdt_anc)), 4
        ),
        "ct15_distribution_pct": ct15_distribution_pct,
        "rdt_bridge6_distribution_pct": bridge6_distribution_pct,
        "ct_cumuliform_distribution_pct": cumuliform_distribution_pct,
        "ct_multilayer_distribution_pct": multilayer_distribution_pct,
        "cma_quality_pct": cma_quality_pct,
        "ct_quality_pct": ct_quality_pct,
        "windows": window_reports,
    }

    out_dir = REPO_ROOT / "CT_original" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ct_seviri_benchmark_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NWC SAF GEO-CMA & GEO-CT on real SEVIRI data.")
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(r"E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "catalogs" / "seviri_pretraining_random_100.csv",
    )
    parser.add_argument("--num-windows", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args.zarr_root, args.manifest, args.num_windows)


if __name__ == "__main__":
    main()
