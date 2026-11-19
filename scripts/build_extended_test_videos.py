#!/usr/bin/env python3
"""Build an extended held-out test gallery (6 original + 10 extra diverse test clips)
and render synchronized 16-frame comparison videos (MP4) for the notebook.

For each clip, renders two 5-column videos (3 rows: CMA, CT, RDT, animated across all 16 frames):
1. vid_kinetics_<id>.mp4 : SEVIRI | Teacher | Leonardo 131K | Kinetics RGB | Random
2. vid_leonardo_<id>.mp4 : SEVIRI | Teacher | Leonardo 131K | Leonardo 10K Spaced | Random
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "CMA_original")]

from emma_gpm.rdt.storage import atomic_json
from emma_gpm.seviri_clips import IncompleteSeviriClipError, SeviriZarrClipReader, normalize_seviri_clip
from emma_gpm.three_head_data import CLASS_NAMES, _crop_sums, valid_origins
from emma_gpm.three_head_model import ThreeHeadProbe, make_encoder

SEVIRI_ROOT = Path(os.environ.get("LOKI_SEVIRI_ROOT", r"E:\Datasets\MSG-SEVIRI"))
OUT_ROOT = ROOT / "results" / "local_probe_loki"
TEACHERS_DIR = OUT_ROOT / "three_head_teachers" / "test"
DATASET_DIR = OUT_ROOT / "three_head_dataset"
PROBES_DIR = OUT_ROOT / "three_head_probes"
GALLERY_DIR = PROBES_DIR / "extended_video_gallery"
VIDEOS_DIR = PROBES_DIR / "notebook_figures" / "videos"

CROP_SIZE = 224
CLIP_FRAMES = 16
WARMUP_FRAMES = 3

MODELS = [
    "Compare-Base-131K-future",
    "Compare-Base-10K-spaced",
    "E0-Rand",
    "E1-Joint3",
]

PALETTE = {
    "cma": ["#1e293b", "#f1f5f9"],
    "ct": ["#1e293b", "#f59e0b", "#10b981", "#ef4444", "#06b6d4"],
    "rdt": ["#1e293b", "#3b82f6", "#fae014", "#ff8800", "#eb0055"],
}


def _load_window_targets(folder: Path, frames_slice: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in (*CLASS_NAMES, "valid", "quality")}
    for row in frames_slice:
        with np.load(folder / row["file"]) as archive:
            for k in arrays:
                arrays[k].append(archive[k])
    return {k: np.stack(v) for k, v in arrays.items()}


def _far_enough(y: int, x: int, chosen: list[tuple[int, int]], min_dist: float) -> bool:
    for cy, cx in chosen:
        if math.hypot(y - cy, x - cx) < min_dist:
            return False
    return True


def select_and_materialize_extra_clips(base_manifest: dict, extra_per_window: int = 5) -> list[dict]:
    """Select diverse extra held-out crops from the two precomputed test windows."""
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    gallery_manifest_path = GALLERY_DIR / "gallery_manifest.json"
    if gallery_manifest_path.exists():
        existing = json.loads(gallery_manifest_path.read_text())
        if len(existing.get("clips", [])) >= 12:
            print(f"[1/3] Reusing existing extended gallery ({len(existing['clips'])} clips)", flush=True)
            return existing["clips"]

    print(f"[1/3] Building extended test gallery (6 core + {2 * extra_per_window} diverse extra clips)...", flush=True)
    clips_meta: list[dict] = []

    # 1. Include the 6 original test clips first
    orig_test = [(i, s) for i, s in enumerate(base_manifest["samples"]) if s["split"] == "test"]
    used_by_period: dict[str, list[tuple[int, int]]] = {}
    for orig_idx, s in orig_test:
        entry = dict(s)
        entry["source_kind"] = "core_test"
        entry["orig_index"] = orig_idx
        entry["clip_abs"] = str((DATASET_DIR / s["clip"]).resolve())
        entry["regime_note"] = (
            "Convezione profonda / OT" if s.get("contains_ot")
            else "Nubi basse marine / stratocumuli (no convezione)"
        )
        clips_meta.append(entry)
        used_by_period.setdefault(s["period_tag"], []).append((int(s["canonical_y"]), int(s["canonical_x"])))

    # 2. Sample extra diverse crops from precomputed full-domain test teacher windows
    report = json.loads((TEACHERS_DIR / "complete.json").read_text())
    with np.load(TEACHERS_DIR / "geometry.npz") as geom:
        lat = geom["latitude"]
        lon = geom["longitude"]
        source_x = geom["source_x"]
        source_y = geom["source_y"]

    all_origins = valid_origins(lat, lon, CROP_SIZE, (-20.0, 40.0, 30.0, 70.0))
    cand_origins = all_origins[(all_origins[:, 0] % 16 == 0) & (all_origins[:, 1] % 16 == 0)]

    frames = report["frames"]
    win_starts = [i for i, f in enumerate(frames) if f["window_local_idx"] == 0]

    reader = SeviriZarrClipReader(SEVIRI_ROOT)
    try:
        for win_idx in win_starts:
            clip_frames_meta = frames[win_idx + WARMUP_FRAMES : win_idx + WARMUP_FRAMES + CLIP_FRAMES]
            period_tag = clip_frames_meta[0]["period_tag"]
            period_label = clip_frames_meta[0]["period_label"]
            start_time = clip_frames_meta[0]["time"]
            end_time = clip_frames_meta[-1]["time"]
            day = start_time[:10]

            targets = _load_window_targets(TEACHERS_DIR, clip_frames_meta)
            valid = targets["valid"]
            cma, ct, rdt = targets["cma"], targets["ct"], targets["rdt"]

            counts = {
                k: _crop_sums(vals.sum(0), cand_origins, CROP_SIZE)
                for k, vals in {
                    "valid": valid,
                    "cma_cloud": (cma == 1) & valid,
                    "low": (ct == 1) & valid,
                    "mid": (ct == 2) & valid,
                    "high_op": (ct == 3) & valid,
                    "cirrus": (ct == 4) & valid,
                    "cold_nc": (rdt == 1) & valid,
                    "growing": (rdt == 2) & valid,
                    "mature": (rdt == 3) & valid,
                    "ot": (rdt == 4) & valid,
                }.items()
            }

            rows = []
            for i, (y, x) in enumerate(cand_origins):
                v = int(counts["valid"][i])
                if v < 0.85 * CLIP_FRAMES * CROP_SIZE * CROP_SIZE:
                    continue
                rows.append({
                    "y": int(y), "x": int(x), "v": v,
                    "cloud_f": counts["cma_cloud"][i] / v,
                    "low_f": counts["low"][i] / v,
                    "mid_f": counts["mid"][i] / v,
                    "high_op_f": counts["high_op"][i] / v,
                    "cirrus_f": counts["cirrus"][i] / v,
                    "cold_nc_f": counts["cold_nc"][i] / v,
                    "growing_n": int(counts["growing"][i]),
                    "mature_n": int(counts["mature"][i]),
                    "ot_n": int(counts["ot"][i]),
                })

            def load_valid_clip(y: int, x: int) -> np.ndarray | None:
                ny = len(source_y) - y - CROP_SIZE if source_y[0] < source_y[-1] else y
                nx = len(source_x) - x - CROP_SIZE if source_x[0] > source_x[-1] else x
                try:
                    clip = reader.load(start_time, origin_y=ny, origin_x=nx)
                except IncompleteSeviriClipError:
                    return None
                if not (np.isfinite(clip).all() and (clip >= 100).all() and (clip <= 400).all()):
                    return None
                return clip.astype(np.float32, copy=False)

            used = list(used_by_period.get(period_tag, []))

            # Pick 5 meteorologically distinct regimes per test window:
            # 1) Triggering / growing convection
            # 2) Additional convective / OT / mature core
            # 3) Dominant Cirrus / multilayer plume
            # 4) Low / mid stratiform & broken clouds (no/little convection)
            # 5) Mixed clear-to-cloudy boundary (~30-60% cloud cover)
            regime_queries = [
                ("Convezione in crescita (Triggering / Growing)",
                 sorted([r for r in rows if r["growing_n"] > 500], key=lambda r: (r["growing_n"], r["ot_n"]), reverse=True)),
                ("Nucleo convettivo maturo / OT secondario",
                 sorted([r for r in rows if r["mature_n"] + r["ot_n"] > 2000], key=lambda r: (r["ot_n"], r["mature_n"]), reverse=True)),
                ("Cirri estesi e nubi alte sottili (Cirrus / Multilayer)",
                 sorted([r for r in rows if r["cirrus_f"] > 0.15 and r["mature_n"] < 20000], key=lambda r: r["cirrus_f"], reverse=True)),
                ("Nubi basse e medie stratiformi (Low / Mid Cloud)",
                 sorted([r for r in rows if r["low_f"] + r["mid_f"] > 0.20 and r["mature_n"] == 0], key=lambda r: r["low_f"] + r["mid_f"], reverse=True)),
                ("Transizione sereno / nuvoloso frammentato",
                 sorted([r for r in rows if 0.25 <= r["cloud_f"] <= 0.65 and r["low_f"] + r["cirrus_f"] > 0.15],
                        key=lambda r: abs(r["cloud_f"] - 0.45) - 0.5 * (r["cirrus_f"] + r["low_f"]))),
            ][:extra_per_window]

            for regime_note, cands in regime_queries:
                picked_row = None
                picked_clip = None
                for min_d in (160.0, 112.0, 64.0):
                    for cand in cands:
                        if not _far_enough(cand["y"], cand["x"], used, min_d):
                            continue
                        arr = load_valid_clip(cand["y"], cand["x"])
                        if arr is not None:
                            picked_row = cand
                            picked_clip = arr
                            break
                    if picked_row is not None:
                        break
                if picked_row is None or picked_clip is None:
                    continue

                y, x = picked_row["y"], picked_row["x"]
                used.append((y, x))
                clip_id = f"{start_time[:16].replace('-', '').replace(':', '')}_{y}_{x}"
                clip_file = GALLERY_DIR / f"{clip_id}.npy"
                tgt_file = GALLERY_DIR / f"{clip_id}_targets.npz"

                np.save(clip_file, picked_clip, allow_pickle=False)
                sl = (slice(None), slice(y, y + CROP_SIZE), slice(x, x + CROP_SIZE))
                v_crop = valid[sl]
                t_cma = np.where(v_crop, cma[sl], 255).astype(np.uint8)
                t_ct = np.where(v_crop, ct[sl], 255).astype(np.uint8)
                t_rdt = np.where(v_crop, rdt[sl], 255).astype(np.uint8)
                np.savez_compressed(tgt_file, target_cma=t_cma, target_ct=t_ct, target_rdt=t_rdt)

                cy, cx = y + CROP_SIZE // 2, x + CROP_SIZE // 2
                clips_meta.append({
                    "id": clip_id,
                    "split": "test_extended",
                    "source_kind": "extra_test",
                    "day": day,
                    "start_time": start_time,
                    "end_time": end_time,
                    "period_tag": period_tag,
                    "period_label": period_label,
                    "canonical_y": y,
                    "canonical_x": x,
                    "center_lat": float(lat[cy, cx]),
                    "center_lon": float(lon[cy, cx]),
                    "contains_ot": bool(picked_row["ot_n"] > 0),
                    "regime_note": regime_note,
                    "clip_abs": str(clip_file.resolve()),
                    "targets_abs": str(tgt_file.resolve()),
                })
                print(f"      Picked extra clip {clip_id} ({period_tag}): {regime_note} [lat={lat[cy, cx]:.1f}, lon={lon[cy, cx]:.1f}]", flush=True)
    finally:
        reader.close()

    atomic_json(gallery_manifest_path, {"clips": clips_meta})
    return clips_meta


def run_inference_on_extra_clips(clips_meta: list[dict], stats: dict) -> None:
    """Run frozen-backbone + best-head CPU inference on extra clips for all 4 models."""
    extra = [c for c in clips_meta if c["source_kind"] == "extra_test"]
    if not extra:
        return

    # Check which models still need inference
    pending_models = []
    for m_name in MODELS:
        out_dir = GALLERY_DIR / "predictions" / m_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(not (out_dir / f"{c['id']}.npz").exists() for c in extra):
            pending_models.append(m_name)

    if not pending_models:
        print("[2/3] All extended clip predictions already cached!", flush=True)
        return

    torch.set_num_threads(6)
    mean_k, std_k = float(stats["global_mean_k"]), float(stats["global_std_k"])

    # Pre-load normalized extra clips
    norm_clips = {}
    for c in extra:
        raw = np.load(c["clip_abs"])
        norm = normalize_seviri_clip(raw, mean_k, std_k).astype(np.float32, copy=False)
        norm_clips[c["id"]] = torch.from_numpy(norm).unsqueeze(0)  # (1, 16, 7, 224, 224)

    print(f"[2/3] Running inference on {len(extra)} extra clips for {len(pending_models)} models...", flush=True)
    for m_name in pending_models:
        t0 = time.monotonic()
        run_dir = PROBES_DIR / m_name
        cfg = json.loads((run_dir / "config.json").read_text())
        ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)

        encoder, _ = make_encoder(cfg["source"], cfg.get("kind", "pretrained"))
        model = ThreeHeadProbe(encoder, tuple(cfg["heads"]), encoder_mode="frozen")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        out_dir = GALLERY_DIR / "predictions" / m_name
        with torch.no_grad():
            for c in extra:
                out_path = out_dir / f"{c['id']}.npz"
                if out_path.exists():
                    continue
                outputs = model(norm_clips[c["id"]])
                preds = {
                    f"prediction_{k}": (
                        (v[0, 0] >= 0).to(torch.uint8).cpu().numpy()
                        if k == "cma"
                        else v[0].argmax(0).to(torch.uint8).cpu().numpy()
                    )
                    for k, v in outputs.items()
                }
                tgt = np.load(c["targets_abs"])
                payload = {
                    "target_cma": tgt["target_cma"],
                    "target_ct": tgt["target_ct"],
                    "target_rdt": tgt["target_rdt"],
                    **preds,
                }
                tgt.close()
                np.savez_compressed(out_path, **payload)
        print(f"      Completed {m_name} ({len(extra)} clips) in {time.monotonic() - t0:.1f}s", flush=True)


def _load_clip_predictions(clip_meta: dict, model_name: str) -> dict[str, np.ndarray]:
    if clip_meta["source_kind"] == "core_test":
        path = PROBES_DIR / model_name / f"example_{clip_meta['orig_index']:05d}.npz"
    else:
        path = GALLERY_DIR / "predictions" / model_name / f"{clip_meta['id']}.npz"
    with np.load(path) as arc:
        return {k: arc[k] for k in arc.files}


def render_comparison_video(
    clip_raw: np.ndarray,
    meta: dict,
    model_entries: list[tuple[str, dict[str, np.ndarray], str]],
    out_path: Path,
    subtitle_tag: str,
) -> None:
    """Fast in-place matplotlib animation -> H.264 MP4 (3 rows CMA/CT/RDT x 5 cols)."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return

    heads = ("cma", "ct", "rdt")
    ncols = 2 + len(model_entries)  # SEVIRI + Teacher + 3 models = 5 cols
    ref = model_entries[0][1]

    fig, axes = plt.subplots(3, ncols, figsize=(15.6, 9.2), dpi=100)
    fig.subplots_adjust(left=0.015, right=0.87, top=0.90, bottom=0.02, wspace=0.06, hspace=0.14)

    cmaps = {}
    norms = {}
    for h in heads:
        cm = ListedColormap(PALETTE[h])
        cm.set_bad("#64748b")
        cmaps[h] = cm
        norms[h] = BoundaryNorm(np.arange(len(CLASS_NAMES[h]) + 1) - 0.5, len(CLASS_NAMES[h]))

    # Precompute BTD WV6.2 - IR10.8 for Row 2 Col 0
    btd = clip_raw[:, 0] - clip_raw[:, 4]

    # Create artists once (t=0)
    seviri_ims = []
    seviri_specs = [
        (clip_raw[0, 4], "gray_r", 205, 305, "IR 10.8 µm (205–305 K)"),
        (clip_raw[0, 0], "gray_r", 210, 255, "WV 6.2 µm (210–255 K)"),
        (btd[0], "magma", -55, 5, "BTD WV6.2 − IR10.8 (K)"),
    ]
    for r, (arr0, cm, vmin, vmax, title) in enumerate(seviri_specs):
        im = axes[r, 0].imshow(arr0, cmap=cm, vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(title, fontsize=9.5, fontweight="bold")
        axes[r, 0].axis("off")
        seviri_ims.append(im)

    mask_ims = []  # list of (row, head, col_idx, im_artist, data_cube)
    for r, h in enumerate(heads):
        valid0 = ref[f"target_{h}"][0] != 255
        im_t = axes[r, 1].imshow(
            np.ma.array(ref[f"target_{h}"][0], mask=~valid0),
            cmap=cmaps[h], norm=norms[h], interpolation="nearest",
        )
        axes[r, 1].set_title(f"{h.upper()} · Originale (Teacher)", fontsize=9.5, fontweight="bold")
        axes[r, 1].axis("off")
        mask_ims.append((h, im_t, ref[f"target_{h}"]))

        for c_off, (label, data_dict, color) in enumerate(model_entries, start=2):
            im_m = axes[r, c_off].imshow(
                np.ma.array(data_dict[f"prediction_{h}"][0], mask=~valid0),
                cmap=cmaps[h], norm=norms[h], interpolation="nearest",
            )
            axes[r, c_off].set_title(f"{h.upper()} · {label}", fontsize=9.5, fontweight="bold", color=color)
            axes[r, c_off].axis("off")
            mask_ims.append((h, im_m, data_dict[f"prediction_{h}"]))

        # Dedicated colorbar positioned cleanly to the right of each row
        pos = axes[r, -1].get_position()
        cax = fig.add_axes([0.885, pos.y0, 0.013, pos.height])
        cbar = fig.colorbar(im_t, cax=cax, ticks=range(len(CLASS_NAMES[h])))
        cbar.ax.set_yticklabels(CLASS_NAMES[h])
        cbar.ax.tick_params(labelsize=8.2)

    loc = f"lat {meta['center_lat']:.1f}°, lon {meta['center_lon']:.1f}°"
    period = meta.get("period_label", meta.get("period_tag", meta["day"]))
    regime = meta.get("regime_note", "")
    base_title = f"{period} · {loc} · {regime}"
    suptitle = fig.suptitle("", fontsize=11.2, fontweight="bold", y=0.975)

    def update(t: int):
        seviri_ims[0].set_data(clip_raw[t, 4])
        seviri_ims[1].set_data(clip_raw[t, 0])
        seviri_ims[2].set_data(btd[t])
        valids = {h: ref[f"target_{h}"][t] != 255 for h in heads}
        for h, artist, cube in mask_ims:
            artist.set_data(np.ma.array(cube[t], mask=~valids[h]))
        suptitle.set_text(
            f"{base_title}\n{subtitle_tag}  |  Frame {t + 1:02d}/16 (t = +{t * 5:02d} min / 75 min)  ·  Fine clip: {meta['end_time'][:16].replace('T', ' ')} UTC"
        )
        return []

    update(0)
    ani = animation.FuncAnimation(fig, update, frames=CLIP_FRAMES, interval=400, blit=False)
    writer = animation.FFMpegWriter(
        fps=2.5,
        codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "25", "-preset", "fast", "-movflags", "+faststart"],
    )
    ani.save(out_path, writer=writer)
    plt.close(fig)


def main():
    base_manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    clips_meta = select_and_materialize_extra_clips(base_manifest, extra_per_window=5)
    run_inference_on_extra_clips(clips_meta, base_manifest["statistics"])

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Rendering synchronized 16-frame MP4 videos for {len(clips_meta)} clips...", flush=True)
    t_all = time.monotonic()

    catalog = []
    for idx, meta in enumerate(clips_meta, 1):
        t0 = time.monotonic()
        clip_raw = np.load(meta["clip_abs"], mmap_mode="r")
        preds = {m: _load_clip_predictions(meta, m) for m in MODELS}

        # Video 1: Leonardo 131K vs Kinetics RGB vs Random
        vid_kin = VIDEOS_DIR / f"vid_kinetics_{meta['id']}.mp4"
        render_comparison_video(
            clip_raw, meta,
            [
                ("Leonardo 131K (from scratch)", preds["Compare-Base-131K-future"], "#1d4ed8"),
                ("Kinetics RGB (Autori)", preds["E1-Joint3"], "#b91c1c"),
                ("Controllo Random", preds["E0-Rand"], "#475569"),
            ],
            vid_kin,
            subtitle_tag="Confronto 1: Leonardo 131K vs Kinetics RGB vs Controllo Random",
        )

        # Video 2: Leonardo 131K vs Leonardo 10K Spaced vs Random
        vid_leo = VIDEOS_DIR / f"vid_leonardo_{meta['id']}.mp4"
        render_comparison_video(
            clip_raw, meta,
            [
                ("Leonardo 131K (150 ep)", preds["Compare-Base-131K-future"], "#1d4ed8"),
                ("Leonardo 10K Spaced (250 ep)", preds["Compare-Base-10K-spaced"], "#047857"),
                ("Controllo Random", preds["E0-Rand"], "#475569"),
            ],
            vid_leo,
            subtitle_tag="Confronto 2: Leonardo 131K vs Leonardo 10K Spaced vs Controllo Random",
        )

        catalog.append({
            **meta,
            "vid_kinetics": str(vid_kin.relative_to(ROOT)).replace("\\", "/"),
            "vid_leonardo": str(vid_leo.relative_to(ROOT)).replace("\\", "/"),
            "vid_kinetics_kb": round(vid_kin.stat().st_size / 1024, 1),
            "vid_leonardo_kb": round(vid_leo.stat().st_size / 1024, 1),
        })
        print(
            f"      [{idx:02d}/{len(clips_meta)}] {meta['id']} rendered in {time.monotonic() - t0:.1f}s "
            f"({catalog[-1]['vid_kinetics_kb']} KB / {catalog[-1]['vid_leonardo_kb']} KB)",
            flush=True,
        )

    atomic_json(VIDEOS_DIR / "video_catalog.json", {"clips": catalog})
    print(f"\nCOMPLETED ALL {len(catalog) * 2} VIDEOS in {time.monotonic() - t_all:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
