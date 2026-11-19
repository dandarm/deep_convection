#!/usr/bin/env python3
"""Controlled A/B experiment: Deterministic SubPatchFPN3D (3D PixelShuffle) vs
legacy trilinear LightweightFPN3D on the exact same frozen backbones, feature caches,
12-train/6-val/6-test dataset, and 15-clip extended held-out video gallery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "CMA_original")]

from emma_gpm.cloud_probe import state_digest
from emma_gpm.rdt.storage import atomic_json
from emma_gpm.seviri_clips import normalize_seviri_clip
from emma_gpm.subpatch_probe import SubPatchThreeHeadProbe
from emma_gpm.three_head_data import CLASS_NAMES
from emma_gpm.three_head_model import SegmentationMetrics, ThreeHeadProbe, class_weights, make_encoder, probe_loss
from emma_gpm.three_head_training import ProbeDataset, _loader


def atomic_torch(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".pt.part")
    torch.save(payload, tmp)
    tmp.replace(path)

OUT_ROOT = ROOT / "results" / "local_probe_loki"
DATASET_DIR = OUT_ROOT / "three_head_dataset"
LEGACY_PROBES_DIR = OUT_ROOT / "three_head_probes"
SUBPATCH_DIR = OUT_ROOT / "three_head_probes_subpatch"
GALLERY_DIR = LEGACY_PROBES_DIR / "extended_video_gallery"
VIDEOS_DIR = SUBPATCH_DIR / "videos"

SOURCE_131K = ROOT / "results" / "leonardo_checkpoints" / "ddp_base_future_150_20261104" / "training"
SOURCE_10K = ROOT / "results" / "leonardo_checkpoints" / "ddp_spaced_iou020_base_future_250_20261110" / "training"
SOURCE_RAND = OUT_ROOT / "videomae_small_7ch_source" / "encoder"

SEED = 20261109

PALETTE = {
    "cma": ["#1e293b", "#f1f5f9"],
    "ct": ["#1e293b", "#f59e0b", "#10b981", "#ef4444", "#06b6d4"],
    "rdt": ["#1e293b", "#3b82f6", "#fae014", "#ff8800", "#eb0055"],
}


def find_existing_feature_cache(encoder, dataset: Path, manifest: dict) -> Path:
    digest = state_digest(encoder)
    identity = {
        "encoder_sha256": digest,
        "statistics": manifest["statistics"],
        "manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "frames": 16,
        "levels": [3, 6, 9, 12],
        "dtype": "float16",
        "precision": "fp32",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = LEGACY_PROBES_DIR / "feature_cache" / key
    if not (folder / "complete.json").exists():
        raise FileNotFoundError(f"Expected precomputed feature cache not found: {folder}")
    return folder / "features.npy"


def evaluate_subpatch(model, loader, weights, nesting_weight: float = 0.1, collect_examples: bool = False):
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    metrics = {head: SegmentationMetrics(2 if head == "cma" else 5) for head in model.head_names}
    examples = {}
    with torch.no_grad():
        for inputs, labels, indices in loader:
            outputs = model(features=[f.float() for f in inputs.unbind(1)])
            loss, terms = probe_loss(outputs, labels, rdt_class_weights=weights, nesting_weight=nesting_weight)
            samples += len(inputs)
            for k, v in {"total": loss, **terms}.items():
                totals[k] = totals.get(k, 0.0) + float(v) * len(inputs)
            for head, logits in outputs.items():
                metrics[head].update(logits, labels[head], convective=(head == "rdt"))
            if collect_examples:
                for i, idx in enumerate(indices):
                    examples[int(idx)] = {
                        **{f"target_{k}": labels[k][i].cpu().numpy() for k in labels},
                        **{
                            f"prediction_{k}": (
                                (v[i, 0] >= 0).to(torch.uint8).cpu().numpy()
                                if k == "cma"
                                else v[i].argmax(0).to(torch.uint8).cpu().numpy()
                            )
                            for k, v in outputs.items()
                        },
                    }
    return (
        {
            "samples": samples,
            "losses": {k: v / samples for k, v in totals.items()},
            "metrics": {head: acc.result(convective=head == "rdt") for head, acc in metrics.items()},
        },
        examples,
    )


def train_subpatch_model(name: str, source: Path, kind: str, manifest: dict, epochs: int) -> dict:
    run_dir = SUBPATCH_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("epochs") == epochs:
            print(f"Reusing completed {name} ({epochs} epochs)", flush=True)
            return existing

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    encoder, prov = make_encoder(str(source), kind)
    features_path = find_existing_feature_cache(encoder, DATASET_DIR, manifest)
    model = SubPatchThreeHeadProbe(encoder, heads=("cma", "ct", "rdt"), out_channels=48, encoder_mode="frozen")
    initial_hash = state_digest(model.encoder)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    n_params = sum(p.numel() for p in head_params)

    print(f"\n{'='*72}", flush=True)
    print(f"TRAINING SUB-PATCH DECODER: {name} ({epochs} epochs, {n_params:,} head params)", flush=True)
    print(f"Feature cache: {features_path.parent.name} | Encoder frozen: {initial_hash[:12]}", flush=True)
    print(f"{'='*72}", flush=True)

    features_arr = np.load(features_path, mmap_mode="r")
    datasets = {
        split: ProbeDataset(DATASET_DIR, manifest, split, features=features_arr)
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: _loader(ds, batch_size=6, training=(split == "train"), seed=SEED, workers=0)
        for split, ds in datasets.items()
    }

    optimizer = torch.optim.AdamW(head_params, lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(e, epochs) / epochs))
    )
    weights = class_weights(manifest["support"]["train"]["rdt"])

    history = []
    best_loss = float("inf")
    best_epoch = 0
    updates = 0
    t_all = time.monotonic()

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        model.train()
        totals: dict[str, float] = {}
        samples = 0
        for inputs, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features=[f.float() for f in inputs.unbind(1)])
            loss, terms = probe_loss(outputs, labels, rdt_class_weights=weights, nesting_weight=0.1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            updates += 1
            samples += len(inputs)
            for k, v in {"total": loss, **terms}.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach()) * len(inputs)

        train_sec = time.monotonic() - t0
        val_t0 = time.monotonic()
        val_report, _ = evaluate_subpatch(model, loaders["validation"], weights, nesting_weight=0.1)
        val_sec = time.monotonic() - val_t0
        score = val_report["losses"]["total"]
        improved = score < best_loss
        if improved:
            best_loss = score
            best_epoch = epoch
            # Save only head + fpn weights (encoder is strictly frozen!)
            head_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            atomic_torch(run_dir / "head_best.pt", {"epoch": epoch, "val_loss": score, "head_state": head_state})

        history.append({
            "epoch": epoch,
            "updates": updates,
            "train": {k: v / samples for k, v in totals.items()},
            "validation": val_report,
            "train_seconds": train_sec,
            "validation_seconds": val_sec,
        })
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  Ep {epoch:03d}/{epochs} | train={totals['total']/samples:.4f} | "
                f"val={score:.4f} (CMA={val_report['losses']['cma']:.3f}, CT={val_report['losses']['ct']:.3f}, RDT={val_report['losses']['rdt']:.3f}) | "
                f"best={best_loss:.4f}@ep{best_epoch} | {train_sec + val_sec:.1f}s",
                flush=True,
            )

    # Verify frozen encoder unchanged
    assert state_digest(model.encoder) == initial_hash, "Frozen encoder weights changed!"

    # Load best head and evaluate on the 6 core test clips
    best_ckpt = torch.load(run_dir / "head_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["head_state"], strict=False)
    test_report, examples = evaluate_subpatch(model, loaders["test"], weights, nesting_weight=0.1, collect_examples=True)
    for idx, payload in examples.items():
        np.savez_compressed(run_dir / f"example_{idx:05d}.npz", **payload)

    atomic_json(run_dir / "history.json", history)
    result = {
        "name": name,
        "source": str(source),
        "kind": kind,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "head_parameters": n_params,
        "encoder_sha256": initial_hash,
        "encoder_unchanged": True,
        "elapsed_seconds": time.monotonic() - t_all,
        "test": test_report,
    }
    atomic_json(result_path, result)
    print(
        f"COMPLETED {name} in {(time.monotonic()-t_all)/60:.1f} min | best_ep={best_epoch} | "
        f"6-clip CMA={test_report['metrics']['cma']['iou'][1]:.3f} | "
        f"Cirrus={test_report['metrics']['ct']['iou'][4]:.3f} | "
        f"ActCSI={test_report['metrics']['rdt']['active_convection']['csi'][1]:.3f} | "
        f"OT={test_report['metrics']['rdt']['iou'][4]:.3f}",
        flush=True,
    )
    return result


def infer_subpatch_on_extra_gallery(specs: list[tuple[str, Path, str]], stats: dict) -> None:
    gallery = json.loads((GALLERY_DIR / "gallery_manifest.json").read_text())["clips"]
    extra = [c for c in gallery if c["source_kind"] == "extra_test"]
    mean_k, std_k = float(stats["global_mean_k"]), float(stats["global_std_k"])

    norm_clips = {}
    for c in extra:
        raw = np.load(c["clip_abs"])
        norm = normalize_seviri_clip(raw, mean_k, std_k).astype(np.float32, copy=False)
        norm_clips[c["id"]] = torch.from_numpy(norm).unsqueeze(0)

    print(f"\nRunning sub-patch inference on {len(extra)} extra held-out clips...", flush=True)
    for name, source, kind in specs:
        out_dir = SUBPATCH_DIR / name / "extra_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        if all((out_dir / f"{c['id']}.npz").exists() for c in extra):
            continue
        t0 = time.monotonic()
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        encoder, _ = make_encoder(str(source), kind)
        ckpt_meta = json.loads((SUBPATCH_DIR / name / "result.json").read_text())
        assert state_digest(encoder) == ckpt_meta["encoder_sha256"], f"Encoder digest mismatch for {name}!"
        model = SubPatchThreeHeadProbe(encoder, heads=("cma", "ct", "rdt"), out_channels=48, encoder_mode="frozen")
        ckpt = torch.load(SUBPATCH_DIR / name / "head_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["head_state"], strict=False)
        model.eval()
        with torch.no_grad():
            for c in extra:
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
                np.savez_compressed(out_dir / f"{c['id']}.npz", **payload)
        print(f"  Completed extra inference for {name} in {time.monotonic() - t0:.1f}s", flush=True)


def _load_pred(clip_meta: dict, is_subpatch: bool, model_name: str) -> dict[str, np.ndarray]:
    if is_subpatch:
        if clip_meta["source_kind"] == "core_test":
            p = SUBPATCH_DIR / model_name / f"example_{clip_meta['orig_index']:05d}.npz"
        else:
            p = SUBPATCH_DIR / model_name / "extra_predictions" / f"{clip_meta['id']}.npz"
    else:
        if clip_meta["source_kind"] == "core_test":
            p = LEGACY_PROBES_DIR / model_name / f"example_{clip_meta['orig_index']:05d}.npz"
        else:
            p = GALLERY_DIR / "predictions" / model_name / f"{clip_meta['id']}.npz"
    with np.load(p) as arc:
        return {k: arc[k] for k in arc.files}


def evaluate_15_clip_summary() -> dict:
    gallery = json.loads((GALLERY_DIR / "gallery_manifest.json").read_text())["clips"]
    entries = [
        ("Legacy-FPN  · Leonardo 10K", False, "Compare-Base-10K-spaced"),
        ("SubPatch-3D · Leonardo 10K", True, "SubPatch-Base-10K-spaced"),
        ("Legacy-FPN  · Leonardo 131K", False, "Compare-Base-131K-future"),
        ("SubPatch-3D · Leonardo 131K", True, "SubPatch-Base-131K-future"),
        ("Legacy-FPN  · Random", False, "E0-Rand"),
        ("SubPatch-3D · Random", True, "SubPatch-Rand"),
    ]
    summary = {}
    print("\n=== 15-CLIP HELD-OUT COMPARISON: LEGACY TRILINEAR FPN vs NEW SUB-PATCH 3D ===")
    for label, is_sp, mname in entries:
        tp = {k: 0 for k in ("cma", "low", "mid", "high", "cir", "act", "ot")}
        fp = dict(tp)
        fn = dict(tp)
        for c in gallery:
            arc = _load_pred(c, is_sp, mname)
            v = arc["target_cma"] != 255
            tc, pc = arc["target_cma"][v], arc["prediction_cma"][v]
            tp["cma"] += int(((tc == 1) & (pc == 1)).sum())
            fp["cma"] += int(((tc != 1) & (pc == 1)).sum())
            fn["cma"] += int(((tc == 1) & (pc != 1)).sum())

            v = arc["target_ct"] != 255
            tt, pt = arc["target_ct"][v], arc["prediction_ct"][v]
            for key, cid in (("low", 1), ("mid", 2), ("high", 3), ("cir", 4)):
                tp[key] += int(((tt == cid) & (pt == cid)).sum())
                fp[key] += int(((tt != cid) & (pt == cid)).sum())
                fn[key] += int(((tt == cid) & (pt != cid)).sum())

            v = arc["target_rdt"] != 255
            tr, pr = arc["target_rdt"][v], arc["prediction_rdt"][v]
            ta, pa = (tr >= 2) & (tr <= 4), (pr >= 2) & (pr <= 4)
            tp["act"] += int((ta & pa).sum())
            fp["act"] += int(((~ta) & pa).sum())
            fn["act"] += int((ta & (~pa)).sum())
            tp["ot"] += int(((tr == 4) & (pr == 4)).sum())
            fp["ot"] += int(((tr != 4) & (pr == 4)).sum())
            fn["ot"] += int(((tr == 4) & (pr != 4)).sum())

        scores = {k: (tp[k] / (tp[k] + fp[k] + fn[k]) if (tp[k] + fp[k] + fn[k]) else 0.0) for k in tp}
        summary[label] = scores
        print(
            f"  {label:29s} | CMA={scores['cma']:.3f} | Low={scores['low']:.3f} | Mid={scores['mid']:.3f} | "
            f"High={scores['high']:.3f} | Cirrus={scores['cir']:.3f} | ActCSI={scores['act']:.3f} | OT={scores['ot']:.3f}",
            flush=True,
        )
    atomic_json(SUBPATCH_DIR / "comparison_15clips_summary.json", summary)
    return summary


def plot_subpatch_vs_legacy_losses():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=150)
    curves = [
        ("Legacy FPN · 10K Spaced", LEGACY_PROBES_DIR / "Compare-Base-10K-spaced" / "history.json", "#047857", "--"),
        ("SubPatch · 10K Spaced", SUBPATCH_DIR / "SubPatch-Base-10K-spaced" / "history.json", "#047857", "-"),
        ("Legacy FPN · 131K", LEGACY_PROBES_DIR / "Compare-Base-131K-future" / "history.json", "#1d4ed8", "--"),
        ("SubPatch · 131K", SUBPATCH_DIR / "SubPatch-Base-131K-future" / "history.json", "#1d4ed8", "-"),
        ("Legacy FPN · Random", LEGACY_PROBES_DIR / "E0-Rand" / "history.json", "#64748b", "--"),
        ("SubPatch · Random", SUBPATCH_DIR / "SubPatch-Rand" / "history.json", "#dc2626", "-"),
    ]
    for ax, head in zip(axes, ("cma", "ct", "rdt")):
        for label, path, color, ls in curves:
            if not path.exists():
                continue
            hist = json.loads(path.read_text())
            ax.plot([r["epoch"] for r in hist], [r["validation"]["losses"][head] for r in hist],
                    label=label, color=color, linestyle=ls, linewidth=1.8 if ls == "-" else 1.2, alpha=0.9 if ls == "-" else 0.65)
        ax.set(xscale="log", yscale="log", xlabel="Epoch (2 updates/epoch)", ylabel="Validation loss", title=head.upper())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Confronto Validation Loss: Vecchia FPN Trilineare (tratteggiata) vs Nuovo Decoder Sub-Patch 3D (continua)", fontweight="bold")
    fig.tight_layout()
    out = SUBPATCH_DIR / "subpatch_vs_legacy_losses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss comparison figure: {out}", flush=True)


def render_subpatch_videos():
    from scripts.build_extended_test_videos import render_comparison_video

    gallery = json.loads((GALLERY_DIR / "gallery_manifest.json").read_text())["clips"]
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering side-by-side Sub-Patch comparison MP4 videos for all {len(gallery)} clips...", flush=True)

    html_blocks = [
        '<!DOCTYPE html><html lang="it"><head><meta charset="utf-8"><title>Confronto Decoder Sub-Patch 3D vs FPN Trilineare</title>'
        '<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:1240px;margin:24px auto;padding:0 16px;color:#0f172a;background:#fff;}'
        'h2{border-bottom:2px solid #334155;padding-bottom:6px;margin-top:28px;}</style></head><body>'
        '<h1>Confronto Decoder Sub-Patch 3D (PixelShuffle) vs Vecchia FPN Trilineare (15 Clip Held-Out)</h1>'
        '<p>Per ogni clip: (1) Confronto diretto Vecchia FPN vs Nuovo Sub-Patch su Leonardo 10K + Random Sub-Patch; '
        '(2) Confronto completo Leonardo 131K Sub-Patch vs Leonardo 10K Sub-Patch vs Random Sub-Patch.</p>'
    ]

    catalog = []
    for idx, meta in enumerate(gallery, 1):
        t0 = time.monotonic()
        clip_raw = np.load(meta["clip_abs"], mmap_mode="r")

        p_leg10 = _load_pred(meta, False, "Compare-Base-10K-spaced")
        p_sp10 = _load_pred(meta, True, "SubPatch-Base-10K-spaced")
        p_sp131 = _load_pred(meta, True, "SubPatch-Base-131K-future")
        p_sprand = _load_pred(meta, True, "SubPatch-Rand")

        v_ab = VIDEOS_DIR / f"vid_ab_trilinear_vs_subpatch_{meta['id']}.mp4"
        render_comparison_video(
            clip_raw, meta,
            [
                ("Leonardo 10K · Vecchia FPN", p_leg10, "#94a3b8"),
                ("Leonardo 10K · Sub-Patch 3D", p_sp10, "#047857"),
                ("Random · Sub-Patch 3D", p_sprand, "#dc2626"),
            ],
            v_ab,
            subtitle_tag="Confronto A/B: Vecchia FPN Trilineare vs Nuovo Decoder Sub-Patch 3D (PixelShuffle)",
        )

        v_all = VIDEOS_DIR / f"vid_subpatch_all_{meta['id']}.mp4"
        render_comparison_video(
            clip_raw, meta,
            [
                ("Leonardo 131K · Sub-Patch 3D", p_sp131, "#1d4ed8"),
                ("Leonardo 10K · Sub-Patch 3D", p_sp10, "#047857"),
                ("Random · Sub-Patch 3D", p_sprand, "#475569"),
            ],
            v_all,
            subtitle_tag="Tutti con Sub-Patch 3D: Leonardo 131K vs Leonardo 10K Spaced vs Controllo Random",
        )

        loc = f"lat {meta['center_lat']:.1f}°, lon {meta['center_lon']:.1f}°"
        html_blocks.append(
            f"<div style='margin:20px 0 8px 0;padding:8px 12px;background:#f8fafc;border-left:4px solid #047857;border-radius:4px;'>"
            f"<strong>Clip {idx:02d}/{len(gallery)} · {meta['id']}</strong> — {meta['period_label']} ({loc}) · <b>{meta.get('regime_note', '')}</b></div>"
            f"<div style='font-weight:600;color:#047857;margin-bottom:4px;'>▶ A/B: Vecchia FPN Trilineare vs Nuovo Sub-Patch 3D (Leonardo 10K + Random)</div>"
            f"<video controls loop muted playsinline preload='metadata' style='width:100%;max-width:1180px;border:1px solid #cbd5e1;border-radius:6px;'>"
            f"<source src='{v_ab.name}' type='video/mp4'></video>"
            f"<details style='margin:6px 0 18px 0;'><summary style='cursor:pointer;font-weight:600;color:#1d4ed8;'>▶ Tutti in Sub-Patch 3D: Leonardo 131K vs Leonardo 10K vs Random</summary>"
            f"<video controls loop muted playsinline preload='metadata' style='width:100%;max-width:1180px;border:1px solid #cbd5e1;border-radius:6px;margin-top:6px;'>"
            f"<source src='{v_all.name}' type='video/mp4'></video></details>"
        )
        catalog.append({**meta, "vid_ab": v_ab.name, "vid_all": v_all.name})
        print(f"  [{idx:02d}/{len(gallery)}] Rendered sub-patch videos for {meta['id']} in {time.monotonic() - t0:.1f}s", flush=True)

    html_blocks.append("</body></html>")
    (VIDEOS_DIR / "gallery_subpatch.html").write_text("\n".join(html_blocks), encoding="utf-8")
    atomic_json(VIDEOS_DIR / "subpatch_video_catalog.json", {"clips": catalog})
    print(f"Saved interactive gallery: {VIDEOS_DIR / 'gallery_subpatch.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    torch.set_num_threads(6)
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"

    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    specs = [
        ("SubPatch-Base-10K-spaced", SOURCE_10K, "pretrained"),
        ("SubPatch-Base-131K-future", SOURCE_131K, "pretrained"),
        ("SubPatch-Rand", SOURCE_RAND, "random"),
    ]

    for name, src, kind in specs:
        train_subpatch_model(name, src, kind, manifest, epochs=args.epochs)

    infer_subpatch_on_extra_gallery(specs, manifest["statistics"])
    evaluate_15_clip_summary()
    plot_subpatch_vs_legacy_losses()
    render_subpatch_videos()
    print("\nSUB-PATCH PROBE EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
