"""Download official NWC SAF GEO-CMA public GIFs, reproject local 7-ch SEVIRI CMA, and render a Top/Bottom comparison MP4 video."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import shapely
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
CMA_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(CMA_ROOT) not in sys.path:
    sys.path.insert(0, str(CMA_ROOT))

from cma_nwcsaf import CmaCloudSnowClass, NwcSafCmaConfig, NwcSafCmaTracker
from emma_gpm.nwcsaf_rdt_archive import GifGeoReference, differential_view_parallax
from emma_gpm.rdt.storage import DomainFrameReader

# Official NWC SAF GEO-CMA Cloud & Snow Mask RGB Palette (v2025_MSG)
CMA_CLOUDSNOW_PALETTE: dict[int, tuple[int, int, int]] = {
    int(CmaCloudSnowClass.CLOUD_FREE): (0, 0, 0),                    # Black: Cloud-free
    int(CmaCloudSnowClass.CLOUD): (255, 0, 0),                       # Red: Cloud (except thin ice over snow)
    int(CmaCloudSnowClass.THIN_ICE_CLOUD_OVER_SNOW): (200, 175, 250),# Lavender: Thin ice cloud over snow/ice
    int(CmaCloudSnowClass.SNOW_OR_ICE): (250, 190, 250),             # Pink: Snow/Ice
}

MAP_BOX_V2025_CMA = (9, 38, 2212, 1188)  # (x0, y0, x1, y1) -> 2203 x 1150


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def download_official_cma_gifs(
    timestamps: list[pd.Timestamp],
    out_dir: Path,
) -> dict[pd.Timestamp, tuple[Path, Path]]:
    """Download official NWC SAF `.cma_cloudsnow.gif` and `.cma.gif` files for the requested timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[pd.Timestamp, tuple[Path, Path]] = {}
    for ts in timestamps:
        doy = ts.timetuple().tm_yday
        ymd = ts.strftime("%Y%m%d")
        hms = ts.strftime("%H%M%S")
        base_url = f"https://www.nwcsaf.org/ARCHIVO/v2025_MSG/GIF/{ts.year}{doy:03d}"

        fname_cs = f"S_NWC_CMA_MSG3_REFSYS-VISIR_{ymd}T{hms}Z.cma_cloudsnow.gif"
        fname_bin = f"S_NWC_CMA_MSG3_REFSYS-VISIR_{ymd}T{hms}Z.cma.gif"
        path_cs = out_dir / fname_cs
        path_bin = out_dir / fname_bin

        for url_file, dst_path in [(fname_cs, path_cs), (fname_bin, path_bin)]:
            if not dst_path.exists() or dst_path.stat().st_size < 1000:
                url = f"{base_url}/{url_file}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    dst_path.write_bytes(resp.read())
        downloaded[ts] = (path_cs, path_bin)
    return downloaded


def build_reprojection_lut(
    reader: DomainFrameReader,
    first_stamp: pd.Timestamp,
    georef: GifGeoReference,
    map_box: tuple[int, int, int, int],
    cloud_height_km: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact MSG4 (9.5E) -> MSG3 (0.0E) GIF pixel lookup table and Natural Earth 50m land mask."""
    _ = reader.load(first_stamp)
    raw_x, raw_y, projection = reader.geometry
    x, y = raw_x.copy(), raw_y.copy()
    if y[0] < y[-1]:
        y = y[::-1]
    if x[0] > x[-1]:
        x = x[::-1]

    xx, yy = np.meshgrid(x, y)
    to_lonlat = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(xx, yy)
    lon = np.asarray(lon, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float32)

    shp_path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    land_union = unary_union(list(shpreader.Reader(shp_path).geometries()))
    shapely.prepare(land_union)
    valid_ll = np.isfinite(lon) & np.isfinite(lat)
    safe_lon = np.where(valid_ll, lon, 0.0).astype(np.float64)
    safe_lat = np.where(valid_ll, lat, 0.0).astype(np.float64)
    is_land = shapely.contains_xy(land_union, safe_lon, safe_lat) & valid_ll

    x0, y0, x1, y1 = map_box
    cols = np.arange(x0, x1, dtype=np.float64)
    rows = np.arange(y0, y1, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)
    tx, ty = georef.pixel_to_projected(rr, cc)
    ref_to_lonlat = Transformer.from_crs(georef.geos_crs, "EPSG:4326", always_xy=True)
    tlon, tlat = ref_to_lonlat.transform(tx, ty)

    slon, slat, pvalid = differential_view_parallax(
        tlon,
        tlat,
        source_longitude_deg=0.0,
        target_longitude_deg=9.5,
        cloud_height_km=cloud_height_km,
    )
    lonlat_to_src = Transformer.from_crs("EPSG:4326", CRS.from_user_input(projection), always_xy=True)
    sx, sy = lonlat_to_src.transform(slon, slat)
    scol = (sx - x[0]) / (x[-1] - x[0]) * (len(x) - 1)
    srow = (sy - y[0]) / (y[-1] - y[0]) * (len(y) - 1)
    inside = (
        pvalid
        & np.isfinite(scol)
        & np.isfinite(srow)
        & (scol >= -0.5)
        & (scol < len(x) - 0.5)
        & (srow >= -0.5)
        & (srow < len(y) - 0.5)
    )
    srow_i = np.clip(np.rint(np.where(inside, srow, 0.0)), 0, len(y) - 1).astype(np.int32)
    scol_i = np.clip(np.rint(np.where(inside, scol, 0.0)), 0, len(x) - 1).astype(np.int32)
    return lat, lon, is_land, inside, pvalid, srow_i, scol_i


def render_bottom_cma_panel(
    ref_gif_rgb: np.ndarray,
    our_cs_mapped: np.ndarray,
    inside_footprint: np.ndarray,
    is_earth_disk: np.ndarray,
    map_box: tuple[int, int, int, int],
    timestamp_str: str,
    metrics: dict[str, float],
) -> np.ndarray:
    """Render our reprojected Pure-IR CMA product using the exact NWC SAF canvas, palette, coastlines, and colorbar."""
    panel = ref_gif_rgb.copy()
    x0, y0, x1, y1 = map_box
    ref_map = ref_gif_rgb[y0:y1, x0:x1]
    H, W, _ = ref_map.shape

    # Identify static reference overlays in the official GIF:
    # - Exclude the top-right NWC SAF logo box and top-left copyright box so no logos/silhouettes are copied
    rr, cc = np.mgrid[0:H, 0:W]
    is_logo_box = (rr < 95) & (cc > W - 430)
    is_copyright_box = (rr < 48) & (cc < 530)
    is_coast_or_grid = (
        (ref_map[:, :, 0] == 255)
        & (ref_map[:, :, 1] == 255)
        & (ref_map[:, :, 2] == 255)
        & (~is_logo_box)
        & (~is_copyright_box)
    )

    # Start with dark slate gray for regions outside our local MSG4 RSS crop
    our_map = np.full_like(ref_map, 55, dtype=np.uint8)
    for cls_val, rgb in CMA_CLOUDSNOW_PALETTE.items():
        m = inside_footprint & (our_cs_mapped == cls_val)
        our_map[m] = rgb

    # Outer space is (100, 100, 100) wherever outside Earth disk
    our_map[~is_earth_disk] = (100, 100, 100)

    # Restore white coastlines and borders on top
    our_map[is_coast_or_grid] = (255, 255, 255)
    panel[y0:y1, x0:x1] = our_map

    # Erase any leftover logo card or copyright notice in outer space
    panel[38:130, 1820:2220] = (100, 100, 100)
    panel[38:82, 0:530] = (100, 100, 100)

    # Annotate header bar on the bottom panel to identify our Pure-IR 7ch model and live metrics
    pil_img = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil_img)
    font_hdr = _load_font(22, bold=True)
    font_sub = _load_font(19, bold=False)

    # Replace top header strip (0..38) across entire width with clean dark banner for our product
    draw.rectangle([0, 0, panel.shape[1], 38], fill=(18, 28, 42))
    header_left = f"NOSTRA IMPLEMENTAZIONE PURE-SEVIRI 7 CANALI IR (CMA_original)  |  {timestamp_str}"
    header_right = (
        f"Acc: {100*metrics['accuracy']:.1f}%  |  "
        f"POD: {100*metrics['pod']:.1f}%  |  "
        f"FAR: {100*metrics['far']:.1f}%  |  "
        f"IoU: {100*metrics['iou']:.1f}%  |  "
        f"Cloudy: Ref {100*metrics['ref_cloudy_frac']:.1f}% vs Our {100*metrics['our_cloudy_frac']:.1f}%"
    )
    draw.text((14, 6), header_left, fill=(130, 220, 255), font=font_hdr)
    draw.text((1120, 7), header_right, fill=(255, 235, 140), font=font_sub)
    return np.asarray(pil_img)


def main() -> None:
    zarr_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_pilot\rss_zarr")
    staging_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot")
    gif_dir = staging_root / "gifs_v2025_msg"
    out_dir = staging_root / "cma_comparison_20261027"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DomainFrameReader(zarr_root)
    georef = GifGeoReference.from_json(ROOT / "configs" / "nwcsaf_rdt_gif_v2025_msg.json")

    all_5min_stamps = list(
        pd.date_range("2026-10-27T09:00:00", "2026-10-27T12:45:00", freq="5min")
    )
    quarter_hour_stamps = [ts for ts in all_5min_stamps if ts.minute in (0, 15, 30, 45)]

    print(f"[1/4] Downloading {len(quarter_hour_stamps)} official NWC SAF GEO-CMA GIF pairs...")
    gif_paths = download_official_cma_gifs(quarter_hour_stamps, gif_dir)

    print("[2/4] Building MSG4 (9.5E) -> MSG3 (0.0E) parallax-corrected reprojection LUT & Land/Sea mask...")
    lat, lon, is_land, inside, pvalid, srow_i, scol_i = build_reprojection_lut(
        reader, all_5min_stamps[0], georef, MAP_BOX_V2025_CMA, cloud_height_km=5.0
    )

    print("[3/4] Running causal 5-min Pure-IR NwcSafCmaTracker & rendering Top/Bottom frames...")
    cfg = NwcSafCmaConfig.full_domain_pure_ir()
    tracker = NwcSafCmaTracker(
        cfg,
        lat_deg=lat,
        lon_deg=lon,
        is_land=is_land,
    )

    video_frames: list[np.ndarray] = []
    metrics_rows: list[dict[str, float | str | int]] = []
    x0, y0, x1, y1 = MAP_BOX_V2025_CMA

    for ts in all_5min_stamps:
        cube = reader.load(ts)
        bt_dict = {
            "WV_062": cube[0],
            "WV_073": cube[1],
            "IR_087": cube[2],
            "IR_097": cube[3],
            "IR_108": cube[4],
            "IR_120": cube[5],
            "IR_134": cube[6],
        }
        cma_out, _ = tracker.step(bt_dict, timestamp=ts.to_pydatetime())

        if ts not in gif_paths:
            continue

        path_cs, _ = gif_paths[ts]
        ref_gif_rgb = np.asarray(Image.open(path_cs).convert("RGB"))
        ref_map = ref_gif_rgb[y0:y1, x0:x1]

        # Decode official reference GIF classes inside map box
        ref_clear = (ref_map[:, :, 0] == 0) & (ref_map[:, :, 1] == 0) & (ref_map[:, :, 2] == 0)
        ref_cloud = (ref_map[:, :, 0] == 255) & (ref_map[:, :, 1] == 0) & (ref_map[:, :, 2] == 0)
        ref_thin_snow = (ref_map[:, :, 0] == 200) & (ref_map[:, :, 1] == 175) & (ref_map[:, :, 2] == 250)
        ref_snow = (ref_map[:, :, 0] == 250) & (ref_map[:, :, 1] == 190) & (ref_map[:, :, 2] == 250)

        ref_any_cloud = ref_cloud | ref_thin_snow
        ref_any_clear = ref_clear | ref_snow

        our_bin_m = cma_out.cma[srow_i, scol_i]
        our_cs_m = cma_out.cma_cloudsnow[srow_i, scol_i]
        valid_eval = inside & (ref_any_cloud | ref_any_clear) & (our_bin_m <= 1)

        our_cloudy = our_bin_m == 1
        tp = int((valid_eval & ref_any_cloud & our_cloudy).sum())
        tn = int((valid_eval & ref_any_clear & (~our_cloudy)).sum())
        fp = int((valid_eval & ref_any_clear & our_cloudy).sum())
        fn = int((valid_eval & ref_any_cloud & (~our_cloudy)).sum())
        n_tot = max(tp + tn + fp + fn, 1)

        acc = (tp + tn) / n_tot
        pod = tp / max(tp + fn, 1)
        far = fp / max(tp + fp, 1)
        iou = tp / max(tp + fp + fn, 1)
        ref_cf = (tp + fn) / n_tot
        our_cf = (tp + fp) / n_tot

        row = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_eval_pixels": n_tot,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": acc,
            "pod": pod,
            "far": far,
            "iou": iou,
            "ref_cloudy_frac": ref_cf,
            "our_cloudy_frac": our_cf,
        }
        metrics_rows.append(row)
        print(
            f"  {row['timestamp']} | Acc={100*acc:5.2f}% | POD={100*pod:5.2f}% | "
            f"FAR={100*far:5.2f}% | IoU={100*iou:5.2f}% | "
            f"Cloudy: Ref={100*ref_cf:5.2f}% Our={100*our_cf:5.2f}%"
        )

        # Render bottom panel and stack Top (Official NWC SAF) over Bottom (Ours)
        bottom_panel = render_bottom_cma_panel(
            ref_gif_rgb,
            our_cs_m,
            inside,
            pvalid,
            MAP_BOX_V2025_CMA,
            ts.strftime("%Y-%m-%d %H:%M UTC"),
            row,
        )

        # Add a thin 4-pixel gold divider between Top and Bottom panels
        divider = np.full((4, ref_gif_rgb.shape[1], 3), (240, 190, 60), dtype=np.uint8)
        stacked = np.vstack([ref_gif_rgb, divider, bottom_panel])

        # Downscale width to 1600 px (even height/width for H.264 macroblocks)
        target_w = 1600
        scale = target_w / stacked.shape[1]
        target_h = int(round(stacked.shape[0] * scale / 2.0)) * 2
        stacked_resized = np.asarray(
            Image.fromarray(stacked).resize((target_w, target_h), Image.Resampling.LANCZOS)
        )
        video_frames.append(stacked_resized)

        if ts == pd.Timestamp("2026-10-27T12:00:00"):
            Image.fromarray(stacked).save(out_dir / "cma_comparison_20261027T1200Z_fullres.png")
            Image.fromarray(stacked_resized).save(out_dir / "cma_comparison_20261027T1200Z_preview.png")

    print("[4/4] Writing MP4 video and summary metrics...")
    video_path = out_dir / "nwcsaf_vs_local_cma_20261027_0900_1245.mp4"
    gif_out_path = out_dir / "nwcsaf_vs_local_cma_20261027_0900_1245.gif"
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    vh, vw, _ = video_frames[0].shape
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{vw}x{vh}",
        "-r",
        "2",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in video_frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"FFmpeg failed ({rc}): {err}")

    pil_frames = [Image.fromarray(fr) for fr in video_frames]
    pil_frames[0].save(
        gif_out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=500,
        loop=0,
    )

    df = pd.DataFrame(metrics_rows)
    df.to_csv(out_dir / "cma_comparison_metrics_20261027.csv", index=False)

    summary = {
        "product": "GEO-CMA v2025 (Cloud Mask & Cloud/Snow Mask)",
        "date": "2026-10-27",
        "period_utc": "09:00 to 12:45 UTC (16 quarter-hour slots)",
        "mean_accuracy": float(df["accuracy"].mean()),
        "mean_pod": float(df["pod"].mean()),
        "mean_far": float(df["far"].mean()),
        "mean_iou": float(df["iou"].mean()),
        "mean_ref_cloudy_frac": float(df["ref_cloudy_frac"].mean()),
        "mean_our_cloudy_frac": float(df["our_cloudy_frac"].mean()),
        "video_path": str(video_path),
        "preview_png": str(out_dir / "cma_comparison_20261027T1200Z_preview.png"),
    }
    (out_dir / "cma_comparison_summary_20261027.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Done! Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
