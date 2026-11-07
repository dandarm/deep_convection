"""Download official NWC SAF GEO-CT public GIFs, reproject local 7-ch SEVIRI CT, and render a Top/Bottom comparison MP4 video."""

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
CT_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(CT_ROOT) not in sys.path:
    sys.path.insert(0, str(CT_ROOT))

from ct_nwcsaf import CtClass, NwcSafCmaCtConfig, NwcSafCmaCtTracker
from emma_gpm.nwcsaf_rdt_archive import GifGeoReference, differential_view_parallax
from emma_gpm.rdt.storage import DomainFrameReader

# Official 15-class NWC SAF GEO-CT RGB Palette (v2025_MSG)
CT_PALETTE: dict[int, tuple[int, int, int]] = {
    int(CtClass.CLOUD_FREE_LAND): (0, 120, 0),
    int(CtClass.CLOUD_FREE_SEA): (0, 0, 0),
    int(CtClass.LAND_SNOW): (250, 190, 250),
    int(CtClass.SEA_ICE): (220, 160, 220),
    int(CtClass.VERY_LOW_CLOUDS): (255, 100, 0),
    int(CtClass.LOW_CLOUDS): (255, 180, 0),
    int(CtClass.MID_LEVEL_CLOUDS): (240, 240, 0),
    int(CtClass.HIGH_OPAQUE_CLOUDS): (215, 215, 150),
    int(CtClass.VERY_HIGH_OPAQUE_CLOUDS): (230, 230, 230),
    int(CtClass.FRACTIONAL_CLOUDS): (200, 0, 200),
    int(CtClass.HIGH_SEMITRANSPARENT_THIN): (0, 80, 215),
    int(CtClass.HIGH_SEMITRANSPARENT_MODERATELY_THICK): (0, 180, 230),
    int(CtClass.HIGH_SEMITRANSPARENT_THICK): (0, 240, 240),
    int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_LOW_OR_MEDIUM): (90, 200, 160),
    int(CtClass.HIGH_SEMITRANSPARENT_ABOVE_SNOW_OR_ICE): (200, 175, 250),
}

# In ct.gif (2418 x 1702), the 2203 x 1150 map box is at x=[206..2409], y=[38..1188],
# shifted by dx=+197 px relative to rdt.gif/cma.gif (x=[9..2212], y=[38..1188])
MAP_BOX_V2025_CT = (206, 38, 2409, 1188)
GEOREF_BOX_V2025 = (9, 38, 2212, 1188)


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


def download_official_ct_gifs(
    timestamps: list[pd.Timestamp],
    out_dir: Path,
) -> dict[pd.Timestamp, Path]:
    """Download official NWC SAF `.ct.gif` files for the requested timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[pd.Timestamp, Path] = {}
    for ts in timestamps:
        doy = ts.timetuple().tm_yday
        ymd = ts.strftime("%Y%m%d")
        hms = ts.strftime("%H%M%S")
        base_url = f"https://www.nwcsaf.org/ARCHIVO/v2025_MSG/GIF/{ts.year}{doy:03d}"

        fname_ct = f"S_NWC_CT_MSG3_REFSYS-VISIR_{ymd}T{hms}Z.ct.gif"
        path_ct = out_dir / fname_ct

        if not path_ct.exists() or path_ct.stat().st_size < 1000:
            url = f"{base_url}/{fname_ct}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                path_ct.write_bytes(resp.read())
        downloaded[ts] = path_ct
    return downloaded


def build_reprojection_lut(
    reader: DomainFrameReader,
    first_stamp: pd.Timestamp,
    georef: GifGeoReference,
    georef_box: tuple[int, int, int, int],
    cloud_height_km: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    gx0, gy0, gx1, gy1 = georef_box
    cols = np.arange(gx0, gx1, dtype=np.float64)
    rows = np.arange(gy0, gy1, dtype=np.float64)
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


def to_regime_id(ct_arr: np.ndarray) -> np.ndarray:
    """Group 15 NWC SAF Cloud Type classes into 6 physical vertical/radiative regimes."""
    r = np.zeros_like(ct_arr, dtype=np.uint8)
    r = np.where(np.isin(ct_arr, [1, 2]), 1, r)              # Clear (Land / Sea)
    r = np.where(np.isin(ct_arr, [3, 4]), 2, r)              # Snow / Sea Ice
    r = np.where(np.isin(ct_arr, [5, 6, 10]), 3, r)          # Low / Very Low / Fractional
    r = np.where(ct_arr == 7, 4, r)                          # Mid-level Opaque
    r = np.where(np.isin(ct_arr, [8, 9]), 5, r)              # High / Very High Opaque
    r = np.where(np.isin(ct_arr, [11, 12, 13, 14, 15]), 6, r)# High Semitransparent / Multilayer
    return r


def render_bottom_ct_panel(
    ref_gif_rgb: np.ndarray,
    our_ct_mapped: np.ndarray,
    inside_footprint: np.ndarray,
    is_earth_disk: np.ndarray,
    map_box: tuple[int, int, int, int],
    timestamp_str: str,
    metrics: dict[str, float],
) -> np.ndarray:
    """Render our reprojected Pure-IR 15-class CT product using the exact NWC SAF canvas, palette, coastlines, and colorbar."""
    panel = ref_gif_rgb.copy()
    x0, y0, x1, y1 = map_box
    ref_map = ref_gif_rgb[y0:y1, x0:x1]
    H, W, _ = ref_map.shape

    # Exclude top-right NWC SAF logo box and top-left copyright box so no logos/silhouettes are copied
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

    our_map = np.full_like(ref_map, 55, dtype=np.uint8)
    for cls_val, rgb in CT_PALETTE.items():
        m = inside_footprint & (our_ct_mapped == cls_val)
        our_map[m] = rgb

    # Outer space is (100, 100, 100) wherever outside Earth disk
    our_map[~is_earth_disk] = (100, 100, 100)

    # Restore white coastlines and borders on top
    our_map[is_coast_or_grid] = (255, 255, 255)
    panel[y0:y1, x0:x1] = our_map

    # Erase any leftover logo card or copyright notice in outer space
    panel[38:130, 2020:2415] = (100, 100, 100)
    panel[38:82, 206:730] = (100, 100, 100)

    pil_img = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil_img)
    font_hdr = _load_font(22, bold=True)
    font_sub = _load_font(19, bold=False)

    # Replace top header strip (0..38) across entire width with clean dark banner for our product
    draw.rectangle([0, 0, panel.shape[1], 38], fill=(18, 28, 42))
    header_left = f"NOSTRA IMPLEMENTAZIONE PURE-SEVIRI 7 CANALI IR (CT_original)  |  {timestamp_str}"
    header_right = (
        f"Cloud/Clear Acc: {100*metrics['binary_cloud_acc']:.1f}%  |  "
        f"6-Regime Acc: {100*metrics['regime_6_acc']:.1f}%  |  "
        f"15-Class Exact Acc: {100*metrics['exact_15_acc']:.1f}%"
    )
    draw.text((14, 6), header_left, fill=(130, 220, 255), font=font_hdr)
    draw.text((1120, 7), header_right, fill=(255, 235, 140), font=font_sub)
    return np.asarray(pil_img)


def main() -> None:
    zarr_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_rdt_pilot\rss_zarr")
    staging_root = Path(r"E:\Datasets\Deep_convection\staging\nwcsaf_cma_ct_pilot")
    gif_dir = staging_root / "gifs_v2025_msg"
    out_dir = staging_root / "ct_comparison_20261027"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DomainFrameReader(zarr_root)
    georef = GifGeoReference.from_json(ROOT / "configs" / "nwcsaf_rdt_gif_v2025_msg.json")

    all_5min_stamps = list(
        pd.date_range("2026-10-27T09:00:00", "2026-10-27T12:45:00", freq="5min")
    )
    quarter_hour_stamps = [ts for ts in all_5min_stamps if ts.minute in (0, 15, 30, 45)]

    print(f"[1/4] Downloading {len(quarter_hour_stamps)} official NWC SAF GEO-CT GIFs...")
    gif_paths = download_official_ct_gifs(quarter_hour_stamps, gif_dir)

    print("[2/4] Building MSG4 (9.5E) -> MSG3 (0.0E) parallax-corrected reprojection LUT & Land/Sea mask...")
    lat, lon, is_land, inside, pvalid, srow_i, scol_i = build_reprojection_lut(
        reader, all_5min_stamps[0], georef, GEOREF_BOX_V2025, cloud_height_km=5.0
    )

    print("[3/4] Running causal 5-min Pure-IR NwcSafCmaCtTracker & rendering Top/Bottom frames...")
    cfg = NwcSafCmaCtConfig.full_domain_pure_ir()
    tracker = NwcSafCmaCtTracker(cfg)

    video_frames: list[np.ndarray] = []
    metrics_rows: list[dict[str, float | str | int]] = []
    x0, y0, x1, y1 = MAP_BOX_V2025_CT

    for ts in all_5min_stamps:
        cube = reader.load(ts)
        ct_step = tracker.step(
            cube,
            lat_deg=lat,
            lon_deg=lon,
            land_mask=is_land,
        )

        if ts not in gif_paths:
            continue

        path_ct = gif_paths[ts]
        ref_gif_rgb = np.asarray(Image.open(path_ct).convert("RGB"))
        ref_map = ref_gif_rgb[y0:y1, x0:x1]

        ref_ct = np.zeros(ref_map.shape[:2], dtype=np.uint8)
        for cls_id, rgb in CT_PALETTE.items():
            m = (ref_map[:, :, 0] == rgb[0]) & (ref_map[:, :, 1] == rgb[1]) & (ref_map[:, :, 2] == rgb[2])
            ref_ct[m] = cls_id

        our_ct_m = ct_step.ct[srow_i, scol_i]
        valid_eval = inside & (ref_ct >= 1) & (ref_ct <= 15) & (our_ct_m >= 1) & (our_ct_m <= 15)
        n_tot = max(int(valid_eval.sum()), 1)

        exact_15_acc = float((ref_ct[valid_eval] == our_ct_m[valid_eval]).mean())
        ref_reg = to_regime_id(ref_ct)
        our_reg = to_regime_id(our_ct_m)
        regime_6_acc = float((ref_reg[valid_eval] == our_reg[valid_eval]).mean())
        binary_cloud_acc = float(((ref_ct[valid_eval] >= 5) == (our_ct_m[valid_eval] >= 5)).mean())

        row: dict[str, float | str | int] = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_eval_pixels": n_tot,
            "binary_cloud_acc": binary_cloud_acc,
            "regime_6_acc": regime_6_acc,
            "exact_15_acc": exact_15_acc,
        }
        metrics_rows.append(row)
        print(
            f"  {row['timestamp']} | Cloud/Clear Acc={100*binary_cloud_acc:5.2f}% | "
            f"6-Regime Acc={100*regime_6_acc:5.2f}% | "
            f"15-Class Exact Acc={100*exact_15_acc:5.2f}%"
        )

        bottom_panel = render_bottom_ct_panel(
            ref_gif_rgb,
            our_ct_m,
            inside,
            pvalid,
            MAP_BOX_V2025_CT,
            ts.strftime("%Y-%m-%d %H:%M UTC"),
            row,
        )

        divider = np.full((4, ref_gif_rgb.shape[1], 3), (240, 190, 60), dtype=np.uint8)
        stacked = np.vstack([ref_gif_rgb, divider, bottom_panel])

        target_w = 1600
        scale = target_w / stacked.shape[1]
        target_h = int(round(stacked.shape[0] * scale / 2.0)) * 2
        stacked_resized = np.asarray(
            Image.fromarray(stacked).resize((target_w, target_h), Image.Resampling.LANCZOS)
        )
        video_frames.append(stacked_resized)

        if ts == pd.Timestamp("2026-10-27T12:00:00"):
            Image.fromarray(stacked).save(out_dir / "ct_comparison_20261027T1200Z_fullres.png")
            Image.fromarray(stacked_resized).save(out_dir / "ct_comparison_20261027T1200Z_preview.png")

    print("[4/4] Writing MP4 video and summary metrics...")
    video_path = out_dir / "nwcsaf_vs_local_ct_20261027_0900_1245.mp4"
    gif_out_path = out_dir / "nwcsaf_vs_local_ct_20261027_0900_1245.gif"
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
    df.to_csv(out_dir / "ct_comparison_metrics_20261027.csv", index=False)

    summary = {
        "product": "GEO-CT v2025 (15-Class Cloud Type)",
        "date": "2026-10-27",
        "period_utc": "09:00 to 12:45 UTC (16 quarter-hour slots)",
        "mean_binary_cloud_acc": float(df["binary_cloud_acc"].mean()),
        "mean_regime_6_acc": float(df["regime_6_acc"].mean()),
        "mean_exact_15_acc": float(df["exact_15_acc"].mean()),
        "video_path": str(video_path),
        "preview_png": str(out_dir / "ct_comparison_20261027T1200Z_preview.png"),
    }
    (out_dir / "ct_comparison_summary_20261027.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Done! Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
