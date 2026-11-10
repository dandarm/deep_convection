#!/usr/bin/env python3
"""Replay accepted dataset crops over observed IR108 on the geographic domain."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from emma_gpm.ssl_spaced_sampling import active_clip_rows
from emma_gpm.spatial_sampling import CropEvent, verify_overlap


def native_perimeter(row, x, y, transform, size=224):
    xs = x[int(row.origin_x):int(row.origin_x)+size]
    ys = y[int(row.origin_y):int(row.origin_y)+size]
    if len(xs) != size or len(ys) != size:
        raise ValueError('Manifest crop extends outside the source grid')
    xx = np.concatenate((xs, np.repeat(xs[-1], size), xs[::-1], np.repeat(xs[0], size)))
    yy = np.concatenate((np.repeat(ys[0], size), ys, np.repeat(ys[-1], size), ys[::-1]))
    lon, lat = transform.transform(xx, yy)
    return np.column_stack((lon, lat))


def map_indices(x, y, projection, domain, width=720, height=480):
    west, east, south, north = domain
    lon = west + (np.arange(width)+.5)*(east-west)/width
    lat = south + (np.arange(height)+.5)*(north-south)/height
    xx, yy = Transformer.from_crs('EPSG:4326', projection, always_xy=True).transform(*np.meshgrid(lon, lat))
    def indices(coordinates, projected):
        order = np.argsort(coordinates)
        fractional = np.interp(projected, coordinates[order], order, left=np.nan, right=np.nan)
        valid = np.isfinite(fractional)
        return np.rint(np.where(valid, fractional, 0)).astype(int), valid
    ix, valid_x = indices(x, xx)
    iy, valid_y = indices(y, yy)
    return iy, ix, valid_x & valid_y


def render(args):
    raw = args.manifest.read_bytes()
    frame = pd.read_csv(args.manifest)
    if frame.duplicated(['start_time', 'origin_y', 'origin_x']).any():
        raise ValueError('The manifest must contain unique clips')
    day = args.day or frame.groupby('day').size().sort_values().index[len(frame.day.unique())//2]
    selected = frame[frame.day == day].copy()
    if selected.empty or selected.store.nunique() != 1:
        raise ValueError('Select one available native-grid observation date')
    if 'clip_id' not in selected:
        selected['clip_id'] = [f'row_{i}' for i in selected.index]
    audit = verify_overlap([CropEvent(pd.Timestamp(row.start_time), row.origin_y, row.origin_x)
        for row in selected.itertuples(index=False)], max_iou=args.max_iou)
    start = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=args.start_hour)
    stamps = pd.date_range(start, periods=args.frames, freq='5min')
    if stamps[-1].normalize() != start.normalize():
        raise ValueError('Animation must stay within the selected observation day')
    with xr.open_zarr(args.zarr_root/selected.store.iloc[0], consolidated=True, chunks=None) as ds:
        times = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True))
        indices = times.get_indexer(stamps)
        if (indices < 0).any():
            raise ValueError('Animation has missing native observation times')
        x, y, projection = ds.x.values, ds.y.values, ds.attrs['projection_wkt']
        ir = ds.seviri.isel(time=indices, channel=list(ds.channel.values).index('IR_108')).values
    transform = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True)
    polygons = {row.clip_id:native_perimeter(row, x, y, transform) for row in selected.itertuples(index=False)}
    iy, ix, valid = map_indices(x, y, projection, args.domain)
    fig = plt.figure(figsize=(10.5, 7.5), facecolor='white')
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent(args.domain, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='#142a3b', linewidth=.65, zorder=4)
    grid = ax.gridlines(draw_labels=True, linewidth=.4, alpha=.4)
    grid.top_labels, grid.right_labels = False, False
    image = ax.imshow(np.full(iy.shape, np.nan), extent=args.domain, origin='lower',
        transform=ccrs.PlateCarree(), cmap='gray_r', vmin=190, vmax=320, zorder=0)
    tiles = PolyCollection([], transform=ccrs.PlateCarree(), linewidths=1.1, zorder=3)
    ax.add_collection(tiles)
    bar = fig.colorbar(image, ax=ax, pad=.03, shrink=.78)
    bar.set_label('IR 10.8 µm brightness temperature (K)')
    ax.legend(handles=[Line2D([], [], color='#0089a8', linewidth=2, label='Active dataset crop')],
        loc='lower left', framealpha=.9)
    fig.subplots_adjust(left=.07, right=.90, bottom=.07, top=.92)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap('tab20')
    records = []
    temporary = args.output.with_name(args.output.stem+'.part.gif')
    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(temporary), dpi=110):
        for index, timestamp in enumerate(stamps):
            mapped = ir[index, iy, ix].astype(np.float32)
            mapped[~valid | ~np.isfinite(mapped) | (mapped < 100) | (mapped > 400)] = np.nan
            image.set_data(mapped)
            active = active_clip_rows(selected, timestamp)
            identifiers = active.clip_id.tolist()
            colors = [palette(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)%20)
                for name in identifiers]
            tiles.set_verts([polygons[name] for name in identifiers])
            tiles.set_edgecolors(colors)
            tiles.set_facecolors([to_rgba(color, .13) for color in colors])
            ax.set_title(f"{timestamp:%d %b %Y · %H:%M} UTC   |   {len(active)} active clips   |   IoU ≤ {args.max_iou:.2f}",
                fontsize=13, pad=15)
            writer.grab_frame()
            if index == args.frames//2:
                fig.savefig(args.output.with_suffix('.png'), dpi=140)
            records.append(dict(timestamp=timestamp.isoformat(), clip_ids=identifiers))
    plt.close(fig)
    temporary.replace(args.output)
    metadata = dict(manifest=str(args.manifest.resolve()), manifest_sha256=hashlib.sha256(raw).hexdigest(),
        background_archive=str(args.zarr_root.resolve()), date=day, frames=args.frames,
        cadence_minutes=5, playback_fps=args.fps, domain_lonlat=args.domain, max_iou_active=args.max_iou,
        measured_maximum_active_iou=audit['maximum_concurrent_iou'],
        source='Exact accepted manifest events; no separate sampling simulation',
        background_display='Nearest native pixel on a regular lon/lat display grid', timeline=records)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), date=day, frames=args.frames, bytes=args.output.stat().st_size)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--zarr-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--day')
    parser.add_argument('--start-hour', type=int, default=2)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--fps', type=float, default=6.25)
    parser.add_argument('--domain', nargs=4, type=float, default=[-20, 40, 30, 70])
    parser.add_argument('--max-iou', type=float, default=.2)
    render(parser.parse_args())
