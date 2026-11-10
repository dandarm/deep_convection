#!/usr/bin/env python3
"""Replay a multi-day training manifest in an MP4 on the native SEVIRI grid."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.render_seviri_sampling_gif import (
    domain_perimeter, geostationary_crs, native_rectangle, pixel_edges)
from emma_gpm.spatial_sampling import CropEvent, verify_overlap
from emma_gpm.ssl_spaced_sampling import active_clip_rows


def ffmpeg_binary():
    binary = shutil.which('ffmpeg')
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError('Install FFmpeg or imageio-ffmpeg to export MP4') from error


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def date_status(day, training_dates, protocol):
    if day in training_dates:
        return 'Training date'
    if day in protocol.get('validation_days', {}):
        return 'Validation date'
    reserved = pd.DatetimeIndex(protocol.get('reserved_days', []))
    if pd.Timestamp(day) in reserved:
        return 'Reserved date'
    gap = pd.Timedelta(hours=protocol.get('temporal_separation_hours', 0))
    if len(reserved) and gap > pd.Timedelta(0) and (abs(reserved-pd.Timestamp(day)) <= gap).any():
        return 'Temporal separation day'
    return 'Outside training selection'


def render(args):
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 8, 'threading.max_workers': 4})
    blosc.set_nthreads(1)
    if args.output.suffix.lower() != '.mp4' or args.output.exists():
        raise ValueError('Choose a new .mp4 output path')
    if args.step_minutes < 5 or args.step_minutes % 5 or args.fps <= 0:
        raise ValueError('Use a positive FPS and a display step divisible by five minutes')
    start = pd.Timestamp(args.start_date, tz='UTC').normalize()
    end = pd.Timestamp(args.end_date, tz='UTC').normalize()+pd.Timedelta(days=1)
    if end <= start:
        raise ValueError('The end date must follow or equal the start date')
    stamps = pd.date_range(start, end, freq=f'{args.step_minutes}min', inclusive='left')
    if len(stamps)/args.fps > args.max_seconds:
        raise ValueError('The requested interval and display rate exceed --max-seconds')
    raw = args.manifest.read_bytes()
    frame = pd.read_csv(args.manifest)
    if frame.clip_id.duplicated().any():
        raise ValueError('The manifest must contain unique clip IDs')
    protocol = json.loads((args.manifest.parent/'protocol.json').read_text())
    domain = protocol['domain_lonlat']
    times = pd.to_datetime(frame.start_time, utc=True)
    selected = frame[(times >= start-pd.Timedelta(minutes=75)) & (times < end)].copy()
    if selected.empty:
        raise ValueError('No selected training clips in this interval')
    audit = verify_overlap([CropEvent(pd.Timestamp(r.start_time), r.origin_y, r.origin_x)
        for r in selected.itertuples(index=False)], max_iou=protocol['max_iou_active'])
    inventory = json.loads(args.inventory.read_text())
    stores = {pd.Timestamp(Path(r['store']).stem).strftime('%Y-%m-%d'):r['store']
              for r in inventory['stores']}
    days = stamps.normalize().unique()
    if any(day.strftime('%Y-%m-%d') not in stores for day in days):
        raise ValueError('Each displayed date must have an inventoried observation store')
    archive = Path(inventory['root'])
    with xr.open_zarr(archive/stores[days[0].strftime('%Y-%m-%d')], consolidated=True, chunks=None) as ds:
        x, y, projection = ds.x.values, ds.y.values, ds.attrs['projection_wkt']
    native = geostationary_crs(projection)
    x_edges, y_edges = pixel_edges(x), pixel_edges(y)
    polygons = {r.clip_id:native_rectangle(r, x_edges, y_edges) for r in selected.itertuples(index=False)}
    ix, iy = np.argsort(x), np.argsort(y)
    extent = [x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
    lon, lat = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True).transform(*np.meshgrid(x,y))
    west, east, south, north = domain
    valid = (np.isfinite(lon) & np.isfinite(lat) & (lon>=west) & (lon<=east) & (lat>=south) & (lat<=north))
    valid = valid[np.ix_(iy, ix)]
    color = '#ed542f'
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='white')
    fig.suptitle(f'{len(frame):,} training clips · {start:%d %b} – {end-pd.Timedelta(days=1):%d %b %Y}',
                 fontsize=16, y=.97)
    ax = fig.add_axes([.045, .19, .77, .67], projection=native)
    ax.set_facecolor('#eef1f4')
    ax.set_xlim(extent[0]-.015*np.ptp(x), extent[1]+.015*np.ptp(x))
    ax.set_ylim(extent[2]-.02*np.ptp(y), native.y_limits[1])
    ax.coastlines(resolution='50m', color='#263746', linewidth=.55, zorder=4)
    grid = ax.gridlines(draw_labels=True, linewidth=.4, color='#536273', alpha=.45,
                       xlocs=np.arange(-60,81,20), ylocs=np.arange(20,81,10))
    grid.top_labels = grid.right_labels = False
    grid.xlabel_style = grid.ylabel_style = {'size':10}
    image = ax.imshow(np.full(valid.shape, np.nan), extent=extent, origin='lower',
        transform=native, interpolation='nearest', cmap='gray_r', vmin=190, vmax=320, zorder=1)
    tiles = PolyCollection([], transform=native, linewidths=1.2, zorder=5,
                           edgecolors=color, facecolors=to_rgba(color, .10))
    ax.add_collection(tiles)
    ax.legend(handles=[Line2D([], [], color=color, linewidth=1.8, label='Active crop · 224 × 224 pixels')],
              loc='lower left', framealpha=.9, fontsize=10)
    overview = fig.add_axes([.83, .59, .155, .30], projection=native)
    overview.set_global()
    overview.set_facecolor('#e5ebef')
    overview.coastlines(resolution='50m', color='#607481', linewidth=.35)
    overview.gridlines(linewidth=.3, alpha=.35, xlocs=np.arange(-180,181,30), ylocs=np.arange(-60,61,30))
    overview.plot(*domain_perimeter(domain), transform=ccrs.PlateCarree(), color='#152d40', linewidth=1.)
    overview.set_title('Satellite view', fontsize=11)
    bar = fig.colorbar(image, cax=fig.add_axes([.87, .22, .020, .30]))
    bar.set_label('IR 10.8 µm brightness temperature (K)')
    timeline_axis = fig.add_axes([.075, .085, .71, .025])
    bounds = mdates.date2num([start, end])
    timeline_axis.plot(bounds, [0, 0], color='#c5ccd2', lw=3)
    marker, = timeline_axis.plot([bounds[0]], [0], 'o', color=color, ms=8)
    timeline_axis.set(xlim=bounds, ylim=(-1, 1), yticks=[])
    timeline_axis.xaxis.set_major_locator(mdates.DayLocator())
    timeline_axis.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    timeline_axis.spines[:].set_visible(False)
    timeline_axis.tick_params(axis='x', length=0, labelsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem+'.part.mp4')
    width, height = fig.canvas.get_width_height()
    command = [ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgba', '-s', f'{width}x{height}', '-r', str(args.fps), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-f', 'mp4', str(temporary)]
    records, missing, date_records = [], [], []
    started = time.monotonic()
    with args.output.with_suffix('.ffmpeg.log').open('w') as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for day in days:
                date = day.strftime('%Y-%m-%d')
                day_stamps = stamps[stamps.normalize() == day]
                with xr.open_zarr(archive/stores[date], consolidated=True, chunks=None) as ds:
                    if (not np.array_equal(x, ds.x.values) or not np.array_equal(y, ds.y.values)
                            or ds.attrs['projection_wkt'] != projection):
                        raise ValueError('All displayed dates must share the native grid')
                    observed = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True))
                    if not observed.is_unique or not observed.is_monotonic_increasing:
                        raise ValueError('Native observation times must be unique and ordered')
                    indices = observed.get_indexer(day_stamps)
                    present = indices >= 0
                    ir = np.full((len(day_stamps), len(y), len(x)), np.nan, np.float16)
                    ir[present] = ds.seviri.isel(time=indices[present], channel=list(ds.channel.values).index('IR_108')).values
                status = date_status(date, set(frame.day), protocol)
                date_records.append(dict(date=date, status=status, training_clips=int((frame.day == date).sum())))
                for index, timestamp in enumerate(day_stamps):
                    mapped = ir[index][np.ix_(iy, ix)].astype(np.float32)
                    mapped[~valid | ~np.isfinite(mapped) | (mapped < 100) | (mapped > 400)] = np.nan
                    image.set_data(mapped)
                    active = active_clip_rows(selected, timestamp)
                    identifiers = active.clip_id.tolist()
                    tiles.set_verts([polygons[name] for name in identifiers])
                    label = status if status != 'Training date' else f'{len(active)} active training clips'
                    if not present[index]:
                        missing.append(timestamp.isoformat())
                        label += ' · Missing IR observation'
                    ax.set_title(f'{timestamp:%d %b %Y · %H:%M} UTC  |  {label}', fontsize=13, pad=15)
                    marker.set_xdata([mdates.date2num(timestamp)])
                    fig.canvas.draw()
                    encoder.stdin.write(fig.canvas.buffer_rgba())
                    if len(records) == len(stamps)//2:
                        fig.savefig(args.output.with_suffix('.png'), dpi=100)
                    records.append(dict(timestamp=timestamp.isoformat(), clip_ids=identifiers,
                                        ir_observation_present=bool(present[index]), date_status=status))
                del ir
                progress = dict(status='rendering', frames=len(records), total_frames=len(stamps),
                                last_timestamp=records[-1]['timestamp'], elapsed_seconds=time.monotonic()-started)
                write_json(args.output.with_suffix('.progress.json'), progress)
                print(json.dumps(progress), flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('FFmpeg failed; inspect the adjacent .ffmpeg.log')
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
        finally:
            plt.close(fig)
    displayed = {identifier for record in records for identifier in record['clip_ids']}
    if displayed != set(selected.clip_id):
        raise ValueError('The display interval must show every selected clip at least once')
    temporary.replace(args.output)
    metadata = dict(status='completed', manifest=str(args.manifest.resolve()),
        manifest_sha256=hashlib.sha256(raw).hexdigest(), dataset_training_clips=len(frame),
        start_utc=start.isoformat(), end_exclusive_utc=end.isoformat(), frames=len(records),
        playback_fps=args.fps, duration_seconds=len(records)/args.fps, width=width, height=height,
        native_cadence_minutes=5, displayed_cadence_minutes=args.step_minutes,
        missing_displayed_observations=missing, missing_observation_display='Blank IR background at the real timestamp',
        source='Exact training manifest events at each displayed observation time',
        source_archive=str(archive), domain_lonlat=domain, projection_wkt=projection,
        tile_color=color, displayed_unique_training_clips=len(displayed),
        measured_maximum_active_iou=audit['maximum_concurrent_iou'],
        dates=date_records, clip_events=selected[['clip_id','start_time','origin_y','origin_x']].to_dict('records'),
        timeline=records, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output.with_suffix('.json'), metadata)
    write_json(args.output.with_suffix('.progress.json'), dict(status='completed', frames=len(records),
        total_frames=len(stamps), duration_seconds=metadata['duration_seconds'], elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(output=str(args.output), frames=len(records), duration_seconds=metadata['duration_seconds'],
                          displayed_training_clips=len(displayed))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True, help='Inclusive UTC date')
    parser.add_argument('--step-minutes', type=int, default=15)
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--max-seconds', type=float, default=30)
    render(parser.parse_args())
