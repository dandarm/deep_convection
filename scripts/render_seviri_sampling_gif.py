#!/usr/bin/env python3
"""Replay accepted crops and observed IR108 in the native geostationary view."""
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
from pyproj import CRS, Transformer
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from emma_gpm.ssl_spaced_sampling import active_clip_rows
from emma_gpm.spatial_sampling import CropEvent, verify_overlap


def geostationary_crs(projection):
    """Preserve the source ellipsoid, satellite position, height and sweep."""
    crs = CRS.from_user_input(projection)
    operation = crs.coordinate_operation
    if operation is None or not operation.method_name.startswith('Geostationary Satellite'):
        raise ValueError('The source must use a geostationary satellite projection')
    parameters = {parameter.name:parameter.value for parameter in operation.params}
    globe = ccrs.Globe(ellipse=None, semimajor_axis=crs.ellipsoid.semi_major_metre,
                      semiminor_axis=crs.ellipsoid.semi_minor_metre)
    return ccrs.Geostationary(
        central_longitude=parameters['Longitude of natural origin'],
        satellite_height=parameters['Satellite Height'],
        false_easting=parameters.get('False easting', 0.),
        false_northing=parameters.get('False northing', 0.),
        sweep_axis='x' if 'Sweep X' in operation.method_name else 'y', globe=globe)


def pixel_edges(coordinates):
    """Native pixel boundaries, preserving either direction of a regular axis."""
    values = np.asarray(coordinates, dtype=float)
    difference = np.diff(values)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or not (np.all(difference > 0) or np.all(difference < 0))
            or not np.allclose(difference, difference[0], rtol=1e-6, atol=1e-6)):
        raise ValueError('Regular monotonic native coordinates are required')
    return np.r_[values[0]-difference[0]/2, (values[:-1]+values[1:])/2,
                 values[-1]+difference[-1]/2]


def native_rectangle(row, x_edges, y_edges, size=224):
    """Exact footprint of one manifest crop on its native pixel boundaries."""
    x, y = int(row.origin_x), int(row.origin_y)
    if x < 0 or y < 0 or x+size >= len(x_edges) or y+size >= len(y_edges):
        raise ValueError('Manifest crop extends outside the source grid')
    left, right = sorted((x_edges[x], x_edges[x+size]))
    bottom, top = sorted((y_edges[y], y_edges[y+size]))
    return np.array([[left,bottom], [right,bottom], [right,top], [left,top]])


def domain_perimeter(domain, samples=100):
    west, east, south, north = domain
    lon = np.linspace(west, east, samples)
    lat = np.linspace(south, north, samples)
    return (np.r_[lon, np.repeat(east,samples), lon[::-1], np.repeat(west,samples), west],
            np.r_[np.repeat(south,samples), lat, np.repeat(north,samples), lat[::-1], south])


def render(args):
    raw = args.manifest.read_bytes()
    frame = pd.read_csv(args.manifest)
    if frame.duplicated(['start_time', 'origin_y', 'origin_x']).any():
        raise ValueError('The manifest must contain unique clips')
    day = args.day or frame.groupby('day').size().sort_values().index[len(frame.day.unique())//2]
    owned = frame[frame.day == day]
    if owned.empty or owned.store.nunique() != 1:
        raise ValueError('Select one available native-grid observation date')
    starts = pd.to_datetime(frame.start_time, utc=True)
    midnight = pd.Timestamp(day, tz='UTC')
    selected = frame[(starts >= midnight-pd.Timedelta(minutes=75)) &
                     (starts < midnight+pd.Timedelta(days=1))].copy()
    if 'clip_id' not in selected:
        selected['clip_id'] = [f'row_{i}' for i in selected.index]
    audit = verify_overlap([CropEvent(pd.Timestamp(row.start_time), row.origin_y, row.origin_x)
        for row in selected.itertuples(index=False)], max_iou=args.max_iou)
    start = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=args.start_hour)
    stamps = pd.date_range(start, periods=args.frames, freq='5min')
    if stamps[-1].normalize() != start.normalize():
        raise ValueError('Animation must stay within the selected observation day')
    with xr.open_zarr(args.zarr_root/owned.store.iloc[0], consolidated=True, chunks=None) as ds:
        times = pd.DatetimeIndex(pd.to_datetime(ds.time.values, utc=True))
        indices = times.get_indexer(stamps)
        if (indices < 0).any():
            raise ValueError('Animation has missing native observation times')
        x, y, projection = ds.x.values, ds.y.values, ds.attrs['projection_wkt']
        ir = ds.seviri.isel(time=indices, channel=list(ds.channel.values).index('IR_108')).values
    for store in selected.store.unique():
        with xr.open_zarr(args.zarr_root/store, consolidated=True, chunks=None) as ds:
            if (not np.array_equal(x, ds.x.values) or not np.array_equal(y, ds.y.values)
                    or projection != ds.attrs['projection_wkt']):
                raise ValueError('Displayed clips must share the same native grid')
    native = geostationary_crs(projection)
    x_edges, y_edges = pixel_edges(x), pixel_edges(y)
    polygons = {row.clip_id:native_rectangle(row,x_edges,y_edges) for row in selected.itertuples(index=False)}
    # Axis ordering changes only the display orientation; every observed pixel
    # remains on its native geostationary grid without a lon/lat interpolation.
    ix, iy = np.argsort(x), np.argsort(y)
    extent = [x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
    lon, lat = Transformer.from_crs(projection,'EPSG:4326',always_xy=True).transform(*np.meshgrid(x,y))
    west, east, south, north = args.domain
    valid = (np.isfinite(lon) & np.isfinite(lat) & (lon>=west) & (lon<=east) & (lat>=south) & (lat<=north))
    valid = valid[np.ix_(iy,ix)]
    fig = plt.figure(figsize=(13, 6.3), facecolor='white')
    ax = fig.add_axes([.045,.13,.77,.76],projection=native)
    ax.set_facecolor('#eef1f4')
    ax.set_xlim(extent[0]-.015*np.ptp(x), extent[1]+.015*np.ptp(x))
    ax.set_ylim(extent[2]-.02*np.ptp(y), native.y_limits[1])
    ax.coastlines(resolution='50m', color='#263746', linewidth=.55, zorder=4)
    grid = ax.gridlines(draw_labels=True, linewidth=.4, color='#536273', alpha=.45,
                       xlocs=np.arange(-60,81,20),ylocs=np.arange(20,81,10))
    grid.top_labels, grid.right_labels = False, False
    grid.xlabel_style=grid.ylabel_style={'size':9}
    image = ax.imshow(np.full(valid.shape,np.nan), extent=extent, origin='lower',
        transform=native, interpolation='nearest', cmap='gray_r', vmin=190, vmax=320, zorder=1)
    tiles = PolyCollection([], transform=native, linewidths=1.1, zorder=5)
    ax.add_collection(tiles)
    overview = fig.add_axes([.83,.60,.155,.32], projection=native)
    overview.set_global()
    overview.set_facecolor('#e5ebef')
    overview.coastlines(resolution='50m',color='#607481',linewidth=.35)
    overview.gridlines(linewidth=.3,alpha=.35,xlocs=np.arange(-180,181,30),ylocs=np.arange(-60,61,30))
    overview.plot(*domain_perimeter(args.domain),transform=ccrs.PlateCarree(),color='#152d40',linewidth=1.)
    overview.set_title('Satellite view',fontsize=10)
    bar = fig.colorbar(image,cax=fig.add_axes([.86,.17,.022,.33]))
    bar.set_label('IR 10.8 µm brightness temperature (K)')
    tile_color=args.tile_color
    ax.legend(handles=[Line2D([],[],color=tile_color,linewidth=1.8,label='Active crop · 224 × 224 pixels')],
        loc='lower left',framealpha=.9,fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    temporary = args.output.with_name(args.output.stem+'.part.gif')
    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(temporary), dpi=110):
        for index, timestamp in enumerate(stamps):
            mapped = ir[index][np.ix_(iy,ix)].astype(np.float32)
            mapped[~valid | ~np.isfinite(mapped) | (mapped < 100) | (mapped > 400)] = np.nan
            image.set_data(mapped)
            active = active_clip_rows(selected, timestamp)
            identifiers = active.clip_id.tolist()
            starting = active.loc[pd.to_datetime(active.start_time,utc=True)==timestamp,'clip_id'].tolist()
            tiles.set_verts([polygons[name] for name in identifiers])
            tiles.set_edgecolors(tile_color)
            tiles.set_facecolors(to_rgba(tile_color,.10))
            ax.set_title(f"{timestamp:%d %b %Y · %H:%M} UTC  |  {len(active)} active clips · {len(starting)} starting  |  pairwise IoU ≤ {args.max_iou:.2f}",
                fontsize=11, pad=15)
            writer.grab_frame()
            if index == args.frames//2:
                fig.savefig(args.output.with_suffix('.png'), dpi=140)
            records.append(dict(timestamp=timestamp.isoformat(), clip_ids=identifiers,starting_clip_ids=starting))
    plt.close(fig)
    temporary.replace(args.output)
    metadata = dict(manifest=str(args.manifest.resolve()), manifest_sha256=hashlib.sha256(raw).hexdigest(),
        background_archive=str(args.zarr_root.resolve()), date=day, frames=args.frames,
        cadence_minutes=5, playback_fps=args.fps, domain_lonlat=args.domain, max_iou_active=args.max_iou,
        measured_maximum_active_iou=audit['maximum_concurrent_iou'],
        source='Exact accepted manifest events; no separate sampling simulation',
        background_display='Native geostationary pixels; north-up, west-left; geographical domain mask',
        projection_wkt=projection, satellite_longitude_deg=native.proj4_params['lon_0'],
        tile_color=tile_color, tile_color_meaning='All accepted active crops use the same color', timeline=records)
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
    parser.add_argument('--tile-color', default='#ed542f')
    render(parser.parse_args())
