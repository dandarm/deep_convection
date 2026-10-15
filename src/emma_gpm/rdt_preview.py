"""Local archive inspection and reproducible examples for the RDT notebook."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from .seviri import THERMAL_CHANNELS
from .seviri_clips import SeviriZarrClipReader, IncompleteSeviriClipError
from .rdt_teacher import label_rdt_clip, projected_pixel_areas


def archive_inventory(root):
    root = Path(root)
    records, all_times = [], []
    geometry = None
    for path in sorted(root.glob('*/*.zarr')):
        with xr.open_zarr(path, consolidated=True, chunks=None) as ds:
            if (tuple(map(str, ds.channel.values)) != THERMAL_CHANNELS or
                ds.attrs.get('units') != 'K' or ds.attrs.get('calibration') != 'brightness_temperature'):
                raise ValueError(f'Invalid spectral/radiometric metadata: {path}')
            times = pd.DatetimeIndex(ds.time.values)
            if times.hasnans or times.has_duplicates:
                raise ValueError(f'Invalid chronological timestamps: {path}')
            source_unsorted = not times.is_monotonic_increasing
            # Acquisition may append earlier slots after existing later ones.
            # Query timestamps chronologically, as the existing strict reader
            # does; never reorder an already loaded clip or invent a slot.
            times = times.sort_values()
            grid = (ds.x.values.copy(), ds.y.values.copy(), ds.attrs['projection_wkt'])
            if geometry is None: geometry = grid
            elif not (np.array_equal(grid[0],geometry[0]) and np.array_equal(grid[1],geometry[1]) and grid[2]==geometry[2]):
                raise ValueError(f'Changing grid geometry: {path}')
            records.append(dict(day=path.stem, frames=len(times), source_unsorted=source_unsorted,
                first=times[0] if len(times) else pd.NaT, last=times[-1] if len(times) else pd.NaT,
                gaps_inside_day=int(np.sum(np.diff(times.asi8)!=5*60*1_000_000_000))))
            all_times.extend(times.tolist())
    if not records: raise FileNotFoundError(f'No published SEVIRI daily Zarr in {root}')
    times = pd.DatetimeIndex(all_times).sort_values()
    if times.has_duplicates: raise ValueError('Duplicate timestamps across published days')
    return pd.DataFrame(records), times, geometry


def load_example(root, start_time, origin_y, origin_x, *, warmup_frames=3, crop_size=224, config=None, teacher_version="legacy", halo=0):
    """Strict warm-up + 16 frames, no fallback if any context frame is missing."""
    if warmup_frames < 0 or int(warmup_frames) != warmup_frames:
        raise ValueError('Warm-up frame count must be a nonnegative integer')
    if teacher_version == 'refined_v2':
        return load_refined_example(root,start_time,origin_y,origin_x,warmup_frames=warmup_frames,
                                    crop_size=crop_size,config=config,halo=halo)
    if teacher_version != 'legacy':raise ValueError('Unknown teacher version')
    start = pd.Timestamp(start_time)
    context_start = start-pd.Timedelta(minutes=5*warmup_frames)
    with SeviriZarrClipReader(root, num_frames=16+warmup_frames, crop_size=crop_size) as reader:
        clip = reader.load(context_start, origin_y=int(origin_y), origin_x=int(origin_x))
        geometries = []
        for ds in reader._stores.values():
            if ds.attrs.get('units') != 'K' or ds.attrs.get('calibration') != 'brightness_temperature':
                raise ValueError('Teacher requires unnormalised brightness-temperature Kelvin metadata')
            x=ds.x.values[int(origin_x):int(origin_x)+crop_size]
            y=ds.y.values[int(origin_y):int(origin_y)+crop_size]
            # Same orientation as the strict reader, deduced from coordinates.
            x=x if x[0]<x[-1] else x[::-1]
            y=y if y[0]>y[-1] else y[::-1]
            geometries.append((x,y,ds.attrs['projection_wkt']))
        geometry=geometries[0]
        if any(not (np.array_equal(g[0],geometry[0]) and np.array_equal(g[1],geometry[1]) and g[2]==geometry[2]) for g in geometries):
            raise ValueError('Geometry changes inside clip')
    area = projected_pixel_areas(*geometry)
    times = pd.date_range(context_start, periods=len(clip), freq='5min')
    result = label_rdt_clip(clip,times,area,config)
    for key in ['classes','tracks','uncertain','levels']: result[key]=result[key][warmup_frames:]
    result['objects']=result['objects'][result['objects'].frame>=warmup_frames].copy()
    result['objects']['frame']-=warmup_frames
    if not result['links'].empty:
        result['links']=result['links'][result['links'].frame>=warmup_frames].copy()
        result['links']['frame']-=warmup_frames
    lon,lat=Transformer.from_crs(CRS.from_user_input(geometry[2]),'EPSG:4326',always_xy=True).transform(
        float(np.mean(geometry[0])),float(np.mean(geometry[1])))
    return dict(clip=clip[warmup_frames:], times=times[warmup_frames:], result=result,
                area=area, longitude=lon, latitude=lat, warmup_frames=warmup_frames,
                start_time=start.isoformat(), origin_y=int(origin_y), origin_x=int(origin_x))


def sample_examples(root, times, geometry, *, count=48, seed=20261012, warmup_frames=3, config=None, progress=None):
    """Seeded, month-balanced spatial samples, diagnostic only, never SSL selection."""
    if count < 1 or int(count) != count or len(geometry[0]) < 224 or len(geometry[1]) < 224:
        raise ValueError('Require positive integer count and a grid large enough for 224-pixel crops')
    rng=np.random.default_rng(seed)
    available=set(times.asi8.tolist()); step=5*60*1_000_000_000
    starts=pd.DatetimeIndex([t for t in times if all(t.value+i*step in available for i in range(16+warmup_frames))])
    if not len(starts):raise ValueError('No complete clips with warm-up context')
    x,y,_=geometry
    records=[]
    months=sorted(set(starts.month))
    with_progress=progress or (lambda value:None)
    for index in range(count):
        month=months[index%len(months)]
        pool=starts[starts.month==month]
        t=pool[int(rng.integers(len(pool)))]+pd.Timedelta(minutes=5*warmup_frames)
        yy=int(rng.integers(0,len(y)-224+1));xx=int(rng.integers(0,len(x)-224+1))
        row=dict(start_time=t.isoformat(),origin_y=yy,origin_x=xx,month=month)
        try:
            example=load_example(root,t,yy,xx,warmup_frames=warmup_frames,config=config)
            objects=example['result']['objects']
            classes=example['result']['classes']
            row.update(status='ok',longitude=example['longitude'],latitude=example['latitude'],
                convective_pixels=int(np.isin(classes,[2,3,4]).sum()),
                ot_pixels=int((classes==4).sum()),cold_pixels=int((classes>0).sum()),
                temporal_unknown_pixels=int(example['result']['uncertain'].sum()),
                tracked_objects=int(objects.track_id.nunique()))
        except (IncompleteSeviriClipError, ValueError) as error:
            row.update(status='rejected',reason=str(error))
        records.append(row);with_progress(index+1)
    return pd.DataFrame(records)


def load_refined_example(root,start_time,origin_y,origin_x,*,warmup_frames=3,crop_size=224,config=None,halo=64):
    from dataclasses import asdict
    from .rdt_refined import RefinedRDTConfig,label_refined_clip
    from .rdt_context import context_window,crop_teacher_result
    config=config if isinstance(config,RefinedRDTConfig) else RefinedRDTConfig(**asdict(config)) if config else RefinedRDTConfig()
    start=pd.Timestamp(start_time);context_start=start-pd.Timedelta(minutes=5*warmup_frames)
    path=Path(root)/f'{context_start:%Y}'/f'{context_start:%Y%m%d}.zarr'
    with xr.open_zarr(path,consolidated=True,chunks=None) as ds:
        gx=ds.x.values.copy();gy=ds.y.values.copy();projection=ds.attrs['projection_wkt']
    yy,xx,size,roi=context_window(int(origin_y),int(origin_x),len(gy),len(gx),crop=crop_size,halo=halo,
                                north_reversed=gy[0]<gy[-1],west_reversed=gx[0]>gx[-1])
    with SeviriZarrClipReader(root,num_frames=16+warmup_frames,crop_size=size) as reader:
        raw=reader.load(context_start,origin_y=yy,origin_x=xx)
        for ds in reader._stores.values():
            if (ds.attrs.get('units')!='K' or ds.attrs.get('calibration')!='brightness_temperature' or
                not np.array_equal(ds.x.values,gx) or not np.array_equal(ds.y.values,gy) or
                ds.attrs.get('projection_wkt')!=projection):raise ValueError('Changing or invalid context metadata')
    x=gx[xx:xx+size];y=gy[yy:yy+size]
    x=x if x[0]<x[-1] else x[::-1];y=y if y[0]>y[-1] else y[::-1]
    area=projected_pixel_areas(x,y,projection)
    times=pd.date_range(context_start,periods=len(raw),freq='5min')
    result=label_refined_clip(raw,times,area,config)
    central_area=area[roi]
    result=crop_teacher_result(result,roi,warmup_frames,central_area)
    lon,lat=Transformer.from_crs(CRS.from_user_input(projection),'EPSG:4326',always_xy=True).transform(
        float(x[roi[1]].mean()),float(y[roi[0]].mean()))
    return dict(clip=raw[warmup_frames:,...,roi[0],roi[1]],times=times[warmup_frames:],result=result,
        area=central_area,longitude=lon,latitude=lat,warmup_frames=warmup_frames,start_time=start.isoformat(),
        origin_y=int(origin_y),origin_x=int(origin_x),teacher_version='refined_v2',analysis_halo=halo)
