"""Temporal availability and geometric clip capacity of the physical archive."""
from __future__ import annotations

import calendar
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .ssl_random_sampling import geographic_origin_mask, origin_rectangles


def record_capacity(record, geographic, *, crop=224, frames=16, memo=None):
    """Count physically supported frames and within-day (time, y, x) candidates.

    The input mask checks all crop pixel centres against the geographic domain.
    Radiometric values are screened when clips are constructed; these counts
    describe capacity before that screening, with one-pixel spatial stride.
    """
    t,channels,h,w=record['shape']
    ct,cc,cy,cx=record['chunks']
    if (ct,cc,channels)!=(1,7,7) or frames<1:
        raise ValueError('Expected seven-channel, single-instant chunks')
    if geographic.shape!=(h-crop+1,w-crop+1):
        raise ValueError('Geographic mask and crop dimensions differ')
    times=pd.DatetimeIndex(pd.to_datetime(record['times'],utc=True)).as_unit('ns')
    if len(times)!=t or times.hasnans or not times.is_unique:
        raise ValueError('Unique, finite timestamps must match the native array')
    support=np.zeros((t,(h+cy-1)//cy,(w+cx-1)//cx),bool)
    for name in record['present']:
        index,channel,iy,ix=map(int,name.split('.'))
        if not (0<=index<t and channel==0 and 0<=iy<support.shape[1] and 0<=ix<support.shape[2]):
            raise ValueError('Unexpected physical chunk index')
        support[index,iy,ix]=True
    # Integer sums make counting each disjoint origin rectangle exact.
    integral=np.pad(geographic.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    memo={} if memo is None else memo
    def capacity(complete):
        key=complete.tobytes()
        if key not in memo:
            total=0
            for y0,y1,x0,x1 in origin_rectangles(complete,h,w,cy,cx,crop):
                total+=int(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
            memo[key]=total
        return memo[key]
    available=np.array([capacity(s)>0 for s in support])
    order=np.argsort(times.asi8)
    sorted_times=times[order]
    windows=[]
    for i in range(t-frames+1):
        stamps=sorted_times[i:i+frames]
        if (not np.all(np.diff(stamps.asi8)==300_000_000_000)
                or stamps[0].normalize()!=stamps[-1].normalize()):
            continue
        count=capacity(support[order[i:i+frames]].all(axis=0))
        if count:
            windows.append((stamps[0],count))
    return times[available],windows


def archive_coverage(inventory, domain=(-20,40,30,70)):
    """Use coordinates and chunk presence only; read no radiance arrays."""
    import xarray as xr
    root=Path(inventory['root'])
    geometries={}
    observations=set()
    starts=set()
    days=[]
    for record in inventory['stores']:
        with xr.open_zarr(root/record['store'],consolidated=True,chunks=None) as ds:
            x,y=np.asarray(ds.x.values),np.asarray(ds.y.values)
            projection=ds.attrs['projection_wkt']
            fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()+str(projection).encode()).hexdigest()
            key=(fingerprint,tuple(record['chunks']))
            if key not in geometries:
                geometries[key]=(geographic_origin_mask(x,y,projection,domain),{})
        geographic,memo=geometries[key]
        times,windows=record_capacity(record,geographic,memo=memo)
        if not len(times):
            continue
        if len(set(times.normalize()))!=1:
            raise ValueError('Expected daily SEVIRI stores')
        if observations.intersection(times) or starts.intersection(t for t,_ in windows):
            raise ValueError('Duplicate observations across stores: de-duplicate the archive first')
        observations.update(times)
        starts.update(t for t,_ in windows)
        days.append(dict(date=times[0].normalize(),frames=len(times),temporal_windows=len(windows),
            crop_candidates=sum(n for _,n in windows),store=record['store']))
    if not days:
        raise ValueError('No geographically supported observations in this inventory')
    daily=pd.DataFrame(days).sort_values('date').reset_index(drop=True)
    summary=dict(frames=int(daily.frames.sum()),observation_dates=len(daily),
        coverage_days=float(daily.frames.sum()/288),temporal_windows=int(daily.temporal_windows.sum()),
        crop_candidates=int(daily.crop_candidates.sum()),first_observation=min(observations).isoformat(),
        last_observation=max(observations).isoformat(),domain_lonlat=list(domain),
        captured_at=inventory.get('captured_at'),
        coverage_definition='Unique five-minute observation slots; disjoint intervals are summed.',
        capacity_definition='Within-day 16-frame windows at five-minute stride times complete 224-pixel crop origins at one-pixel stride, inside the geographic domain and supported by physical seven-channel chunks; before radiometric screening and split reservations.')
    return daily,summary


def manifest_frame_counts(manifest):
    """Count each observed instant once across all overlapping selected clips."""
    starts=pd.DatetimeIndex(pd.to_datetime(manifest.start_time,utc=True).unique()).as_unit('ns')
    instants=np.unique(np.concatenate([starts.asi8+i*300_000_000_000 for i in range(16)]))
    return pd.Series(1,index=pd.to_datetime(instants,utc=True)).groupby(
        pd.to_datetime(instants,utc=True).normalize()).sum()


def archive_coverage_figure(daily, summary, *, selected=None, year=2022):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    daily=daily.copy()
    dates=pd.to_datetime(daily.date,utc=True)
    years=range(dates.dt.year.min(),dates.dt.year.max()+1)
    monthly=daily.groupby([dates.dt.year,dates.dt.month]).frames.sum()
    fractions=np.array([[100*monthly.get((y,m),0)/(calendar.monthrange(y,m)[1]*288)
                         for m in range(1,13)] for y in years])
    fig,(months,detail)=plt.subplots(2,1,figsize=(13,9),height_ratios=(1,1),layout='constrained')
    cmap=plt.get_cmap('Blues').copy()
    image=months.imshow(fractions,aspect='auto',vmin=0,vmax=100,cmap=cmap)
    months.set(xticks=range(12),xticklabels=list(calendar.month_abbr)[1:],
        yticks=range(len(years)),yticklabels=list(years),ylabel='Year',
        title='Monthly archive coverage')
    fig.colorbar(image,ax=months,label='Available five-minute slots (%)',shrink=.8)
    subset=daily.loc[dates.dt.year==year]
    detail.bar(pd.to_datetime(subset.date),subset.frames,width=.95,color='#bfd7ea',label='Archive observations')
    for (label,manifest),colour in zip((selected or {}).items(),('#D55E00','#009E73')):
        counts=manifest_frame_counts(manifest)
        counts=counts[counts.index.year==year]
        detail.bar(counts.index,counts.values,width=.55,color=colour,label=label)
    detail.set(xlim=(pd.Timestamp(year=year,month=1,day=1),pd.Timestamp(year=year+1,month=1,day=1)),
        ylim=(0,310),ylabel='Distinct SEVIRI frames per day',xlabel='Observation date (UTC)',
        title=f'Daily coverage and selected observations · {year}')
    detail.xaxis.set_major_locator(mdates.MonthLocator())
    detail.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    detail.grid(axis='y',alpha=.2)
    handles,labels=detail.get_legend_handles_labels()
    texts=[f"All-archive frames: {summary['frames']:,}",
        f"Coverage at 5-min cadence: {summary['coverage_days']:.1f} days",
        f"16-frame time windows: {summary['temporal_windows']:,}",
        f"Theoretical crop clips: {summary['crop_candidates']/1e9:.2f} billion"]
    handles.extend(Line2D([],[],linestyle='none') for _ in texts)
    detail.legend(handles,labels+texts,loc='upper right',fontsize=9,framealpha=.95,
                  title=f"Archive totals: {min(years)}–{max(years)}",title_fontsize=9)
    return fig
