"""Reproducible, space/time separated diagnostic pool and occupancy metrics."""
import numpy as np
import pandas as pd


def diagnostic_manifest(times, height, width, *, count=8192, seed=20261012, warmup=3):
    if count < 1 or height < 224 or width < 224 or warmup < 0:
        raise ValueError('Invalid diagnostic manifest dimensions/count/context')
    times=pd.DatetimeIndex(times).sort_values()
    if times.hasnans or times.has_duplicates:raise ValueError('Invalid timestamps')
    step=5*60*1_000_000_000; available=set(times.asi8.tolist())
    length=16+warmup; previous=None; windows=[]
    for t in times:
        if previous is not None and t-previous<pd.Timedelta(minutes=5*length):continue
        if all(t.value+i*step in available for i in range(length)):
            windows.append(t);previous=t
    rng=np.random.default_rng(seed);pool=[]
    for context in windows:
        oy=int(rng.integers(height%224+1));ox=int(rng.integers(width%224+1))
        start=context+pd.Timedelta(minutes=5*warmup)
        for y in range(oy,height-223,224):
            for x in range(ox,width-223,224):
                pool.append(dict(start_time=start.isoformat(),context_start=context.isoformat(),
                                 origin_y=y,origin_x=x,month=int(start.month)))
    frame=pd.DataFrame(pool)
    if frame.empty:raise ValueError('No complete diagnostic windows')
    capacity=frame.groupby('month').size().to_dict()
    target=min(count,len(frame)); quotas={m:0 for m in capacity}
    # Equal month quotas until a stratum is exhausted; deterministic surplus.
    while sum(quotas.values())<target:
        for m in capacity:
            if quotas[m]<capacity[m] and sum(quotas.values())<target:quotas[m]+=1
    chosen=[]
    for month,quota in quotas.items():
        candidates=frame[frame.month==month]
        subset=candidates.iloc[rng.permutation(len(candidates))[:quota]].copy()
        subset['pool_month_candidates']=capacity[month]
        subset['sample_month_candidates']=quota
        chosen.append(subset)
    result=pd.concat(chosen,ignore_index=True)
    result=result.iloc[rng.permutation(len(result))].reset_index(drop=True)
    result.insert(0,'clip_id',[f'rdt_{i:06d}' for i in range(len(result))])
    return result, dict(pool_candidates=len(frame),complete_windows=len(windows),
        clips_per_window=(height//224)*(width//224),month_capacity=capacity,month_quotas=quotas,
        requested=count,selected=len(result),stride_minutes=5*length,warmup_frames=warmup,
        seed=seed,sampling='month-balanced random-offset lattice, no overlap including context')


def occupancy_metrics(result, area):
    classes=result['classes'];objects=result['objects']; conv=np.isin(classes,[2,3,4]);ot=classes==4
    cold=classes>0; known=cold & ~result['uncertain']
    size=classes.shape[1]*classes.shape[2]
    conv_objects=objects[objects.class_id.isin([2,3])]
    fractions=conv.reshape(len(classes),-1).mean(1)
    ot_fractions=ot.reshape(len(classes),-1).mean(1)
    return dict(convective_pixels=int(conv.sum()),ot_pixels=int(ot.sum()),cold_pixels=int(cold.sum()),
        temporal_unknown_pixels=int(result['uncertain'].sum()),tracked_objects=int(objects.track_id.nunique()),
        convective_tracks=int(conv_objects.track_id.nunique()),
        ot_tracks=int(objects.loc[objects.ot_pixels>0,'track_id'].nunique()),
        convective_fraction=float(fractions.mean()),convective_peak_fraction=float(fractions.max()),
        peak_frame=int(fractions.argmax()),ot_fraction=float(ot_fractions.mean()),ot_peak_fraction=float(ot_fractions.max()),
        known_cold_pixels=int(known.sum()),
        convective_among_known_cold=float(conv.sum()/known.sum()) if known.any() else np.nan,
        unknown_fraction=float(result['uncertain'].sum()/classes.size),
        convective_peak_area_km2=float((conv*area[None]).sum((1,2)).max()),
        pixels_per_frame=size,frames=len(classes))


def track_metrics(objects):
    lengths=objects.groupby('track_id').size()
    total=len(objects);tracks=len(lengths)
    return dict(visible_track_samples=total,single_point_tracks=int((lengths==1).sum()),
        visible_track_count=tracks,valid_temporal_objects=int(objects.temporal_valid.sum()) if total else 0,
        mean_visible_track_frames=float(lengths.mean()) if tracks else np.nan,
        single_point_fraction=float((lengths==1).mean()) if tracks else np.nan,
        temporal_object_valid_fraction=float(objects.temporal_valid.mean()) if total else np.nan)
