"""Read-only ranking of completed teacher frames and event study crops.

These diagnostics never influence the teacher or the self-supervised manifests.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from .rdt.storage import open_fields, read_table, frame_key

BTD_LIMITS = (-80., 10.)


def btd_norm():
    """Fixed display limits, NOT observed archive extrema; zero is neutral."""
    return TwoSlopeNorm(vmin=BTD_LIMITS[0], vcenter=0., vmax=BTD_LIMITS[1])


def summarize_frame(directory, *, min_proxy_confidence=.35):
    directory = Path(directory)
    with open_fields(directory) as fields:
        labels = np.asarray(fields['semantic_label'][:])
        valid = np.asarray(fields['valid_observation'][:], bool)
    valid_pixels = int(valid.sum())
    convective_pixels = int((np.isin(labels, [2,3,4]) & valid).sum())
    ot_pixels = int(((labels == 4) & valid).sum())
    row = dict(time=pd.to_datetime(directory.name, format='%Y%m%dT%H%M').isoformat(),
               valid_pixels=valid_pixels, convective_pixels=convective_pixels, ot_pixels=ot_pixels,
               convective_percent=100*convective_pixels/valid_pixels if valid_pixels else np.nan,
               ot_percent=100*ot_pixels/valid_pixels if valid_pixels else np.nan)
    objects = read_table(directory/'objects.csv.gz')
    events = []
    if not objects.empty:
        # Separate study selection from the deterministic teacher labels.
        support = (objects.proxy_confidence >= min_proxy_confidence) & ~objects.border
        convection = support & objects.class_id.isin([2,3,4]) & objects.temporal_valid & objects.phase_valid
        ot = support & (objects.ot_pixels > 0)
        for kind, mask, metric in [('convection', convection, 'area_km2'), ('ot', ot, 'ot_pixels')]:
            for cell in objects[mask].itertuples():
                events.append(dict(track_id=int(cell.track_id), kind=kind, time=row['time'],
                                   score=float(getattr(cell, metric)), area_km2=float(cell.area_km2),
                                   ot_pixels=int(cell.ot_pixels), min_k=float(cell.min_k),
                                   proxy_confidence=float(cell.proxy_confidence)))
    return row, events


def rank_frames(metrics, column, *, limit=100, separation_minutes=90):
    """Rank occupancy, suppress nearby timestamps; this does not identify storms."""
    if metrics.empty or column not in metrics:
        return pd.DataFrame(columns=metrics.columns)
    ranked = metrics[metrics[column].notna() & (metrics[column] > 0)].copy()
    ranked['time'] = pd.to_datetime(ranked.time)
    ranked = ranked.sort_values([column, 'time'], ascending=[False, True], kind='stable')
    selected = []
    for i, row in ranked.iterrows():
        if all(abs(row.time-ranked.loc[j,'time']) >= pd.Timedelta(minutes=separation_minutes) for j in selected):
            selected.append(i)
            if len(selected) >= limit:
                break
    return ranked.loc[selected].reset_index(drop=True)


def study_window(committed, peak, *, frames=16):
    """Center on the peak when possible; only complete observed windows qualify."""
    peak = pd.Timestamp(peak)
    available = set(pd.DatetimeIndex(committed))
    for offset in sorted(range(frames), key=lambda i: (abs(i-8), i)):
        start = peak-pd.Timedelta(minutes=5*offset)
        times = pd.date_range(start, periods=frames, freq='5min')
        if all(stamp in available for stamp in times):
            return start
    raise ValueError('Non ci sono ancora 16 frame consecutivi attorno al picco; nessun frame viene sostituito.')


def study_validity(output, times, shape, config_hash):
    """Intersect saved full-domain quality masks over the actual study window."""
    valid=np.ones(shape,bool)
    for stamp in times:
        directory=Path(output)/'frames'/frame_key(stamp)
        if not (directory/'record.json').exists():
            raise FileNotFoundError(f'Full-domain labels not available for {stamp}')
        with open_fields(directory) as group:
            if group.attrs.get('config_hash')!=config_hash or pd.Timestamp(group.attrs['time'])!=pd.Timestamp(stamp):
                raise ValueError('Frame version/timestamp differs from its label archive')
            current=np.asarray(group['valid_observation'][:],bool)
        if current.shape!=tuple(shape):
            raise ValueError('Study quality mask has a different domain shape')
        valid &= current
    return valid


def study_origin(tracks, track_id, native_x, native_y, *, crop_size=224, valid_mask=None):
    """Center on the object mask, convert north-up coordinates to native ROI."""
    yy,xx = np.where(tracks == int(track_id))
    if not len(yy):
        raise ValueError('Traccia non presente nel frame del picco')
    height,width = tracks.shape
    if min(height,width) < crop_size:
        raise ValueError('Dominio troppo piccolo per il crop richiesto')
    top = int(np.clip(round(yy.mean()-crop_size/2), 0, height-crop_size))
    left = int(np.clip(round(xx.mean()-crop_size/2), 0, width-crop_size))
    if valid_mask is not None:
        valid=np.asarray(valid_mask,bool)
        if valid.shape!=tracks.shape:
            raise ValueError('Study quality mask has a different domain shape')
        def window_sums(values):
            integral=np.pad(np.asarray(values,np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
            return (integral[crop_size:,crop_size:]-integral[:-crop_size,crop_size:]
                    -integral[crop_size:,:-crop_size]+integral[:-crop_size,:-crop_size])
        candidates=(window_sums(valid)==crop_size**2)&(window_sums(tracks==int(track_id))>0)
        rows,columns=np.where(candidates)
        if not len(rows):
            raise ValueError('Nessun crop 224×224 interamente valido nei 16 frame contiene la traccia; usare il dominio completo.')
        nearest=np.argmin((rows-top)**2+(columns-left)**2)
        top,left=int(rows[nearest]),int(columns[nearest])
    origin_y = height-top-crop_size if native_y[0] < native_y[-1] else top
    origin_x = width-left-crop_size if native_x[0] > native_x[-1] else left
    return origin_y,origin_x
