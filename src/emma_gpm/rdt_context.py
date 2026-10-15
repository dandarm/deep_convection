"""Expanded teacher context with exact native-coordinate central crop mapping."""
import numpy as np


def context_window(y,x,height,width,*,crop=224,halo=64,north_reversed=False,west_reversed=False):
    if halo<0 or crop<1 or y<0 or x<0 or y+crop>height or x+crop>width:
        raise ValueError('Invalid crop/context bounds')
    size=min(crop+2*halo,height,width)
    yy=min(max(y-halo,0),height-size);xx=min(max(x-halo,0),width-size)
    cy=y-yy;cx=x-xx
    if north_reversed:cy=size-cy-crop
    if west_reversed:cx=size-cx-crop
    return yy,xx,size,(slice(cy,cy+crop),slice(cx,cx+crop))


def crop_teacher_result(result,roi,warmup=3,area=None):
    """Crop outputs only; keep whole-context object metrics for visible objects."""
    for field in ['classes','tracks','uncertain','phase_uncertain','levels']:
        if field in result:result[field]=result[field][warmup:,...,roi[0],roi[1]]
    frames=result['objects']
    keep=[];visible_areas=[]
    for row in frames.itertuples():
        t=int(row.frame)-warmup
        visible=(result['tracks'][t]==row.track_id) if 0<=t<len(result['tracks']) else None
        present=visible is not None and visible.any()
        keep.append(present)
        visible_areas.append(float(area[visible].sum()) if present and area is not None else np.nan)
    result['objects']=frames.loc[keep].copy()
    result['objects']['frame']-=warmup
    result['objects']['visible_area_km2']=np.array(visible_areas)[keep]
    links=result['links']
    if not links.empty:
        result['links']=links[links.frame>=warmup].copy();result['links']['frame']-=warmup
    return result
