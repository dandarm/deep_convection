"""SEVIRI-only RDT-CW-inspired refinement, not operational statistical RDT.

Reference: NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection v1.1 (2016), pp. 25-33.
Adaptive tower contours, advection/correlation recovery, primary lineage.
The project's deterministic convective/OT thresholds remain unchanged.
"""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from scipy import ndimage,signal
from .rdt_teacher import RDTConfig,OBJECT_COLUMNS,_components


@dataclass(frozen=True)
class RefinedRDTConfig(RDTConfig):
    cold_limit_k: float = 198.
    temperature_step_k: float = 1.
    tower_depth_k: float = 6.
    max_base_top_ratio: float = 4.
    max_motion_pixels: int = 12
    recovery_dilation_pixels: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (not 100<=self.cold_limit_k<240 or self.temperature_step_k<=0 or
            self.tower_depth_k<=0 or self.max_base_top_ratio<1 or
            self.max_motion_pixels<0 or self.recovery_dilation_pixels<0):
            raise ValueError('Invalid refined RDT configuration')


class ThermalContours:
    def __init__(self,ir,area,config):
        self.ir,self.area,self.config=ir,area,config
        self.cache={}

    def level(self,threshold):
        key=float(threshold)
        if key not in self.cache:
            labels,n=ndimage.label(self.ir<=key,structure=np.ones((3,3)))
            sizes=np.bincount(labels.ravel(),weights=self.area.ravel(),minlength=n+1)
            minima=ndimage.minimum(self.ir,labels,index=np.arange(1,n+1))
            self.cache[key]=(labels,sizes,np.r_[np.inf,minima])
        return self.cache[key]

    def containing(self,threshold,seed):
        labels,sizes,_=self.level(threshold)
        component=int(labels[seed])
        return float(sizes[component]) if component else 0.


def adaptive_cells(ir,area,config=None):
    """Warmest contours separating >=6 K towers, within the project's 240 K cap.

    A component-tree approximation on a 1 K grid. Leaves shallower than 6 K
    or below minimum physical area are not towers. Single-tower oversized
    bases are capped at 4 times the Tmin+6 K top contour when that top passes
    the same minimum area. This is explicit/limited, not full operational BT
    morphology redefinition or statistical discrimination.
    """
    cfg=config or RefinedRDTConfig();contours=ThermalContours(ir,area,cfg)
    roots,sizes,minima=contours.level(240.)
    selected=[]
    def descend(mask,threshold,minimum):
        # No child can have >=6 K depth below parent_minimum+6 K.
        limit=max(cfg.cold_limit_k,minimum+cfg.tower_depth_k)
        for lower in np.arange(threshold-cfg.temperature_step_k,limit-1e-8,-cfg.temperature_step_k):
            labels,child_sizes,child_min=contours.level(lower)
            children=np.unique(labels[mask]);children=children[children>0]
            children=[int(c) for c in children if child_sizes[c]>=cfg.min_area_km2
                      and lower-child_min[c]>=cfg.tower_depth_k]
            if len(children)>=2:
                for c in children:descend(labels==c,float(lower),float(child_min[c]))
                return
        # Exactly one significant tower: refine an excessive warm cloud base.
        seed=np.unravel_index(np.argmin(np.where(mask,ir,np.inf)),ir.shape)
        top=min(threshold,float(minimum+cfg.tower_depth_k))
        top_size=contours.containing(top,seed)
        if top_size>=cfg.min_area_km2 and area[mask].sum()>cfg.max_base_top_ratio*top_size:
            for lower in np.arange(threshold-cfg.temperature_step_k,top-1e-8,-cfg.temperature_step_k):
                labels,lower_sizes,_=contours.level(lower);c=int(labels[seed])
                if c and cfg.min_area_km2<=lower_sizes[c]<=cfg.max_base_top_ratio*top_size:
                    mask=labels==c;threshold=float(lower);break
        selected.append((mask,threshold,seed))
    for c in range(1,len(sizes)):
        if sizes[c]>=cfg.min_area_km2 and 240.-minima[c]>=cfg.tower_depth_k:
            descend(roots==c,240.,float(minima[c]))
    labels=np.zeros(ir.shape,dtype=np.int32);cells={}
    for c,(mask,threshold,seed) in enumerate(selected,1):
        if np.any(labels[mask]):raise RuntimeError('Overlapping adaptive contours')
        labels[mask]=c;yy,xx=np.where(mask)
        weights=area[mask]*np.maximum(threshold-ir[mask],1e-6)
        cells[c]=dict(mask=mask,threshold_k=threshold,seed=seed,area_km2=float(area[mask].sum()),
            min_k=float(ir[mask].min()),mean_k=float(ir[mask].mean()),
            centroid=np.array([np.average(yy,weights=weights),np.average(xx,weights=weights)]),
            border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()))
    return labels,cells,contours


def correlation_motion(before,now,max_pixels):
    """Bounded cold-pattern cross-correlation; current minus previous offset.

    Only IR108 patterns, no NWP/HRW. Estimated motion moves masks, not input
    radiances. Mean-centred correlation with deterministic smallest-shift tie.
    """
    a=np.maximum(240-before,0.);b=np.maximum(240-now,0.)
    a=a-a.mean();b=b-b.mean()
    if np.linalg.norm(a)<1e-8 or np.linalg.norm(b)<1e-8:return np.zeros(2,dtype=int)
    cross=signal.fftconvolve(b,a[::-1,::-1],mode='full')
    centre=np.array(a.shape)-1;r=int(max_pixels)
    window=cross[centre[0]-r:centre[0]+r+1,centre[1]-r:centre[1]+r+1]
    candidates=np.argwhere(np.isclose(window,window.max(),rtol=1e-12,atol=1e-12))-r
    best=min(candidates.tolist(),key=lambda v:(v[0]**2+v[1]**2,v[0],v[1]))
    return np.asarray(best,dtype=int)


def shifted(mask,motion):
    return ndimage.shift(mask.astype(np.uint8),np.rint(motion),order=0,mode='constant',cval=0,prefilter=False).astype(bool)


def association_graph(previous,current,area,guess,cfg):
    edges={};parents={c:[] for c in current};children={p:[] for p in previous}
    labels=np.zeros(area.shape,dtype=np.int32)
    for c,now in current.items():labels[now['mask']]=c
    cache={}
    for c,now in current.items():
        for method in ['advection','correlation','enlarged_recovery']:
            found=[]
            if method=='enlarged_recovery' and (now['mask'].sum()>5 or cfg.recovery_dilation_pixels==0):break
            cache_key='advection' if method=='advection' else 'correlation'
            if cache_key not in cache:
                translated={}
                for p,before in previous.items():
                    motion=before.get('motion',np.zeros(2)) if cache_key=='advection' else guess
                    mask=shifted(before['mask'],motion)
                    sums=np.bincount(labels[mask],weights=area[mask],minlength=max(current,default=0)+1)
                    translated[p]=(mask,float(area[mask].sum()),sums)
                cache[cache_key]=translated
            target=now['mask']
            target_area=now['area_km2']
            if method=='enlarged_recovery':
                target=ndimage.binary_dilation(target,structure=np.ones((3,3)),iterations=cfg.recovery_dilation_pixels)
                target_area=float(area[target].sum())
            for p,before in previous.items():
                translated,translated_area,sums=cache[cache_key][p]
                intersection=float(area[translated & target].sum()) if method=='enlarged_recovery' else float(sums[c])
                denominator=min(translated_area,target_area)
                aor=intersection/denominator if denominator else 0.
                if intersection and aor>=cfg.min_aor:found.append((p,aor,intersection))
            if found:
                for p,aor,intersection in found:
                    parents[c].append(p);children[p].append(c)
                    edges[p,c]=dict(aor=aor,intersection=intersection,method=method)
                break
    return parents,children,edges


def primary_lineage(previous,current,parents,children,previous_contours,current_contours):
    """Primary continuation for 1:1, simple merges and simple splits only.

    Dominance uses area at a common thermal level, then native-contour area,
    then component ID for deterministic ties. Many-to-many graph components
    have no primary links, even if individual vertices look unambiguous.
    """
    primary={};events={};visited=set()
    for c in current:
        if c in visited:continue
        ps=set();cs={c};todo=[c]
        while todo:
            cc=todo.pop()
            for p in parents[cc]:
                if p not in ps:
                    ps.add(p)
                    for child in children[p]:
                        if child not in cs:cs.add(child);todo.append(child)
        visited.update(cs)
        common=max([current[k]['threshold_k'] for k in cs]+[previous[k]['threshold_k'] for k in ps])
        if not ps:events[c]='new'
        elif len(ps)==1 and len(cs)==1:
            primary[c]=next(iter(ps));events[c]='continuation'
        elif len(cs)==1:
            p=max(ps,key=lambda k:(previous_contours.containing(common,previous[k]['seed']),previous[k]['area_km2'],-k))
            primary[c]=p;events[c]='merge_main'
        elif len(ps)==1:
            p=next(iter(ps));main=max(cs,key=lambda k:(current_contours.containing(common,current[k]['seed']),current[k]['area_km2'],-k))
            primary[main]=p
            for cc in cs:events[cc]='split_main' if cc==main else 'split_new'
        else:
            for cc in cs:events[cc]='complex'
    return primary,events


EXTRA_COLUMNS=['threshold_k','primary_parent','association_method','motion_y_pixels','motion_x_pixels',
               'growth_valid','phase_valid','common_threshold_k','history_length']


def label_refined_clip(clip,times,pixel_area_km2,config=None):
    cfg=config or RefinedRDTConfig();clip=np.asarray(clip);times=pd.DatetimeIndex(times);area=np.asarray(pixel_area_km2,dtype=float)
    if (clip.ndim!=4 or clip.shape[1]!=7 or len(times)!=len(clip) or not len(times) or
        area.shape!=clip.shape[2:] or not np.isfinite(clip).all() or not np.isfinite(area).all() or
        np.any(area<=0) or times.hasnans or not np.all(np.diff(times.asi8)==5*60*1_000_000_000) or
        clip.min()<100 or clip.max()>400):raise ValueError('Invalid finite chronological raw Kelvin clip/geometry')
    shape=(len(clip),*area.shape)
    classes=np.zeros(shape,dtype=np.uint8);tracks=np.zeros(shape,dtype=np.int32)
    uncertain=np.zeros(shape,dtype=bool);phase_uncertain=np.zeros(shape,dtype=bool)
    levels=np.zeros((len(clip),4,*area.shape),dtype=np.int32)
    previous={};before_contours=None;next_id=1;rows=[];links=[]
    for t in range(len(clip)):
        ir,wv=clip[t,4],clip[t,0]
        for k,threshold in enumerate(cfg.thresholds_k):levels[t,k]=_components(ir,area,threshold,cfg.min_area_km2)
        cold=levels[t,0]>0;classes[t,cold]=1;uncertain[t,cold]=True
        labels,current,contours=adaptive_cells(ir,area,cfg)
        guess=correlation_motion(clip[t-1,4],ir,min(cfg.max_motion_pixels,min(ir.shape)-1)) if t else np.zeros(2,dtype=int)
        parents,children,edges=association_graph(previous,current,area,guess,cfg)
        primary,events=primary_lineage(previous,current,parents,children,before_contours,contours)
        for c,now in current.items():
            p=primary.get(c);before=previous[p] if p is not None else None
            if before is not None:now['track_id']=before['track_id']
            else:now['track_id']=next_id;next_id+=1
            valid=before is not None and not now['border'] and not before['border']
            contrast=now['mean_k']-now['min_k']
            cooling=3*(now['min_k']-before['min_k']) if valid else np.nan
            common=max(now['threshold_k'],before['threshold_k']) if before else np.nan
            # Topology changes can produce area jumps unrelated to expansion.
            # Keep identity/cooling, but don't turn the merge/split into growth.
            growth_valid=valid and events[c]=='continuation'
            growth=3*(contours.containing(common,now['seed'])-before_contours.containing(common,before['seed'])) if growth_valid else np.nan
            developing=growth_valid and contrast>=cfg.contrast_k and cooling<=cfg.cooling_k_15min and growth>0
            inherited=valid and before['convective'] and contrast>=cfg.contrast_k
            category=2 if developing else 3 if inherited else 1
            now['convective']=category in (2,3)
            phase_valid=valid and (growth_valid or cooling>cfg.cooling_k_15min)
            known=valid and (growth_valid or inherited)
            now['history_length']=(before['history_length']+1) if before else 1
            motion=guess.astype(float)
            # Family-weighted centres reduce spurious velocity from topology
            # changes (ATBD section 3.2.1.2.2.1). Same motion for linked siblings.
            family_p=set(parents[c]);family_c={c};pending=list(family_p)
            while pending:
                pp=pending.pop()
                for cc in children[pp]:
                    if cc not in family_c:
                        family_c.add(cc)
                        for other in parents[cc]:
                            if other not in family_p:family_p.add(other);pending.append(other)
            if family_p:
                wp=[previous[k]['area_km2'] for k in sorted(family_p)]
                wc=[current[k]['area_km2'] for k in sorted(family_c)]
                previous_centre=np.average([previous[k]['centroid'] for k in sorted(family_p)],axis=0,weights=wp)
                current_centre=np.average([current[k]['centroid'] for k in sorted(family_c)],axis=0,weights=wc)
                displacement=current_centre-previous_centre
                prior_motion=np.average([previous[k].get('motion',guess) for k in sorted(family_p)],axis=0,weights=wp)
                if np.linalg.norm(displacement)<=cfg.max_motion_pixels:
                    motion=.5*prior_motion+.5*displacement
            now['motion']=motion
            mask=now['mask'];classes[t,mask]=category;tracks[t,mask]=now['track_id']
            uncertain[t,mask]=not known;phase_uncertain[t,mask]=not phase_valid
            ot=mask & now['convective'] & ((wv-ir)>=cfg.ot_btd_k) & (ir<=cfg.ot_ir_k)
            classes[t,ot]=4
            reason=('development_criteria' if developing else 'inherited_main_link' if inherited else
                    'temporal_unknown' if not known else 'criteria_not_met')
            method=edges[p,c]['method'] if p is not None else 'none'
            rows.append(dict(frame=t,time=times[t].isoformat(),component=c,track_id=now['track_id'],
                parents=tuple(previous[k]['track_id'] for k in parents[c]),event=events[c],
                area_km2=now['area_km2'],min_k=now['min_k'],mean_k=now['mean_k'],contrast_k=contrast,
                cooling_k_15min=cooling,growth_km2_15min=growth,border=now['border'],temporal_valid=valid,
                class_id=category,ot_pixels=int(ot.sum()),reason=reason,threshold_k=now['threshold_k'],
                primary_parent=before['track_id'] if before else None,association_method=method,
                motion_y_pixels=float(motion[0]),motion_x_pixels=float(motion[1]),growth_valid=growth_valid,
                phase_valid=phase_valid,common_threshold_k=common,history_length=now['history_length']))
            for pp in parents[c]:
                links.append(dict(frame=t,parent_component=pp,parent_track=previous[pp]['track_id'],component=c,
                    aor=edges[pp,c]['aor'],method=edges[pp,c]['method'],primary=pp==p))
        previous,before_contours=current,contours
    return dict(classes=classes,tracks=tracks,uncertain=uncertain,phase_uncertain=phase_uncertain,
                levels=levels,objects=pd.DataFrame(rows,columns=OBJECT_COLUMNS+EXTRA_COLUMNS),
                links=pd.DataFrame(links,columns=['frame','parent_component','parent_track','component','aor','method','primary']),
                config=asdict(cfg),teacher_version='refined_v2')
