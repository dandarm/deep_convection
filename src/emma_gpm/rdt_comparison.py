"""Paired diagnostics on common valid clips; no accuracy/winner claim."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def compare_catalogs(new_dir,baseline_dir):
    new_dir,baseline_dir=Path(new_dir),Path(baseline_dir)
    new=pd.read_csv(new_dir/'catalog.csv');old=pd.read_csv(baseline_dir/'catalog.csv')
    old=old[old.clip_id.isin(new.clip_id)].copy()
    pairs=new[new.status=='ok'].merge(old[old.status=='ok'],on='clip_id',suffixes=('_new','_old'),validate='one_to_one')
    for column in ['start_time','origin_y','origin_x']:
        if not np.array_equal(pairs[column+'_new'],pairs[column+'_old']):raise ValueError('Unpaired clip identity')
    columns=['clip_id','convective_fraction_new','convective_fraction_old','convective_peak_fraction_new',
        'convective_peak_fraction_old','tracked_objects_new','tracked_objects_old',
        'single_point_fraction','baseline_single_point_fraction','mean_visible_track_frames',
        'baseline_mean_visible_track_frames','temporal_object_valid_fraction','baseline_temporal_object_valid_fraction']
    pairs[[c for c in columns if c in pairs]].to_csv(new_dir/'paired_comparison.csv',index=False)
    summary=dict(common_valid=len(pairs),baseline_valid=int((old.status=='ok').sum()),new_valid=int((new.status=='ok').sum()),
        baseline_only_valid=int(((old.status=='ok')&~old.clip_id.isin(pairs.clip_id)).sum()),
        note='Same central clips, expanded context and adaptive object definition; geometric continuity diagnostics, not detection accuracy.')
    for name in ['single_point_fraction','mean_visible_track_frames','temporal_object_valid_fraction']:
        values=pairs[[name,'baseline_'+name]].replace([np.inf,-np.inf],np.nan).dropna()
        summary[name]=dict(clips=len(values),refined=float(values[name].mean()) if len(values) else None,
                          baseline=float(values['baseline_'+name].mean()) if len(values) else None)
    for version in ['new','old']:
        frame=pairs
        summary[version+'_convective_clips']=int((frame['convective_pixels_'+version]>0).sum())
        summary[version+'_ot_clips']=int((frame['ot_pixels_'+version]>0).sum())
        summary[version+'_known_cold_share']=float(frame['known_cold_pixels_'+version].sum()/frame['cold_pixels_'+version].sum()) if frame['cold_pixels_'+version].sum()>0 else None
    (new_dir/'paired_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
