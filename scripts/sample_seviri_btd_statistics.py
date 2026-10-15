#!/usr/bin/env python3
"""Seeded month-stratified BTD diagnostics; not exhaustive annual extrema."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.rdt.storage import DomainFrameReader,atomic_json
from emma_gpm.seviri_clips import SeviriZarrClipReader,IncompleteSeviriClipError


def stats(arrays):
    if not arrays:return None
    values=np.concatenate(arrays)
    q=np.percentile(values,[0,1,50,99,100])
    return dict(pixels=len(values),minimum_k=float(q[0]),p01_k=float(q[1]),median_k=float(q[2]),p99_k=float(q[3]),maximum_k=float(q[4]),negative_percent=float(100*(values<0).mean()),positive_percent=float(100*(values>0).mean()),below_minus40_percent=float(100*(values<-40).mean()),above_plus10_percent=float(100*(values>10).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label-root',type=Path,required=True);args=parser.parse_args()
    labels=args.label_root;protocol=json.loads((labels/'protocol.json').read_text());root=Path(protocol['root'])
    _,times,_=pickle.loads((labels/'inventory.pkl').read_bytes())
    area=np.load(next((labels/'segments').glob('*/pixel_area_north_up.npy')))
    rng=np.random.default_rng(20261014);available=set(times)
    complete=pd.DatetimeIndex([t for t in times if all(t+pd.Timedelta(minutes=5*i) in available for i in range(16))])
    full,crops,selection,rejections=[],[],[],[]
    reader=DomainFrameReader(root)
    try:
        for month in sorted(set(times.month)):
            pool=times[times.month==month]
            chosen=rng.choice(len(pool),size=min(3,len(pool)),replace=False)
            for i in chosen:
                stamp=pool[i];raw=reader.load(stamp)
                valid=np.isfinite(raw).all(axis=0)&((raw>=100)&(raw<=400)).all(axis=0)&np.isfinite(area)
                full.append((raw[0]-raw[4])[valid]);selection.append(dict(kind='domain_frame',time=stamp.isoformat()))
            starts=complete[complete.month==month]
            if not len(starts):continue
            for attempt in range(12):
                start=starts[int(rng.integers(len(starts)))];y=int(rng.integers(area.shape[0]//224))*224;x=int(rng.integers(area.shape[1]//224))*224
                try:
                    with SeviriZarrClipReader(root) as clip_reader:clip=clip_reader.load(start,origin_y=y,origin_x=x)
                    if not ((clip>=100)&(clip<=400)).all():raise ValueError('Invalid radiometry')
                except (IncompleteSeviriClipError,ValueError) as error:
                    rejections.append(dict(time=start.isoformat(),origin_y=y,origin_x=x,reason=str(error)));continue
                crops.append((clip[:,0]-clip[:,4]).ravel());selection.append(dict(kind='clip',time=start.isoformat(),origin_y=y,origin_x=x));break
            print(f'Month {month}: {len(full)} frames, {len(crops)} complete clips',flush=True)
    finally:reader.close()
    report=dict(seed=20261014,sampling='3 random whole-domain frames and up to 1 complete random grid clip per available month; sample statistics, not exhaustive annual bounds',original_display_limits_k=[-40,10],domain=stats(full),clips=stats(crops),selection=selection,rejections=rejections)
    out=labels/'exploration';out.mkdir(exist_ok=True);atomic_json(out/'btd_statistics.json',report)
    print(json.dumps({k:report[k] for k in ('domain','clips')},indent=2),flush=True)


if __name__=='__main__':main()
