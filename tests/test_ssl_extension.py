import numpy as np
import pandas as pd
import pytest
from argparse import Namespace
import json
import os

from scripts.build_seviri_ssl_extension import assemble, unused_candidates, keys
from scripts.build_seviri_ssl_scaling_dataset import global_moments, MANIFEST_COLUMNS
from scripts.extend_ssl_clip_files import collection_plan, build as extend_files
from scripts.run_seviri_ssl_selected import selected_stages
from emma_gpm.seviri_clip_files import clip_key, codec, digest_bytes, SHAPE, ClipFileDataset
from emma_gpm.seviri import THERMAL_CHANNELS


def clips(start, count):
    rows=[]
    for i in range(start,start+count):
        rows.append(dict(start_time='2020-09-12T00:00:00Z',origin_y=i,origin_x=0,
            day='2020-09-12',time_bin=0,store='2020/day.zarr',mean_abs_change_k=float(i),
            p90_abs_change_k=float(i+1),fraction_change_gt2k=.1,mean_k=260.+i,variance_k2=4.))
    return pd.DataFrame(rows)


def test_extension_preserves_base_and_stratifies_only_disjoint_additions():
    base,pool=clips(0,4),clips(4,8)
    result,extra,indices,sampling=assemble(base,pool,8,20261024)
    pd.testing.assert_frame_equal(result.iloc[:4].reset_index(drop=True),base[MANIFEST_COLUMNS])
    assert len(result)==len(set(keys(result)))==8
    assert set(keys(extra)).issubset(set(keys(pool)))
    assert sampling['uniform_count']==sampling['variability_count']==2
    again,_,_,_=assemble(base,pool,8,20261024)
    pd.testing.assert_frame_equal(result,again)
    with pytest.raises(ValueError,match='overlaps'):
        assemble(base,pd.concat([pool,base.iloc[:1]]),8,1)


def test_extension_excludes_timezone_equivalent_existing_clip_keys():
    base=clips(0,2);candidate=clips(0,4)
    candidate['start_time']='2020-09-12T02:00:00+02:00'
    remaining=unused_candidates(candidate,base)
    assert remaining.origin_y.tolist()==[2,3]


def test_pooled_extension_statistics_include_between_clip_temperature_variation():
    values=[np.array([258.,262.]),np.array([268.,272.])]
    moments=pd.DataFrame({'mean_k':[v.mean() for v in values],
                          'variance_k2':[v.var() for v in values]})
    mean,std=global_moments(moments)
    assert mean==np.concatenate(values).mean()
    assert std==pytest.approx(np.concatenate(values).std())


def test_collection_plan_reuses_keys_and_preserves_manifest_order(tmp_path):
    base=clips(0,2);extended=clips(0,4)
    path=tmp_path/'train.csv';extended.to_csv(path,index=False)
    val=tmp_path/'validation.csv';base.iloc[::-1].to_csv(val,index=False)
    records={f'clip_{i:06d}.blosc':dict(key=key,bytes=1,sha256='unused') for i,key in enumerate(keys(base))}
    shared,tasks,bindings=collection_plan([path,val],{'clips':records})
    assert shared==records
    assert len(tasks)==2 and {t[1] for t in tasks}==set(keys(extended))-set(keys(base))
    assert bindings[digest_bytes(val.read_bytes())]['files']==['clip_000001.blosc','clip_000000.blosc']


def test_linked_collection_leaves_parent_bytes_and_metadata_unchanged(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'extension';out.mkdir()
    raw=tmp_path/'zarr';raw.mkdir();manifest=tmp_path/'train.csv';clips(0,1).to_csv(manifest,index=False)
    pixels=np.full(SHAPE,260.,dtype=np.float16);encoded=codec().encode(pixels)
    name='clip_000000.blosc';(parent/name).write_bytes(encoded)
    metadata=dict(version=1,shape=list(SHAPE),dtype='float16',units='K',channels=list(THERMAL_CHANNELS),
        orientation='north-up west-left',cadence_minutes=5,lossless=True,codec=codec().get_config(),
        zarr_root=str(raw),clips={name:dict(key=keys(clips(0,1))[0],bytes=len(encoded),sha256=digest_bytes(encoded))})
    (parent/'metadata.json').write_text(json.dumps(metadata));before=(parent/'metadata.json').read_bytes()
    args=Namespace(reuse_root=parent,output_dir=out,zarr_root=raw,manifest=[manifest],workers=1)
    extend_files(args)
    assert os.path.samefile(parent/name,out/name)
    assert (parent/'metadata.json').read_bytes()==before
    ds=ClipFileDataset(out,manifest,zarr_root=raw)
    restored=ds.decode_into(0,np.empty(SHAPE,dtype=np.float16))
    np.testing.assert_array_equal(restored.view(np.uint16),pixels.view(np.uint16))


def test_selected_queue_contains_only_approved_objectives_and_full_batches(tmp_path):
    args=Namespace(zarr_root=tmp_path/'zarr',epochs=75,batch_size=32,workers=4,
        seed=20261024,clip_root=tmp_path/'clips',reference_series=tmp_path/'previous')
    stages=selected_stages(args,tmp_path/'dataset',tmp_path/'n16384')
    assert [s['name'] for s in stages]==['future','joint']
    for stage in stages:
        assert stage['epochs']==75 and stage['kind']=='pretraining'
        flags=stage['flags']
        for flag,value in [('--batch-size','32'),('--loader-batch-size','32'),
                           ('--gradient-accumulation','1'),('--prefetch-factor','4'),('--epochs','75')]:
            assert flags[flags.index(flag)+1]==value
        assert '--freeze-encoder' not in flags and '--encoder-from' not in flags
