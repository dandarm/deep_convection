import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_seviri_ssl_scaling_dataset import expansion_partition, global_moments, collect_cached
from scripts.run_seviri_ssl_scaling import make_plans, stage_command, stage_completed, immutable_json
from emma_gpm.ssl_scaling import collect_scaling, paired_comparisons, publish_summary


def candidate(day, hour=0, minute=0):
    return dict(start_time=f'2020-09-{day:02d}T{hour:02d}:{minute:02d}:00Z',
                day=f'2020-09-{day:02d}',origin_y=0,origin_x=0,time_bin=hour//6,store=f'2020/day{day}.zarr')


def test_expansion_preserves_historical_holdout_and_full_day_separation():
    base=pd.DataFrame([candidate(12)])
    old=pd.DataFrame([candidate(14,12)])
    pool=pd.DataFrame([candidate(12),candidate(14),candidate(15,22,40),
                       candidate(15,22,45),candidate(16),candidate(17)])
    additions,validation,boundary=expansion_partition(pool,base,old)
    assert additions.start_time.tolist()==['2020-09-15T22:40:00Z']
    assert validation.day.tolist()==['2020-09-17']
    assert pd.to_datetime(boundary)==pd.Timestamp('2020-09-17',tz='UTC')
    with pytest.raises(ValueError,match='insufficient'):
        expansion_partition(pool[pool.day<'2020-09-17'],base,old)


def test_pooled_normalization_includes_between_clip_mean_changes():
    clips=np.array([[250.,252.,254.],[270.,274.,278.]])
    frame=pd.DataFrame({'mean_k':clips.mean(1),'variance_k2':clips.var(1)})
    mean,std=global_moments(frame)
    assert np.isclose(mean,clips.mean())
    assert np.isclose(std,clips.std())
    assert std>np.sqrt(frame.variance_k2.mean())


def setup_plans(tmp_path):
    ready={'cases':[]}
    for size in (4096,8192):
        case=dict(name=f'n{size}',train_samples=size,validation_samples=512)
        ready['cases'].append(case)
        path=tmp_path/case['name']/'dataset'
        path.mkdir(parents=True)
        for name in ('train.csv','channel_stats.json','source_snapshot.json'):
            (path/name).write_text(str(size))
        (path/'validation.csv').write_text('same later validation')
    (tmp_path/'dataset_ready.json').write_text(json.dumps(ready))
    args=argparse.Namespace(output_dir=tmp_path,zarr_root=tmp_path/'data',batch_size=16,
                           workers=8,seed=20261024,epochs=150,probe_epochs=150)
    return args,ready


def test_scaling_plan_covers_every_objective_and_probe_at_both_sizes(tmp_path):
    args,ready=setup_plans(tmp_path)
    plan,plans=make_plans(args,ready)
    assert len(plan['order'])==18
    assert plan['order'][:2]==[{'case':'n4096','stage':'future'},{'case':'n8192','stage':'future'}]
    for name,case in plans.items():
        assert {stage['name'] for stage in case['stages']}=={
            'future','mvm','joint','residual','probe_random','probe_mvm','probe_future','probe_joint','probe_residual'}
        assert all(stage['epochs']==150 for stage in case['stages'])
        for stage in case['stages']:
            flags=stage['flags']
            assert flags[flags.index('--epochs')+1]=='150'
            assert flags[flags.index('--manifest')+1]==str(tmp_path/name/'dataset/train.csv')
            assert flags[flags.index('--variability-reference-manifest')+1]==str(tmp_path/'n4096/dataset/train.csv')
    (tmp_path/'n8192/dataset/validation.csv').write_text('different holdout')
    with pytest.raises(ValueError,match='identical validation'):
        make_plans(args,ready)


def test_resume_requires_checkpoint_and_completion_requires_artifacts(tmp_path):
    stage=dict(name='future',epochs=150,flags=['--epochs','150'])
    output=tmp_path/'future'
    output.mkdir()
    assert '--resume' not in stage_command(tmp_path,stage)
    (output/'checkpoint_latest.pt').touch()
    assert '--resume' in stage_command(tmp_path,stage)
    (output/'progress.json').write_text(json.dumps(dict(status='completed',completed_epochs=149)))
    assert not stage_completed(tmp_path,stage)
    (output/'progress.json').write_text(json.dumps(dict(status='completed',completed_epochs=150)))
    with pytest.raises(ValueError,match='missing required artifacts'):
        stage_completed(tmp_path,stage)
    (output/'encoder').mkdir()
    for rel in ('validation_latest.json','encoder/config.json','encoder/model.safetensors'):
        (output/rel).touch()
    assert stage_completed(tmp_path,stage)


def test_existing_plan_is_immutable(tmp_path):
    path=tmp_path/'plan.json'
    immutable_json(path,{'epochs':150})
    immutable_json(path,{'epochs':150})
    with pytest.raises(ValueError,match='Frozen plan changed'):
        immutable_json(path,{'epochs':151})


def test_pairing_distinguishes_epochs_from_optimizer_updates():
    curves=pd.DataFrame([
        dict(model='future',kind='pretraining',training_clips=size,epoch=epoch,
             optimizer_steps=epoch*size//32,validation_rmse_k=rmse,validation_mvm_mse_k2=np.nan)
        for size,epoch,rmse in ((4096,1,4.),(4096,2,3.),(8192,1,2.5))])
    pairs=paired_comparisons(curves).set_index('matched_budget')
    assert pairs.loc['epoch','small_epoch']==1
    assert pairs.loc['epoch','large_epoch']==1
    assert pairs.loc['epoch','validation_rmse_k_gain']==1.5
    assert pairs.loc['optimizer_steps','small_epoch']==2
    assert pairs.loc['optimizer_steps','large_epoch']==1
    assert pairs.loc['optimizer_steps','validation_rmse_k_gain']==.5


def test_summary_uses_each_dataset_kelvin_scale_and_committed_epochs(tmp_path):
    (tmp_path/'scaling_plan.json').write_text(json.dumps({'cases':[dict(name='n4096',train_samples=4096)]}))
    queue=tmp_path/'n4096'
    (queue/'dataset').mkdir(parents=True)
    (queue/'dataset/channel_stats.json').write_text(json.dumps({'global_std_k':20.}))
    (queue/'comparison_plan.json').write_text(json.dumps({'stages':[
        dict(name='mvm',kind='pretraining',epochs=150),dict(name='future',kind='pretraining',epochs=150)]}))
    (queue/'mvm').mkdir()
    progress=dict(status='training',completed_epochs=1,current_epoch=2,current_step=16,steps_per_epoch=256,
        epochs=[dict(epoch=1,optimizer_steps=128,train_seconds=50.,validation_seconds=10.,
                     persistence_rmse_k=4.5,advection_rmse_k=4.1,train_mvm=.1,validation_mvm=.2)])
    (queue/'mvm/progress.json').write_text(json.dumps(progress))
    curves,status,channels=publish_summary(tmp_path)
    assert len(curves)==1
    assert curves.iloc[0].validation_mvm_mse_k2==80.
    assert status.set_index('model').loc['future','status']=='queued'
    assert status.set_index('model').loc['mvm','completed_epochs']==1
    assert channels.empty
    assert (tmp_path/'training_status.csv').exists()


def test_preparation_cache_resumes_without_revalidating_committed_candidates(tmp_path,monkeypatch):
    import scripts.build_seviri_ssl_scaling_dataset as builder
    frames=pd.DataFrame([candidate(15,hour) for hour in range(3)])
    pd.DataFrame([dict(**frames.iloc[0].to_dict(),valid=True,mean_k=270.,variance_k2=10.)]).to_csv(
        tmp_path/'pool_checked.csv',index=False)
    calls=[]
    def validate(row):
        calls.append(row['start_time'])
        return dict(**row,valid=True,mean_k=270.,variance_k2=10.)
    class Executor:
        def map(self,function,rows,chunksize):
            return map(function,rows)
    monkeypatch.setattr(builder,'validate_candidate',validate)
    result=collect_cached(Executor(),frames,3,tmp_path,'pool')
    assert len(result)==3
    assert frames.iloc[0].start_time not in calls
    assert len(calls)==2


def test_variability_reference_uses_shared_training_subset_without_validation_leakage(tmp_path):
    from scripts.pretrain_seviri_ssl import reference_variability
    training=pd.DataFrame([dict(**candidate(12),mean_abs_change_k=.5),
                           dict(**candidate(13),mean_abs_change_k=1.),
                           dict(**candidate(15),mean_abs_change_k=9.)])
    reference=tmp_path/'reference.csv'
    # Scores come from the prepared training data, even if a reference column differs.
    training.iloc[:2].assign(mean_abs_change_k=999).to_csv(reference,index=False)
    scores=reference_variability(training,reference)
    assert scores.tolist()==[.5,1.]
    pd.DataFrame([candidate(17)]).to_csv(reference,index=False)
    with pytest.raises(ValueError,match='subset of the training'):
        reference_variability(training,reference)
