import numpy as np
import pytest
import torch
from transformers import VideoMAEModel

from emma_gpm.three_head_data import (CT_MAPPING, RDT_MAPPING, map_classes, quality_gate,
    gate_counts, valid_history, valid_origins, clip_bucket, sample_weights,
    teacher_acceptance, validate_manifest_acceptance)
from emma_gpm.three_head_model import (ThreeHeadProbe, probe_loss, SegmentationMetrics,
    diagnostic_input, class_weights)
from emma_gpm.videomae import videomae_small_7ch_config


def test_reused_teacher_periods_preserve_gate_and_reject_mixed_definitions(tmp_path):
    import json
    from emma_gpm.three_head_data import materialize_dataset
    plan=dict(source_root=str(tmp_path/'raw'),dataset_output=str(tmp_path/'dataset'),
              pretraining_manifests=[],splits={'train':['2022-04-04'],
                  'validation':['2022-07-04'],'test':['2022-07-07']},teacher_periods={})
    protocols={}
    for split,days in plan['splits'].items():
        folder=tmp_path/('period_'+split);folder.mkdir()
        plan['teacher_periods'][split]=str(folder)
        # Generation roles may differ from the final downstream assignment.
        protocol=dict(version=2,split='original_diagnostic_role',days=days,
            source_root=plan['source_root'],ct_config={'same':True},rdt_config_hash='teacher_v3',
            ct_mapping=CT_MAPPING,rdt_mapping=RDT_MAPPING,valid_history_minutes=15,
            cell_history='same real trajectory definition',teacher_domain='full',orientation='north-up')
        (folder/'protocol.json').write_text(json.dumps(protocol))
        failed_gate=quality_gate(dict(convective=1000,inside_cma=1000,inside_high_opaque=850,inside_clear_low=0))
        (folder/'complete.json').write_text(json.dumps({'quality_gate':failed_gate}))
        protocols[split]=protocol
    with pytest.raises(ValueError,match='Inter-teacher quality gate failed'):
        materialize_dataset(plan)
    assert not (tmp_path/'dataset/manifest.json').exists()
    protocols['test']['rdt_config_hash']='incompatible_teacher'
    (tmp_path/'period_test/protocol.json').write_text(json.dumps(protocols['test']))
    with pytest.raises(ValueError,match='different teacher or target-validity'):
        materialize_dataset(plan)
    assert not (tmp_path/'dataset/manifest.json').exists()


def test_native_mappings_preserve_fill_and_reduce_rdt_seven_to_five():
    assert map_classes(np.arange(16), CT_MAPPING).tolist() == [255,0,0,0,0,1,1,2,3,3,1,4,4,4,4,4]
    assert map_classes(np.array([0,1,2,3,4,5,6,255]), RDT_MAPPING).tolist() == [0,1,2,2,3,3,4,255]
    with pytest.raises(ValueError):
        map_classes(np.array([17]), CT_MAPPING)


def test_background_validity_and_real_fifteen_minute_history():
    observations = [np.ones((2,2), bool) for _ in range(4)]
    fields = dict(valid_observation=observations[-1], cell_id_map=np.array([[0,1],[2,0]]),
                  time='2022-07-01T00:15', cells={
                      1:dict(track_id=1,bbox=(slice(0,1),slice(1,2)),mask=np.ones((1,1),bool),cooling_15m_k=-1.),
                      2:dict(track_id=2,bbox=(slice(1,2),slice(0,1)),mask=np.ones((1,1),bool),cooling_15m_k=-1.)})
    trajectory=[{'timestamp':f'2022-07-01T00:{minute:02}'} for minute in (0,5,10,15)]
    tracks={1:trajectory[-2:],2:trajectory}
    assert not valid_history(observations[:3],fields,tracks).any()
    np.testing.assert_array_equal(valid_history(observations,fields,tracks), [[True,False],[True,True]])
    observations[0][0,0] = False
    assert not valid_history(observations,fields,tracks)[0,0]
    tracks[2]=[{'timestamp':'2022-06-30T23:55'},*trajectory[-3:]]
    assert not valid_history(observations,fields,tracks)[1,0]
    tracks[2]=trajectory
    fields['cells'][2]['cooling_15m_k']=np.nan
    assert not valid_history(observations,fields,tracks)[1,0]


def test_gate_enforces_thresholds_and_undefined_empty_support():
    assert not quality_gate(dict(convective=0,inside_cma=0,inside_high_opaque=0,inside_clear_low=0))['passed']
    counts = dict(convective=1000,inside_cma=995,inside_high_opaque=900,inside_clear_low=0)
    assert quality_gate(counts)['passed']
    assert not quality_gate({**counts,'inside_clear_low':1})['passed']
    assert not quality_gate({**counts,'inside_high_opaque':899})['passed']
    assert gate_counts(np.array([1,1,1]),np.array([3,3,3]),np.array([4,255,0]),np.ones(3,bool))['convective'] == 1


def test_exploratory_acceptance_retains_original_failure_and_remaining_requirements():
    counts=dict(convective=1000,inside_cma=995,inside_high_opaque=850,inside_clear_low=0)
    gate=quality_gate(counts)
    assert not teacher_acceptance(gate)['accepted']
    accepted=teacher_acceptance(gate,'ct_high_opaque_diagnostic')
    assert accepted['accepted'] and not accepted['original_gate_passed'] and not gate['passed']
    for changed in [{'inside_cma':994},{'inside_clear_low':1}]:
        assert not teacher_acceptance(quality_gate({**counts,**changed}),'ct_high_opaque_diagnostic')['accepted']
    with pytest.raises(ValueError,match='Unknown'):
        teacher_acceptance(gate,'typo')
    manifest=dict(teacher_quality_gates={split:gate for split in ('train','validation','test')},
                  teacher_acceptance_policy='ct_high_opaque_diagnostic')
    manifest['teacher_acceptance']=validate_manifest_acceptance(manifest)
    assert all(row['accepted'] for row in validate_manifest_acceptance(manifest).values())
    manifest['teacher_acceptance']['test']['original_gate_passed']=True
    with pytest.raises(ValueError,match='Persisted'):
        validate_manifest_acceptance(manifest)


def test_cold_cloud_centre_can_select_a_mixed_scene_without_relabelling_it():
    ct=np.full((4,8,8),3,dtype=np.uint8)
    rdt=np.ones_like(ct);rdt[:,:,0]=2;rdt[0,0,0]=4
    before=rdt.copy()
    assert clip_bucket(ct,rdt)==('A',True)
    assert clip_bucket(ct,rdt,policy='cold_cloud_centre')==('B',True)
    np.testing.assert_array_equal(rdt,before)
    rdt[-1,4,4]=2
    assert clip_bucket(ct,rdt,policy='cold_cloud_centre')==('A',True)
    rdt[-1,4,4]=255
    assert clip_bucket(ct,rdt,policy='cold_cloud_centre')==('A',True)
    rows=[dict(bucket=b,contains_ot=ot) for b,ot in [('A',False),('A',True),('B',False),('B',True),('C',False)]]
    np.testing.assert_allclose(sample_weights(rows),[.125,.375,.15,.15,.2])


def test_crop_domain_checks_entire_area_and_sampler_probabilities():
    lat = np.full((5,6),40.)
    lon = np.broadcast_to(np.arange(-22,-16),(5,6))
    origins = valid_origins(lat,lon,size=3)
    assert origins[:,1].min() == 2
    assert set(origins[:,1]) == {2,3}
    rows = [dict(bucket=b,contains_ot=ot) for b,ot in [('A',False),('A',True),('B',False),('C',False)]]
    np.testing.assert_allclose(sample_weights(rows), [.125,.375,.3,.2])
    with pytest.raises(ValueError,match='Missing'):
        sample_weights(rows[:2])


def tiny_model(heads=('cma','ct','rdt'),mode='frozen'):
    cfg = videomae_small_7ch_config(image_size=32, num_frames=4, patch_size=16,
        hidden_size=24, num_attention_heads=3, intermediate_size=48, num_hidden_layers=12)
    return ThreeHeadProbe(VideoMAEModel(cfg), heads, width=8, encoder_mode=mode)


def test_conditioned_heads_shapes_gradients_and_frozen_weights():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = tiny_model()
    before = {k:v.clone() for k,v in model.encoder.state_dict().items()}
    model.train()
    assert not model.encoder.training
    outputs = model(torch.randn(1,4,7,32,32))
    assert outputs['cma'].shape == (1,1,4,32,32)
    assert outputs['ct'].shape == outputs['rdt'].shape == (1,5,4,32,32)
    # RDT-only gradient reaches both conditioning heads.
    outputs['rdt'].square().mean().backward()
    assert model.heads['cma'].weight.grad.abs().sum() > 0
    assert model.heads['ct'].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    assert all(torch.equal(before[k],v) for k,v in model.encoder.state_dict().items())


def test_selective_transfer_only_unfreezes_last_four_blocks():
    model = tiny_model(('rdt',),'last4')
    for i,block in enumerate(model.encoder.encoder.layer):
        assert all(p.requires_grad == (i>=8) for p in block.parameters())
    assert not any(p.requires_grad for p in model.encoder.embeddings.parameters())
    model.set_encoder_mode('frozen')
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_project_before_upsampling_preserves_full_resolution_conditioning():
    model = tiny_model().double()
    features = [torch.randn(1,24,2,2,2,dtype=torch.float64) for _ in range(4)]
    actual = model.decode(features)
    shared = model.fpn(features,(4,32,32))
    reference = {}
    for head in ('cma','ct','rdt'):
        inputs = [shared]
        if head in ('ct','rdt'):
            inputs.append(reference['cma'].sigmoid())
        if head == 'rdt':
            inputs.append(reference['ct'].softmax(1))
        reference[head] = model.heads[head](torch.cat(inputs,1))
        torch.testing.assert_close(actual[head],reference[head],rtol=1e-10,atol=1e-10)


def test_ignored_pixels_do_not_change_loss_or_gradients():
    outputs = {head:torch.randn(1,n,2,2,2,requires_grad=True) for head,n in [('cma',1),('ct',5),('rdt',5)]}
    labels = {head:torch.zeros((1,2,2,2),dtype=torch.uint8) for head in outputs}
    for values in labels.values():
        values[:,:,:,1] = 255
    loss,_ = probe_loss(outputs,labels,rdt_class_weights=class_weights([8,0,0,0,0]))
    loss.backward()
    assert torch.isfinite(loss)
    for logits in outputs.values():
        assert torch.count_nonzero(logits.grad[:,:,:,:,1]) == 0
    for values in labels.values():
        values.fill_(255)
    loss,_ = probe_loss(outputs,labels)
    assert loss == 0 and torch.isfinite(loss)


def test_metrics_pool_pixels_and_brier_and_active_probability():
    metric = SegmentationMetrics(5)
    logits = torch.full((1,5,1,1,3),-20.)
    logits[:,2,0,0,0]=20; logits[:,0,0,0,1]=20; logits[:,4,0,0,2]=20
    target = torch.tensor([[[[2,3,255]]]],dtype=torch.uint8)
    metric.update(logits,target,convective=True)
    result=metric.result(convective=True)
    assert result['pixels']==2
    assert result['support']==[0,0,1,1,0]
    assert result['iou'][4] is None
    assert result['active_convection']['csi'][1] == .5
    assert result['brier_multiclass'] == pytest.approx(1.)
    assert result['active_convection']['brier'] == pytest.approx(.5)


def test_diagnostic_interventions_are_explicit_and_leave_original_unchanged():
    x=torch.arange(1*16*7*2*2).reshape(1,16,7,2,2).float()
    before=x.clone()
    repeated=diagnostic_input(x,'repeat_last')
    assert torch.equal(repeated[:,0],x[:,-1])
    ablated=diagnostic_input(x,'without_ir134')
    assert torch.count_nonzero(ablated[:,:,6])==0
    assert torch.equal(ablated[:,:,:6],x[:,:,:6])
    assert torch.equal(before,x)


def test_training_checkpoint_resume_preserves_optimizer_scheduler_and_rng(tmp_path,monkeypatch):
    import json
    from emma_gpm import three_head_training as training
    def encoder(*args,**kwargs):
        cfg=videomae_small_7ch_config(image_size=32,num_frames=4,patch_size=16,
            hidden_size=24,num_attention_heads=3,intermediate_size=48,num_hidden_layers=12)
        return VideoMAEModel(cfg), {'synthetic_test':True}
    monkeypatch.setattr(training,'make_encoder',encoder)
    data=tmp_path/'dataset';data.mkdir()
    rows=[]
    for i,(split,bucket) in enumerate([('train','A'),('train','B'),('train','C'),('validation','A'),('test','C')]):
        np.save(data/f'{i}.npy', np.random.default_rng(i).uniform(240,280,(4,7,32,32)).astype('float16'))
        ct=np.full((4,32,32),3 if bucket!='C' else 0,dtype=np.uint8)
        rdt=np.full_like(ct,2 if bucket=='A' else 1 if bucket=='B' else 0)
        np.savez_compressed(data/f'{i}.npz',ct=ct,rdt=rdt,cma=(ct>0).astype(np.uint8))
        rows.append(dict(id=str(i),split=split,bucket=bucket,contains_ot=False,clip=f'{i}.npy',targets=f'{i}.npz'))
    accepted=quality_gate(dict(convective=4096,inside_cma=4096,inside_high_opaque=4096,inside_clear_low=0))
    manifest=dict(samples=rows,statistics=dict(global_mean_k=260.,global_std_k=12.),
                  support={'train':{'rdt':[4096,4096,4096,0,0]}},
                  teacher_quality_gates={s:accepted for s in ('train','validation','test')})
    (data/'manifest.json').write_text(json.dumps(manifest))
    config=dict(name='same',source='synthetic_unit_test',heads=['cma','ct','rdt'],encoder_mode='full',
                epochs=2,batch_size=2,workers=0,device='cpu',precision='fp32',seed=42)
    training.run_probe(config,data,tmp_path/'uninterrupted')
    original=training.atomic_torch
    def interrupt_after_committed_epoch(path,payload):
        original(path,payload)
        if path.name=='checkpoint_latest.pt' and payload['epoch']==1:
            assert path.with_name('checkpoint_best.pt').exists()
            raise InterruptedError('Simulated process interruption after committed epoch')
    monkeypatch.setattr(training,'atomic_torch',interrupt_after_committed_epoch)
    with pytest.raises(InterruptedError):training.run_probe(config,data,tmp_path/'resumed')
    monkeypatch.setattr(training,'atomic_torch',original)
    training.run_probe(config,data,tmp_path/'resumed')
    a=torch.load(tmp_path/'uninterrupted/same/checkpoint_latest.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/same/checkpoint_latest.pt',weights_only=False)
    assert a['updates']==b['updates']==4 and a['scheduler']==b['scheduler']
    for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
    for name,params in a['optimizer']['state'].items():
        for key,value in params.items():torch.testing.assert_close(value,b['optimizer']['state'][name][key],rtol=0,atol=0)
