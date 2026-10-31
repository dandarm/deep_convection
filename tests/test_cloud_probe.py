import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import VideoMAEModel

from emma_gpm.cloud_probe_data import binary_target, canonical_axes, check_split_days, check_times
from emma_gpm.cloud_probe import (CloudProbeHead, confusion_counts, encode_last_time,
    masked_bce, metrics_from_counts, state_digest, unpatchify_logits)
from emma_gpm.videomae import videomae_small_7ch_config


def test_binary_mask_fill_and_snow_categories_are_explicit():
    assert binary_target(np.array([0,1,255])).tolist() == [0,1,255]
    with pytest.raises(ValueError,match='binary cma'):
        binary_target(np.array([0,1,2,3]))  # cloud-cover categories require their own mapping
    target = torch.tensor([0,1,255,0,1],dtype=torch.uint8)
    logits = torch.tensor([-3.,2.,100.,2.,-1.],requires_grad=True)
    assert confusion_counts(logits,target).tolist() == [1,1,1,1]
    masked_bce(logits,target).backward()
    assert logits.grad[2] == 0
    m = metrics_from_counts([1,1,1,1])
    assert m['cloud_iou'] == pytest.approx(1/3)
    assert m['cloud_dice'] == pytest.approx(.5)
    assert m['balanced_accuracy'] == pytest.approx(.5)
    with pytest.raises(ValueError,match='no valid'):
        masked_bce(torch.zeros(1),torch.tensor([255]))
    assert metrics_from_counts([0,0,0,0])['cloud_iou'] is None


def test_temporal_partition_and_cadence_reject_leakage():
    splits = {'train':['2020-12-13','2020-12-14'],'validation':['2020-12-16'],'test':['2020-12-26']}
    check_split_days(splits,['2020-09-12'])
    with pytest.raises(ValueError,match='overlap'):
        check_split_days(splits,['2020-12-14'])
    with pytest.raises(ValueError,match='unused calendar'):
        check_split_days({**splits,'validation':['2020-12-15']},[])
    times = pd.date_range('2020-12-13',periods=16,freq='5min')
    assert len(check_times(times)) == 16
    with pytest.raises(ValueError):
        check_times(times[::-1])
    with pytest.raises(ValueError):
        check_times(times.insert(4,times[4]))
    with pytest.raises(ValueError):
        check_times(pd.date_range('2020-12-13',periods=16,freq='10min'))


def test_orientation_is_derived_from_both_axes():
    x,y = canonical_axes([3,2,1],[1,2,3])
    assert x.tolist() == [1,2,3] and y.tolist() == [3,2,1]
    with pytest.raises(ValueError):
        canonical_axes([1,3,2],[1,2,3])


def test_patch_pixels_are_placed_in_spatial_order():
    image = torch.arange(36).reshape(1,6,6)
    patches = image.reshape(1,2,3,2,3).permute(0,1,3,2,4).reshape(1,4,9)
    assert torch.equal(unpatchify_logits(patches,grid=2,patch=3),image)
    with pytest.raises(ValueError):
        unpatchify_logits(torch.zeros(1,8,9),grid=2,patch=3)


def test_head_training_leaves_encoder_and_earlier_temporal_tokens_frozen():
    torch.set_num_threads(1)
    cfg = videomae_small_7ch_config(image_size=32,num_frames=4,hidden_size=24,
        num_hidden_layers=1,num_attention_heads=3,intermediate_size=48)
    torch.manual_seed(1)
    encoder = VideoMAEModel(cfg).eval().requires_grad_(False)
    before = state_digest(encoder)
    pixels = torch.randn(2,4,7,32,32)
    features = encode_last_time(encoder,pixels,precision='fp32')
    with torch.inference_mode():
        expected = encoder(pixels).last_hidden_state[:,-4:].float()
    assert torch.equal(features,expected)
    head = CloudProbeHead(24,width=16,grid=2,patch=16)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.02)
    features = features.clone()  # materialize cached inference features for autograd
    labels = torch.zeros(2,32,32,dtype=torch.uint8);labels[:,:,16:] = 1
    initial = masked_bce(head(features),labels).item()
    for _ in range(10):
        optimizer.zero_grad();loss=masked_bce(head(features),labels);loss.backward();optimizer.step()
    assert masked_bce(head(features),labels).item() < initial
    assert state_digest(encoder) == before
    assert all(p.grad is None for p in encoder.parameters())
    encoder.train()
    with pytest.raises(ValueError,match='frozen'):
        encode_last_time(encoder,pixels,precision='fp32')


def test_status_flags_dead_or_stale_workers(tmp_path,monkeypatch):
    import json,time
    path = Path(__file__).parents[1]/'scripts/status_seviri_cloud_probes.py'
    spec = importlib.util.spec_from_file_location('cloud_status',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p = tmp_path/'progress.json'
    p.write_text(json.dumps({'status':'running','pid':123,'updated_unix':time.time()}))
    monkeypatch.setattr(module,'process_alive',lambda pid:False)
    assert 'ARRESTATO' in module.inspect(tmp_path)['health']
    monkeypatch.setattr(module,'process_alive',lambda pid:True)
    p.write_text(json.dumps({'status':'running','pid':123,'updated_unix':time.time()-1000}))
    assert 'BLOCCO' in module.inspect(tmp_path)['health']


def test_real_feature_extraction_and_atomic_head_resume(tmp_path,monkeypatch):
    """Interrupt after a committed epoch and recover the exact final head."""
    import json
    import emma_gpm.cloud_probe as module
    from emma_gpm.rdt.storage import atomic_json,sha256
    torch.set_num_threads(1)
    dataset,run = tmp_path/'dataset',tmp_path/'run'
    dataset.mkdir();(run/'encoder').mkdir(parents=True)
    (run/'encoder/model.safetensors').write_bytes(b'test fixture; weights supplied below')
    config = videomae_small_7ch_config(num_frames=10,hidden_size=12,num_hidden_layers=1,
        num_attention_heads=3,intermediate_size=24)
    monkeypatch.setattr(module,'encoder_source_config',lambda *args:config)
    monkeypatch.setattr(module,'load_encoder_weights',lambda *args:None)
    monkeypatch.setattr(module,'audit_pretraining_days',lambda *args:['2020-09-12'])
    atomic_json(run/'run_config.json',dict(arguments={'epochs':1,'model_size':'small'},
        identity={'seed':3,'objective':'future','stats':{'global_mean_k':260.,'global_std_k':20.,'samples':100}}))
    atomic_json(run/'progress.json',{'status':'completed'})
    atomic_json(dataset/'protocol.json',{'test':True});atomic_json(dataset/'ancillary.json',{'test':True})
    rows=[]
    rng=np.random.default_rng(2)
    for i,split in enumerate(['train','validation','test']):
        clip=rng.normal(260,5,(16,7,224,224)).astype(np.float16)
        label=np.zeros((224,224),np.uint8);label[:,112:]=1;label[:2]=255
        np.save(dataset/f'clip{i}.npy',clip);np.save(dataset/f'label{i}.npy',label)
        rows.append(dict(id=str(i),split=split,clip=f'clip{i}.npy',label=f'label{i}.npy',
            clip_sha256=sha256(dataset/f'clip{i}.npy'),label_sha256=sha256(dataset/f'label{i}.npy')))
    atomic_json(dataset/'manifest.json',dict(samples=rows,protocol_sha256=sha256(dataset/'protocol.json'),
        ancillary_sha256=sha256(dataset/'ancillary.json')))
    reference=module.run_cloud_probe(run,dataset,tmp_path/'reference',epochs=2,batch_size=1,precision='fp32')
    def interrupt(p):
        if p['stage']=='head_training' and p['epoch']==2:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        module.run_cloud_probe(run,dataset,tmp_path/'resumed',epochs=2,batch_size=1,precision='fp32',heartbeat=interrupt)
    resumed=module.run_cloud_probe(run,dataset,tmp_path/'resumed',epochs=2,batch_size=1,precision='fp32')
    assert reference['test']==resumed['test']
    a=torch.load(tmp_path/'reference/head_best.pt',weights_only=False)['head']
    b=torch.load(tmp_path/'resumed/head_best.pt',weights_only=False)['head']
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert reference['encoder_frozen_verified'] and reference['test']['valid_pixels']==222*224

    # A longer budget continues the final optimizer state and reuses features.
    module.run_cloud_probe(run,dataset,tmp_path/'one_epoch',epochs=1,batch_size=1,precision='fp32')
    original_digest=sha256(tmp_path/'one_epoch/head_checkpoint.pt')
    def forbidden_extraction(*args,**kwargs):
        raise AssertionError('An extension must reuse the existing feature cache')
    monkeypatch.setattr(module,'encode_last_time',forbidden_extraction)
    extended=module.run_cloud_probe(run,dataset,tmp_path/'extended',epochs=2,batch_size=1,
        precision='fp32',extend_from=tmp_path/'one_epoch')
    assert extended['test']==reference['test']
    assert sha256(tmp_path/'one_epoch/head_checkpoint.pt')==original_digest
    assert (tmp_path/'extended/features.npy').samefile(tmp_path/'one_epoch/features.npy')
    ref=torch.load(tmp_path/'reference/head_checkpoint.pt',weights_only=False)
    ext=torch.load(tmp_path/'extended/head_checkpoint.pt',weights_only=False)
    assert all(torch.equal(ref['head'][k],ext['head'][k]) for k in ref['head'])
    for parameter,state in ref['optimizer']['state'].items():
        assert all(torch.equal(value,ext['optimizer']['state'][parameter][key]) for key,value in state.items())
    with pytest.raises(ValueError,match='Only the epoch budget'):
        module.run_cloud_probe(run,dataset,tmp_path/'bad_extension',epochs=2,batch_size=1,
            precision='fp32',learning_rate=.02,extend_from=tmp_path/'one_epoch')


def test_extension_accepts_only_the_audited_source_migration(tmp_path):
    from emma_gpm.cloud_probe import extension_provenance,COMPATIBLE_V1_SOURCE_SHA256
    from emma_gpm.rdt.storage import atomic_json
    previous=dict(epochs=30,seed=7,sources={'cloud_probe.py':COMPATIBLE_V1_SOURCE_SHA256,'ssl_model.py':'same'})
    current=dict(epochs=150,seed=7,sources={'cloud_probe.py':'new','ssl_model.py':'same'})
    atomic_json(tmp_path/'identity.json',previous);atomic_json(tmp_path/'progress.json',{'status':'completed'})
    for name in ['head_checkpoint.pt','features.json','result.json']:
        (tmp_path/name).write_bytes(b'fixture')
    assert extension_provenance(tmp_path,current)['parent_epochs']==30
    with pytest.raises(ValueError,match='epoch budget must exceed'):
        extension_provenance(tmp_path,{**current,'epochs':30})
    with pytest.raises(ValueError,match='sources are incompatible'):
        extension_provenance(tmp_path,{**current,'sources':{**current['sources'],'ssl_model.py':'changed'}})
    previous['sources']['cloud_probe.py']='unknown'
    atomic_json(tmp_path/'identity.json',previous)
    with pytest.raises(ValueError,match='sources are incompatible'):
        extension_provenance(tmp_path,current)
