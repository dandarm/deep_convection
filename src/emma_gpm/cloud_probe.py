"""Frozen VideoMAE cloud segmentation, with matched random-encoder controls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import VideoMAEModel

from .cloud_probe_data import binary_target
from .forecasting import ForecastWindow
from .rdt.storage import atomic_json, sha256
from .seviri_clips import normalize_seviri_clip
from .ssl_model import encoder_source_config, load_encoder_weights


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def unpatchify_logits(patches, grid=14, patch=16):
    if patches.ndim != 3 or tuple(patches.shape[1:]) != (grid*grid, patch*patch):
        raise ValueError('Expected one spatial plane of patch pixel logits')
    return patches.reshape(-1,grid,grid,patch,patch).permute(0,1,3,2,4).reshape(-1,grid*patch,grid*patch)


class CloudProbeHead(nn.Module):
    """Two shared Linear layers decode each last-time token into a 16x16 mask."""
    def __init__(self, dimension, width=128, grid=14, patch=16):
        super().__init__()
        self.grid, self.patch = grid, patch
        self.network = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension,width),
                                     nn.GELU(),nn.Linear(width,patch*patch))

    def forward(self, features):
        return unpatchify_logits(self.network(features),self.grid,self.patch)


def masked_bce(logits, labels):
    valid = (labels == 0) | (labels == 1)
    if not valid.any():
        raise ValueError('Batch has no valid target pixels')
    if not ((labels == 255) | valid).all():
        raise ValueError('Unknown binary target class')
    return F.binary_cross_entropy_with_logits(logits[valid],labels[valid].float())


def confusion_counts(logits, labels):
    """[TN, FP, FN, TP], ignoring fill; zero logit corresponds to probability .5."""
    valid = (labels == 0) | (labels == 1)
    if not ((labels == 255) | valid).all():
        raise ValueError('Unknown binary target class')
    pred = logits >= 0
    return torch.bincount((2*labels[valid].long()+pred[valid].long()),minlength=4).cpu().numpy()


def metrics_from_counts(counts):
    tn, fp, fn, tp = [int(x) for x in counts]
    def ratio(a,b):
        return a/b if b else None
    cloud_iou, clear_iou = ratio(tp,tp+fp+fn), ratio(tn,tn+fp+fn)
    recall, specificity = ratio(tp,tp+fn), ratio(tn,tn+fp)
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,valid_pixels=tn+fp+fn+tp,
        cloud_iou=cloud_iou,clear_iou=clear_iou,
        mean_iou=(cloud_iou+clear_iou)/2 if cloud_iou is not None and clear_iou is not None else None,
        cloud_dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=recall,
        specificity=specificity,balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
        accuracy=ratio(tp+tn,tp+tn+fp+fn),target_cloud_fraction=ratio(tp+fn,tp+tn+fp+fn),
        predicted_cloud_fraction=ratio(tp+fp,tp+tn+fp+fn))


def encode_last_time(encoder, history, *, precision='bf16'):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Probe encoder must be frozen and in evaluation mode')
    cfg = encoder.config
    if tuple(history.shape[1:]) != (cfg.num_frames,7,cfg.image_size,cfg.image_size):
        raise ValueError('Historical input shape differs from encoder configuration')
    with torch.inference_mode(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=precision=='bf16'):
        tokens = encoder(history).last_hidden_state
        grid = cfg.image_size//cfg.patch_size
        count = (cfg.num_frames//cfg.tubelet_size)*grid*grid
        if tokens.shape[1] != count:
            raise ValueError('Unexpected encoder token layout')
        return tokens.reshape(len(history),cfg.num_frames//cfg.tubelet_size,grid*grid,cfg.hidden_size)[:,-1].float().clone()


def atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.part')
    with temporary.open('wb') as stream:
        torch.save(payload,stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def audit_pretraining_days(run, manifest):
    probe_days = {r['day'] for r in manifest['samples']}
    all_days = set()
    for key in ['manifest','validation_manifest']:
        path = Path(run['arguments'][key])
        rows = pd.read_csv(path,usecols=['start_time'])
        for stamp in pd.to_datetime(rows.start_time,utc=True).unique():
            start = pd.Timestamp(stamp)
            all_days.update(d.strftime('%Y-%m-%d') for d in pd.date_range(start.normalize(),(start+pd.Timedelta(minutes=75)).normalize()))
    if probe_days & all_days:
        raise ValueError('Probe data overlap encoder pretraining or validation data')
    return sorted(all_days)


def run_cloud_probe(run_dir, dataset, output, *, kind='pretrained', epochs=30, batch_size=16,
                    learning_rate=1e-3, weight_decay=1e-4, seed=20261031, precision='bf16', heartbeat=None):
    """CPU-only resumable extraction and training. Test is evaluated after selection."""
    from .file_lock import exclusive_file_lock
    run_dir, dataset, output = Path(run_dir),Path(dataset),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if kind not in ('pretrained','random') or precision not in ('bf16','fp32'):
        raise ValueError('Unknown encoder kind or CPU precision')
    if min(epochs,batch_size) < 1:
        raise ValueError('Positive epoch and batch counts are required')
    with exclusive_file_lock(output/'run.lock'):
        manifest = json.loads((dataset/'manifest.json').read_text())
        if sha256(dataset/'protocol.json') != manifest['protocol_sha256'] or sha256(dataset/'ancillary.json') != manifest['ancillary_sha256']:
            raise ValueError('Dataset protocol or ancillary metadata changed')
        run = json.loads((run_dir/'run_config.json').read_text())
        if json.loads((run_dir/'progress.json').read_text()).get('status') != 'completed':
            raise ValueError('Encoder pretraining must be completed')
        if run['arguments'].get('freeze_encoder') or run['arguments'].get('synthetic_smoke'):
            raise ValueError('Expected a real-data pretrained encoder')
        config = encoder_source_config(run_dir/'encoder',ForecastWindow())
        stats = run['identity']['stats']
        excluded_days = audit_pretraining_days(run,manifest)
        mean,std = stats['global_mean_k'],stats['global_std_k']
        normalize_seviri_clip(np.zeros(1),mean,std)  # validates finite scalar statistics
        encoder_seed = int(run['identity']['seed'])
        torch.manual_seed(encoder_seed)
        encoder = VideoMAEModel(config)
        if kind == 'pretrained':
            load_encoder_weights(encoder,run_dir/'encoder')
        encoder.eval().requires_grad_(False)
        frozen_hash = state_digest(encoder)
        sources = {name:sha256(Path(__file__).parent/name) for name in ['cloud_probe.py','cloud_probe_data.py','ssl_model.py','seviri_clips.py']}
        identity = dict(version=1,kind=kind,run_dir=str(run_dir.resolve()),
            encoder_weights_sha256=sha256(run_dir/'encoder/model.safetensors'),encoder_state_sha256=frozen_hash,
            encoder_seed=encoder_seed,model_size=run['arguments'].get('model_size','small'),
            pretraining_objective=run['identity']['objective'],pretraining_clips=stats['samples'],
            pretraining_epochs=run['arguments']['epochs'],pretraining_days=excluded_days,
            manifest_sha256=sha256(dataset/'manifest.json'),dataset=str(dataset.resolve()),
            global_mean_k=mean,global_std_k=std,history_frames=10,target_frame=9,
            feature='final encoder layer, last temporal tubelet, all 14x14 spatial tokens',
            feature_precision=precision,head='LayerNorm -> Linear(D,128) -> GELU -> Linear(128,256)',
            seed=seed,epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,weight_decay=weight_decay,
            selection='maximum validation mean IoU; first epoch in ties',probability_threshold=0.5,
            sources=sources,torch_version=torch.__version__)
        if (output/'identity.json').exists():
            if json.loads((output/'identity.json').read_text()) != identity:
                raise ValueError('Probe identity changed; use a new output directory')
        else:
            atomic_json(output/'identity.json',identity)
        if (output/'result.json').exists():
            return json.loads((output/'result.json').read_text())
        started = time.monotonic()
        def progress(stage,**fields):
            payload = dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),
                elapsed_seconds=time.monotonic()-started,**fields)
            atomic_json(output/'progress.json',payload)
            if heartbeat is not None:
                heartbeat(payload)
        rows = manifest['samples']
        feature_path = output/'features.npy'
        shape = (len(rows),196,config.hidden_size)
        if feature_path.exists():
            if sha256(feature_path) != json.loads((output/'features.json').read_text())['sha256']:
                raise ValueError('Feature cache changed')
        else:
            partial = output/'features.npy.part'
            progress_file = output/'feature_resume.json'
            count = 0
            if partial.exists() and progress_file.exists():
                count = json.loads(progress_file.read_text())['completed']
                cached = np.lib.format.open_memmap(partial,mode='r+')
                if cached.shape != shape or cached.dtype != np.float32:
                    raise ValueError('Feature resume shape or dtype changed')
            else:
                cached = np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=shape)
            for index in range(count,len(rows)):
                row = rows[index]
                if sha256(dataset/row['clip']) != row['clip_sha256']:
                    raise ValueError('Input clip hash changed')
                clip = np.load(dataset/row['clip'],allow_pickle=False)
                if clip.shape != (16,7,224,224) or not np.isfinite(clip).all():
                    raise ValueError('Invalid canonical clip')
                history = torch.from_numpy(normalize_seviri_clip(clip[:10],mean,std)[None])
                features = encode_last_time(encoder,history,precision=precision).numpy()[0]
                if not np.isfinite(features).all():
                    raise ValueError('Nonfinite encoder features')
                cached[index] = features
                if (index+1)%16 == 0 or index+1 == len(rows):
                    cached.flush()
                    with partial.open('rb') as stream:
                        os.fsync(stream.fileno())
                    atomic_json(progress_file,dict(completed=index+1))
                progress('feature_extraction',completed=index+1,total=len(rows))
            cached.flush(); del cached
            if state_digest(encoder) != frozen_hash:
                raise ValueError('Frozen encoder changed during feature extraction')
            # Metadata precedes publication so an interrupted rename is safely retried.
            atomic_json(output/'features.json',dict(sha256=sha256(partial),shape=shape,encoder_unchanged=True))
            partial.replace(feature_path)
        if state_digest(encoder) != frozen_hash:
            raise ValueError('Frozen encoder state changed')
        del encoder
        features = torch.from_numpy(np.load(feature_path,allow_pickle=False))
        labels = []
        for row in rows:
            if sha256(dataset/row['label']) != row['label_sha256']:
                raise ValueError('Target mask hash changed')
            labels.append(binary_target(np.load(dataset/row['label'],allow_pickle=False)))
        labels = torch.from_numpy(np.stack(labels))
        indices = {s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
        if any(not len(idx) for idx in indices.values()):
            raise ValueError('Every probe split must be nonempty')
        torch.manual_seed(seed)
        head = CloudProbeHead(config.hidden_size)
        optimizer = torch.optim.AdamW(head.parameters(),lr=learning_rate,weight_decay=weight_decay)
        start_epoch, best_score, best_epoch, best_state, history_rows = 0,-1.,None,None,[]
        checkpoint = output/'head_checkpoint.pt'
        if checkpoint.exists():
            state = torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Head checkpoint identity differs')
            head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
            start_epoch,best_score,best_epoch,best_state,history_rows = (state[k] for k in ['epoch','best_score','best_epoch','best_state','history'])
        def evaluate(idx, return_examples=False):
            counts = np.zeros(4,dtype=np.int64)
            individual, bce_sum, pixels = [],0.,0
            head.eval()
            with torch.inference_mode():
                for first in range(0,len(idx),batch_size):
                    selected = idx[first:first+batch_size]
                    logits = head(features[selected]); target = labels[selected]
                    valid = ((target == 0) | (target == 1)).sum().item()
                    if valid:
                        bce_sum += masked_bce(logits,target).item()*valid; pixels += valid
                    for j,index in enumerate(selected.tolist()):
                        current = confusion_counts(logits[j],target[j]); counts += current
                        individual.append(dict(id=rows[index]['id'],**metrics_from_counts(current)))
            metrics = metrics_from_counts(counts); metrics['bce'] = bce_sum/pixels if pixels else None
            return metrics,individual
        for epoch in range(start_epoch,epochs):
            tick = time.monotonic()
            generator = torch.Generator().manual_seed(seed+epoch)
            order = indices['train'][torch.randperm(len(indices['train']),generator=generator)]
            head.train(); train_sum,train_pixels = 0.,0
            for first in range(0,len(order),batch_size):
                selected = order[first:first+batch_size]
                optimizer.zero_grad(set_to_none=True)
                target = labels[selected]
                loss = masked_bce(head(features[selected]),target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite head loss')
                loss.backward(); optimizer.step()
                valid = ((target == 0) | (target == 1)).sum().item()
                train_sum += loss.item()*valid; train_pixels += valid
                progress('head_training',epoch=epoch+1,epochs=epochs,batch=first//batch_size+1,
                         batches=(len(order)+batch_size-1)//batch_size)
            validation,_ = evaluate(indices['validation'])
            score = validation['mean_iou']
            if score is None:
                raise ValueError('Validation must contain both clear and cloudy pixels')
            if score > best_score:
                best_score,best_epoch = score,epoch+1
                best_state = {k:v.detach().clone() for k,v in head.state_dict().items()}
            history_rows.append(dict(epoch=epoch+1,train_bce=train_sum/train_pixels,validation=validation,seconds=time.monotonic()-tick))
            atomic_torch(checkpoint,dict(identity=identity,head=head.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch+1,best_score=best_score,best_epoch=best_epoch,best_state=best_state,history=history_rows))
            atomic_json(output/'history.json',history_rows)
        head.load_state_dict(best_state)
        validation,_ = evaluate(indices['validation'])
        test,per_clip = evaluate(indices['test'])
        atomic_torch(output/'head_best.pt',dict(identity=identity,head=best_state,epoch=best_epoch))
        example_ids = indices['test'][torch.linspace(0,len(indices['test'])-1,6).long()]
        head.eval()
        with torch.inference_mode():
            probabilities = head(features[example_ids]).sigmoid().numpy()
        np.savez_compressed(output/'examples.npz',probability=probabilities,
            target=labels[example_ids].numpy(),ids=np.asarray([rows[i]['id'] for i in example_ids.tolist()]))
        result = dict(identity=identity,best_epoch=best_epoch,validation=validation,test=test,
            split_clips={s:len(i) for s,i in indices.items()},head_parameters=sum(p.numel() for p in head.parameters()),
            encoder_frozen_verified=True,elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'test_per_clip.json',per_clip)
        atomic_json(output/'result.json',result)
        atomic_json(output/'progress.json',dict(status='completed',stage='test_evaluated',pid=os.getpid(),
            updated_unix=time.time(),best_epoch=best_epoch,test=test))
        return result
