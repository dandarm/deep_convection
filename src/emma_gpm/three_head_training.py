"""Resumable GPU probing, frozen feature caches and sequential transfer runs."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .cloud_probe import atomic_torch, state_digest
from .file_lock import exclusive_file_lock
from .rdt.storage import atomic_json
from .ssl_distributed import capture_rng, restore_rng
from .ssl_masking import make_ssl_mask
from .three_head_data import CLASS_NAMES, sample_weights, validate_manifest_acceptance
from .three_head_model import (ThreeHeadProbe, make_encoder, probe_loss, class_weights,
                               SegmentationMetrics, diagnostic_input)
from .videomae_v2 import VideoMAEV2ForPreTraining, running_cell_mask


class ProbeDataset(Dataset):
    def __init__(self, directory, manifest, split, *, features=None):
        self.directory = Path(directory)
        self.manifest = manifest
        self.indices = [i for i, row in enumerate(manifest['samples']) if row['split'] == split]
        self.samples = [manifest['samples'][i] for i in self.indices]
        self.features = features
        self.mean = manifest['statistics']['global_mean_k']
        self.std = manifest['statistics']['global_std_k']
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError('Invalid persisted global normalization')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        if self.features is None:
            values = np.load(self.directory / row['clip']).astype(np.float32)
            values = (values-self.mean) / self.std
        else:
            values = np.array(self.features[self.indices[index]], copy=True)
        with np.load(self.directory / row['targets']) as archive:
            labels = {head: torch.from_numpy(archive[head].copy()) for head in CLASS_NAMES}
        return torch.from_numpy(values), labels, self.indices[index]


def feature_cache(model, dataset, manifest, output, device, precision, batch_size=4, heartbeat=None):
    """One encoder pass per immutable cache; identity checks read metadata only."""
    if model.encoder_mode != 'frozen':
        raise ValueError('Feature caching requires a frozen encoder')
    digest = state_digest(model.encoder)
    identity = dict(encoder_sha256=digest, statistics=manifest['statistics'],
        manifest_sha256=hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),
        frames=16, levels=[3,6,9,12], dtype='float16', precision=precision)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(output) / 'feature_cache' / key
    folder.mkdir(parents=True, exist_ok=True)
    shape = (len(manifest['samples']), 4, model.encoder.config.hidden_size, 8, 14, 14)
    with exclusive_file_lock(folder/'build.lock'):
        if (folder/'complete.json').exists():
            previous = json.loads((folder/'complete.json').read_text())
            if previous['identity'] != identity or previous['shape'] != list(shape):
                raise ValueError('Frozen feature cache identity differs')
            return np.load(folder/'features.npy', mmap_mode='r')
        values = np.lib.format.open_memmap(folder/'features.part.npy', mode='w+', dtype='float16', shape=shape)
        model.eval()
        written = 0
        for split in ('train', 'validation', 'test'):
            loader = DataLoader(ProbeDataset(dataset, manifest, split), batch_size=batch_size,
                                num_workers=2, pin_memory=device.type=='cuda')
            for clips, _, indices in loader:
                clips = clips.to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision=='bf16'):
                    features = torch.stack(model.encode(clips), 1).to(torch.float16).cpu().numpy()
                values[indices.numpy()] = features
                written += len(indices)
                atomic_json(folder/'progress.json', dict(status='extracting', clips=written, total=shape[0]))
                if heartbeat is not None:
                    heartbeat(written, shape[0])
        values.flush(); del values
        if state_digest(model.encoder) != digest:
            raise RuntimeError('Frozen encoder weights changed during extraction')
        (folder/'features.part.npy').replace(folder/'features.npy')
        atomic_json(folder/'complete.json', dict(identity=identity, shape=list(shape), encoder_unchanged=True))
    return np.load(folder/'features.npy', mmap_mode='r')


def _loader(dataset, batch_size, *, training=False, seed=0, workers=2):
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights(dataset.samples), len(dataset), replacement=True,
                                    generator=generator) if training else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed+1))


def _forward(model, inputs, cached):
    return model(features=list(inputs.unbind(1))) if cached else model(inputs)


@torch.no_grad()
def evaluate(model, loader, device, weights, *, precision='bf16', cached=False,
             intervention='all_channels', collect_examples=False, nesting_weight=.1):
    model.eval()
    metrics = {head: SegmentationMetrics(len(CLASS_NAMES[head])) for head in model.head_names}
    totals, samples = {}, 0
    examples = []
    example_indices = set()
    if collect_examples:
        candidates = list(zip(loader.dataset.indices, loader.dataset.samples))
        for criterion in (lambda row: row['contains_ot'], lambda row: row['bucket']=='A',
                          lambda row: row['bucket']=='B', lambda row: row['bucket']=='C'):
            chosen = [index for index,row in candidates if criterion(row) and index not in example_indices][:2]
            example_indices.update(chosen)
        example_indices.update([index for index,_ in candidates if index not in example_indices][:8-len(example_indices)])
    for inputs, labels, indices in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = {head: value.to(device, non_blocking=True) for head, value in labels.items()}
        if intervention != 'all_channels':
            if cached:
                raise ValueError('Temporal/spectral interventions require recomputed encoder features')
            # Per-clip seeds keep interventions invariant to batch size.
            inputs = torch.cat([diagnostic_input(inputs[i:i+1], intervention, seed=int(index))
                                for i, index in enumerate(indices)], 0)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision=='bf16'):
            outputs = _forward(model, inputs, cached)
            loss, terms = probe_loss(outputs, labels, rdt_class_weights=weights, nesting_weight=nesting_weight)
        for key, value in {'total': loss, **terms}.items():
            totals[key] = totals.get(key, 0.) + float(value)*len(inputs)
        samples += len(inputs)
        for head in outputs:
            metrics[head].update(outputs[head], labels[head], convective=head=='rdt')
        if collect_examples and len(examples) < 8:
            for i, index in enumerate(indices):
                if int(index) not in example_indices:
                    continue
                example = dict(index=int(index), targets={key: val[i].cpu().numpy() for key, val in labels.items()},
                    predictions={key: (val[i,0] >= 0).to(torch.uint8).cpu().numpy() if key == 'cma'
                                 else val[i].argmax(0).to(torch.uint8).cpu().numpy() for key, val in outputs.items()})
                examples.append(example)
    return dict(samples=samples, losses={key: value/samples for key, value in totals.items()},
        metrics={head: value.result(convective=head=='rdt') for head, value in metrics.items()}), examples


def _auxiliary_loss(model, inputs):
    cfg = model.encoder.config
    mask = make_ssl_mask(cfg, len(inputs), .9, device=inputs.device)
    decode = running_cell_mask(cfg, len(inputs), .5, device=inputs.device)
    return model.auxiliary_mae(inputs, mask, decode_mask=decode).loss


def run_probe(config, dataset, output_root):
    """Best validation selection; test and interventions use that checkpoint only."""
    output_root, dataset = Path(output_root), Path(dataset)
    output = output_root / config['name']
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(output/'run.lock'):
        manifest = json.loads((dataset/'manifest.json').read_text())
        validate_manifest_acceptance(manifest)
        if (output/'result.json').exists():
            if json.loads((output/'config.json').read_text()) != config:
                raise ValueError('Completed run uses different settings')
            return json.loads((output/'result.json').read_text())
        seed = config.get('seed', 20261109)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(config.get('device', 'cuda'))
        precision = config.get('precision', 'bf16')
        if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('This protocol requires a working bf16 CUDA device')
        source = config['source']
        if config.get('parent'):
            parent = output_root / config['parent']
            if not (parent/'result.json').exists():
                raise ValueError('Sequential parent has not completed')
            source = str(parent/'encoder_best')
        encoder, provenance = make_encoder(source, config.get('kind', 'pretrained'))
        mode = config.get('encoder_mode', 'frozen')
        # Reset decoder RNG for every run: fresh heads with matched initialization.
        torch.manual_seed(seed+101)
        model = ThreeHeadProbe(encoder, config['heads'], encoder_mode=mode).to(device)
        if config.get('auxiliary_mae', False):
            model.auxiliary_mae = VideoMAEV2ForPreTraining(model.encoder.config).to(device)
            original = torch.load(config['source'], map_location='cpu', weights_only=False)['model']
            auxiliary_weights = {key[len('mae.'):]: value for key,value in original.items()
                                 if key.startswith('mae.')}
            if not auxiliary_weights:
                raise ValueError('Cumulative MAE retention requires the original MVM decoder checkpoint')
            # Strict decoder restoration; the current encoder may already have
            # been adapted by an earlier supervised stage.
            keys = model.auxiliary_mae.state_dict()
            for key in list(auxiliary_weights):
                translated = key.replace('.q_bias', '.query.bias').replace('.v_bias', '.value.bias')
                if key not in keys and translated in keys:
                    auxiliary_weights[translated] = auxiliary_weights.pop(key)
            model.auxiliary_mae.load_state_dict(auxiliary_weights, strict=True)
            model.auxiliary_mae.videomae = model.encoder
        if config.get('retain_cma', False):
            parent_state = torch.load(output_root/config['parent']/'checkpoint_best.pt', map_location='cpu', weights_only=False)
            # Keep the learned shared decoder and CMA head during CT adaptation.
            selected = {key: value for key, value in parent_state['model'].items()
                        if key.startswith(('fpn.', 'heads.cma.'))}
            incompat = model.load_state_dict(selected, strict=False)
            if incompat.unexpected_keys:
                raise ValueError('Cumulative parent decoder is incompatible')
        initial_hash = state_digest(model.encoder)
        anchor = None
        if config.get('anchor_weight', 0.):
            initial, _ = make_encoder(config['source'])
            anchor = {name: value.detach().to(device) for name, value in initial.named_parameters()
                      if dict(model.encoder.named_parameters())[name].requires_grad}
            del initial
        identity = dict(config=config, dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
            encoder=provenance, initial_encoder_sha256=initial_hash, statistics=manifest['statistics'],
            head_parameters=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('encoder.')),
            trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
        identity['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('three_head_training.py','three_head_model.py','three_head_data.py')}
        if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('Resume identity differs; select a new output directory')
        atomic_json(output/'identity.json', identity); atomic_json(output/'config.json', config)
        cached = mode == 'frozen'
        def feature_heartbeat(done, total):
            atomic_json(output/'progress.json', dict(status='extracting_features', clips=done, total_clips=total,
                completed_epochs=0, epochs=config.get('epochs',60), heartbeat_utc=pd_timestamp()))
        features = feature_cache(model, dataset, manifest, output_root, device, precision,
                                 heartbeat=feature_heartbeat) if cached else None
        datasets = {split: ProbeDataset(dataset, manifest, split, features=features) for split in ('train','validation','test')}
        batch_size = config.get('batch_size', 8 if cached else 2)
        epochs = config.get('epochs', 60)
        loaders = {split: _loader(data, batch_size, training=split=='train', seed=seed,
                                  workers=config.get('workers', 2)) for split, data in datasets.items()}
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        groups = [dict(params=head_params, lr=config.get('head_lr', 1e-3))]
        trained_encoder = [p for p in model.encoder.parameters() if p.requires_grad]
        if trained_encoder:
            groups.append(dict(params=trained_encoder, lr=config.get('encoder_lr', 1e-5)))
        optimizer = torch.optim.AdamW(groups, weight_decay=config.get('weight_decay', 1e-4), fused=device.type=='cuda')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: .1+.9*.5*(1.+math.cos(math.pi*min(e,epochs)/epochs)))
        weights = class_weights(manifest['support']['train']['rdt']).to(device)
        history, first, best, best_epoch, updates = [], 1, float('inf'), 0, 0
        if (output/'checkpoint_latest.pt').exists():
            state = torch.load(output/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Checkpoint identity differs from requested protocol')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']); restore_rng(state['rng'], device)
            loaders['train'].sampler.generator.set_state(state['sampler_rng'])
            history, first, best, best_epoch, updates = state['history'], state['epoch']+1, state['best'], state['best_epoch'], state['updates']
        for epoch in range(first, epochs+1):
            started = time.monotonic()
            model.train()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            totals, samples = {}, 0
            for batch, (inputs, labels, _) in enumerate(loaders['train'], 1):
                inputs = inputs.to(device, non_blocking=True)
                labels = {head: value.to(device, non_blocking=True) for head, value in labels.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision=='bf16'):
                    outputs = _forward(model, inputs, cached)
                    loss, terms = probe_loss(outputs, labels, rdt_class_weights=weights,
                                              nesting_weight=config.get('nesting_weight', .1))
                    if model.auxiliary_mae is not None:
                        terms['auxiliary_mae'] = _auxiliary_loss(model, inputs)
                        loss = loss + config.get('auxiliary_weight', .1)*terms['auxiliary_mae']
                    if anchor is not None:
                        differences = [(value-anchor[name]).float().square().sum()
                                       for name,value in model.encoder.named_parameters() if name in anchor]
                        terms['ssl_anchor'] = sum(differences)/sum(anchor[name].numel() for name in anchor)
                        loss = loss + config['anchor_weight']*terms['ssl_anchor']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite probe loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); updates += 1
                samples += len(inputs)
                for key, value in {'total':loss, **terms}.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach())*len(inputs)
                atomic_json(output/'progress.json', dict(status='training', epoch=epoch, completed_epochs=epoch-1,
                    epochs=epochs, batch=batch, batches=len(loaders['train']), updates=updates,
                    heartbeat_utc=pd_timestamp(), seconds_in_epoch=time.monotonic()-started))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_seconds = time.monotonic()-started
            atomic_json(output/'progress.json', dict(status='validation', epoch=epoch, completed_epochs=epoch-1,
                epochs=epochs, updates=updates, heartbeat_utc=pd_timestamp()))
            val_start = time.monotonic()
            validation, _ = evaluate(model, loaders['validation'], device, weights, precision=precision, cached=cached,
                                      nesting_weight=config.get('nesting_weight',.1))
            score = validation['losses']['total']
            improved = score < best
            if improved:
                best, best_epoch = score, epoch
            history.append(dict(epoch=epoch, updates=updates, train={k:v/samples for k,v in totals.items()},
                validation=validation, train_seconds=train_seconds, validation_seconds=time.monotonic()-val_start,
                clips_per_second=samples/train_seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None))
            scheduler.step()
            state = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=capture_rng(device),
                sampler_rng=loaders['train'].sampler.generator.get_state(), epoch=epoch,
                history=history, best=best, best_epoch=best_epoch, updates=updates)
            if improved:
                atomic_torch(output/'checkpoint_best.pt', state)
            atomic_torch(output/'checkpoint_latest.pt', state)
            atomic_json(output/'history.json', history)
            atomic_json(output/'progress.json', dict(status='epoch_complete', completed_epochs=epoch,
                epochs=epochs, updates=updates, best_epoch=best_epoch, heartbeat_utc=pd_timestamp()))
        model.load_state_dict(torch.load(output/'checkpoint_best.pt', map_location=device, weights_only=False)['model'])
        final_hash = state_digest(model.encoder)
        if cached and final_hash != initial_hash:
            raise RuntimeError('Scientific frozen probe modified its encoder')
        model.encoder.save_pretrained(output/'encoder_best')
        test, examples = evaluate(model, loaders['test'], device, weights, precision=precision, cached=cached,
                                  collect_examples=True, nesting_weight=config.get('nesting_weight',.1))
        for example in examples:
            payload = {f'target_{key}':value for key,value in example['targets'].items()}
            payload.update({f'prediction_{key}':value for key,value in example['predictions'].items()})
            np.savez_compressed(output/f"example_{example['index']:05d}.npz", **payload)
        interventions = {}
        if config.get('ablations', False):
            raw_loader = _loader(ProbeDataset(dataset, manifest, 'test'), min(batch_size,4))
            for intervention in ('shuffle_time','repeat_last','teacher_two_channels','wv_and_ir108','without_ir134'):
                atomic_json(output/'progress.json', dict(status='diagnostic_evaluation', intervention=intervention,
                    completed_epochs=epochs, heartbeat_utc=pd_timestamp()))
                interventions[intervention], _ = evaluate(model, raw_loader, device, weights, precision=precision,
                    intervention=intervention, nesting_weight=config.get('nesting_weight',.1))
        result = dict(config=config, best_epoch=best_epoch, completed_epochs=epochs, updates=updates,
            best_validation=history[best_epoch-1]['validation'], test=test, interventions=interventions,
            encoder_unchanged=initial_hash==final_hash, initial_encoder_sha256=initial_hash,
            selected_encoder_sha256=final_hash)
        atomic_json(output/'result.json', result)
        atomic_json(output/'progress.json', dict(status='completed', completed_epochs=epochs, best_epoch=best_epoch,
            epochs=epochs, updates=updates, heartbeat_utc=pd_timestamp()))
        del model, optimizer, loaders, datasets, features
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return result


def pd_timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
