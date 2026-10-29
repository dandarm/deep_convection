#!/usr/bin/env python3
"""Evaluate temporal controls, frozen probes, embeddings and reproducible examples."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

from emma_gpm.advection import advection_forecast
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.forecasting import ForecastWindow,temporal_ablation,persistence
from emma_gpm.seviri_clips import SeviriVideoMAEDataset,SeviriZarrClipReader
from emma_gpm.ssl_dataset import verify_snapshot
from emma_gpm.ssl_diagnostics import (load_run_model,encoder_digest,pool_embeddings,
    radiometric_descriptors,purely_advected_history,ridge_probe,DESCRIPTOR_NAMES)
from emma_gpm.ssl_evaluation import evaluate_ssl
from emma_gpm.ssl_masking import fixed_ssl_mask
from emma_gpm.ssl_model import load_encoder_weights,SeviriSSL,history_config
from emma_gpm.videomae import restore_masked_videomae_pixels
from scripts.pretrain_seviri_ssl import write_json,digest
from scripts.pretrain_videomae import initialize_loader_worker


def loader(dataset,batch_size,workers):
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
        pin_memory=True,**(dict(multiprocessing_context='spawn',persistent_workers=True,
            prefetch_factor=2,worker_init_fn=initialize_loader_worker) if workers else {}))


def analyze(args):
    queue=args.queue_dir
    output=queue/'analysis'
    output.mkdir(exist_ok=True)
    plan=json.loads((queue/'comparison_plan.json').read_text())
    dataset=queue/'dataset'
    analysis_identity=dict(batch_size=args.batch_size,precision='bf16',
        checkpoints={stage['name']:digest(queue/stage['name']/'checkpoint_latest.pt') for stage in plan['stages']},
        dataset={name:digest(dataset/name) for name in ('train.csv','validation.csv','channel_stats.json','source_snapshot.json')},
        sources={name:digest(ROOT/name) for name in ('scripts/analyze_seviri_ssl_comparison.py',
            'src/emma_gpm/ssl_diagnostics.py','src/emma_gpm/ssl_evaluation.py','src/emma_gpm/forecast_metrics.py')})
    identity_path=output/'analysis_plan.json'
    if identity_path.exists() and read_json_identity(identity_path)!=analysis_identity:
        raise ValueError('Analysis sources, data or checkpoints changed; use a separate analysis directory')
    write_json(identity_path,analysis_identity)
    stats=json.loads((dataset/'channel_stats.json').read_text())
    root=Path(stats['zarr_root'])
    mean,std=stats['global_mean_k'],stats['global_std_k']
    verify_snapshot(root,json.loads((dataset/'source_snapshot.json').read_text()))
    window=ForecastWindow()
    torch.set_num_threads(4)
    train_frame=pd.read_csv(dataset/'train.csv')
    val_frame=pd.read_csv(dataset/'validation.csv')
    training=SeviriVideoMAEDataset(train_frame,SeviriZarrClipReader(root),global_mean=mean,global_std=std)
    validation=SeviriVideoMAEDataset(val_frame,SeviriZarrClipReader(root),global_mean=mean,global_std=std)
    train_loader=loader(training,args.batch_size,args.workers)
    val_loader=loader(validation,args.batch_size,args.workers)
    edges=json.loads((queue/'future/progress.json').read_text())['variability_edges_k']
    selected=val_frame.mean_abs_change_k.sort_values().index.to_numpy()
    examples=selected[np.linspace(0,len(selected)-1,4).round().astype(int)].tolist()
    example_arrays={}
    for index in examples:
        clip=validation[index]['pixel_values'][None].cuda()
        history,target=window.split(clip)
        adv,_,_=advection_forecast(history,window.steps)
        example_arrays[index]=dict(observed=(clip[0]*std+mean).cpu().numpy().astype(np.float16),
            persistence=(persistence(history,2)[0]*std+mean).cpu().numpy().astype(np.float16),
            advection=(adv[0]*std+mean).cpu().numpy().astype(np.float16))
    metrics,stratified_metrics,ablations,probe_rows,embedding_rows,frozen_checks=[],[],[],[],[],[]
    sources={}
    for stage in plan['stages']:
        name=stage['name']
        run=queue/name
        progress=json.loads((run/'progress.json').read_text())
        if progress['status']!='completed' or progress['completed_epochs']!=stage['epochs']:
            raise ValueError(f'Incomplete run: {name}')
        sources[name]=digest(run/'checkpoint_latest.pt')
        write_json(output/'progress.json',dict(status='running',run=name))
        print('Analyzing',name,flush=True)
        model,config=load_run_model(run,'cuda')
        report=json.loads((run/'validation_latest.json').read_text())
        for method in ('persistence','advection','neural'):
            if method not in report or (method!='neural' and name!='future'):
                continue
            label=name if method=='neural' else method
            for row in report[method]['per_channel_horizon']:
                metrics.append(dict(run=label,kind=stage['kind'] if method=='neural' else 'baseline',**row))
            for level,by_method in report.get('by_historical_variability',{}).items():
                values=by_method.get(method)
                if values:
                    stratified_metrics.extend(dict(run=label,stratum=level,samples=values['samples'],
                        kind=stage['kind'] if method=='neural' else 'baseline',**row)
                        for row in values['per_channel_horizon'])
        if model.head is not None:
            for ablation in ('chronological','shuffle','last_frame'):
                write_json(output/'progress.json',dict(status='running',run=name,phase=f'evaluation_{ablation}'))
                path=output/f'{name}_{ablation}.json'
                if ablation=='chronological':
                    result=report
                elif path.exists():
                    cached=json.loads(path.read_text())
                    if cached.get('checkpoint_sha256')!=sources[name]:
                        raise ValueError('Ablation cache checkpoint differs')
                    result=cached['evaluation']
                else:
                    result=evaluate_ssl(model,val_loader,window,torch.device('cuda'),mean,std,
                        precision='bf16',ablation=ablation,variability_edges=edges)
                write_json(path,dict(checkpoint_sha256=sources[name],evaluation=result))
                ablations.append(dict(run=name,kind=stage['kind'],ablation=ablation,
                    rmse_k=result['neural']['rmse_k'],mse=result['neural']['mse']))
        if stage['kind']=='frozen_common_head':
            current_hash=encoder_digest(model.encoder)
            torch.manual_seed(config['identity']['seed'])
            reference=SeviriSSL(history_config(window),objective='future',freeze_encoder=True)
            if name!='probe_random':
                load_encoder_weights(reference.encoder,queue/name.removeprefix('probe_')/'encoder')
            expected_hash=encoder_digest(reference.encoder)
            if current_hash!=expected_hash:
                raise ValueError(f'Frozen encoder changed: {name}')
            frozen_checks.append(dict(run=name,encoder_sha256=current_hash,unchanged=True))
            del reference
            features,descriptors={},{}
            for split,data_loader in (('train',train_loader),('validation',val_loader)):
                write_json(output/'progress.json',dict(status='running',run=name,phase=f'embeddings_{split}',
                    samples=0,total_samples=len(data_loader.dataset)))
                cache=output/f'{name}_{split}_embeddings.npz'
                cache_meta=output/f'{name}_{split}_embeddings.json'
                identity=dict(checkpoint_sha256=sources[name],manifest_sha256=digest(dataset/('train.csv' if split=='train' else 'validation.csv')),
                    source_sha256=digest(ROOT/'src/emma_gpm/ssl_diagnostics.py'))
                controls_path=output/f'{name}_embedding_controls.csv'
                if (cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==identity
                        and (split=='train' or controls_path.exists())):
                    with np.load(cache) as saved:
                        features[split],descriptors[split]=saved['features'],saved['descriptors']
                    continue
                vectors,targets=[],[]
                count=0
                with torch.inference_mode():
                    for batch in data_loader:
                        history,target=window.split(batch['pixel_values'].cuda(non_blocking=True))
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            embedding=pool_embeddings(model.encode(history),model.config)
                        vectors.append(embedding.cpu().numpy())
                        targets.append(radiometric_descriptors(history*std+mean,target*std+mean).cpu().numpy())
                        if split=='validation':
                            for intervention in ('shuffle','last_frame','advection_only'):
                                if intervention=='advection_only':
                                    changed,source_support=purely_advected_history(history,return_support=True)
                                else:
                                    changed=temporal_ablation(history,intervention,seed=314159,indices=range(count,count+len(history)))
                                    source_support=torch.ones(len(history),device=history.device)
                                with torch.autocast('cuda',dtype=torch.bfloat16):
                                    altered=pool_embeddings(model.encode(changed),model.config)
                                cosine=1-torch.nn.functional.cosine_similarity(embedding,altered,dim=1)
                                relative=(embedding-altered).norm(dim=1)/embedding.norm(dim=1).clamp_min(1e-12)
                                raw=(history-changed).square().mean((1,2,3,4)).sqrt()*std
                                embedding_rows.extend(dict(run=name,clip_index=count+i,intervention=intervention,
                                    cosine_distance=float(cosine[i]),relative_l2=float(relative[i]),input_rmse_k=float(raw[i]),
                                    advected_source_support_fraction=float(source_support[i])) for i in range(len(history)))
                        count+=len(history)
                        if count%512==0 or count==len(data_loader.dataset):
                            write_json(output/'progress.json',dict(status='running',run=name,phase=f'embeddings_{split}',
                                samples=count,total_samples=len(data_loader.dataset)))
                            print(name,split,count,'/',len(data_loader.dataset),flush=True)
                features[split]=np.concatenate(vectors)
                descriptors[split]=np.concatenate(targets)
                if split=='validation':
                    pd.DataFrame([row for row in embedding_rows if row['run']==name]).to_csv(controls_path,index=False)
                np.savez_compressed(cache,features=features[split],descriptors=descriptors[split])
                write_json(cache_meta,identity)
            rows,predicted=ridge_probe(features['train'],descriptors['train'],features['validation'],descriptors['validation'])
            probe_rows.extend(dict(run=name,**row) for row in rows)
            np.save(output/f'{name}_ridge_predictions.npy',predicted)
        with torch.inference_mode():
            for index in examples:
                clip=validation[index]['pixel_values'][None].cuda()
                history,target=window.split(clip)
                adv,_,_=advection_forecast(history,window.steps)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if model.head is not None:
                        predicted=model.head(model.encode(history)).float()
                        if model.objective=='residual':
                            predicted=predicted+adv
                        example_arrays[index][name]=(predicted[0]*std+mean).cpu().numpy().astype(np.float16)
                    if name=='mvm':
                        mask=fixed_ssl_mask(model.config,[index],.9,'tube',314159,'cuda')
                        reconstruction=model.mae(history,mask)
                        restored,pixel_mask=restore_masked_videomae_pixels(history,reconstruction.logits.float(),mask,model.config)
                        example_arrays[index]['mvm_reconstruction']=(restored[0].float()*std+mean).cpu().numpy().astype(np.float16)
                        example_arrays[index]['mvm_mask']=pixel_mask[0].cpu().numpy()
        pd.DataFrame(metrics).to_csv(output/'forecast_metrics.csv',index=False)
        pd.DataFrame(stratified_metrics).to_csv(output/'forecast_metrics_by_variability.csv',index=False)
        pd.DataFrame(ablations).to_csv(output/'temporal_ablations.csv',index=False)
        pd.DataFrame(probe_rows).to_csv(output/'radiometric_probes.csv',index=False)
        # Preserve previously completed per-run diagnostics on resumed analyses.
        per_run=[r for r in embedding_rows if r['run']==name]
        if per_run:
            pd.DataFrame(per_run).to_csv(output/f'{name}_embedding_controls.csv',index=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (output/'examples').mkdir(exist_ok=True)
    for index,arrays in example_arrays.items():
        np.savez_compressed(output/'examples'/f'clip_{index:04d}.npz',**arrays)
    write_json(output/'examples/catalog.json',dict(indices=examples,selection='fixed validation variability ranks 0,1/3,2/3,1',
        clips=[dict(index=index,**val_frame.iloc[index].to_dict()) for index in examples],
        display_dtype='float16 cache only; metrics computed before display quantization',
        mvm='all masked historical tokens reconstructed; visible pixels copied; not the decoder-subsampled training loss'))
    write_json(output/'frozen_encoder_checks.json',frozen_checks)
    write_json(output/'descriptor_schema.json',dict(names=DESCRIPTOR_NAMES,
        ridge_alpha=10,fit_split='training only',pooling='spatial mean per tubelet, temporal mean and signed linear slope',
        note='descriptors are radiometric observables, not convection labels; absolute input normalization unchanged'))
    verify_snapshot(root,json.loads((dataset/'source_snapshot.json').read_text()))
    write_json(output/'progress.json',dict(status='completed',checkpoints=sources,source_snapshot_verified=True,
        examples=examples,frozen_encoders_verified=len(frozen_checks)))


def read_json_identity(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.queue_dir=args.queue_dir.resolve()
    with exclusive_file_lock(args.queue_dir/'analysis/.lock'):
        analyze(args)


if __name__=='__main__':
    main()
