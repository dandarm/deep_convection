#!/usr/bin/env python3
"""Evaluate saved best/final checkpoints with identical 50/75/90% holdout masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from emma_gpm.seviri_clips import SeviriVideoMAEDataset, SeviriZarrClipReader
from emma_gpm.pretraining_validation import evaluate_reconstruction
from pretrain_videomae import initialize_loader_worker


def main():
    import torch
    from transformers import VideoMAEConfig,VideoMAEForPreTraining
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=args.queue_dir.resolve()
    results_path=output/'validation_mask_sweep.csv'
    results=pd.read_csv(results_path).to_dict('records') if results_path.exists() else []
    manifest=output/'validation.csv'
    for name in ['mask50','mask75','mask90']:
        folder=output/name
        for kind,checkpoint_name in [('best','checkpoint_best.pt'),('final','checkpoint_latest.pt')]:
            path=folder/checkpoint_name
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            identity=checkpoint['identity']; epoch=checkpoint['progress']['completed_epochs']
            if identity['validation_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
                raise ValueError('Checkpoint validation manifest differs')
            model_class=VideoMAEForPreTraining
            if identity['config'].get('videomae_version')=='v2':
                from emma_gpm.videomae_v2 import VideoMAEV2ForPreTraining
                model_class=VideoMAEV2ForPreTraining
            model=model_class(VideoMAEConfig(**identity['config']))
            model.load_state_dict(checkpoint['model']);model=model.cuda().eval()
            del checkpoint
            reader=SeviriZarrClipReader(identity['zarr_root'])
            dataset=SeviriVideoMAEDataset(manifest,reader,global_mean=identity['mean'],global_std=identity['std'])
            loader=torch.utils.data.DataLoader(dataset,batch_size=8,num_workers=4,pin_memory=True,
                multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2,
                worker_init_fn=initialize_loader_worker,generator=torch.Generator().manual_seed(identity['validation_seed']))
            for ratio in [.5,.75,.9]:
                if any(row['run']==name and row['checkpoint']==kind and row['epoch']==epoch and row['validation_mask_ratio']==ratio for row in results):
                    continue
                metrics=evaluate_reconstruction(model,loader,torch.device('cuda'),identity['std'],ratio=ratio,
                    seed=identity['validation_seed'],precision=identity['precision'])
                row=dict(run=name,checkpoint=kind,epoch=epoch,training_mask_ratio=identity['mask_ratio'],
                    validation_mask_ratio=ratio,validation_mse=metrics['mse'],validation_rmse_k=metrics['rmse_k'],
                    samples=metrics['samples'],validation_seed=metrics['mask_seed'],
                    decoder_mask_ratio=metrics['decoder_mask_ratio'],
                    validation_manifest_sha256=identity['validation_sha256'])
                results.append(row)
                temporary=results_path.with_suffix('.csv.part')
                pd.DataFrame(results).to_csv(temporary,index=False);temporary.replace(results_path)
                print(json.dumps(row),flush=True)
            del loader,model,dataset
            reader.close();torch.cuda.empty_cache()
    (output/'validation_mask_sweep_status.json').write_text(json.dumps(dict(status='completed',evaluations=len(results)),indent=2)+'\n')


if __name__=='__main__':
    main()
