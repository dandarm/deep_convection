#!/usr/bin/env python3
"""Evaluate a fixed trusted checkpoint with baseline and candidate CUDA kernels."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.forecasting import ForecastWindow
from emma_gpm.seviri_clip_files import ClipBatchLoader
from emma_gpm.seviri_frame_dataset import FrameFileDataset
from emma_gpm.ssl_model import SeviriSSL,history_config,ssl_loss
from emma_gpm.ssl_masking import make_ssl_mask
from emma_gpm.videomae_v2 import running_cell_mask
from emma_gpm.ssl_runtime import (CudaRuntime,configure_runtime,make_optimizer,
    apply_optimizer_runtime,compile_model_regions)
from emma_gpm.ssl_evaluation import evaluate_ssl
from scripts.pretrain_seviri_ssl import write_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--frame-cache',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    identity=checkpoint['identity'];stats=identity['stats']
    data=FrameFileDataset(args.frame_cache,args.dataset/'validation.csv',zarr_root=stats['zarr_root'])
    size='base' if identity['model_config']['hidden_size']==768 else 'small'
    results=dict(checkpoint_sha256=digest(args.checkpoint),epoch=checkpoint['progress']['completed_epochs'],
        validation_manifest_sha256=digest(args.dataset/'validation.csv'),variants={})
    for name,settings in [('baseline',CudaRuntime()),('candidate',CudaRuntime(**json.loads(args.runtime.read_text())))]:
        configure_runtime(settings,'cuda')
        window=ForecastWindow()
        model=SeviriSSL(history_config(window,model_size=size),objective=identity['objective']).cuda()
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer=make_optimizer(model,settings,lr=identity['learning_rate'],weight_decay=identity['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        apply_optimizer_runtime(optimizer,settings)
        # Check every moment and count after changing the execution backend.
        for group,old_group in zip(optimizer.param_groups,checkpoint['optimizer']['param_groups'],strict=True):
            for parameter,identifier in zip(group['params'],old_group['params'],strict=True):
                state=optimizer.state.get(parameter,{})
                old=checkpoint['optimizer']['state'].get(identifier,{})
                for key in ('step','exp_avg','exp_avg_sq'):
                    if key in old: torch.testing.assert_close(state[key].cpu(),old[key],rtol=0,atol=0)
        keys=set(model.state_dict())
        compiled=compile_model_regions(model,settings)
        assert set(model.state_dict())==keys
        loader=ClipBatchLoader(data,batch_size=32,workers=4,prefetch=4,mean=stats['global_mean_k'],
            std=stats['global_std_k'],device='cuda',seed=314159,shuffle=False)
        started=time.perf_counter()
        report=evaluate_ssl(model,loader,window,torch.device('cuda'),stats['global_mean_k'],stats['global_std_k'],
            precision='bf16',variability_edges=checkpoint['progress']['variability_edges_k'])
        torch.cuda.synchronize()
        results['variants'][name]=dict(runtime=settings.to_dict(),compiled_regions=compiled,
            seconds_including_compile=time.perf_counter()-started,optimizer_state_exact=True,
            checkpoint_keys_preserved=True,report=report)
        write_json(args.output_dir/'result.json',results)
        print(name,report['neural']['rmse_k'],flush=True)
        # Execute a resumed training update with loaded moments and compiled
        # backward, after the frozen-weight validation has been recorded.
        iterator=iter(loader)
        try: pixels=next(iterator)['pixel_values']
        finally: iterator.close()
        history,target=window.split(pixels)
        model.train();mask=decode=None
        if model.mae is not None:
            rng=torch.Generator().manual_seed(731)
            mask=make_ssl_mask(model.config,len(history),generator=rng,device='cuda')
            decode=running_cell_mask(model.config,len(history),generator=rng).cuda()
        counts={p:float(s['step']) for p,s in optimizer.state.items() if 'step' in s}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            if compiled: torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(history,mask=mask,decode_mask=decode)
                loss,_=ssl_loss(output,target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite resumed loss')
            prediction=output['prediction'][:2].detach().float().cpu()
            torch.save(prediction,args.output_dir/(name+'_prediction.pt'))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        profiler.export_chrome_trace(str(args.output_dir/(name+'_training_update_trace.json')))
        results['variants'][name]['attention_operators']=[event.key for event in profiler.key_averages()
            if 'scaled_dot_product' in event.key or 'flash_attention' in event.key]
        for parameter,count in counts.items():
            if float(optimizer.state[parameter]['step'])!=count+1:
                raise RuntimeError('Resumed optimizer update count differs')
        results['variants'][name]['resumed_update_passed']=True
        write_json(args.output_dir/'result.json',results)
        del optimizer,model,loader,output,loss,history,target,pixels,counts,parameter,prediction
        gc.collect();torch.cuda.empty_cache()
    old=results['variants']['baseline']['report']['neural']['rmse_k']
    new=results['variants']['candidate']['report']['neural']['rmse_k']
    results['rmse_difference_k']=new-old
    old_rows=results['variants']['baseline']['report']['neural']['per_channel_horizon']
    new_rows=results['variants']['candidate']['report']['neural']['per_channel_horizon']
    results['channel_horizon_rmse_differences_k']=[dict(channel=a['channel'],horizon_minutes=a['horizon_minutes'],
        difference_k=b['rmse_k']-a['rmse_k']) for a,b in zip(old_rows,new_rows,strict=True)]
    maximum=max(abs(r['difference_k']) for r in results['channel_horizon_rmse_differences_k'])
    results['max_channel_horizon_rmse_difference_k']=maximum
    a=torch.load(args.output_dir/'baseline_prediction.pt',weights_only=True)
    b=torch.load(args.output_dir/'candidate_prediction.pt',weights_only=True)
    results['prediction_difference_rms_k']=float((a-b).square().mean().sqrt())*stats['global_std_k']
    results['within_0_02_k']=max(abs(new-old),maximum)<=.02
    write_json(args.output_dir/'result.json',results)
    data.close()
    if not results['within_0_02_k']: raise RuntimeError('Validation RMSE changed by more than 0.02 K')


if __name__=='__main__':main()
