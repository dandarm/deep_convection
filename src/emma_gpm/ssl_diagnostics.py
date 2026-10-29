"""Read completed pilots and compare frozen representations on common tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import VideoMAEConfig

from .advection import advection_forecast, estimate_translation, warp_translation
from .ssl_model import SeviriSSL


def load_run_model(path, device='cpu'):
    path=Path(path)
    config=json.loads((path/'run_config.json').read_text())
    identity=config['identity']
    model=SeviriSSL(VideoMAEConfig.from_dict(identity['model_config']),
        horizons=len(identity['window']['horizons_minutes']),objective=identity['objective'],
        head_width=identity['head_width'],freeze_encoder=identity['frozen_encoder'])
    checkpoint=torch.load(path/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    if json.loads(json.dumps(checkpoint['identity']))!=identity:
        raise ValueError('Checkpoint and run configuration differ')
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),config


def encoder_digest(encoder):
    digest=hashlib.sha256()
    for name,tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pool_embeddings(tokens, config):
    time=config.num_frames//config.tubelet_size
    spatial=(config.image_size//config.patch_size)**2
    if tokens.shape[1]!=time*spatial:
        raise ValueError('Unexpected temporal/spatial token geometry')
    pooled=tokens.float().reshape(len(tokens),time,spatial,-1).mean(2)
    coordinate=torch.linspace(-1,1,time,device=tokens.device)
    slope=(pooled*coordinate[None,:,None]).sum(1)/coordinate.square().sum()
    return torch.cat((pooled.mean(1),slope),dim=-1)


DESCRIPTOR_NAMES=[f'history_mean_{name}_k' for name in
    ('WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134')]+[
    'history_ir108_mean_abs_change_k','history_ir108_net_change_k_per_15min',
    'last_ir108_gradient_k_per_pixel','future_ir108_mean_change_15min_k',
    'future_ir108_mean_change_30min_k','future_ir108_advection_residual_mae_15min_k',
    'future_ir108_advection_residual_mae_30min_k']


@torch.no_grad()
def radiometric_descriptors(history_k,target_k):
    means=history_k.mean((1,3,4))
    infrared=history_k[:,:,4]
    change=infrared.diff(dim=1).abs().mean((1,2,3))
    tendency=(infrared[:,-1]-infrared[:,0]).mean((1,2))*15/(5*(history_k.shape[1]-1))
    gradient=(infrared[:,-1].diff(dim=-1).abs().mean((1,2))+
              infrared[:,-1].diff(dim=-2).abs().mean((1,2)))/2
    future_change=(target_k[:,:,4]-infrared[:,-1,None]).mean((2,3))
    adv,_,_=advection_forecast(history_k,(3,6))
    residual=(target_k[:,:,4]-adv[:,:,4]).abs().mean((2,3))
    return torch.cat((means,change[:,None],tendency[:,None],gradient[:,None],future_change,residual),1)


@torch.no_grad()
def purely_advected_history(history,return_support=False):
    # Diagnostic synthesis: preserve the first observed image and apply the
    # motion estimated within the real history, without any future target.
    motion=estimate_translation(history[:,-2,4],history[:,-1,4],12)
    frames,masks=zip(*(warp_translation(history[:,0],motion*step)
                      for step in range(history.shape[1])))
    result=torch.stack(frames,dim=1)
    # Where a source leaves the crop, warp_translation retains the first
    # image. Report the supported fraction so this boundary effect is visible.
    support=torch.stack(masks,dim=1).float().mean((1,2,3,4))
    return (result,support) if return_support else result


def ridge_probe(train_x,train_y,val_x,val_y,alpha=10.):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    # This standardizes diagnostic embeddings only, never SEVIRI radiometry.
    scale=StandardScaler().fit(train_x)
    regressor=Ridge(alpha=alpha).fit(scale.transform(train_x),train_y)
    predicted=regressor.predict(scale.transform(val_x))
    baseline=np.broadcast_to(train_y.mean(0),val_y.shape)
    return [dict(descriptor=name,mae=float(np.abs(predicted[:,i]-val_y[:,i]).mean()),
        baseline_mae=float(np.abs(baseline[:,i]-val_y[:,i]).mean()),
        r2=float(r2_score(val_y[:,i],predicted[:,i],force_finite=False)))
        for i,name in enumerate(DESCRIPTOR_NAMES)],predicted
