import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from emma_gpm.ssl_distributed import GlobalBatchSampler
from emma_gpm.forecast_metrics import ForecastMetrics


def test_global_permutation_has_no_rank_padding_and_preserves_tail_weights():
    samplers=[GlobalBatchSampler(20,3,r,2,17) for r in range(2)]
    batches=[list(s) for s in samplers]
    assert [len(b) for b in batches[0]]==[3,3,3,1]
    assert [len(b) for b in batches[1]]==[3,3,3,1]
    order=[i for pair in zip(*batches) for row in zip(*pair) for i in row]
    assert order==torch.randperm(20,generator=torch.Generator().manual_seed(17)).tolist()
    assert len(set(order))==20
    for s in samplers:s.epoch=1
    assert list(samplers[0])!=batches[0]
    validation=[i for r in range(4) for b in GlobalBatchSampler(20,3,r,4,1,False) for i in b]
    assert validation==list(range(20))
    with pytest.raises(ValueError,match='padding'):
        GlobalBatchSampler(21,3,0,2,1)


def test_metric_merge_weights_pixels_before_computing_rmse():
    torch.manual_seed(19)
    target=torch.randn(6,2,7,8,8)
    prediction=target+torch.tensor([1,2,3,4,5,6.])[:,None,None,None,None]
    full=ForecastMetrics((15,30),2.);full.update(prediction,target)
    left,right=ForecastMetrics((15,30),2.),ForecastMetrics((15,30),2.)
    left.update(prediction[:1],target[:1]);right.update(prediction[1:],target[1:])
    report=left.merge(right).report()
    assert report['samples']==6
    assert report['rmse_k']==pytest.approx(full.report()['rmse_k'])
    for a,b in zip(report['per_channel_horizon'],full.report()['per_channel_horizon']):
        assert a==pytest.approx(b)


def launch(out,objective='future',epochs=1,resume=False):
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         'scripts/pretrain_seviri_ssl_ddp.py','--synthetic-smoke','--device','cpu',
         '--precision','fp32','--compile-mode','off','--cpu-threads','1','--objective',objective,
         '--batch-size-per-gpu','2','--validation-batch-size','2','--epochs',str(epochs),'--output-dir',str(out)]
    if resume:cmd+=['--resume',str(out/'checkpoint_latest.pt')]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=150)
    assert result.returncode==0,result.stdout[-8000:]
    return torch.load(out/'checkpoint_latest.pt',weights_only=False,map_location='cpu')


@pytest.mark.parametrize('objective',['mvm','future','joint'])
def test_two_rank_training_and_validation_match_unsharded_evaluation(tmp_path,objective):
    from transformers import VideoMAEConfig
    from scripts.pretrain_seviri_ssl import SyntheticClips
    from emma_gpm.forecasting import ForecastWindow
    from emma_gpm.ssl_model import SeviriSSL
    from emma_gpm.ssl_evaluation import evaluate_ssl
    saved=launch(tmp_path/'run',objective)
    identity=saved['identity'];stats=identity['stats']
    model=SeviriSSL(VideoMAEConfig.from_dict(identity['model_config']),objective=objective,head_width=12)
    model.load_state_dict(saved['model']);torch.set_num_threads(1)
    data=SyntheticClips(8,identity['seed']+10000,stats['global_mean_k'],stats['global_std_k'])
    expected=evaluate_ssl(model,torch.utils.data.DataLoader(data,batch_size=2),ForecastWindow(),
                          torch.device('cpu'),stats['global_mean_k'],stats['global_std_k'])
    actual=json.loads((tmp_path/'run/validation_latest.json').read_text())
    for name in ('persistence','advection','neural'):
        if name in expected:
            assert actual[name]['samples']==8
            assert actual[name]['rmse_k']==pytest.approx(expected[name]['rmse_k'],rel=1e-7)
    assert actual['mvm_mse']==pytest.approx(expected['mvm_mse'])
    assert saved['progress']['optimizer_steps']==4
    assert len(saved['rank_rng'])==2


def test_distributed_epoch_resume_preserves_parameters(tmp_path):
    continuous=launch(tmp_path/'continuous',epochs=2)
    launch(tmp_path/'resumed',epochs=1)
    resumed=launch(tmp_path/'resumed',epochs=2,resume=True)
    assert continuous['progress']['optimizer_steps']==resumed['progress']['optimizer_steps']==8
    for name,value in continuous['model'].items():
        assert torch.equal(value,resumed['model'][name]),name
