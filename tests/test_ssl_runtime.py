"""Runtime changes must preserve learned parameters, optimizer state and masks."""
import copy
import pytest
import torch

from emma_gpm.ssl_runtime import CudaRuntime, configure_runtime, make_optimizer, apply_optimizer_runtime


def test_runtime_rejects_unknown_modes_and_gpu_options_on_cpu():
    with pytest.raises(ValueError): CudaRuntime(compile_mode='eager')
    with pytest.raises(ValueError): CudaRuntime(sdpa_kernel='magic')
    with pytest.raises(ValueError): configure_runtime(CudaRuntime(fused_adamw=True),'cpu')
    configure_runtime(CudaRuntime(),'cpu')


def test_optimizer_runtime_survives_restore_without_resetting_moments():
    torch.manual_seed(23)
    model=torch.nn.Linear(7,3)
    optimizer=make_optimizer(model,CudaRuntime(),lr=.001,weight_decay=.05)
    model(torch.ones(2,7)).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    saved=copy.deepcopy(optimizer.state_dict())
    destination=make_optimizer(model,CudaRuntime(fused_adamw=True),lr=.02,weight_decay=.1)
    destination.load_state_dict(saved)
    assert destination.param_groups[0]['fused'] is None
    apply_optimizer_runtime(destination,CudaRuntime(fused_adamw=True))
    assert destination.param_groups[0]['fused'] is True
    assert destination.param_groups[0]['lr']==.001
    assert destination.param_groups[0]['weight_decay']==.05
    for parameter, old in zip(model.parameters(),saved['state'].values()):
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(destination.state[parameter][key].cpu(),old[key])
    apply_optimizer_runtime(destination,CudaRuntime())
    assert destination.param_groups[0]['fused'] is None


def test_execution_revision_requires_exact_cuda_settings():
    from emma_gpm.ssl_execution import validate_migration
    parent=dict(batch_size=32,loader_batch_size=32,gradient_accumulation=1,num_workers=4,
                sources={"trainer":"old"},objective="future",learning_rate=.00015,
                input_pipeline={"metadata_sha256":"cache"})
    current={**parent,"sources":{"trainer":"new"},"cuda_runtime":CudaRuntime(fused_adamw=True).to_dict()}
    revision=dict(parent_checkpoints={"checkpoint":"parent.pt"},parent_sources=parent["sources"],
        sources=current["sources"],clip_metadata_sha256="cache",cuda_runtime=current["cuda_runtime"],
        runtime={k:current[k] for k in ("batch_size","loader_batch_size","gradient_accumulation","num_workers")})
    validate_migration(parent,current,"checkpoint",revision)
    with pytest.raises(ValueError,match="CUDA"):
        validate_migration(parent,{**current,"cuda_runtime":CudaRuntime().to_dict()},"checkpoint",revision)
    with pytest.raises(ValueError,match="scientific"):
        validate_migration(parent,{**current,"learning_rate":.002},"checkpoint",revision)


def test_fixed_positions_keep_outputs_and_checkpoint_keys():
    from emma_gpm.forecasting import ForecastWindow
    from emma_gpm.ssl_model import history_config,SeviriSSL
    from emma_gpm.ssl_runtime import compile_model_regions
    config=history_config(ForecastWindow(),image_size=32,patch_size=8,hidden_size=48,
        num_hidden_layers=1,num_attention_heads=3,intermediate_size=96,
        decoder_hidden_size=24,decoder_num_hidden_layers=1,decoder_num_attention_heads=3,
        decoder_intermediate_size=48)
    model=SeviriSSL(config).eval()
    pixels=torch.randn(2,10,7,32,32)
    expected=model(pixels)["prediction"]
    state=copy.deepcopy(model.state_dict())
    compile_model_regions(model,CudaRuntime(static_positions=True))
    assert "position_embeddings" in model.encoder.embeddings._buffers
    assert model.state_dict().keys()==state.keys()
    torch.testing.assert_close(model(pixels)["prediction"],expected,rtol=0,atol=0)
    restored=SeviriSSL(config).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored(pixels)["prediction"],expected,rtol=0,atol=0)


def test_revision_refuses_modified_scientific_model_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import json
    from scripts import run_seviri_ssl_runtime_revision as runner
    live=tmp_path/"live";output=tmp_path/"output"
    filename="src/emma_gpm/ssl_model.py"
    original=output/"source_archive"/filename;original.parent.mkdir(parents=True)
    original.write_text("original scientific model")
    current=live/filename;current.parent.mkdir(parents=True)
    current.write_text("different scientific model")
    settings=tmp_path/"runtime.json";settings.write_text(json.dumps(CudaRuntime(fused_adamw=True).to_dict()))
    monkeypatch.setattr(runner,"ROOT",live)
    args=SimpleNamespace(output_dir=output,revision_dir=tmp_path/"revision",cuda_runtime=settings)
    with pytest.raises(ValueError,match="Scientific source changed"):
        runner.prepare(args,{"sources":{filename:runner.digest(original)}},{})
