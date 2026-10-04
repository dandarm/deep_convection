import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from emma_gpm.videomae_v2 import (build_videomae_v2_small_7ch_from_scratch,
                                running_cell_mask, fixed_running_cell_mask)
from emma_gpm.videomae import forward_seviri_mae, patchify_videomae_pixels


def tiny_model():
    return build_videomae_v2_small_7ch_from_scratch(image_size=32, num_frames=8,
        hidden_size=48, num_hidden_layers=1, num_attention_heads=3, intermediate_size=96,
        decoder_hidden_size=24, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=3, decoder_intermediate_size=48)


def test_running_cells_cover_every_position_and_match_official_phase_maps():
    model = tiny_model()
    mask = running_cell_mask(model.config, 8, .5).reshape(8, 4, 2, 2)
    assert (mask.sum((2, 3)) == 2).all()
    assert (mask.sum(1) == 2).all()
    queue = torch.tensor([True, True, False, False])
    expected = [torch.stack([queue[(torch.arange(4)+phase+t+1)%4].reshape(2,2)
                            for t in range(4)]) for phase in range(4)]
    assert all(any(torch.equal(row, candidate) for candidate in expected) for row in mask)
    fixed = fixed_running_cell_mask(model.config, [3, 8], .5, 123, 'cpu')
    assert torch.equal(fixed[1], fixed_running_cell_mask(model.config, [8], .5, 123, 'cpu')[0])


def test_dual_mask_loss_excludes_encoder_visible_targets_and_reduces_decoder_tokens():
    model = tiny_model()
    pixels = torch.randn(2, 8, 7, 32, 32)
    encoder_mask = torch.tensor([[True, False, True, False]*4]*2)
    decoder_mask = running_cell_mask(model.config, 2, .5)
    tokens_seen=[]
    hook=model.decoder.register_forward_pre_hook(lambda module,args:tokens_seen.append(args[0].shape[1]))
    output=forward_seviri_mae(model,pixels,encoder_mask,decoder_mask)
    hook.remove()
    assert tokens_seen==[8+8]
    assert output.logits.shape==(2,8,3584)
    targets=patchify_videomae_pixels(pixels,model.config)[~decoder_mask].reshape_as(output.logits)
    expected=(output.logits-targets).square()[output.loss_mask].mean()
    assert torch.allclose(output.loss,expected)
    assert int(output.loss_mask.sum())==8
    output.loss.backward()
    assert model.videomae.embeddings.patch_embeddings.projection.weight.grad.abs().sum()>0
    assert model.decoder.head.weight.grad.abs().sum()>0


def test_v2_export_reload_and_full_reconstruction(tmp_path):
    from emma_gpm.videomae_v2 import VideoMAEV2ForPreTraining
    model=tiny_model().eval()
    pixels=torch.randn(1,8,7,32,32)
    mask=torch.tensor([[True,False,True,False]*4])
    with torch.no_grad():
        first=forward_seviri_mae(model,pixels,mask)
    assert torch.equal(first.loss_mask,torch.ones_like(first.loss_mask))
    model.save_pretrained(tmp_path)
    loaded=VideoMAEV2ForPreTraining.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        second=forward_seviri_mae(loaded,pixels,mask)
    assert torch.equal(first.logits,second.logits)
    with pytest.raises(ValueError,match='No encoder-hidden'):
        forward_seviri_mae(model,pixels,mask,mask)


def test_v2_validation_aggregates_intersection_and_restores_training_mode():
    from emma_gpm.pretraining_validation import evaluate_reconstruction
    model=tiny_model().train()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.decoder.head.bias.copy_(torch.arange(1,8).repeat(512))
    samples=[{'pixel_values':torch.zeros(8,7,32,32)} for _ in range(3)]
    first=evaluate_reconstruction(model,torch.utils.data.DataLoader(samples,batch_size=2),
                                 torch.device('cpu'),2.,ratio=.5)
    second=evaluate_reconstruction(model,torch.utils.data.DataLoader(samples,batch_size=3),
                                  torch.device('cpu'),2.,ratio=.5)
    assert first==second
    assert first['mse']==20.
    assert first['channel_rmse_k']==[2.,4.,6.,8.,10.,12.,14.]
    assert model.training


def test_v2_dual_mask_checkpoint_restores_both_masks_and_optimizer_update(tmp_path):
    from emma_gpm.training_checkpoint import save_checkpoint,restore_checkpoint
    from scripts.pretrain_videomae import make_tube_mask
    model=tiny_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    pixels=torch.randn(1,8,7,32,32)
    generators={'sampler':torch.Generator().manual_seed(42)}
    def step():
        mask=make_tube_mask(model.config,1,.5,'cpu')
        decode=running_cell_mask(model.config,1,.5)
        optimizer.zero_grad()
        output=forward_seviri_mae(model,pixels,mask,decode)
        output.loss.backward();optimizer.step();scheduler.step()
        return mask,decode,output.loss.detach()
    step()
    path=tmp_path/'resume.pt';identity={'version':'v2'}
    save_checkpoint(path,model,optimizer,scheduler,generators,{'completed_epochs':1},identity)
    expected=step()
    weights={k:v.clone() for k,v in model.state_dict().items()}
    restore_checkpoint(path,model,optimizer,scheduler,generators,identity)
    actual=step()
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())


def test_v2_can_overfit_a_fixed_dual_mask_target():
    torch.manual_seed(42)
    model=tiny_model()
    pixels=torch.randn(1,8,7,32,32)
    mask=torch.tensor([[True,False,True,False]*4])
    decode=running_cell_mask(model.config,1,.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    initial=float(forward_seviri_mae(model,pixels,mask,decode).loss.detach())
    for _ in range(25):
        optimizer.zero_grad()
        loss=forward_seviri_mae(model,pixels,mask,decode).loss
        loss.backward();optimizer.step()
    final=float(forward_seviri_mae(model,pixels,mask,decode).loss.detach())
    assert final<initial*.9


def test_v2_full_decoder_matches_v1_logits_with_identical_weights():
    from transformers import VideoMAEForPreTraining,VideoMAEConfig
    v2=tiny_model().eval()
    values=v2.config.to_dict();values['videomae_version']='v1'
    v1=VideoMAEForPreTraining(VideoMAEConfig(**values)).eval()
    v1.load_state_dict(v2.state_dict())
    pixels=torch.randn(1,8,7,32,32)
    mask=torch.tensor([[True,False,True,False]*4])
    with torch.no_grad():
        out1=forward_seviri_mae(v1,pixels,mask)
        out2=forward_seviri_mae(v2,pixels,mask)
    assert torch.equal(out1.logits,out2.logits)
    assert torch.allclose(out1.loss,out2.loss)
