import numpy as np
import pandas as pd
import pytest

from emma_gpm.pretraining_validation import split_temporal_starts, assert_temporal_holdout, fixed_tube_mask, evaluate_reconstruction


def test_split_embargo_applies_to_entire_clip_and_rejects_leakage():
    starts = pd.DatetimeIndex(['2020-01-01', '2020-01-02T23:00', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'])
    training, validation = split_temporal_starts(starts)
    assert list(training) == [starts[0]]
    assert list(validation) == list(starts[-3:])
    assert_temporal_holdout(pd.DataFrame({'start_time':training}), pd.DataFrame({'start_time':validation}))
    with pytest.raises(ValueError, match='leakage'):
        assert_temporal_holdout(pd.DataFrame({'start_time':[starts[1]]}), pd.DataFrame({'start_time':validation}))


def test_fixed_masks_are_batch_independent_tubes_and_do_not_advance_rng():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    config = SimpleNamespace(image_size=224, patch_size=16, num_frames=16, tubelet_size=2)
    state = torch.get_rng_state().clone()
    both = fixed_tube_mask(config, [4,5], .75, 42, 'cpu')
    separate = torch.cat([fixed_tube_mask(config, [i], .75, 42, 'cpu') for i in [4,5]])
    assert torch.equal(both, separate)
    assert torch.equal(state, torch.get_rng_state())
    assert both[0].sum() == 147 * 8
    assert torch.equal(both.reshape(2,8,196)[:,0], both.reshape(2,8,196)[:,7])


def test_validation_aggregates_only_masked_pixels_per_channel(monkeypatch):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    import emma_gpm.videomae as mae
    config = SimpleNamespace(image_size=32, patch_size=16, num_frames=4, tubelet_size=2, num_channels=7)
    model = torch.nn.Linear(1,1)
    model.config = config
    def forward(model, pixels, mask):
        target = mae.patchify_videomae_pixels(pixels, config)[mask].reshape(len(pixels),-1,3584)
        return SimpleNamespace(logits=target + torch.arange(1,8).repeat(512))
    monkeypatch.setattr(mae, 'forward_seviri_mae', forward)
    pixels = torch.zeros(3,4,7,32,32)
    loader = torch.utils.data.DataLoader([{'pixel_values':p} for p in pixels], batch_size=2)
    result = evaluate_reconstruction(model,loader,torch.device('cpu'),2.,ratio=.5)
    assert result['samples'] == 3
    assert result['channel_rmse_k'] == list(np.arange(1,8)*2.)
    assert result['rmse_k'] == pytest.approx(np.sqrt(np.mean(np.arange(1,8)**2))*2)
    assert model.training


def test_trainer_accumulation_matches_full_batches_including_partial_tail(tmp_path, monkeypatch):
    """Exercise actual optimizer updates, including a one-sample final group."""
    torch = pytest.importorskip('torch')
    import scripts.pretrain_videomae as trainer
    from emma_gpm.videomae import build_videomae_small_7ch_from_scratch
    import sys
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([dict(start_time='2020-01-01', origin_y=0, origin_x=i) for i in range(5)]).to_csv(manifest,index=False)
    class Reader:
        def __init__(self, root):
            pass
        def close(self):
            pass
        def load(self, start, *, origin_y, origin_x, global_mean, global_std):
            return np.full((4,7,32,32), .2 + origin_x * .1, dtype=np.float32)
    monkeypatch.setattr(trainer, 'SeviriZarrClipReader', Reader)
    monkeypatch.setattr(trainer, 'estimate_global_stats', lambda *args: (260.,20.))
    monkeypatch.setattr(trainer, 'save_epoch_loss_plot', lambda *args: None)
    def model(args):
        return build_videomae_small_7ch_from_scratch(image_size=32,num_frames=4,
            hidden_size=24,num_hidden_layers=1,num_attention_heads=3,intermediate_size=48,
            decoder_hidden_size=12,decoder_num_hidden_layers=1,decoder_num_attention_heads=3,
            decoder_intermediate_size=24)
    monkeypatch.setattr(trainer, 'build_model', model)
    validation_manifest=tmp_path / 'validation.csv'
    pd.DataFrame([dict(start_time='2020-01-03',origin_y=0,origin_x=0)]).to_csv(validation_manifest,index=False)
    states=[]
    for batch, accumulation in [(4,1),(2,2)]:
        destination=tmp_path / f'batch{batch}'
        monkeypatch.setattr(sys, 'argv', ['pretrain', '--manifest', str(manifest),
            '--zarr-root', str(tmp_path), '--output-dir', str(destination), '--device', 'cpu',
            '--cpu-threads','1','--batch-size',str(batch),'--gradient-accumulation',str(accumulation),
            '--checkpoint-every','1','--mask-ratio','.5','--validation-manifest',str(validation_manifest)])
        assert trainer.main() == 0
        metrics=pd.read_csv(destination/'loss_epochs.csv')
        assert np.isfinite(metrics.validation_own_mse.iloc[0])
        assert np.isfinite(metrics.validation_mse.iloc[0])
        states.append(torch.load(destination/'checkpoint_latest.pt',weights_only=False)['model'])
    assert all(torch.allclose(states[0][k],states[1][k],atol=2e-5,rtol=1e-4) for k in states[0])
