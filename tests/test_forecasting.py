from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from emma_gpm.advection import advection_forecast, estimate_translation, warp_translation
from emma_gpm.forecasting import ForecastWindow, temporal_ablation, variability_statistics
from emma_gpm.forecast_metrics import ForecastMetrics
from emma_gpm.ssl_masking import make_ssl_mask, fixed_ssl_mask
from emma_gpm.ssl_model import SeviriSSL, history_config, load_encoder_weights, ssl_loss
from emma_gpm.videomae_v2 import build_videomae_v2_small_7ch_from_scratch
from scripts.pretrain_seviri_ssl import validate_manifests


def tiny_config():
    return history_config(ForecastWindow(), image_size=16, patch_size=4,
        hidden_size=24, num_hidden_layers=1, num_attention_heads=3, intermediate_size=48,
        decoder_hidden_size=12, decoder_num_hidden_layers=1, decoder_num_attention_heads=3,
        decoder_intermediate_size=24)


def test_window_offsets_and_future_perturbation_have_no_encoder_effect():
    torch.set_num_threads(1)
    window = ForecastWindow()
    clip = torch.arange(16.).reshape(1, 16, 1, 1, 1).expand(2, 16, 7, 16, 16).clone()
    history, targets = window.split(clip)
    assert history.shape == (2, 10, 7, 16, 16)
    assert window.target_indices == (12, 15)
    assert torch.equal(targets[0, :, 0, 0, 0], torch.tensor([12., 15.]))
    model = SeviriSSL(tiny_config(), objective="future", head_width=12).eval()
    observed = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: observed.append(args[0].shape))
    before = model(history)["prediction"].detach()
    clip[:, 10:] = 10000
    perturbed_history, _ = window.split(clip)
    after = model(perturbed_history)["prediction"].detach()
    hook.remove()
    assert torch.equal(before, after)
    assert observed == [history.shape, history.shape]
    with pytest.raises(ValueError, match="historical"):
        model(clip)


@pytest.mark.parametrize("arguments", [dict(history_frames=11), dict(history_frames=12),
    dict(horizons_minutes=(30, 15)), dict(horizons_minutes=(0, 15)),
    dict(horizons_minutes=(10, 17)), dict(horizons_minutes=(15, 15))])
def test_invalid_forecast_windows_are_rejected(arguments):
    with pytest.raises(ValueError):
        ForecastWindow(**arguments)


def test_future_targets_are_finite_and_not_repaired():
    clip = torch.ones(1, 16, 7, 2, 2)
    clip[0, 15, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        ForecastWindow().split(clip)


@pytest.mark.parametrize("dx,dy", [(2, -1), (-3, 2), (0, 0)])
def test_advection_translation_sign_horizon_and_boundary_support(dx, dy):
    torch.manual_seed(12)
    previous = torch.randn(2, 32, 32)
    current = torch.roll(previous, shifts=(dy, dx), dims=(-2, -1))
    shift = estimate_translation(previous, current, max_shift=4)
    assert torch.equal(shift, torch.tensor([[dx, dy], [dx, dy]], dtype=torch.float32))
    history = torch.stack((previous, current), 1)[:, :, None].expand(-1, -1, 7, -1, -1)
    forecasts, masks, _ = advection_forecast(history, (1, 3), max_shift=4)
    for index, steps in enumerate((1, 3)):
        expected = torch.roll(history[:, -1], (dy * steps, dx * steps), (-2, -1))
        supported = masks[:, index].expand_as(expected)
        assert torch.allclose(forecasts[:, index][supported], expected[supported], atol=2e-5)
        assert torch.equal(forecasts[:, index][~supported], history[:, -1][~supported])
    assert int(masks[:, 0].sum()) == 2 * (32 - abs(dx)) * (32 - abs(dy))


def test_uniform_motion_and_fractional_warp():
    assert torch.equal(estimate_translation(torch.ones(1, 8, 8), torch.ones(1, 8, 8) * 2), torch.zeros(1, 2))
    frame = torch.arange(8.).reshape(1, 1, 1, 8).expand(1, 7, 8, 8)
    warped, valid = warp_translation(frame, torch.tensor([[.5, 0.]]))
    assert torch.allclose(warped[..., 1:], frame[..., 1:] - .5, atol=1e-6)
    assert not valid[..., 0].any()


@pytest.mark.parametrize("objective", ["mvm", "future", "joint", "residual"])
def test_all_ssl_objectives_backpropagate_with_history_only(objective):
    torch.manual_seed(1)
    config = tiny_config()
    model = SeviriSSL(config, objective=objective, head_width=12)
    history = torch.randn(2, 10, 7, 16, 16)
    mask = make_ssl_mask(config, 2, .75)
    target = torch.randn(2, 2, 7, 16, 16)
    advection = torch.randn_like(target, requires_grad=True)
    output = model(history, mask=mask, advection=advection)
    loss, terms = ssl_loss(output, target, lambda_mvm=.3, lambda_future=.7)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    if objective in {"future", "joint", "residual"}:
        assert output["prediction"].shape == target.shape
        assert any(p.grad is not None for p in model.head.parameters())
    if objective == "joint":
        assert torch.allclose(loss, .3 * terms["mvm"] + .7 * terms["future"])
    if objective == "residual":
        assert advection.grad is None
        assert torch.equal(output["prediction"], output["residual"] + advection.detach())


@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_residual_and_reconstructed_losses_are_equivalent(loss):
    target, adv, residual = (torch.randn(2, 2, 7, 4, 4) for _ in range(3))
    reconstructed, _ = ssl_loss({"prediction": adv + residual}, target, loss=loss)
    residual_loss, _ = ssl_loss({"prediction": residual}, target - adv, loss=loss)
    assert torch.allclose(reconstructed, residual_loss)


def test_frozen_probe_keeps_encoder_in_eval_and_updates_only_head():
    model = SeviriSSL(tiny_config(), objective="future", head_width=12, freeze_encoder=True).train()
    before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    prediction = model(torch.randn(1, 10, 7, 16, 16))["prediction"]
    prediction.square().mean().backward()
    optimizer.step()
    assert not model.encoder.training
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(torch.equal(value, model.encoder.state_dict()[key]) for key, value in before.items())
    assert any(p.grad is not None for p in model.head.parameters())


def test_existing_v2_encoder_weights_transfer_from_16_to_10_frames(tmp_path):
    config = tiny_config().to_dict()
    config.pop("model_type", None)
    config["num_frames"] = 16
    old = build_videomae_v2_small_7ch_from_scratch(**config)
    new = SeviriSSL(tiny_config(), objective="future", head_width=12)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": old.state_dict()}, path)
    load_encoder_weights(new.encoder, path)
    assert all(torch.equal(value, new.encoder.state_dict()[key]) for key, value in old.videomae.state_dict().items())
    assert new(torch.randn(1, 10, 7, 16, 16))["prediction"].shape == (1, 2, 7, 16, 16)
    old.save_pretrained(tmp_path / "model")
    load_encoder_weights(new.encoder, tmp_path / "model")


@pytest.mark.parametrize("strategy", ["tube", "random", "block"])
def test_masks_have_equal_counts_and_fixed_masks_do_not_depend_on_batches(strategy):
    config = tiny_config()
    mask = make_ssl_mask(config, 3, .75, strategy, generator=torch.Generator().manual_seed(42))
    assert mask.shape == (3, 80)
    assert mask.dtype == torch.bool and torch.all(mask.sum(1) == 60)
    together = fixed_ssl_mask(config, [7, 9], .75, strategy, 42, "cpu")
    apart = torch.cat([fixed_ssl_mask(config, [index], .75, strategy, 42, "cpu") for index in [7, 9]])
    assert torch.equal(together, apart)
    if strategy == "tube":
        assert torch.equal(mask.reshape(3, 5, 16)[:, 0], mask.reshape(3, 5, 16)[:, -1])


def test_physical_metrics_units_aggregation_gradient_and_empty_support():
    target = torch.zeros(3, 2, 7, 4, 4)
    predicted = torch.ones_like(target)
    predicted[-1] = 3
    metric = ForecastMetrics((15, 30), std_k=2.)
    metric.update(predicted[:2], target[:2], torch.zeros(2, 2, 1, 4, 4, dtype=torch.bool))
    metric.update(predicted[2:], target[2:], torch.zeros(1, 2, 1, 4, 4, dtype=torch.bool))
    report = metric.report()
    row = report["per_channel_horizon"][0]
    assert row["mae_k"] == pytest.approx(10 / 3)
    assert row["rmse_k"] == pytest.approx(np.sqrt(44 / 3))
    assert row["gradient_mae_k_per_pixel"] == 0
    assert row["minimum_temperature_mae_k"] == pytest.approx(10 / 3)
    assert row["contrast_p95_p05_mae_k"] == 0
    assert row["tendency_mae_k_per_min"] == pytest.approx((10 / 3) / 15)
    assert row["supported_mae_k"] is None
    assert report["advection_support_fraction"] == 0
    assert report["per_channel_horizon"][7]["interval_tendency_mae_k_per_min"] == 0
    gradient = torch.arange(4.).reshape(1, 1, 1, 1, 4).expand_as(target)
    metric = ForecastMetrics((15, 30), 2.)
    metric.update(gradient, target)
    assert metric.report()["per_channel_horizon"][0]["gradient_mae_k_per_pixel"] == pytest.approx(1.)


def test_ablation_preserves_reference_time_and_sample_seeds():
    history = torch.arange(10.).reshape(1, 10, 1, 1, 1).expand(2, 10, 7, 2, 2)
    shuffled = temporal_ablation(history, "shuffle", seed=2, indices=[7, 9])
    assert torch.equal(shuffled[:, -1], history[:, -1])
    assert not torch.equal(shuffled[:, :-1], history[:, :-1])
    assert torch.equal(shuffled[1:], temporal_ablation(history[1:], "shuffle", seed=2, indices=[9]))
    repeated = temporal_ablation(history, "last_frame")
    assert torch.all(repeated == 9)
    stats = variability_statistics(history)
    assert torch.all(stats["mean_abs_change_k"] == 1)


def test_manifest_validation_embargo_reserved_days_and_fractional_origins():
    def manifest(start):
        return pd.DataFrame(dict(start_time=[start], origin_y=[0], origin_x=[0]))
    validate_manifests(manifest("2020-01-01"), manifest("2020-01-03"))
    with pytest.raises(ValueError, match="leakage"):
        validate_manifests(manifest("2020-01-01"), manifest("2020-01-02"))
    with pytest.raises(ValueError, match="reserved"):
        validate_manifests(manifest("2020-03-02T23:00Z"), manifest("2020-03-12"))
    frame = manifest("2020-01-01")
    frame["origin_x"] = .5
    with pytest.raises(ValueError, match="integers"):
        validate_manifests(frame, manifest("2020-01-03"))


def test_stratified_sampling_retains_uniform_component_and_is_reproducible():
    from emma_gpm.ssl_sampling import stratified_indices
    scores = np.linspace(0., 10., 1000)
    indices, report = stratified_indices(scores, 200, seed=11)
    again, _ = stratified_indices(scores, 200, seed=11)
    assert np.array_equal(indices, again) and len(set(indices)) == 200
    assert report["uniform_count"] == report["variability_count"] == 100
    assert report["selected_stratum_counts"][2] > report["selected_stratum_counts"][0]
    equal, report = stratified_indices(np.zeros(100), 20)
    assert len(set(equal)) == 20
    with pytest.raises(ValueError, match="uniform"):
        stratified_indices(scores, 20, uniform_fraction=0)


def test_compensated_score_removes_pure_translation_but_retains_cooling():
    from emma_gpm.ssl_sampling import compensated_variability
    torch.manual_seed(1)
    base = torch.randn(1, 7, 32, 32)
    history = torch.stack([torch.roll(base, shift, -1) for shift in range(4)], dim=1)
    assert compensated_variability(history).item() < 1e-5
    assert variability_statistics(history)["mean_abs_change_k"].item() > .1
    cooling = history - torch.arange(4.)[None, :, None, None, None] * 2
    assert compensated_variability(cooling).item() == pytest.approx(2., abs=1e-5)


def test_shared_encoder_and_prediction_head_initialization_are_objective_independent():
    models = []
    for objective in ("future", "joint", "residual"):
        torch.manual_seed(42)
        models.append(SeviriSSL(tiny_config(), objective=objective, head_width=12))
    for model in models[1:]:
        for key, value in models[0].encoder.state_dict().items():
            assert torch.equal(value, model.encoder.state_dict()[key])
        for key, value in models[0].head.state_dict().items():
            assert torch.equal(value, model.head.state_dict()[key])
