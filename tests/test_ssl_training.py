import json

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.pretrain_seviri_ssl import main


def test_four_objectives_start_with_identical_encoders_and_prediction_heads():
    from emma_gpm.forecasting import ForecastWindow
    from emma_gpm.ssl_model import history_config, SeviriSSL
    config = history_config(ForecastWindow(), image_size=32, patch_size=8, hidden_size=48,
        num_hidden_layers=1, num_attention_heads=3, intermediate_size=96,
        decoder_hidden_size=24, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=3, decoder_intermediate_size=48)
    encoders, heads = [], []
    for objective in ('future','mvm','joint','residual'):
        torch.manual_seed(14)
        model=SeviriSSL(config,objective=objective,head_width=12)
        encoders.append(model.encoder.state_dict())
        if model.head is not None:
            heads.append(model.head.state_dict())
    for states in (encoders,heads):
        for other in states[1:]:
            assert all(torch.equal(value,other[key]) for key,value in states[0].items())


@pytest.mark.parametrize("objective", ["future", "mvm", "joint", "residual"])
def test_trainer_resume_matches_uninterrupted_training(tmp_path, objective):
    common = ["--synthetic-smoke", "--device", "cpu", "--cpu-threads", "1", "--objective", objective,
              "--batch-size", "3", "--gradient-accumulation", "2", "--head-width", "12"]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    main(common + ["--epochs", "2", "--output-dir", str(continuous)])
    main(common + ["--epochs", "1", "--output-dir", str(resumed)])
    main(common + ["--epochs", "2", "--output-dir", str(resumed), "--resume", str(resumed / "checkpoint_latest.pt")])
    expected = torch.load(continuous / "checkpoint_latest.pt", weights_only=False)
    actual = torch.load(resumed / "checkpoint_latest.pt", weights_only=False)
    assert actual["progress"]["optimizer_steps"] == 2
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    assert actual["progress"]["best_score"] == expected["progress"]["best_score"]
    assert (resumed / "encoder/config.json").exists()
    report = json.loads((resumed / "validation_latest.json").read_text())
    assert len(report["advection"]["per_channel_horizon"]) == 14
    assert report["persistence"]["samples"] == 2
    if objective == "future":
        main(common + ["--output-dir", str(resumed), "--resume", str(resumed / "checkpoint_latest.pt"),
                       "--evaluate-only", "--temporal-ablation", "shuffle"])
        assert (resumed / "evaluation_shuffle.json").exists()
        with pytest.raises(ValueError, match="differs"):
            main(common + ["--epochs", "3", "--output-dir", str(resumed),
                           "--resume", str(resumed / "checkpoint_latest.pt"), "--learning-rate", ".1"])


def test_future_head_can_overfit_a_fixed_synthetic_batch():
    from scripts.pretrain_seviri_ssl import SyntheticClips
    from emma_gpm.forecasting import ForecastWindow
    from emma_gpm.ssl_model import history_config, SeviriSSL, ssl_loss

    torch.set_num_threads(1)
    torch.manual_seed(1)
    config = history_config(ForecastWindow(), image_size=32, patch_size=8, hidden_size=48,
        num_hidden_layers=1, num_attention_heads=3, intermediate_size=96)
    model = SeviriSSL(config, head_width=24)
    history, target = ForecastWindow().split(SyntheticClips(1, 1, 260., 20.)[0]["pixel_values"][None])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    initial = float(ssl_loss(model(history), target)[0].detach())
    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = ssl_loss(model(history), target)
        loss.backward()
        optimizer.step()
    final = float(ssl_loss(model(history), target)[0].detach())
    assert final < .2 * initial
