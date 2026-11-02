import pandas as pd
import pytest

from emma_gpm.ssl_validation_set import assert_day_separation, select_balanced_validation


def test_day_reservation_checks_both_sides_and_final_training_frame():
    def training(start): return pd.DataFrame({'start_time':[start]})
    assert_day_separation(training('2022-01-01T22:35:00Z'), ['2022-01-03'])
    with pytest.raises(ValueError, match='reserved validation'):
        assert_day_separation(training('2022-01-01T22:45:00Z'), ['2022-01-03'])
    with pytest.raises(ValueError, match='reserved validation'):
        assert_day_separation(training('2022-01-04T23:55:00Z'), ['2022-01-03'])
    assert_day_separation(training('2022-01-05T00:00:00Z'), ['2022-01-03'])


def test_selection_keeps_equal_day_and_time_weights_after_rejections():
    rows = [dict(day=day,time_bin=hour//6,start_time=f'{day}T{hour:02d}:00:00Z',origin_y=y,origin_x=0)
            for day in ('2022-01-01','2022-02-01') for hour in (0,6,12,18) for y in range(10)]
    pool = pd.DataFrame(rows).iloc[3:]
    selected = select_balanced_validation(pool, ['2022-01-01','2022-02-01'], 16)
    assert len(selected) == 32
    assert selected.groupby(['day','time_bin']).size().eq(4).all()
    pd.testing.assert_frame_equal(selected, select_balanced_validation(pool, ['2022-01-01','2022-02-01'],16))
    with pytest.raises(ValueError, match='Insufficient'):
        select_balanced_validation(pool.iloc[8:], ['2022-01-01','2022-02-01'],16)


def test_training_entrypoint_reserves_registered_days():
    from scripts.pretrain_seviri_ssl import validate_manifests
    training = pd.DataFrame(dict(start_time=['2022-01-05T12:00:00Z'],origin_y=[0],origin_x=[0]))
    validation = training.assign(start_time='2023-01-01T12:00:00Z')
    with pytest.raises(ValueError, match='reserved validation'):
        validate_manifests(training, validation)


def test_relocated_validation_inventory_requires_unique_filenames():
    from scripts.evaluate_seviri_ssl_validation import expected_digest
    ready = dict(files={'/original/validation.csv':'digest'})
    assert expected_digest(ready, 'validation.csv') == 'digest'
    with pytest.raises(ValueError, match='exactly one'):
        expected_digest(ready, 'metadata.json')
    ready['files']['/another/validation.csv'] = 'digest'
    with pytest.raises(ValueError, match='exactly one'):
        expected_digest(ready, 'validation.csv')


@pytest.mark.parametrize('objective', ['future','mvm','joint'])
def test_daily_reports_preserve_global_metrics_and_mask_indices(objective):
    import torch
    from scripts.pretrain_seviri_ssl import SyntheticClips
    from emma_gpm.forecasting import ForecastWindow
    from emma_gpm.ssl_model import history_config, SeviriSSL
    from emma_gpm.ssl_evaluation import evaluate_ssl
    torch.set_num_threads(1)
    torch.manual_seed(9)
    window = ForecastWindow()
    config = history_config(window, image_size=32, patch_size=8, hidden_size=48,
        num_hidden_layers=1, num_attention_heads=3, intermediate_size=96,
        decoder_hidden_size=24, decoder_num_hidden_layers=1, decoder_num_attention_heads=3,
        decoder_intermediate_size=48)
    model = SeviriSSL(config, objective=objective, head_width=12)
    loader = torch.utils.data.DataLoader(SyntheticClips(4,1,260.,20.), batch_size=2)
    args = (model,loader,window,torch.device('cpu'),260.,20.)
    original = evaluate_ssl(*args)
    progress = []
    report = evaluate_ssl(*args, sample_days=['2022-01-01']*2+['2022-02-01']*2,
                          progress_callback=progress.append)
    assert {k:v for k,v in report.items() if k!='by_day'} == original
    assert progress == [2,4]
    assert all(day['samples']==2 for day in report['by_day'].values())
    for method in ('persistence','advection'):
        assert sum(day[method]['samples'] for day in report['by_day'].values()) == original[method]['samples']
    if objective in ('mvm','joint'):
        assert sum(day['mvm_mse'] for day in report['by_day'].values())/2 == pytest.approx(original['mvm_mse'])
        with pytest.raises(ValueError, match='contained within'):
            evaluate_ssl(*args, sample_days=['a','b','b','b'])
