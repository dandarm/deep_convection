import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from emma_gpm import ssl_study as study
from emma_gpm.seviri import THERMAL_CHANNELS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metric(scale=1.):
    rows = [dict(horizon_minutes=h, channel=c, mse_k2=mse*scale**2,
                 rmse_k=np.sqrt(mse)*scale, mae_k=np.sqrt(mse)*scale,
                 gradient_mae_k_per_pixel=.2*scale)
            for h, mse in ((15, 1.), (30, 9.)) for c in THERMAL_CHANNELS]
    return dict(samples=2, rmse_k=np.sqrt(5)*scale, per_channel_horizon=rows)


@pytest.fixture
def series(tmp_path):
    cases = [dict(name=f'n{n}', train_samples=n, validation_samples=2) for n in (2, 4)]
    write(tmp_path/'scaling_plan.json', dict(cases=cases))
    for case in cases:
        name, n = case['name'], case['train_samples']
        queue = tmp_path/name
        data = queue/'dataset'
        data.mkdir(parents=True)
        pd.DataFrame(dict(start_time=['2020-09-12']*n, origin_y=range(n), origin_x=[0]*n)).to_csv(data/'train.csv', index=False)
        pd.DataFrame(dict(start_time=['2020-09-17']*2, origin_y=[0, 1], origin_x=[0, 0])).to_csv(data/'validation.csv', index=False)
        write(data/'channel_stats.json', dict(global_mean_k=260., global_std_k=2.))
        names = ('future', 'mvm', 'joint', 'residual', *study.FAMILIES['frozen_common_head'])
        stages = [dict(name=model, epochs=4, kind='frozen_common_head' if model.startswith('probe_') else 'pretraining') for model in names]
        write(queue/'comparison_plan.json', dict(stages=stages, dataset={file: hashlib.sha256((data/file).read_bytes()).hexdigest()
            for file in ('train.csv', 'validation.csv', 'channel_stats.json')}))
        for model in names:
            if model.startswith('probe_') and model != 'probe_random':
                continue
            count = (2 if n == 2 else 1) if model == 'probe_random' else 3 if n == 4 and model == 'joint' else 4
            epochs = [dict(epoch=e, optimizer_steps=e*n//2, train_mvm=.25, train_future=.25,
                           validation_mvm=.25, validation_future=5*e**2/4,
                           validation_rmse_k=float(np.sqrt(5)*e)) for e in range(1, count+1)]
            write(queue/model/'progress.json', dict(status='completed' if count == 4 else 'training',
                completed_epochs=count, epochs=epochs))
            for e in range(1, count+1):
                report = dict(persistence=metric(), advection=metric(), mvm_mse=.25,
                              variability_edges_k=[.75, 1.44], by_historical_variability={})
                if model != 'mvm':
                    report['neural'] = metric(e)
                write(queue/model/'validation_epochs'/f'epoch_{e:03d}.json', report)
            # A live/latest report can precede the publication of its checkpoint epoch.
            write(queue/model/'validation_latest.json', dict(neural=metric(99)))
    return tmp_path


def test_committed_reports_and_correct_rmse_pooling(series):
    snapshot = study.load_study(series)
    selected, coverage = study.select_budget(snapshot)
    assert selected['n2', 'future'] == 4
    assert selected['n4', 'future'] == selected['n4', 'joint'] == 3
    scores = study.metrics_frame(snapshot, selected)
    summary = study.forecast_summary(scores)
    row = summary[(summary.Model == 'Future') & (summary['Training clips'] == 2)].iloc[0]
    assert row['RMSE (K)'] == pytest.approx(np.sqrt(5)*4)
    assert row['RMSE +15 min (K)'] == 4
    assert row['RMSE +30 min (K)'] == 12
    assert row['RMSE (K)'] != (4+12)/2
    assert coverage.Comparison.tolist() == ['Final', 'Interim / partial']


def test_matching_updates_uses_different_epochs_for_nested_sizes(series):
    snapshot = study.load_study(series)
    selected, _ = study.select_budget(snapshot, budget='optimizer_steps', across_sizes=True)
    assert all(epoch == (4 if case == 'n2' else 2) for (case, _), epoch in selected.items())
    table = study.size_comparison(snapshot)
    matched = table[table['Matched budget'] == 'Optimizer updates']
    assert (matched.Budget == 4).all()
    assert (matched['Small-set epoch'] == 4).all()
    assert (matched['Large-set epoch'] == 2).all()


def test_pending_frozen_probes_are_explicit_and_mvm_is_a_separate_task(series):
    snapshot = study.load_study(series)
    selected, coverage = study.select_budget(snapshot, 'frozen_common_head')
    assert all(name == 'probe_random' for _, name in selected)
    assert (coverage['Available models'] == 1).all()
    assert (coverage['Planned models'] == 5).all()
    assert (coverage.Comparison == 'Interim / partial').all()
    reconstruct, _ = study.reconstruction_summary(snapshot)
    assert set(reconstruct.Model) == {'MVM', 'Joint'}
    np.testing.assert_allclose(reconstruct['Historical reconstruction RMSE (K)'], 1.)


def test_validation_manifest_must_match_across_sizes(series):
    path = series/'n4/dataset/validation.csv'
    path.write_text(path.read_text().replace('2020-09-17', '2020-09-18'))
    plan_path = series/'n4/comparison_plan.json'
    plan = json.loads(plan_path.read_text())
    plan['dataset']['validation.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(plan_path, plan)
    with pytest.raises(ValueError, match='exact validation manifest'):
        study.load_study(series)


def test_missing_epoch_never_falls_back_to_latest_report(series):
    snapshot = study.load_study(series)
    (series/'n2/future/validation_epochs/epoch_004.json').unlink()
    with pytest.raises(FileNotFoundError):
        study.epoch_report(snapshot, 'n2', 'future', 4)
    with pytest.raises(ValueError, match='not been committed'):
        study.epoch_report(snapshot, 'n4', 'joint', 4)


def test_inconsistent_baseline_is_rejected(series):
    path = series/'n2/joint/validation_epochs/epoch_004.json'
    value = json.loads(path.read_text())
    value['advection'] = metric(2.)
    write(path, value)
    snapshot = study.load_study(series)
    selected, _ = study.select_budget(snapshot)
    with pytest.raises(ValueError, match='Baselines differ'):
        study.metrics_frame(snapshot, selected)


def test_lead_times_and_sizes_share_the_bar_scale(series):
    snapshot = study.load_study(series)
    selected, _ = study.select_budget(snapshot)
    scores = study.metrics_frame(snapshot, selected)
    fig = study.channel_figure(scores)
    assert len(fig.axes) == 4
    assert len({axis.get_ylim() for axis in fig.axes}) == 1
    fig.canvas.draw()
    plt.close(fig)


def test_probe_reference_matches_size_and_epoch_and_preserves_sign():
    scores = pd.DataFrame([
        {'Training clips': 4096, 'Epoch': 150, 'Model': 'Probe (Random)', 'RMSE (K)': 4.3},
        {'Training clips': 4096, 'Epoch': 150, 'Model': 'Probe (Future)', 'RMSE (K)': 4.0},
        {'Training clips': 4096, 'Epoch': 150, 'Model': 'Probe (MVM)', 'RMSE (K)': 4.5},
        {'Training clips': 8192, 'Epoch': 150, 'Model': 'Probe (Random)', 'RMSE (K)': 4.1},
        {'Training clips': 8192, 'Epoch': 150, 'Model': 'Probe (Future)', 'RMSE (K)': 3.7},
        {'Training clips': 8192, 'Epoch': float('nan'), 'Model': 'Advection', 'RMSE (K)': 5.2},
    ])
    result = study.probe_reference_summary(scores)
    np.testing.assert_allclose(result['RMSE reduction vs Random (K)'], [0, .3, -.2, 0, .4])
    np.testing.assert_allclose(result['RMSE reduction vs Random (%)'], [0, 100*.3/4.3, -100*.2/4.3, 0, 100*.4/4.1])
    scores.loc[scores.Model == 'Probe (Future)', 'Epoch'] = 75
    with pytest.raises(ValueError, match='same size and epoch'):
        study.probe_reference_summary(scores)


def test_selected_extension_compares_registered_models_at_equal_updates(series):
    reference = study.load_study(series)
    extension = {**reference, 'cases': {'n4': reference['cases']['n4']},
                 'runs': {('n4', name): reference['runs']['n4', name] for name in ('future', 'joint')}}
    combined = study.extension_study(reference, extension, reference_case='n2')
    selected, coverage = study.select_budget(combined, budget='optimizer_steps', across_sizes=True)
    assert selected == {('n2', 'future'): 4, ('n2', 'joint'): 4, ('n4', 'future'): 2, ('n4', 'joint'): 2}
    assert coverage['Planned models'].tolist() == [2, 2]
    assert set(combined['runs']) == set(selected)
    summary = study.forecast_summary(study.metrics_frame(combined, selected))
    assert set(summary[summary.Model.isin(['Future', 'Joint'])]['Optimizer updates']) == {4}
    with pytest.raises(ValueError, match='same validation'):
        study.extension_study(reference, {**extension, 'validation_sha256': 'different'}, reference_case='n2')


def test_extension_rejects_training_sets_without_strict_nesting(series):
    reference = study.load_study(series)
    path = series/'n4/dataset/train.csv'
    frame = pd.read_csv(path)
    frame['origin_y'] += 10
    frame.to_csv(path, index=False)
    plan_path = series/'n4/comparison_plan.json'
    plan = json.loads(plan_path.read_text())
    plan['dataset']['train.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(plan_path, plan)
    loaded = study.load_study(series)
    extension = {**loaded, 'cases': {'n4': loaded['cases']['n4']},
                 'runs': {('n4', name): loaded['runs']['n4', name] for name in ('future', 'joint')}}
    with pytest.raises(ValueError, match='every reference clip'):
        study.extension_study(reference, extension, reference_case='n2')


def test_unified_snapshot_preserves_runs_and_rejects_incompatible_data(series):
    loaded = study.load_study(series)
    parts = [{**loaded, 'cases': {case: loaded['cases'][case]},
              'runs': {key: run for key, run in loaded['runs'].items() if key[0] == case}}
             for case in ('n2', 'n4')]
    combined = study.merge_studies(*parts)
    assert set(combined['runs']) == set(loaded['runs'])
    assert list(combined['cases']) == ['n2', 'n4']
    with pytest.raises(ValueError, match='unique across studies'):
        study.merge_studies(parts[0], parts[0])
    with pytest.raises(ValueError, match='same validation'):
        study.merge_studies(parts[0], {**parts[1], 'validation_sha256': 'changed'})
    manifest = series/'n4/dataset/train.csv'
    frame = pd.read_csv(manifest)
    frame['origin_x'] = 100
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='every smaller-set'):
        study.merge_studies(*parts)


def test_joint_curves_use_weighted_total_including_reconstruction(series):
    snapshot = study.load_study(series)
    run = snapshot['runs']['n2', 'joint']
    run['loss_weights'] = {'mvm': 2., 'future': .5}
    run['epochs'] = [dict(epoch=1, optimizer_steps=1, train_mvm=3., train_future=2.,
                         train_total=7., validation_mvm=5., validation_future=4., validation_score=4.)]
    history = study.native_loss_history(snapshot, run)
    assert history.train_objective_k2.iloc[0] == 28.
    assert history.validation_objective_k2.iloc[0] == 48.
    fig = study.learning_figure(snapshot)
    try:
        for axis in fig.axes:
            labels = [line.get_label() for line in axis.lines]
            assert labels.count('Joint · 2') == 1
            assert labels.count('Joint · 4') == 1
            assert len(labels) == 8
        joint = next(line for line in fig.axes[2].lines if line.get_label() == 'Joint · 2')
        np.testing.assert_allclose(joint.get_ydata(), [48.])
    finally:
        plt.close(fig)


def test_size_comparison_matches_only_objectives_shared_by_each_pair(series):
    snapshot = study.load_study(series)
    # A third size has only Future and a longer run; unrelated partial Joint
    # results at the middle size must not restrict this Future-only pair.
    snapshot['cases']['n8'] = {**snapshot['cases']['n4'], 'train_samples': 8, 'encoder': 'Base'}
    snapshot['runs']['n8', 'future'] = {**snapshot['runs']['n4', 'future'], 'case': 'n8',
        'epochs': [{**row, 'optimizer_steps': row['epoch']*4} for row in snapshot['runs']['n4', 'future']['epochs']]}
    table = study.size_comparison(snapshot)
    rows = table[(table['Smaller training clips'] == 4) & (table['Larger training clips'] == 8)]
    assert set(rows.Model) == {'Future'}
    matched = rows[rows['Matched budget'] == 'Optimizer updates'].iloc[0]
    assert matched.Budget == 8 and matched['Small-set epoch'] == 4 and matched['Large-set epoch'] == 2
    assert matched['Encoder comparison'] == 'Small → Base'


def test_channel_aggregation_pools_squared_errors_and_requires_complete_channels(series):
    snapshot = study.load_study(series)
    selected, _ = study.select_budget(snapshot)
    scores = study.metrics_frame(snapshot, selected)
    chosen = (scores.case == 'n2') & (scores.model == 'future') & (scores.horizon_minutes == 15)
    squared_errors = np.array([1., 1., 1., 9., 9., 9., 9.])
    scores.loc[chosen, 'mse_k2'] = squared_errors
    scores.loc[chosen, 'rmse_k'] = np.sqrt(squared_errors)
    scores.loc[chosen, 'mae_k'] = np.sqrt(squared_errors)
    result = study.aggregate_channels(scores)
    row = result[(result.case == 'n2') & (result.model == 'future') &
                 (result.horizon_minutes == 15) & (result.channel == 'All channels')].iloc[0]
    assert row.rmse_k == pytest.approx(np.sqrt(squared_errors.mean()))
    assert row.rmse_k != pytest.approx(np.sqrt(squared_errors).mean())
    assert row.mae_k == pytest.approx(np.sqrt(squared_errors).mean())
    with pytest.raises(ValueError, match='each thermal channel exactly once'):
        study.aggregate_channels(scores.drop(scores[chosen].index[0]))


def test_channel_training_comparison_preserves_missing_runs_and_shared_references(series):
    snapshot = study.load_study(series)
    selected, _ = study.select_budget(snapshot)
    scores = study.metrics_frame(snapshot, selected)
    scores = scores[~((scores.case == 'n4') & (scores.model == 'residual'))]
    fig = study.channel_training_figure(scores, channels=['All channels', 'IR_108'])
    try:
        assert len(fig.axes) == 4
        for axis in fig.axes:
            # Two sizes for Future and Joint, and one actual Residual run.
            assert len(axis.patches) == 5
            assert len(axis.lines) == 2
        for first, second in zip(fig.axes[::2], fig.axes[1::2]):
            assert first.get_ylim() == second.get_ylim()
        fig.canvas.draw()
    finally:
        plt.close(fig)
    changed = scores.copy()
    changed.loc[(changed.case == 'n4') & (changed.model == 'advection'), 'rmse_k'] += 1
    with pytest.raises(ValueError, match='Reference forecasts differ'):
        try:
            study.channel_training_figure(changed, channels=['IR_108'])
        finally:
            plt.close('all')
