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


def test_complementary_objectives_require_identical_case_data(series):
    snapshot = study.load_study(series)
    reference = {**snapshot, 'runs': {key:run for key,run in snapshot['runs'].items() if key[1] != 'mvm'}}
    addition = {**snapshot, 'runs': {key:run for key,run in snapshot['runs'].items() if key[1] == 'mvm'}}
    combined = study.add_completed_experiments(reference,addition)
    assert set(combined['runs']) == set(snapshot['runs'])
    with pytest.raises(ValueError,match='already exists'):
        study.add_completed_experiments(snapshot,addition)
    with pytest.raises(ValueError,match='same validation'):
        study.add_completed_experiments(reference,{**addition,'validation_sha256':'changed'})
    alias = {**addition, 'cases': {'renamed': addition['cases']['n4']},
             'runs': {('renamed','mvm'): {**addition['runs']['n4','mvm'], 'case':'renamed'}}}
    combined = study.add_completed_experiments(reference,alias)
    assert 'renamed' not in combined['cases']
    assert combined['runs']['n4','mvm']['case'] == 'n4'


def test_distributed_history_uses_configured_budget_and_physical_scaling(series,tmp_path):
    snapshot = study.load_study(series)
    source = tmp_path/'hpc'
    case = snapshot['cases']['n4']
    identity = dict(validation_sha256=snapshot['validation_sha256'],
        train_sha256=hashlib.sha256((case['queue']/'dataset/train.csv').read_bytes()).hexdigest(),
        model_config={'hidden_size':384},stats={**case['stats'],'samples':4},objective='future',batch_size=608,world_size=4)
    write(source/'run_config.json',dict(identity=identity,arguments={'epochs':2}))
    write(source/'progress.json',dict(status='completed',completed_epochs=2,epochs=[
        dict(epoch=e,optimizer_steps=e*2,train_loss=.25,validation_score=.5) for e in (1,2)]))
    snapshot['runs'].pop(('n4','future'))
    result = study.attach_distributed_training(snapshot,source)
    run = result['runs']['n4','future']
    assert run['complete'] and run['stage']['epochs']==2
    assert run['epochs'][0]['validation_future']==.5
    assert run['execution']==dict(site='Leonardo',global_batch=608,world_size=4)
    with pytest.raises(ValueError,match='replace'):
        study.attach_distributed_training(result,source)
    identity['train_sha256']='changed'
    write(source/'run_config.json',dict(identity=identity,arguments={'epochs':2}))
    with pytest.raises(ValueError,match='manifest differs'):
        study.attach_distributed_training(snapshot,source)


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
        joint = next(line for line in fig.axes[1].lines if line.get_label() == 'Joint · 2')
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


def test_mixed_architecture_plan_uses_case_model_size(series):
    path = series/'scaling_plan.json'
    plan = json.loads(path.read_text())
    plan['cases'][1]['model_size'] = 'base'
    write(path, plan)
    snapshot = study.load_study(series)
    assert snapshot['cases']['n2']['encoder'] == 'Small'
    assert snapshot['cases']['n4']['encoder'] == 'Base'


def test_architectures_can_share_exactly_the_same_training_set(series,tmp_path):
    original = study.load_study(series)
    shared = {**original, 'cases': {'base_n2': {**original['cases']['n2'], 'encoder': 'Base'}},
              'runs': {('base_n2',name): {**run,'case':'base_n2'}
                       for (case,name),run in original['runs'].items() if case == 'n2'}}
    combined = study.merge_studies(original,shared)
    assert len(combined['cases']) == 3
    assert combined['cases']['base_n2']['encoder'] == 'Base'
    # Matching counts alone are insufficient: equal-size cases must share clips.
    other = tmp_path/'other'
    (other/'dataset').mkdir(parents=True)
    frame = pd.read_csv(series/'n2/dataset/train.csv')
    frame['origin_y'] += 100
    frame.to_csv(other/'dataset/train.csv',index=False)
    shared['cases']['base_n2']['queue'] = other
    with pytest.raises(ValueError,match='every smaller-set'):
        study.merge_studies(original,shared)


def test_objective_panels_preserve_architecture_size_and_partial_history(series):
    snapshot = study.load_study(series)
    # Five sizes and two architectures exceed the old four-style limit.
    base = snapshot['runs']['n2','future']
    snapshot['cases'] = {};snapshot['runs'] = {}
    for size in [4096,8192,16384,32768,65536]:
        for encoder in ['Small','Base']:
            case=f'{encoder}_{size}'
            snapshot['cases'][case]=dict(train_samples=size,encoder=encoder,stats={'global_std_k':2.})
            for objective in ['future','joint']:
                snapshot['runs'][case,objective]={**base,'case':case,'name':objective,
                    'complete':(size,encoder,objective)!=(65536,'Base','joint'),
                    'epochs':[dict(epoch=1,optimizer_steps=size//32,train_mvm=3.,train_future=2.,
                        train_total=5. if objective=='joint' else 2.,validation_mvm=5.,validation_future=4.)]}
    figures=study.objective_learning_figures(snapshot)
    assert set(figures)=={'future','joint'}
    try:
        for objective,fig in figures.items():
            assert len(fig.axes)==2
            for axis in fig.axes:
                assert len(axis.lines)==10
                assert axis.get_xscale()==axis.get_yscale()=='log'
            lines={line.get_label():line for line in fig.axes[0].lines}
            for size in [4096,8192,16384,32768,65536]:
                small,large=lines[f'Small · {size:,} clips'],lines[f'Base · {size:,} clips']
                assert small.get_color()==large.get_color()
                assert small.get_linestyle()=='-' and large.get_linestyle()=='--'
            val=next(line for line in fig.axes[1].lines
                     if line.get_label()=='Base · 65,536 clips')
            np.testing.assert_allclose(val.get_ydata(),[36. if objective=='joint' else 16.])
            assert val.get_marker()==('o' if objective=='joint' else 'None')
            assert val.get_xdata().tolist()==[65536//32]
    finally:
        for fig in figures.values():plt.close(fig)


def test_learning_overlay_preserves_each_validation_population_and_temperature_scale(series):
    historical = study.load_study(series)
    source_run = historical['runs']['n4', 'future']
    geographic = {**historical, 'validation_sha256': 'resampled-manifest',
        'cases': {'base_n131072': dict(train_samples=131072, encoder='Base', stats={'global_std_k': 10.})},
        'runs': {('base_n131072', 'future'): {**source_run, 'case': 'base_n131072',
            'complete': True, 'epochs': [dict(epoch=1, optimizer_steps=216, train_future=.2,
                                            train_total=.2, validation_future=.1)]}}}
    with pytest.raises(ValueError, match='same validation'):
        study.merge_studies(historical, geographic)
    figures = study.objective_learning_figures(historical,
        additional_studies={'Resampled validation · 512 clips': geographic})
    try:
        axes = figures['future'].axes
        labels = [line.get_label() for line in axes[1].lines]
        assert len(labels) == 3
        extra = next(line for line in axes[1].lines if '131,072' in line.get_label())
        np.testing.assert_allclose(extra.get_ydata(), [10.])
        np.testing.assert_allclose(extra.get_xdata(), [216])
        assert extra.get_marker() == 'D' and extra.get_linestyle() == '--'
        old = next(line for line in axes[1].lines if '2 clips' in line.get_label())
        np.testing.assert_allclose(old.get_ydata(), [5., 20., 45., 80.])
        legend = [item.get_text() for item in figures['future'].legends[0].get_texts()]
        assert 'Resampled validation · 512 clips' in legend
        assert 'Historical validation · 512 clips' in legend
        assert len(figures['mvm'].axes[1].lines) == 2
        figures['future'].canvas.draw()
    finally:
        for fig in figures.values():
            plt.close(fig)
    assert historical['validation_sha256'] != geographic['validation_sha256']
    assert ('base_n131072', 'future') not in historical['runs']


def test_extended_validation_overlay_uses_physical_mse_and_preserves_all_points(series, tmp_path):
    snapshot = study.load_study(series)
    directory = tmp_path/'distributed'
    write(directory/'run_config.json', dict(identity=dict(objective='future', stats={'global_std_k': 10.})))
    write(directory/'progress.json', dict(completed_epochs=2, epochs=[
        dict(epoch=1, optimizer_steps=216), dict(epoch=2, optimizer_steps=432)]))
    for epoch, mse, rmse in [(1, .09, 3.), (2, .04, 2.), (3, .01, 1.)]:
        write(directory/'validation_extended'/f'epoch_{epoch:03d}.json',
              dict(neural=dict(samples=10240, mse=mse, rmse_k=rmse)))
    history = study.distributed_validation_history(directory)
    assert history.epoch.tolist() == [1, 2]
    assert history.optimizer_steps.tolist() == [216, 432]
    np.testing.assert_allclose(history.validation_objective_k2, [9., 4.])
    assert history.validation_clips.tolist() == [10240, 10240]
    figures = study.objective_learning_figures(snapshot, validation_zoom=True,
        extra_validation=[dict(objective='future', encoder='Small', training_clips=4,
                               label='Small · 4 · 10,240 validation clips', history=history)])
    try:
        training, validation, detail = figures['future'].axes
        assert len(training.lines) == 2  # A second evaluation never duplicates training loss.
        assert len(validation.lines) == len(detail.lines) == 3
        assert validation.get_yscale() == 'log' and detail.get_yscale() == 'linear'
        for axis in (validation, detail):
            extra = next(line for line in axis.lines if '10,240' in line.get_label())
            np.testing.assert_allclose(extra.get_xdata(), [216, 432])
            np.testing.assert_allclose(extra.get_ydata(), [9., 4.])
        for full, zoom in zip(validation.lines, detail.lines):
            np.testing.assert_array_equal(full.get_xdata(), zoom.get_xdata())
            np.testing.assert_array_equal(full.get_ydata(), zoom.get_ydata())
        assert len(figures['mvm'].axes[1].lines) == 2
        figures['future'].canvas.draw()
    finally:
        for figure in figures.values():
            plt.close(figure)
    write(directory/'validation_extended/epoch_002.json', dict(neural=dict(samples=10240, mse=.04, rmse_k=9.)))
    with pytest.raises(ValueError, match='inconsistent'):
        study.distributed_validation_history(directory)


def test_global_dataset_budgets_use_fixed_references_and_actual_saved_epochs(series):
    snapshot = study.load_study(series)
    for encoder, size in [('Small', 8), ('Base', 4), ('Base', 8)]:
        case = f'{encoder}_{size}'
        snapshot['cases'][case] = {**snapshot['cases']['n4'], 'encoder': encoder, 'train_samples': size}
        for model in ('future', 'joint'):
            source = snapshot['runs']['n4', model]
            snapshot['runs'][case, model] = {**source, 'case': case, 'completed_epochs': 2,
                'epochs': [{**row, 'optimizer_steps': row['epoch']*size//2} for row in source['epochs'][:2]]}
    frame = study.dataset_budget_comparison(snapshot)
    assert len(frame) == 24
    epochs = frame[frame['Matched budget'] == 'Epochs']
    updates = frame[frame['Matched budget'] == 'Optimizer updates']
    assert set(epochs.Budget) == {2} and set(epochs.Epoch) == {2}
    assert set(updates.Budget) == {4} and set(updates['Clip presentations']) == {8}
    assert set(updates.loc[updates['Training clips'] == 8, 'Epoch']) == {1}
    assert set(frame.loc[frame.Encoder == 'Small', 'Reference clips']) == {2}
    assert set(frame.loc[frame.Encoder == 'Base', 'Reference clips']) == {4}
    small = updates[(updates.Encoder == 'Small') & (updates['Training clips'] == 8)]
    np.testing.assert_allclose(small['RMSE reduction (K)'], 3*np.sqrt(5))
    base = updates[(updates.Encoder == 'Base') & (updates['Training clips'] == 8)]
    np.testing.assert_allclose(base['RMSE reduction (K)'], np.sqrt(5))
    assert not ((frame['Training clips'] == 8) & (frame.Model == 'Residual')).any()
    reference = frame[frame['Training clips'] == frame['Reference clips']]
    assert (reference['RMSE reduction (K)'] == 0).all()


def test_dataset_budgets_require_exact_common_saved_reports(series):
    snapshot = study.load_study(series)
    for (case, _), run in snapshot['runs'].items():
        if case == 'n2':
            run['epochs'] = [row for row in run['epochs'] if row['optimizer_steps'] % 2 == 1]
    frame = study.dataset_budget_comparison(snapshot)
    assert set(frame['Matched budget']) == {'Epochs'}
    assert set(frame.Budget) == {3}
    # A common saved history entry still requires its committed validation report.
    (series/'n2/future/validation_epochs/epoch_003.json').unlink()
    snapshot['reports'].clear()
    with pytest.raises(FileNotFoundError):
        study.dataset_budget_comparison(snapshot)


def test_differences_align_exact_updates_and_use_all_past_minima(series):
    snapshot = study.load_study(series)
    snapshot['runs'] = {key: run for key, run in snapshot['runs'].items() if key[1] == 'future'}
    snapshot['cases']['n4']['stats'] = {'global_std_k': 3.}
    for case, updates, values in [('n2', [1, 2, 3, 4, 5], [10., 8., 1., 9., .01]),
                                  ('n4', [2, 4, 6], [2., 3., 1.])]:
        run = snapshot['runs'][case, 'future']
        run['epochs'] = [dict(epoch=i+1, optimizer_steps=u, train_future=v, validation_future=v)
                         for i, (u, v) in enumerate(zip(updates, values))]
    frame = study.learning_differences(snapshot, 'dataset', reference_sizes={'Small': 2})
    assert frame.optimizer_steps.tolist() == [2, 4]
    assert frame.epoch_candidate.tolist() == [1, 2]
    assert frame.epoch_reference.tolist() == [2, 4]
    np.testing.assert_allclose(frame.validation_difference_k2, [-14., -9.])
    # The reference minimum at update 3 contributes at update 4, before alignment.
    # Its still lower loss at update 5 cannot affect either earlier comparison.
    np.testing.assert_allclose(frame.best_difference_k2, [-14., 14.])
    figures = study.learning_difference_figures(frame)
    try:
        assert len(figures['future'].axes) == 2
        for axis in figures['future'].axes:
            curve = next(line for line in axis.lines if line.get_label() == '4 − 2 clips')
            assert curve.get_xdata().tolist() == [2, 4]
            assert axis.get_xscale() == 'log' and axis.get_yscale() == 'linear'
        figures['future'].canvas.draw()
    finally:
        for fig in figures.values(): plt.close(fig)


def test_architecture_differences_hold_dataset_and_objective_fixed(series):
    snapshot = study.load_study(series)
    snapshot['cases']['base_n2'] = {**snapshot['cases']['n2'], 'encoder': 'Base'}
    source = snapshot['runs']['n2', 'future']
    snapshot['runs']['base_n2', 'future'] = {**source, 'case': 'base_n2',
        'epochs': [{**row, 'validation_future': row['validation_future']*.5} for row in source['epochs']]}
    frame = study.learning_differences(snapshot, 'architecture')
    assert set(frame.objective) == {'future'}
    assert set(frame.training_clips) == set(frame.reference_clips) == {2}
    assert set(frame.encoder) == {'Base'} and set(frame.reference_encoder) == {'Small'}
    assert (frame.validation_difference_k2 < 0).all()
    datasets = study.learning_differences(snapshot, 'dataset', reference_sizes={'Small': 2, 'Base': 2})
    assert (datasets.encoder == datasets.reference_encoder).all()
    assert set(datasets.encoder) == {'Small'}
    snapshot['runs']['base_n2', 'future']['loss_weights'] = {'mvm': 1., 'future': 2.}
    with pytest.raises(ValueError, match='same objective weights'):
        study.learning_differences(snapshot, 'architecture')


def test_dataset_differences_use_smallest_available_set_for_each_encoder(series):
    snapshot = study.load_study(series)
    for size in (2, 4):
        name = f'base_n{size}'
        snapshot['cases'][name] = {**snapshot['cases'][f'n{size}'], 'encoder': 'Base'}
        snapshot['runs'][name, 'future'] = {**snapshot['runs'][f'n{size}', 'future'], 'case': name}
    frame = study.learning_differences(snapshot, 'dataset')
    assert set(frame.encoder) == {'Small', 'Base'}
    assert set(frame.training_clips) == {4}
    assert set(frame.reference_clips) == {2}
    assert (frame.encoder == frame.reference_encoder).all()
    # A larger explicit reference remains supported for historical views.
    explicit = study.learning_differences(snapshot, 'dataset', reference_sizes={'Small': 2, 'Base': 4})
    assert set(explicit.encoder) == {'Small'}


def test_validation_minimum_uses_joint_objective_and_earliest_tie(series):
    snapshot = study.load_study(series)
    run = snapshot['runs']['n2', 'joint']
    run['epochs'] = [dict(epoch=i+1, optimizer_steps=i+1, train_mvm=m, train_future=f,
                         validation_mvm=m, validation_future=f)
                     for i, (m, f) in enumerate([(1., 10.), (10., 2.), (1., 10.)])]
    snapshot['runs'] = {('n2', 'joint'): run}
    minima = study.objective_minima(snapshot).iloc[0]
    assert minima['Minimum validation loss (K²)'] == 44.
    assert minima['Updates at minimum'] == minima['Epoch of minimum'] == 1
    assert minima['Last saved updates'] == 3
