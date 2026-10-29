import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_seviri_ssl_new_dates import freeze_store, rebind_existing_cache
from scripts.pretrain_seviri_ssl import digest
from scripts.run_seviri_ssl_matrix import make_plans, DATA_FILES
from scripts.status_seviri_ssl import collect_status, format_report


def matrix_config(tmp_path):
    raw = tmp_path/'raw'
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'metadata.json').write_text('{}')
    config = dict(epochs=150, batch_size=4, workers=1, seed=20261024, cases=[])
    for model, size in [('small',16),('base',8),('small',32),('base',32)]:
        dataset = tmp_path/f'dataset_{size}'
        if not dataset.exists():
            dataset.mkdir()
            train = pd.DataFrame([dict(start_time='2020-09-12T00:00:00Z', origin_y=i, origin_x=0) for i in range(size)])
            validation = train.iloc[:2].assign(start_time='2020-09-17T00:00:00Z')
            train.to_csv(dataset/'train.csv', index=False)
            validation.to_csv(dataset/'validation.csv', index=False)
            (dataset/'channel_stats.json').write_text(json.dumps(dict(zarr_root=str(raw),
                manifest_sha256=digest(dataset/'train.csv'), global_mean_k=260., global_std_k=20.)))
            for name in ('source_snapshot.json','protocol.json','inventory.json'):
                (dataset/name).write_text('{}')
        config['cases'].append(dict(name=f'{model}_n{size}', model_size=model, train_samples=size,
            dataset=str(dataset), zarr_root=str(raw), frame_cache=str(cache)))
    config['variability_reference_manifest'] = str(tmp_path/'dataset_8/train.csv')
    return config


def test_matrix_preserves_architectures_common_data_and_all_eight_fresh_runs(tmp_path):
    config = matrix_config(tmp_path)
    plan, local = make_plans(tmp_path/'series', config)
    assert len(plan['cases']) == 4 and len(plan['order']) == 8
    assert plan['strict_nesting'] and plan['validation_identical']
    assert {p['files']['validation.csv'] for p in plan['cases']} == {digest(tmp_path/'dataset_8/validation.csv')}
    for case in plan['cases']:
        assert case['planned_optimizer_updates'] == 150*case['train_samples']//4
        assert {s['name'] for s in local[case['name']]['stages']} == {'future','joint'}
        for stage in local[case['name']]['stages']:
            flags = stage['flags']
            assert flags[flags.index('--model-size')+1] == case['model_size']
            assert flags[flags.index('--gradient-accumulation')+1] == '1'
            assert '--encoder-from' not in flags and '--resume' not in flags
    assert make_plans(tmp_path/'series', config) == (plan, local)


def test_matrix_rejects_changed_holdout_and_non_nested_data(tmp_path):
    config = matrix_config(tmp_path)
    val = tmp_path/'dataset_32/validation.csv'
    val.write_text(val.read_text().replace('09-17','09-18'))
    with pytest.raises(ValueError, match='identical validation'):
        make_plans(tmp_path/'bad_validation', config)
    val.write_bytes((tmp_path/'dataset_8/validation.csv').read_bytes())
    train = tmp_path/'dataset_16/train.csv'
    frame = pd.read_csv(train);frame.loc[0,'origin_y'] = 100
    frame.to_csv(train, index=False)
    stats = tmp_path/'dataset_16/channel_stats.json'
    value = json.loads(stats.read_text());value['manifest_sha256'] = digest(train)
    stats.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='strict nested'):
        make_plans(tmp_path/'bad_nesting', config)


def test_matrix_rejects_changed_frozen_copy(tmp_path):
    config = matrix_config(tmp_path)
    output = tmp_path/'series'
    make_plans(output, config)
    (output/'small_n32/dataset/train.csv').write_text('changed')
    with pytest.raises(ValueError, match='differs from its source'):
        make_plans(output, config)


def test_freeze_copies_physical_files_and_detects_later_source_changes(tmp_path):
    upload, frozen = tmp_path/'upload', tmp_path/'frozen'
    header = '2010/day.zarr/.zmetadata'; chunk = '2010/day.zarr/seviri/0.0.0.0'
    for name, value in [(header,b'header'),(chunk,b'radiance')]:
        p = upload/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(value)
    record = dict(store='2010/day.zarr', headers={header:digest(upload/header)}, present=['0.0.0.0'])
    hashes = freeze_store(upload, frozen, record)
    assert set(hashes) == {header,chunk}
    assert (upload/chunk).stat().st_ino != (frozen/chunk).stat().st_ino
    (upload/chunk).write_bytes(b'changed')
    assert (frozen/chunk).read_bytes() == b'radiance'
    with pytest.raises(ValueError, match='destination differs'):
        freeze_store(upload, frozen, record)


def test_rebinding_rejects_a_different_raw_archive(tmp_path):
    reuse = tmp_path/'reuse';reuse.mkdir()
    (reuse/'metadata.json').write_text(json.dumps(dict(zarr_root=str(tmp_path/'old'), stores={'2020/day.zarr':{}})))
    with pytest.raises(ValueError, match='identical historical source'):
        rebind_existing_cache(reuse, tmp_path/'new_cache', tmp_path/'new_raw')


def test_monitor_distinguishes_same_size_encoders(tmp_path):
    config = matrix_config(tmp_path)
    plan, local = make_plans(tmp_path/'series', config)
    output = tmp_path/'series'
    (output/'scaling_plan.json').write_text(json.dumps(plan))
    for case, value in local.items():
        (output/case/'comparison_plan.json').write_text(json.dumps(value))
    report = collect_status(output, process_probe=lambda *args:dict(pid=None,state='absent'))
    same_size = [r for r in report['rows'] if r['training_clips'] == 32]
    assert [r['model_size'] for r in same_size] == ['small','small','base','base']
    formatted = format_report(report)
    assert 'Small' in formatted and 'Base' in formatted and '0/8' in formatted
