import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import numpy as np

from scripts import prepare_seviri_spaced_hpc as prepare
from scripts.transfer_seviri_spaced_hpc import REMOTE_VERIFY


@pytest.mark.parametrize('samples,updates,benchmark_steps',[(35000,58,32),(10000,17,12)])
def test_spaced_hpc_freezes_actual_update_budget_and_shared_clip_ram(tmp_path, monkeypatch,samples,updates,benchmark_steps):
    root = tmp_path/'repo'
    for name in ('src', 'scripts', 'configs'):
        (root/name).mkdir(parents=True)
        (root/name/'source.txt').write_text('frozen')
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(prepare.subprocess, 'check_output', lambda *a, **k: 'test-commit\n')
    data = tmp_path/'data'
    data.mkdir()
    (data/'dataset_ready.json').write_text(json.dumps(dict(status='completed', files={},
        cache_gib=200, train_samples=samples)))
    sampling = data/'sampling_ready.json'
    sampling.write_text(json.dumps(dict(max_iou_active=.2, measured_maximum_active_iou=.1999)))
    (data/'preparation_plan.json').write_text('{}')
    experiment = tmp_path/'experiment.json'
    experiment.write_text(json.dumps(dict(initialization='scratch', global_batch_size=608,
        batch_size_per_gpu=152, world_size=4, epochs=250)))
    args = SimpleNamespace(dataset=data, output=tmp_path/'run', experiment=experiment,
        remote_dataset=tmp_path/'remote-data', remote_run=tmp_path/'remote-run')
    prepare.prepare(args)
    plan = json.loads((args.output/'full_training.json').read_text())
    assert plan['updates_per_epoch'] == updates
    assert plan['planned_total_updates'] == updates*250
    assert plan['pipeline_benchmark_steps'] == benchmark_steps
    assert plan['pipeline_benchmark_steps']+plan['pipeline_benchmark_warmup'] <= samples//608
    locations = json.loads((args.output/'data_locations.json').read_text())
    assert locations['ram_backend'] == 'tmpfs'
    assert locations['canonical_clip_root'] == str(data/'clips')
    assert 'canonical_frame_root' not in locations
    assert json.loads((args.output/'submission.json').read_text())['status'] == 'prepared'
    sampling.write_text(json.dumps(dict(max_iou_active=.2, measured_maximum_active_iou=.21)))
    with pytest.raises(ValueError, match='constraint'):
        prepare.prepare(args)


def test_new_clip_ingestion_verifies_destination_bytes_once_before_publication(tmp_path):
    (tmp_path/'clips').mkdir()
    target = tmp_path/'clips/example.blosc'
    target.write_bytes(b'encoded test data')
    expected = dict(bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (tmp_path/'transfer_plan.json').write_text(json.dumps(dict(clips={target.name:expected}, payload={})))
    target.write_bytes(b'corrupt test data')
    result = subprocess.run([sys.executable, '-', str(tmp_path)], input=REMOTE_VERIFY,
        text=True, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path/'verification.json').exists()
    target.write_bytes(b'encoded test data')
    subprocess.run([sys.executable, '-', str(tmp_path)], input=REMOTE_VERIFY, text=True, check=True, capture_output=True)
    assert json.loads((tmp_path/'verification.json').read_text())['status'] == 'verified'


def test_rank_alignment_removes_surplus_and_rebinds_statistics_and_metadata(tmp_path):
    from scripts.prepare_seviri_spaced_validation import align_training_ranks
    for directory in ('training/dataset', 'clips', 'daily'):
        (tmp_path/directory).mkdir(parents=True)
    rows = [dict(day=f'2022-{month:02d}-01', start_time=f'2022-{month:02d}-01T00:{i*5:02d}:00Z',
        time_bin=0, origin_y=i, origin_x=i, clip_id=f'{month}_{i}', file=f'{month}_{i}.blosc',
        mean_k=float(250+i), variance_k2=4., store=f'2022/2022{month:02d}01.zarr')
        for month in (1, 2) for i in range(5)]
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path/'training_crop_moments.csv', index=False)
    manifest = tmp_path/'training/dataset/train.csv'
    frame.drop(columns=['file', 'mean_k', 'variance_k2']).to_csv(manifest, index=False)
    old_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    def write(name, data):
        (tmp_path/name).write_text(json.dumps(data))
    write('clips/metadata.json', dict(clips={row['file']:dict(bytes=100) for row in rows},
        manifests={old_hash:dict(files=frame.file.tolist())}))
    write('sampling_ready.json', dict(train_samples=10))
    write('training/dataset/channel_stats.json', dict(samples=10))
    write('training/dataset/protocol.json', dict(train_samples=10))
    for day in frame.day.unique():
        write('daily/'+day+'.json', dict(shape=[288, 7, 763, 1762]))
    align_training_ranks(tmp_path)
    selected = pd.read_csv(tmp_path/'training_crop_moments.csv')
    assert len(selected) == 8
    assert selected.groupby('day').size().tolist() == [4, 4]
    stats = json.loads((tmp_path/'training/dataset/channel_stats.json').read_text())
    assert stats['samples'] == 8 and stats['pixel_count'] == 8*16*7*224*224
    assert stats['global_mean_k'] == pytest.approx(selected.mean_k.mean())
    assert stats['global_std_k'] == pytest.approx(np.sqrt(np.mean(selected.variance_k2 +
        (selected.mean_k-selected.mean_k.mean())**2)))
    metadata = json.loads((tmp_path/'clips/metadata.json').read_text())
    assert set(metadata['clips']) == set(selected.file)
    assert old_hash not in metadata['manifests']
    assert set(metadata['manifests']) == {hashlib.sha256(manifest.read_bytes()).hexdigest()}
    align_training_ranks(tmp_path)  # Already aligned; no further edits.
    assert len(pd.read_csv(manifest)) == 8


def test_fixed_validation_reuse_preserves_files_and_rejects_changed_manifest(tmp_path):
    import os
    from scripts.prepare_seviri_spaced_validation import reuse_validation
    from emma_gpm.seviri_clip_files import clip_key
    source,reference,output = (tmp_path/name for name in ('source','reference','output'))
    for root in (source,reference,output):
        for directory in ('training/dataset','extended/dataset','clips','validation_clips'):
            (root/directory).mkdir(parents=True)
    frame = pd.DataFrame([dict(start_time='2022-07-01T00:00Z',origin_y=0,origin_x=0,
                               day='2022-07-01',store='2022/20220701.zarr')])
    for root in (source,reference):
        for role in ('training','extended'):
            frame.to_csv(root/role/'dataset/validation.csv',index=False)
    key=clip_key(frame.iloc[0].start_time,0,0)
    name=hashlib.sha256(key.encode()).hexdigest()[:24]+'.blosc'
    payload=b'preexisting immutable clip'
    (source/'clips'/name).write_bytes(payload)
    record=dict(key=key,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    (source/'clips/metadata.json').write_text(json.dumps(dict(clips={name:record})))
    (source/'dataset_ready.json').write_text(json.dumps(dict(status='completed')))
    result=reuse_validation(output,reference,source)
    assert result[0]['clips']=={name:record}
    assert os.path.samefile(source/'clips'/name,output/'validation_clips'/name)
    (reference/'extended/dataset/validation.csv').write_text('changed')
    with pytest.raises(ValueError,match='holdout'):
        reuse_validation(output,reference,source)
