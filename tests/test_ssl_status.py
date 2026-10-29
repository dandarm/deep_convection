import json
import os
from pathlib import Path
import sys

import pytest

from scripts.status_seviri_ssl import collect_status, format_report, inspect_process


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def series(tmp_path):
    write(tmp_path/'scaling_plan.json', dict(cases=[dict(name='n4096', train_samples=4096)],
        order=[dict(case='n4096', stage=name) for name in ('future', 'mvm', 'joint')]))
    write(tmp_path/'n4096/comparison_plan.json', dict(stages=[dict(name=name, epochs=10)
        for name in ('future', 'mvm', 'joint')]))
    write(tmp_path/'n4096/future/progress.json', dict(status='completed', completed_epochs=10))
    write(tmp_path/'n4096/mvm/progress.json', dict(status='training', completed_epochs=3,
        current_epoch=4, current_step=64, steps_per_epoch=128,
        execution_history=[dict(after_epoch=1)], epochs=[
            dict(epoch=1, train_seconds=100, validation_seconds=10),
            dict(epoch=2, train_seconds=20, validation_seconds=5),
            dict(epoch=3, train_seconds=20, validation_seconds=5)]))
    write(tmp_path/'n4096/mvm/timing_epochs/epoch_003.json', dict(epoch_wall_seconds=27))
    write(tmp_path/'queue_status.json', dict(status='running', case='n4096', stage='mvm',
        pid=101, supervisor_pid=100, command=['python', 'trainer.py']))
    write(tmp_path/'launcher.json', dict(pid=100, command=['python', 'supervisor.py']))
    return tmp_path


def probe(state):
    return lambda pid, command: dict(pid=pid, state=state)


def test_progress_budget_recent_runtime_and_terminal_report(series):
    report = collect_status(series, process_probe=probe('active'))
    assert report['completed_runs'] == 1
    assert report['completed_epochs'] == 13
    assert report['planned_epochs'] == 30
    assert report['state'] == 'in esecuzione'
    assert report['current']['epoch_seconds'] == 26
    assert report['current']['remaining_seconds_estimate'] == 172
    text = format_report(report)
    assert 'batch 64/128' in text
    assert '1/3' in text and '13/30' in text
    assert report['rows'][-1]['state'] == 'in attesa'


@pytest.mark.parametrize('state', ['absent', 'different'])
def test_stale_running_file_does_not_imply_an_active_training(series, state):
    report = collect_status(series, process_probe=probe(state))
    assert report['state'] == 'ferma'
    assert report['rows'][1]['state'] == 'fermo'
    assert 'remaining_seconds_estimate' not in report['current']


def test_paused_process_and_old_progress_are_distinguished(series):
    assert collect_status(series, process_probe=probe('paused'))['state'] == 'sospesa'
    path = series/'n4096/mvm/progress.json'
    os.utime(path, (0, 0))
    report = collect_status(series, now=1000, process_probe=probe('active'), stale_after=180)
    assert report['state'] == 'processo presente; avanzamento da verificare'
    assert 'remaining_seconds_estimate' not in report['current']


def test_analysis_continues_after_all_training_epochs_are_complete(series):
    for name in ('mvm', 'joint'):
        write(series/f'n4096/{name}/progress.json', dict(status='completed', completed_epochs=10))
    write(series/'queue_status.json', dict(status='running', stage='analysis', case='n4096',
        pid=101, supervisor_pid=100, command=['python', 'analysis.py']))
    report = collect_status(series, process_probe=probe('active'))
    assert report['completed_runs'] == 3 and report['epoch_percent'] == 100
    assert report['state'] == 'analisi in esecuzione'
    assert 'analisi finale n4096' in format_report(report)
    write(series/'queue_status.json', dict(status='completed'))
    assert collect_status(series, process_probe=probe('absent'))['state'] == 'completata'


@pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='Linux process inspection')
def test_process_identity_rejects_reused_pid():
    command = [part.decode() for part in Path('/proc/self/cmdline').read_bytes().split(b'\0') if part]
    assert inspect_process(os.getpid(), command)['state'] == 'active'
    assert inspect_process(os.getpid(), [sys.executable, 'unrelated_script.py'])['state'] == 'different'
    assert inspect_process(-1, command)['state'] == 'absent'
