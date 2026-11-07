import json
import multiprocessing
import os
import subprocess

import pytest

from scripts import guard_seviri_slurm_pair as guard
from scripts.guard_seviri_slurm_pair import claim, write_json


class FakeSlurm:
    def __init__(self, members):
        self.jobs = {key: dict(value, uid=os.getuid(), state='RUNNING') for key, value in members.items()}
        self.cancelled = []

    def job(self, key):
        return self.jobs.get(key)

    def cancel(self, key):
        self.cancelled.append(key)
        self.jobs[key]['state'] = 'CANCELLED'


@pytest.fixture
def pair(tmp_path):
    path = tmp_path/'pair.json'
    members = {key: dict(root=str(tmp_path/key), name='seviri_'+key) for key in ('101', '102')}
    write_json(path, dict(uid=os.getuid(), members=members))
    return path, members


def test_winner_cancels_peer_and_can_resume_same_job(pair):
    path, members = pair
    slurm = FakeSlurm(members)
    assert claim(path, members['102']['root'], '102', slurm=slurm)
    assert slurm.cancelled == ['101']
    assert not claim(path, members['101']['root'], '101', slurm=slurm)
    assert claim(path, members['102']['root'], '102', slurm=slurm)
    assert slurm.cancelled == ['101']


def test_foreign_job_is_never_cancelled(pair):
    path, members = pair
    slurm = FakeSlurm(members)
    slurm.jobs['102']['uid'] += 1
    with pytest.raises(ValueError, match='ownership'):
        claim(path, members['101']['root'], '101', slurm=slurm)
    assert slurm.cancelled == []
    assert not (path.parent/'winner.json').exists()


def test_unconfirmed_cancellation_prevents_training(pair):
    path, members = pair
    slurm = FakeSlurm(members)
    slurm.cancel = lambda job: None
    with pytest.raises(RuntimeError, match='unconfirmed'):
        claim(path, members['101']['root'], '101', slurm=slurm, timeout=0)
    assert json.loads((path.parent/'winner.json').read_text())['status'] == 'cancelling_peer'
    assert not claim(path, members['102']['root'], '102', slurm=slurm)


def test_pending_or_unregistered_job_cannot_claim(pair):
    path, members = pair
    slurm = FakeSlurm(members)
    slurm.jobs['101']['state'] = 'PENDING'
    with pytest.raises(ValueError, match='running'):
        claim(path, members['101']['root'], '101', slurm=slurm)
    with pytest.raises(ValueError, match='registered'):
        claim(path, members['101']['root'], '999', slurm=slurm)
    assert not (path.parent/'winner.json').exists()


def compete(path, members, key, event, results):
    event.wait()
    results.put((key, claim(path, members[key]['root'], key, slurm=FakeSlurm(members))))


def test_simultaneous_starts_select_exactly_one_winner(pair):
    path, members = pair
    context = multiprocessing.get_context('spawn')
    event, results = context.Event(), context.Queue()
    processes = [context.Process(target=compete, args=(path, members, key, event, results)) for key in members]
    for process in processes:
        process.start()
    event.set()
    outcomes = dict(results.get(timeout=20) for _ in processes)
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert sum(outcomes.values()) == 1
    winner = json.loads((path.parent/'winner.json').read_text())
    assert outcomes[winner['job_id']]
    assert winner['status'] == 'ready'


def test_slurm_identity_is_parsed_before_cancellation(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0,
            'JobId=101 JobName=seviri_101 UserId=ddarmien(31656) '
            'JobState=RUNNING WorkDir=/work/run101/logs', '')

    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard.Slurm().job('101') == dict(uid=31656, name='seviri_101',
                                          root='/work/run101', state='RUNNING')
    assert calls == [['scontrol', 'show', 'job', '101', '-o']]


def test_losing_entrypoint_does_not_execute_training(pair, monkeypatch):
    path, members = pair
    slurm = FakeSlurm(members)
    assert claim(path, members['101']['root'], '101', slurm=slurm)
    monkeypatch.setenv('SLURM_JOB_ID', '102')
    monkeypatch.setattr(guard.os, 'execv', lambda *args: pytest.fail('Loser launched training'))
    assert guard.main(['--pair-config', str(path), '--run-root', members['102']['root'],
                       '--', '/usr/bin/python3', 'training.py']) == 0
