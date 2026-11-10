import hashlib
import json
import subprocess
import sys

import pytest

from scripts.finalize_seviri_calendar_hpc import REMOTE_SUBMIT, REMOTE_VERIFY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def transferred(tmp_path):
    data = tmp_path/'data'
    (data/'frames').mkdir(parents=True)
    (data/'frames/day.npy').write_bytes(b'previously verified array')
    (data/'frames/metadata.json').write_text('{}')
    (data/'manifest.csv').write_text('a,b\n1,2\n')
    frames = {p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in (data/'frames').iterdir()}
    payload = {'manifest.csv':dict(bytes=(data/'manifest.csv').stat().st_size, sha256=digest(data/'manifest.csv'))}
    (data/'transfer_plan.json').write_text(json.dumps(dict(frames=frames, payload=payload)))
    (data/'frame_transfer.json').write_text(json.dumps(dict(status='completed', files={'day.npy':frames['day.npy']})))
    return data


def run_remote(script, *arguments):
    return subprocess.run([sys.executable, '-', *map(str, arguments)], input=script,
        text=True, capture_output=True)


def test_finalization_rejects_metadata_changed_after_transfer(transferred):
    assert run_remote(REMOTE_VERIFY, transferred).returncode == 0
    (transferred/'manifest.csv').write_text('changed')
    assert run_remote(REMOTE_VERIFY, transferred).returncode != 0


def test_submission_is_idempotent_and_preserves_the_authorized_budget(transferred, tmp_path):
    assert run_remote(REMOTE_VERIFY, transferred).returncode == 0
    run = tmp_path/'run'
    (run/'slurm_logs').mkdir(parents=True)
    (run/'runtime_files.json').write_text('{}')
    plan = dict(epochs=250, learning_rate=5e-5, world_size=4, batch_size_per_gpu=152, initialization='scratch')
    (run/'full_training.json').write_text(json.dumps(plan))
    fake = tmp_path/'sbatch_fixture.py'
    calls = tmp_path/'calls.txt'
    fake.write_text('from pathlib import Path\n' +
        f'with Path({str(calls)!r}).open("a") as f: f.write("submitted\\n")\nprint("12345")\n')
    (run/'submission.json').write_text(json.dumps(dict(status='prepared', command=[sys.executable, str(fake)])))
    for _ in range(2):
        result = run_remote(REMOTE_SUBMIT, run, transferred, 'yes')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['job_id'] == '12345'
    assert calls.read_text() == 'submitted\n'


def test_wrong_learning_rate_is_rejected_before_sbatch(transferred, tmp_path):
    assert run_remote(REMOTE_VERIFY, transferred).returncode == 0
    run = tmp_path/'run'
    run.mkdir()
    (run/'runtime_files.json').write_text('{}')
    (run/'full_training.json').write_text(json.dumps(dict(epochs=250, learning_rate=1.5e-4,
        world_size=4, batch_size_per_gpu=152, initialization='scratch')))
    (run/'submission.json').write_text(json.dumps(dict(status='prepared', command=['invalid-unused-command'])))
    result = run_remote(REMOTE_SUBMIT, run, transferred, 'yes')
    assert result.returncode != 0
    assert 'AssertionError' in result.stderr
