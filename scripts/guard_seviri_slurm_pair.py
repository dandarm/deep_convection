#!/usr/bin/env python3
"""Select one Slurm submission before either duplicate loads training data.

The shared election survives logout and requeue. The winner verifies ownership
and output directories, cancels its peer, and waits for cancellation to finish.
Slurm/controller errors stop startup instead of allowing concurrent training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TERMINAL = frozenset(('CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
                      'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.{}.part'.format(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Slurm:
    def job(self, job_id):
        result = subprocess.run(['scontrol', 'show', 'job', job_id, '-o'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        if result.returncode:
            if 'Invalid job id' in result.stderr:
                return None
            raise RuntimeError('Cannot inspect job {}: {}'.format(job_id, result.stderr.strip()))
        fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', result.stdout))
        user = re.fullmatch(r'[^()]+\((\d+)\)', fields.get('UserId', ''))
        if fields.get('JobId') != job_id or user is None:
            raise ValueError('Unexpected Slurm job identity')
        return dict(uid=int(user.group(1)), root=str(Path(fields['WorkDir']).parent),
                    name=fields['JobName'], state=fields['JobState'].split('+')[0])

    def cancel(self, job_id):
        result = subprocess.run(['scancel', '--quiet', job_id], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        if result.returncode and 'Invalid job id' not in result.stderr:
            raise RuntimeError('Cannot cancel duplicate {}: {}'.format(job_id, result.stderr.strip()))


def check_job(info, member, uid):
    if info is not None and (info['uid'] != uid or
            Path(info['root']).resolve() != Path(member['root']).resolve() or
            info['name'] != member['name']):
        raise ValueError('Job ownership, name or output directory differs from the registered pair')


def claim(pair_path, run_root, job_id, *, slurm=None, timeout=60, poll_seconds=1):
    pair_path = Path(pair_path)
    slurm = Slurm() if slurm is None else slurm
    if not re.fullmatch(r'[1-9][0-9]*', job_id):
        raise ValueError('A valid SLURM_JOB_ID is required')
    with (pair_path.parent/'election.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pair = json.loads(pair_path.read_text())
        members = pair['members']
        if pair['uid'] != os.getuid() or job_id not in members:
            raise ValueError('This job is not registered in the current user\'s pair')
        if Path(members[job_id]['root']).resolve() != Path(run_root).resolve():
            raise ValueError('The training root differs from the registered job')
        winner_path = pair_path.parent/'winner.json'
        winner = json.loads(winner_path.read_text()) if winner_path.exists() else None
        if winner is not None and winner['job_id'] not in members:
            raise ValueError('The selected winner is absent from the pair')
        if winner is not None and winner['job_id'] != job_id:
            write_json(pair_path.parent/'attempts'/(job_id+'.json'),
                       dict(status='not_selected', winner=winner['job_id'], updated_unix=time.time()))
            return False
        states = {key: slurm.job(key) for key in members}
        for key, info in states.items():
            check_job(info, members[key], pair['uid'])
        if states[job_id] is None or states[job_id]['state'] != 'RUNNING':
            raise ValueError('Only a running Slurm job can claim training')
        if winner is None:
            winner = dict(job_id=job_id, selected_unix=time.time(), status='cancelling_peer')
            write_json(winner_path, winner)
        for peer, info in states.items():
            if peer != job_id and info is not None and info['state'] not in TERMINAL:
                slurm.cancel(peer)
        deadline = time.monotonic()+timeout
        while True:
            active = []
            for peer in members:
                if peer == job_id:
                    continue
                info = slurm.job(peer)
                check_job(info, members[peer], pair['uid'])
                if info is not None and info['state'] not in TERMINAL:
                    active.append(peer)
            if not active:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Peer cancellation is unconfirmed; training remains disabled')
            time.sleep(poll_seconds)
        winner.update(status='ready', updated_unix=time.time(), cancelled_peers=[p for p in members if p != job_id])
        write_json(winner_path, winner)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('The guarded training command is required')
    job_id = os.environ.get('SLURM_JOB_ID', '')
    if claim(args.pair_config, args.run_root, job_id):
        print('Selected Slurm job {}; peer cancellation confirmed'.format(job_id), flush=True)
        os.execv(command[0], command)
    print('Slurm job {} exits because its peer was selected'.format(job_id), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
