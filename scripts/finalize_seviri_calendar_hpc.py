#!/usr/bin/env python3
"""Publish verified metadata and a frozen runtime, then optionally submit one job."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_VERIFY = '''import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
plan=json.loads((root/'transfer_plan.json').read_text())
records=json.loads((root/'frame_transfer.json').read_text())
assert records['status']=='completed'
for name,expected in plan['frames'].items():
    path=root/'frames'/name
    assert path.stat().st_size==expected['bytes'],name
    if name=='metadata.json':
        assert digest(path)==expected['sha256'],name
    else:
        # Array bytes were checked on this host immediately after each transfer.
        assert records['files'][name]['sha256']==expected['sha256'],name
        assert records['files'][name]['bytes']==expected['bytes'],name
for name,expected in plan['payload'].items():
    path=root/name
    path.resolve().relative_to(root.resolve())
    assert digest(path)==expected['sha256'],name
report=dict(status='verified',transfer_plan_sha256=digest(root/'transfer_plan.json'),
    native_arrays=len(plan['frames'])-1,payload_files=len(plan['payload']),verified_unix=time.time(),
    array_verification='one destination SHA256 check per new transferred array; immutable validation arrays reused')
temporary=root/'verification.json.part'
temporary.write_text(json.dumps(report,indent=2)+'\\n')
temporary.replace(root/'verification.json')
print(json.dumps(report))
'''

REMOTE_SUBMIT = '''import fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[1]);data=Path(sys.argv[2]);submit=sys.argv[3]=='yes'
with (run/'.submission.lock').open('w') as lock:
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    path=run/'submission.json';state=json.loads(path.read_text())
    if state.get('job_id'):
        print(json.dumps(state));sys.exit(0)
    verification=json.loads((data/'verification.json').read_text())
    assert verification['status']=='verified'
    assert hashlib.sha256((data/'transfer_plan.json').read_bytes()).hexdigest()==verification['transfer_plan_sha256']
    files=json.loads((run/'runtime_files.json').read_text())
    for name,expected in files.items():
        assert hashlib.sha256((run/'source_archive'/name).read_bytes()).hexdigest()==expected,name
    plan=json.loads((run/'full_training.json').read_text())
    assert plan['epochs']==250 and plan['learning_rate']==5e-5
    assert plan['world_size']==4 and plan['batch_size_per_gpu']==152
    assert plan['initialization']=='scratch' and not plan.get('parent_checkpoint')
    if submit:
        result=subprocess.check_output(state['command'],cwd=str(run/'slurm_logs'),universal_newlines=True).strip()
        job=result.split(';')[0]
        assert job.isdigit(),result
        state.update(status='submitted',job_id=job,submitted_unix=time.time())
        temporary=path.with_suffix('.json.part');temporary.write_text(json.dumps(state,indent=2)+'\\n');temporary.replace(path)
    print(json.dumps(state))
'''


def main(args):
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host]
    inspect = 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"submission.json"; print(p.read_text() if p.exists() else "{}")'
    command = 'python3 -c '+shlex.quote(inspect)+' '+shlex.quote(str(args.remote_run))
    existing = json.loads(subprocess.check_output([*ssh, command], text=True))
    if existing.get('job_id'):
        print(json.dumps(existing))
        (args.bundle/'remote_submission.json').write_text(json.dumps(existing, indent=2)+'\n')
        return
    data = args.dataset.resolve()
    transfer = json.loads((data/'transfer_plan.json').read_text())
    completed = json.loads(args.transfer_state.read_text())
    if completed['status'] != 'completed':
        raise ValueError('Daily array transfers must finish first')
    for name, expected in transfer['frames'].items():
        if name != 'metadata.json' and completed['files'][name]['sha256'] != expected['sha256']:
            raise ValueError('Transferred array identity differs: '+name)
    (data/'frame_transfer.json').write_text(args.transfer_state.read_text())
    names = sorted(set(transfer['payload']) | {'frames/metadata.json', 'transfer_plan.json',
        'training/transfer_plan.json', 'extended/transfer_plan.json', 'training/frames', 'extended/frames', 'frame_transfer.json'})
    file_list = args.bundle/'payload_files.txt'
    file_list.write_text('\n'.join(names)+'\n')
    remote_data = str(args.remote_dataset)
    remote_run = str(args.remote_run)
    rsh = 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30'
    subprocess.run(['rsync', '-a', '--checksum', '-e', rsh, '--files-from='+str(file_list),
        str(data)+'/', args.host+':'+remote_data+'/'], check=True)
    def remote(script, *parameters):
        command = 'python3 - '+ ' '.join(shlex.quote(str(p)) for p in parameters)
        result = subprocess.run([*ssh, command], input=script, text=True, capture_output=True, check=True)
        print(result.stdout, end='', flush=True)
        return json.loads(result.stdout)
    verification = remote(REMOTE_VERIFY, remote_data)
    (args.bundle/'transfer_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    # Preserve remote progress and the submission record if this command is resumed.
    subprocess.run(['rsync', '-a', '--checksum', '-e', rsh, '--exclude=submission.json',
        str(args.bundle)+'/', args.host+':'+remote_run+'/'], check=True)
    subprocess.run(['rsync', '-a', '--ignore-existing', '-e', rsh, str(args.bundle/'submission.json'),
        args.host+':'+remote_run+'/'], check=True)
    submission = remote(REMOTE_SUBMIT, remote_run, remote_data, 'yes' if args.submit else 'no')
    (args.bundle/'remote_submission.json').write_text(json.dumps(submission, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'transfer-state', 'bundle', 'remote-dataset', 'remote-run'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--submit', action='store_true')
    main(p.parse_args())
