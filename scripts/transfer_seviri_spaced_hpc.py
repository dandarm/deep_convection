#!/usr/bin/env python3
"""Transfer immutable precomputed clips, verify ingestion once, and publish a run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from scripts.finalize_seviri_calendar_hpc import REMOTE_SUBMIT


REMOTE_VERIFY = '''import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
identity=digest(root/'transfer_plan.json')
done=root/'verification.json'
if done.exists():
    report=json.loads(done.read_text())
    assert report['status']=='verified' and report['transfer_plan_sha256']==identity
    print(json.dumps(report));sys.exit(0)
plan=json.loads((root/'transfer_plan.json').read_text())
files=dict(plan['payload'])
files.update({'clips/'+name:entry for name,entry in plan['clips'].items()})
for i,(name,entry) in enumerate(files.items()):
    path=root/name
    path.resolve().relative_to(root.resolve())
    assert path.stat().st_size==entry['bytes'],name
    assert digest(path)==entry['sha256'],name
    if (i+1)%1000==0:print('Verified',i+1,'/',len(files),file=sys.stderr,flush=True)
report=dict(status='verified',transfer_plan_sha256=identity,files=len(files),verified_unix=time.time(),
    verification='One SHA256 comparison per newly transferred immutable file; no audits during RAM staging or epoch reads')
temporary=done.with_suffix('.json.part');temporary.write_text(json.dumps(report,indent=2)+'\\n');temporary.replace(done)
print(json.dumps(report))
'''


def main(args):
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host]
    rsh = 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30'
    data, bundle = args.dataset.resolve(), args.bundle.resolve()
    transfer = json.loads((data/'transfer_plan.json').read_text())
    def remote(script, *values):
        command = 'python3 - '+ ' '.join(shlex.quote(str(value)) for value in values)
        result = subprocess.run([*ssh, command], input=script, text=True, stdout=subprocess.PIPE, check=True)
        return json.loads(result.stdout)
    setup = 'import json,sys; from pathlib import Path; [Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]; print("{}")'
    remote(setup, args.remote_dataset, args.remote_run)
    inspect = 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"submission.json"; print(p.read_text() if p.exists() else "{}")'
    state = remote(inspect, args.remote_run)
    if state.get('job_id'):
        print(json.dumps(state));return
    names = ['clips/'+name for name in sorted(transfer['clips'])]
    shard_lists = []
    for index in range(args.workers):
        path = bundle/f'clip_transfer_{index}.txt'
        path.write_text('\n'.join(names[index::args.workers])+'\n')
        shard_lists.append(path)
    def upload(file_list):
        subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '--partial-dir=.rsync-partial',
            '--info=progress2', '-e', rsh, '--files-from='+str(file_list), str(data)+'/',
            args.host+':'+str(args.remote_dataset)+'/'], check=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(upload, shard_lists))
    payload = bundle/'payload_transfer.txt'
    payload.write_text('\n'.join(sorted(set(transfer['payload']) | {'transfer_plan.json',
        'training/transfer_plan.json', 'extended/transfer_plan.json', 'training/clips', 'extended/clips'}))+'\n')
    upload(payload)
    verification = remote(REMOTE_VERIFY, args.remote_dataset)
    (bundle/'transfer_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    subprocess.run(['rsync', '-a', '-e', rsh, '--exclude=submission.json', str(bundle)+'/',
        args.host+':'+str(args.remote_run)+'/'], check=True)
    subprocess.run(['rsync', '-a', '--ignore-existing', '-e', rsh, str(bundle/'submission.json'),
        args.host+':'+str(args.remote_run)+'/'], check=True)
    state = remote(REMOTE_SUBMIT, args.remote_run, args.remote_dataset, 'yes' if args.submit else 'no')
    (bundle/'remote_submission.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'bundle', 'remote-dataset', 'remote-run'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--submit', action='store_true')
    main(p.parse_args())
