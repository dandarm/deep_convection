#!/usr/bin/env python3
"""Transfer committed daily arrays while preparation continues; verify each copy once."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shlex
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    state = args.progress
    completed = json.loads(state.read_text()).get('files', {}) if state.exists() else {}
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host]
    remote = args.destination.rstrip('/')
    reused = json.loads(subprocess.check_output([*ssh,
        'cat ' + shlex.quote(remote+'/reused_validation_metadata.json')], text=True))
    old = {row['file']:row for row in reused['stores'].values()}
    def upload(row):
        name = row['file']
        started = time.monotonic()
        if name in old:
            if old[name]['sha256'] != row['sha256'] or old[name]['bytes'] != row['bytes']:
                raise ValueError('Existing validation array belongs to another dataset: '+name)
            method = 'shared immutable validation array already verified at ingestion'
        else:
            subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '--ignore-existing',
                '--partial-dir=.rsync-partial', '--compress', '--compress-level=1',
                '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30',
                str(args.source/'frames'/name), args.host+':'+remote+'/frames/'], check=True)
            script = """import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as stream:
    for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
print(h.hexdigest())
"""
            result = subprocess.run([*ssh, 'python3 - '+shlex.quote(remote+'/frames/'+name)],
                input=script, text=True, capture_output=True, check=True)
            if result.stdout.strip() != row['sha256']:
                raise ValueError('Transferred array checksum differs: '+name)
            method = 'SHA256 verified on destination after transfer'
        result = dict(bytes=row['bytes'], sha256=row['sha256'], method=method,
            elapsed_seconds=time.monotonic()-started, verified_unix=time.time())
        print(f'Verified {name}: {row["bytes"]/2**30:.1f} GiB in {time.monotonic()-started:.0f} s', flush=True)
        return name, result
    while True:
        ready = sorted((args.source/'daily').glob('*.json'))
        pending = []
        for path in ready:
            row = json.loads(path.read_text())['entry']
            name = row['file']
            if name in completed:
                if completed[name]['sha256'] != row['sha256']:
                    raise ValueError('Committed array changed: '+name)
                continue
            pending.append(row)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(upload, row) for row in pending]
            for future in as_completed(futures):
                name, record = future.result()
                completed[name] = record
                save(state, dict(status='transferring', files=completed, updated_unix=time.time()))
        if (args.source/'dataset_ready.json').exists():
            metadata = json.loads((args.source/'frames/metadata.json').read_text())
            if {row['file'] for row in metadata['stores'].values()} <= completed.keys():
                save(state, dict(status='completed', files=completed, updated_unix=time.time()))
                return
        time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--progress', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    try:
        main(args)
    except BaseException as error:
        previous = json.loads(args.progress.read_text()) if args.progress.exists() else {}
        save(args.progress, dict(previous, status='failed', error=str(error), updated_unix=time.time()))
        raise
