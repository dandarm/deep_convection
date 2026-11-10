#!/usr/bin/env python3
"""Copy previously verified immutable frame archives into shared node RAM."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from emma_gpm.ssl_ram import cache_files,digest,stage_frame_cache,stage_memfd_cache
from scripts.run_seviri_ssl_ddp_trials import write_json


def stage(root):
    root=Path(root).resolve()
    ram=Path('/dev/shm')/f'seviri_ssl_{os.getuid()}_{root.name}'
    if subprocess.check_output(['findmnt','-n','-o','FSTYPE','--target','/dev/shm'],text=True).strip()!='tmpfs':
        raise ValueError('/dev/shm must be a RAM filesystem')
    ram.mkdir(mode=0o700,exist_ok=True)
    if ram.is_symlink() or ram.stat().st_uid!=os.getuid():raise ValueError('Unexpected RAM directory owner')
    owner=dict(run_root=str(root),uid=os.getuid())
    marker=ram/'owner.json'
    if marker.exists() and json.loads(marker.read_text())!=owner:raise ValueError('RAM directory belongs to another run')
    write_json(marker,owner)
    mapping=json.loads((root/'storage_roots.json').read_text())
    locations=json.loads((root/'data_locations.json').read_text())
    backend=locations.get('ram_backend','tmpfs')
    if backend not in ('tmpfs','memfd'):
        raise ValueError('Unknown RAM storage backend')
    datasets=[];needed=0;shared={}
    for name in ('training','extended'):
        directory=Path(locations[name]);plan=json.loads((directory/'transfer_plan.json').read_text())
        expected=plan['frames']['metadata.json']['sha256']
        if plan['canonical_frame_root'] not in mapping['roots']:
            raise ValueError('Canonical frame root is absent from the storage mapping')
        canonical=plan['canonical_frame_root']
        if canonical in shared:
            if shared[canonical]['metadata_sha256']!=expected:
                raise ValueError('Shared frame root has inconsistent metadata')
            continue
        shared[canonical]=dict(dataset=name,metadata_sha256=expected)
        files=cache_files(directory/'frames',expected)
        needed+=sum(row['bytes'] for filename,row in files.items() if not (ram/name/filename).exists())
        datasets.append((name,directory,expected,plan['canonical_frame_root']))
    if backend=='tmpfs':
        if shutil.disk_usage(ram).free<needed+8*1024**3:
            raise ValueError('The frame archives exceed available tmpfs space')
    else:
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
        allocated=os.environ.get('SLURM_MEM_PER_NODE','')
        if allocated.isdigit():available=min(available,int(allocated)*1024**2)
        if available<needed+64*1024**3:
            raise ValueError('Insufficient node memory for the inputs and 64 GiB training reserve')
    report=dict(status='staging',backend=backend,ram_root=str(ram),datasets={},shared_roots=shared,updated_unix=time.time())
    write_json(root/'ram_stage.json',report)
    for name,directory,expected,canonical in datasets:
        def progress(filename):
            report.update(current_dataset=name,current_file=filename,updated_unix=time.time())
            write_json(root/'ram_stage.json',report)
        if backend=='tmpfs':
            report['datasets'][name]=stage_frame_cache(directory/'frames',ram/name,expected,progress=progress)
            mapping['roots'][canonical]=str(ram/name)
        else:
            report['datasets'][name]=stage_memfd_cache(directory/'frames',ram/name,expected,progress=progress)
            mapping['roots'][canonical]=str(ram/name)
    write_json(root/'ram_storage_roots.json',mapping)
    report.update(status='ready',mapping_sha256=digest(root/'ram_storage_roots.json'),updated_unix=time.time())
    write_json(root/'ram_stage.json',report)
    return report


def cleanup(root):
    root=Path(root).resolve();ram=Path('/dev/shm')/f'seviri_ssl_{os.getuid()}_{root.name}'
    if not ram.exists():return
    if ram.is_symlink() or ram.stat().st_uid!=os.getuid():raise ValueError('Unexpected RAM directory owner')
    if json.loads((ram/'owner.json').read_text())!=dict(run_root=str(root),uid=os.getuid()):
        raise ValueError('RAM staging ownership does not match this run')
    shutil.rmtree(ram)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cleanup',action='store_true')
    args=p.parse_args()
    if args.cleanup:cleanup(args.root)
    else:print(json.dumps(stage(args.root),indent=2))
