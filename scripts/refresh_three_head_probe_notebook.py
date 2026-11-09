#!/usr/bin/env python3
"""Refresh saved probe plots as experiments complete, without model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.rdt.storage import atomic_json


def refresh(notebook, output):
    original = notebook.read_bytes()
    document = nbformat.reads(original.decode(), as_version=4)
    NotebookClient(document, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    # Respect edits made in the editor while the kernel was generating figures.
    if notebook.read_bytes() != original:
        return False
    temporary = notebook.with_suffix('.refreshing.ipynb')
    nbformat.write(document, temporary)
    temporary.replace(notebook)
    atomic_json(output / 'notebook_refresh.json', {
        'status': 'completed', 'notebook': str(notebook),
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'completed_experiments': sorted(p.parent.name for p in output.glob('*/result.json')),
    })
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data1/deep_convection_results/three_head_probes_20261109'))
    parser.add_argument('--notebook', type=Path,
                        default=ROOT / 'notebooks/study_seviri_cloud_mask_probe.ipynb')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=300,
                        help='Minimum seconds between refreshes of in-progress loss curves')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(args.root / 'notebook_refresh.lock'):
        previous, last = None, 0.
        while True:
            state_path = args.root / 'suite_status.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            complete = tuple(sorted(p.parent.name for p in args.root.glob('*/result.json')))
            fingerprint = (complete, state.get('status'))
            if fingerprint != previous or time.monotonic() - last >= args.interval:
                try:
                    updated = refresh(args.notebook, args.root)
                except Exception as error:
                    atomic_json(args.root / 'notebook_refresh.json', {
                        'status': 'failed', 'error': str(error),
                        'updated_utc': datetime.now(timezone.utc).isoformat(),
                    })
                    raise
                if updated:
                    previous, last = fingerprint, time.monotonic()
                    print(f'Refreshed notebook: {len(complete)} completed experiments', flush=True)
                else:
                    print('Notebook edited during execution; retrying from current content', flush=True)
            if not args.watch or (state.get('status') in ('completed', 'failed') and fingerprint == previous):
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
