#!/usr/bin/env python3
"""Execute one frozen Slurm-array entry, resuming its latest full epoch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_seviri_ssl_hpc import digest, write_json
from emma_gpm.file_lock import exclusive_file_lock


def build_command(plan, index, python):
    run = plan["runs"][index]
    archive = Path(plan["source_archive"])
    for name, expected in plan["frozen_files"].items():
        if digest(archive / name) != expected:
            raise ValueError(f"Frozen HPC source changed: {name}")
    if digest(run["config"]) != run["config_sha256"]:
        raise ValueError("Frozen training configuration changed")
    if digest(plan["storage_roots"]) != plan["storage_roots_sha256"]:
        raise ValueError("Storage mapping changed")
    dataset = Path(plan["dataset"])
    if digest(dataset / "transfer_plan.json") != plan["transfer_plan_sha256"]:
        raise ValueError("Transferred data identity changed")
    config = json.loads(Path(run["config"]).read_text())
    if digest(config["cuda_runtime"]) != plan["cuda_runtime_sha256"]:
        raise ValueError("Frozen CUDA runtime changed")
    command = [python, "-u", str(archive / "scripts/pretrain_seviri_ssl.py"),
               "--config", run["config"], "--output-dir", run["output"]]
    checkpoint = Path(run["output"]) / "checkpoint_latest.pt"
    if checkpoint.exists():
        command += ["--resume", str(checkpoint)]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.index < 0 or args.index >= len(plan["runs"]):
        parser.error("Array index is outside the frozen plan")
    command = build_command(plan, args.index, sys.executable)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    run = plan["runs"][args.index]
    stop = Path(run["stop_file"])
    output = Path(run["output"])
    with exclusive_file_lock(args.plan.parent / f".hpc-{args.index}.lock"):
        stop.unlink(missing_ok=True)
        os.environ["SEVIRI_STORAGE_ROOTS"] = plan["storage_roots"]
        status_file = args.plan.parent / f"job_status_{args.index}.json"
        status = dict(status="running", index=args.index, pid=os.getpid(),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      hostname=os.uname().nodename, started_at=time.time(), command=command)
        write_json(status_file, status)
        def request_stop(signum, frame):
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.touch()
        previous = signal.signal(signal.SIGUSR1, request_stop)
        try:
            result = subprocess.run(command, cwd=plan["source_archive"])
        finally:
            signal.signal(signal.SIGUSR1, previous)
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        paused = result.returncode == 0 and progress.get("status") == "paused" and stop.exists()
        code = 75 if paused else result.returncode
        status.update(status="paused" if paused else "completed" if code == 0 else "failed",
                      returncode=code, finished_at=time.time(),
                      completed_epochs=progress.get("completed_epochs"),
                      optimizer_steps=progress.get("optimizer_steps"))
        write_json(status_file, status)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
