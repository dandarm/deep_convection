#!/usr/bin/env python3
"""Read-only terminal status for the SEVIRI SSL experiment series (stdlib only)."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'future': 'Future', 'mvm': 'MVM', 'joint': 'Joint', 'residual': 'Residual',
          'probe_random': 'Probe (Random)', 'probe_mvm': 'Probe (MVM)',
          'probe_future': 'Probe (Future)', 'probe_joint': 'Probe (Joint)',
          'probe_residual': 'Probe (Residual)'}
PHASES = {'training': 'training', 'validating': 'validation', 'running': 'preparazione / salvataggio',
          'completed': 'finalizzazione', 'queued': 'in attesa'}


def read_json(path, required=False):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if required:
            raise
        return {}


def inspect_process(pid, expected_command):
    """Check Linux process state and command identity without sending signals."""
    result = dict(pid=pid, state='absent')
    if not isinstance(pid, int) or pid <= 0:
        return result
    try:
        directory = Path('/proc') / str(pid)
        state = (directory / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        actual = [part.decode(errors='replace') for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
    except (FileNotFoundError, ProcessLookupError):
        return result
    except (PermissionError, OSError):
        return dict(result, state='unknown')
    if state in ('Z', 'X'):
        return result
    if not expected_command:
        return dict(result, state='unknown')
    # Interpreter paths may be equivalent symlinks. Script and arguments must match.
    if actual[1:] != expected_command[1:]:
        return dict(result, state='different')
    return dict(result, state='paused' if state in ('T', 't') else 'active')


def collect_status(root, *, now=None, stale_after=180, process_probe=inspect_process):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    plan = read_json(root / 'scaling_plan.json', required=True)
    queue = read_json(root / 'queue_status.json')
    launcher = read_json(root / 'launcher.json')
    cases = {case['name']: case for case in plan['cases']}
    stages = {case: {stage['name']: stage for stage in read_json(root / case / 'comparison_plan.json', required=True)['stages']}
              for case in cases}
    worker = process_probe(queue.get('pid'), queue.get('command'))
    supervisor_pid = queue.get('supervisor_pid') or launcher.get('pid')
    supervisor_command = launcher.get('command') if supervisor_pid == launcher.get('pid') else None
    supervisor = process_probe(supervisor_pid, supervisor_command)
    current = None
    rows = []
    for item in plan['order']:
        case, name = item['case'], item['stage']
        stage = stages[case][name]
        directory = root / case / name
        progress = read_json(directory / 'progress.json')
        done, target = progress.get('completed_epochs', 0), stage['epochs']
        selected = (case, name) == (queue.get('case'), queue.get('stage'))
        complete = done == target and progress.get('status') == 'completed'
        state = ('completato' if complete else 'in attesa' if not progress else 'parziale')
        if selected and not complete:
            if worker['state'] == 'active':
                state = PHASES.get(progress.get('status'), 'avvio')
            elif worker['state'] == 'paused':
                state = 'sospeso'
            elif supervisor['state'] == 'active':
                state = 'avvio / transizione'
            elif queue.get('status') == 'failed':
                state = 'errore'
            elif progress:
                state = 'fermo'
        row = dict(case=case, stage=name, label=LABELS.get(name, name),
                   model_size=cases[case].get('model_size', plan.get('model_size', 'small')),
                   training_clips=cases[case]['train_samples'], completed_epochs=done,
                   requested_epochs=target, complete=complete, state=state)
        rows.append(row)
        if selected:
            current = {**row, 'phase': progress.get('status', 'starting'),
                       'current_epoch': progress.get('current_epoch'),
                       'current_step': progress.get('current_step'),
                       'steps_per_epoch': progress.get('steps_per_epoch')}
            path = directory / 'progress.json'
            current['progress_age_seconds'] = max(0., now-path.stat().st_mtime) if path.exists() else None
            epochs = progress.get('epochs', [])
            # Use timings from the current execution revision after a runtime change.
            after_epoch = max((revision['after_epoch'] for revision in progress.get('execution_history', [])), default=0)
            recent = [epoch for epoch in epochs if epoch['epoch'] > after_epoch][-5:]
            times = []
            for epoch in recent:
                timing = read_json(directory / 'timing_epochs' / f"epoch_{epoch['epoch']:03d}.json")
                times.append(timing.get('epoch_wall_seconds', epoch['train_seconds'] + epoch['validation_seconds']))
            if recent:
                current.update(recent_epochs=len(recent), epoch_seconds=median(times),
                    train_seconds=median(epoch['train_seconds'] for epoch in recent),
                    validation_seconds=median(epoch['validation_seconds'] for epoch in recent),
                    last_epoch=recent[-1])
                if worker['state'] == 'active' and not complete:
                    elapsed_training = 0
                    if current['phase'] == 'validating':
                        elapsed_training = current['train_seconds']
                    elif current['steps_per_epoch'] and current['current_step']:
                        elapsed_training = current['train_seconds'] * current['current_step'] / current['steps_per_epoch']
                    current['remaining_seconds_estimate'] = max(0, (target-done)*current['epoch_seconds']-elapsed_training)
    completed_runs = sum(row['complete'] for row in rows)
    completed_epochs = sum(row['completed_epochs'] for row in rows)
    planned_epochs = sum(row['requested_epochs'] for row in rows)
    health = 'in attesa'
    if queue.get('status') == 'completed' and completed_runs == len(rows):
        health = 'completata'
    elif worker['state'] == 'active':
        health = 'analisi in esecuzione' if queue.get('stage') == 'analysis' else 'in esecuzione'
        age = current.get('progress_age_seconds') if current else None
        # A freshly started process may still be restoring an older checkpoint.
        try:
            launch_age = max(0, now-datetime.fromisoformat(queue['started_at_utc']).timestamp())
            age = min(age, launch_age) if age is not None else launch_age
        except (KeyError, ValueError):
            pass
        if age is not None and age > stale_after and current and not current['complete']:
            health = 'processo presente; avanzamento da verificare'
            current.pop('remaining_seconds_estimate', None)
    elif 'paused' in (worker['state'], supervisor['state']):
        health = 'sospesa'
    elif supervisor['state'] == 'active':
        health = 'supervisore attivo; avvio / passaggio tra esperimenti'
    elif queue.get('status') == 'failed':
        health = 'errore'
    elif queue.get('status', '').startswith('paused'):
        health = 'sospesa'
    elif completed_epochs or queue.get('status') == 'running':
        health = 'ferma'
    return dict(series=str(root), checked_at=datetime.fromtimestamp(now, ZoneInfo('Europe/Paris')).isoformat(),
        state=health, supervisor=supervisor, worker=worker, completed_runs=completed_runs,
        planned_runs=len(rows), completed_epochs=completed_epochs, planned_epochs=planned_epochs,
        epoch_percent=100*completed_epochs/planned_epochs if planned_epochs else 0.,
        current=current, queue_stage=queue.get('stage'), queue_case=queue.get('case'),
        log=queue.get('log') or launcher.get('log'), rows=rows)


def duration(seconds):
    minutes = math.ceil(seconds / 60)
    return f'{minutes//60} h {minutes%60:02d} min' if minutes >= 60 else f'{minutes} min'


def format_report(report):
    stamp = datetime.fromisoformat(report['checked_at']).astimezone(ZoneInfo('Europe/Paris'))
    states = {'active': 'attivo', 'absent': 'assente', 'paused': 'sospeso',
              'different': 'PID assegnato a un altro comando', 'unknown': 'identità da verificare'}
    processes = '; '.join(f'{label}: {states[report[key]["state"]]} (PID {report[key]["pid"] or "—"})'
                          for key, label in [('supervisor', 'Supervisore'), ('worker', 'Processo del run')])
    lines = [f'SEVIRI SSL — {stamp:%d/%m/%Y %H:%M:%S %Z}', f'Serie: {report["series"]}',
             f'Stato: {report["state"].upper()}', processes,
             f'Esperimenti conclusi: {report["completed_runs"]}/{report["planned_runs"]}',
             f'Epoche salvate: {report["completed_epochs"]}/{report["planned_epochs"]} ({report["epoch_percent"]:.1f}%)']
    current = report['current']
    if current and report['worker']['state'] in ('active', 'paused'):
        line = f'In corso: {current["label"]}, {current["model_size"].title()}, {current["training_clips"]:,} clip; {PHASES.get(current["phase"], current["phase"])}'
        if current['current_epoch'] is not None:
            line += f'; epoca {current["current_epoch"]}/{current["requested_epochs"]}'
        if current['current_step'] is not None:
            line += f'; batch {current["current_step"]}/{current["steps_per_epoch"]}'
        lines.append(line)
        if current['progress_age_seconds'] is not None:
            lines.append(f'Ultimo aggiornamento avanzamento: {current["progress_age_seconds"]:.0f} s fa')
        if 'epoch_seconds' in current:
            lines.append(f'Durata recente: {current["epoch_seconds"]:.0f} s/epoca (mediana delle ultime {current["recent_epochs"]}); '
                         f'training {current["train_seconds"]:.0f} s, validation {current["validation_seconds"]:.0f} s')
        if 'remaining_seconds_estimate' in current:
            lines.append(f'Tempo residuo stimato per questo esperimento: circa {duration(current["remaining_seconds_estimate"])}')
    elif report['queue_stage'] == 'analysis' and report['worker']['state'] == 'active':
        lines.append(f'In corso: analisi finale {report["queue_case"]}')
    lines += ['', f'{"#":>2}  {"Encoder":<7} {"Clip":>7}  {"Esperimento":<18} {"Epoche":>9}  Stato']
    for index, row in enumerate(report['rows'], 1):
        epochs = f'{row["completed_epochs"]}/{row["requested_epochs"]}'
        lines.append(f'{index:2d}  {row["model_size"].title():<7} {row["training_clips"]:7,d}  {row["label"]:<18} {epochs:>9}  {row["state"]}')
    if report['log']:
        lines += ['', f'Log corrente: {report["log"]}']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = read_json(ROOT/'results/ssl_active_series.json')
    default_series = Path(selection['series']) if selection else ROOT/'results/ssl_scaling_20261025'
    parser.add_argument('--series', type=Path, default=default_series)
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS', help='Refresh interval; Ctrl+C closes the monitor')
    parser.add_argument('--json', action='store_true', help='Structured status, one JSON object per refresh')
    parser.add_argument('--stale-after', type=float, default=180, metavar='SECONDS')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.watch) or args.watch < 0 or 0 < args.watch < 1
            or not math.isfinite(args.stale_after) or args.stale_after <= 0):
        parser.error('Watch interval must be zero or at least 1 second; stale threshold must be positive')
    try:
        while True:
            report = collect_status(args.series, stale_after=args.stale_after)
            if args.watch and not args.json and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(json.dumps(report, ensure_ascii=False) if args.json else format_report(report), flush=True)
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'Stato non disponibile: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
