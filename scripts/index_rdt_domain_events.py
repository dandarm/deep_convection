#!/usr/bin/env python3
"""Incremental read-only occupancy index and strongest candidate per track."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.rdt.storage import atomic_json,sha256
from emma_gpm.rdt_event_explorer import summarize_frame
import pandas as pd


def worker_init():
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency':2,'threading.max_workers':1})
    blosc.set_nthreads(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--follow',action='store_true')
    args=parser.parse_args()
    source=args.label_root.resolve();out=source/'exploration';out.mkdir(exist_ok=True)
    lock=(out/'index.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol_sha=sha256(source/'protocol.json')
    min_confidence=json.loads((source/'protocol.json').read_text())['config']['min_proxy_confidence']
    prior=out/'protocol.json'
    identity=dict(label_root=str(source),teacher_protocol_sha256=protocol_sha,
                  selection='convection: supported, nonborder, temporal/phase-valid classes 2/3/4; OT: supported, nonborder persistent pixels',
                  support_minimum=min_confidence,units='occupancy: % valid pixels; convection events: km2; OT events: pixels',
                  source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/emma_gpm/rdt_event_explorer.py']})
    if prior.exists() and json.loads(prior.read_text()) != identity:
        raise ValueError('Index protocol changed; use a new exploration directory')
    atomic_json(prior,identity)
    db=sqlite3.connect(out/'index.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS frames (time TEXT PRIMARY KEY,valid_pixels INTEGER,convective_pixels INTEGER,ot_pixels INTEGER,convective_percent REAL,ot_percent REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS events (track_id INTEGER,kind TEXT,time TEXT,score REAL,area_km2 REAL,ot_pixels INTEGER,min_k REAL,proxy_confidence REAL,PRIMARY KEY(track_id,kind))')
    db.execute('CREATE INDEX IF NOT EXISTS event_rank ON events(kind,score DESC,time,track_id)')
    known={r[0] for r in db.execute('SELECT time FROM frames')}
    started=time.monotonic();last_publish=-float('inf');total=0

    def publish(status,error=None):
        frame=pd.read_sql_query('SELECT * FROM frames ORDER BY time',db)
        parts=[pd.read_sql_query('SELECT * FROM events WHERE kind=? ORDER BY score DESC,time,track_id LIMIT 200',db,params=[kind]) for kind in ['convection','ot']]
        nonempty=[part for part in parts if not part.empty]
        events=pd.concat(nonempty,ignore_index=True) if nonempty else parts[0]
        for name,table in [('frame_index',frame),('events',events)]:
            temporary=out/f'{name}.csv.part';table.to_csv(temporary,index=False);temporary.replace(out/f'{name}.csv')
        atomic_json(out/'progress.json',dict(status=status,indexed=len(known),available=total,elapsed_seconds=time.monotonic()-started,error=error))

    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
            while True:
                directories=sorted(p for p in (source/'frames').glob('*') if (p/'record.json').exists() and not p.name.endswith('.part'))
                total=len(directories)
                pending=[p for p in directories if pd.to_datetime(p.name,format='%Y%m%dT%H%M').isoformat() not in known]
                # Inspect promising frames and every currently represented month
                # first, then exhaustively process the remaining committed frames.
                metrics_path=source/'frame_metrics.csv'
                if metrics_path.exists() and pending:
                    from emma_gpm.rdt.storage import read_table,frame_key
                    metrics=read_table(metrics_path)
                    if not metrics.empty:
                        metrics['time']=pd.to_datetime(metrics.time)
                        wanted=[]
                        for metric in ['ot_pixels','convective_objects']:
                            wanted.extend(metrics.sort_values([metric,'time'],ascending=[False,True]).head(100).time.tolist())
                        for _,month in metrics.groupby(metrics.time.dt.month):
                            wanted.extend(month.iloc[[0,-1]].time.tolist())
                        order={frame_key(stamp):i for i,stamp in reversed(list(enumerate(wanted)))}
                        pending.sort(key=lambda p:(order.get(p.name,len(order)),p.name))
                # Revisit the live frontier regularly instead of holding a stale
                # queue until the whole previous backlog has finished.
                if args.follow:pending=pending[:512]
                for begin in range(0,len(pending),64):
                    for row,events in pool.map(partial(summarize_frame,min_proxy_confidence=min_confidence),pending[begin:begin+64],chunksize=4):
                        db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?)',tuple(row.values()))
                        for event in events:
                            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(track_id,kind) DO UPDATE SET time=excluded.time,score=excluded.score,area_km2=excluded.area_km2,ot_pixels=excluded.ot_pixels,min_k=excluded.min_k,proxy_confidence=excluded.proxy_confidence WHERE excluded.score>events.score OR (excluded.score=events.score AND excluded.time<events.time)',tuple(event.values()))
                        known.add(row['time'])
                    db.commit()
                    if time.monotonic()-last_publish>=15:
                        publish('running');last_publish=time.monotonic()
                state=json.loads((source/'progress.json').read_text()) if (source/'progress.json').exists() else {}
                if not args.follow or state.get('status') in ('error','partial') or (state.get('status')=='complete' and len(known)==total):
                    publish('complete' if state.get('status')=='complete' and len(known)==total else 'partial');break
                publish('following')
                if len(pending)<512:time.sleep(15)
    except BaseException as exc:
        db.commit()
        publish('error',str(exc));raise
    finally:
        db.close();lock.close()


if __name__=='__main__':main()
