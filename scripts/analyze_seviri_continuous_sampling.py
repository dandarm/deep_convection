#!/usr/bin/env python3
"""Audit and plot actual continuous clip selections against their fixed parent."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from emma_gpm.ssl_continuous_sampling import event_array, temporal_context
from emma_gpm.ssl_incremental_sampling import audit_added_pairs
from emma_gpm.ssl_overlap_audit import overlap_summary
from emma_gpm.ssl_archive_coverage import manifest_frame_counts
from emma_gpm.ssl_dashboard import manifest_coverage_figure
from emma_gpm.ssl_random_sampling import geographic_origin_mask
from emma_gpm.seviri_immutable_clips import ImmutableClipFileDataset
from emma_gpm.seviri_clip_files import SHAPE
from emma_gpm.seviri_clips import SeviriZarrClipReader
from emma_gpm.ssl_validation_set import assert_day_separation
from scripts.build_seviri_ssl_calendar_dataset import write_json


def temporal_counts(manifest, timestamps):
    """Count all active clips, including those starting on a previous UTC date."""
    starts = np.sort(pd.DatetimeIndex(pd.to_datetime(manifest.start_time, utc=True)).as_unit('ns').asi8)
    stamps = pd.DatetimeIndex(timestamps).as_unit('ns').asi8
    return (np.searchsorted(starts, stamps, side='right') -
            np.searchsorted(starts, stamps-75*60*10**9, side='left'))


def analyze(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = output/'audit'
    audit.mkdir(exist_ok=True)
    plan = json.loads((dataset/'preparation_plan.json').read_text())
    config = plan['config']
    parent, original = Path(config['parent_dataset']), Path(config['reference_dataset'])
    frame = pd.read_csv(dataset/'training/dataset/train.csv')
    old = pd.read_csv(parent/'training/dataset/train.csv')
    baseline = pd.read_csv(original/'training/dataset/train.csv')
    candidates = pd.read_csv(dataset/'candidates.csv')
    previous = pd.read_csv(parent/'candidates.csv')
    if not set(old.clip_id) <= set(frame.clip_id) or not frame.clip_id.is_unique:
        raise ValueError('The selected extension must retain every parent clip exactly once')
    pd.testing.assert_frame_equal(old.sort_values('clip_id').reset_index(drop=True),
        frame[frame.clip_id.isin(old.clip_id)].sort_values('clip_id').reset_index(drop=True))
    pd.testing.assert_frame_equal(previous.sort_values('clip_id').reset_index(drop=True),
        candidates[candidates.clip_id.isin(previous.clip_id)].sort_values('clip_id').reset_index(drop=True))
    for row in previous.itertuples():
        if not (parent/'candidate_clips'/row.file).samefile(dataset/'candidate_clips'/row.file):
            raise ValueError('An immutable candidate file was replaced')
    for role in ('training', 'extended'):
        if ((dataset/role/'dataset/validation.csv').read_bytes() !=
                (parent/role/'dataset/validation.csv').read_bytes()):
            raise ValueError('A fixed validation manifest changed')
    assert_day_separation(candidates, config['reserved_days'], config['temporal_separation_hours'])
    checks = []
    # Independent matrix checks include both neighboring dates. Only pairs of
    # original asynchronous clips retain the historical shared-fraction rule.
    for day in sorted(candidates.day.unique()):
        context = temporal_context(candidates, day)
        is_original = context.clip_id.isin(baseline.clip_id)
        ordered = pd.concat([context[is_original], context[~is_original]])
        report = audit_added_pairs(event_array(ordered), int(is_original.sum()))
        if (report['maximum_active_iou'] > config['max_iou']+1e-12 or
                report['maximum_new_pair_shared_fraction'] > config['max_shared_fraction']+1e-12):
            raise ValueError('An overlap constraint failed across the date boundary')
        checks.append(dict(day=day, **report))
    pd.DataFrame(checks).to_csv(audit/'pair_checks.csv', index=False)
    first_grid = None
    for store, group in candidates.groupby('store'):
        with xr.open_zarr(dataset/'sources'/store, consolidated=True, chunks=None) as ds:
            grid = (ds.x.values, ds.y.values, ds.attrs['projection_wkt'])
        if first_grid is None:
            first_grid = grid
            mask = geographic_origin_mask(*grid, config['domain'])
        elif (not np.array_equal(grid[0], first_grid[0]) or not np.array_equal(grid[1], first_grid[1])
                or grid[2] != first_grid[2]):
            raise ValueError('Candidate coordinate grids differ')
        if not mask[group.origin_y.to_numpy(int), group.origin_x.to_numpy(int)].all():
            raise ValueError('A crop leaves the fixed geographic domain')
    starts = pd.to_datetime(frame.start_time, utc=True)
    crossing = frame[starts.dt.normalize() != (starts+pd.Timedelta(minutes=75)).dt.normalize()]
    loader = ImmutableClipFileDataset(dataset/'clips', dataset/'training/dataset/train.csv',
                                      zarr_root=dataset/'sources')
    raw = np.empty(SHAPE, np.float16)
    verified = []
    with SeviriZarrClipReader(config['archive']) as reader:
        for index in crossing.index[::max(1, len(crossing)//3)][:3]:
            row = frame.loc[index]
            loader.decode_into(int(index), raw)
            native = reader.load(row.start_time, origin_y=int(row.origin_y), origin_x=int(row.origin_x))
            np.testing.assert_array_equal(raw, native)
            verified.append(row.clip_id)
    summary, daily, neighbors = overlap_summary(frame)
    if summary['distinct_native_frames'] != int(manifest_frame_counts(frame).sum()):
        raise ValueError('Independent frame counts disagree')
    daily.to_csv(audit/'daily.csv', index=False)
    neighbors.to_parquet(audit/'neighbors.parquet', index=False)
    summary.update(parent_samples=len(old), added_selected=len(frame)-len(old),
        old_candidates_preserved=len(previous), cross_midnight_clips=len(crossing),
        cross_midnight_native_value_checks=verified, domain_lonlat=config['domain'],
        maximum_new_pair_shared_fraction=max(r['maximum_new_pair_shared_fraction'] for r in checks),
        parent_files_reused=True, validation_manifests_identical=True)
    write_json(audit/'summary.json', summary)
    observed = pd.DatetimeIndex(pd.to_datetime([t for r in plan['records'].values()
                                               for t in r['times']], utc=True)).as_unit('ns')
    timestamps = pd.DatetimeIndex(np.concatenate([pd.date_range(day, periods=288, freq='5min', tz='UTC').asi8
        for day in sorted(plan['records'])])).tz_localize('UTC')
    series, comparison = {}, {}
    boundary_stamps = []
    for day in sorted(plan['records']):
        next_day = (pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if next_day not in plan['records']:
            continue
        midnight = pd.Timestamp(next_day, tz='UTC')
        span = pd.date_range(midnight-pd.Timedelta(minutes=75), midnight+pd.Timedelta(minutes=75), freq='5min')
        if span.isin(observed).all():
            boundary_stamps.append(span)
    for label, manifest in [('daily_windows', old), ('continuous_windows', frame)]:
        counts = temporal_counts(manifest, timestamps)
        table = pd.DataFrame(dict(timestamp=timestamps, active=counts, observed=timestamps.isin(observed)))
        table['minute'] = table.timestamp.dt.hour*60+table.timestamp.dt.minute
        table.to_csv(audit/f'temporal_{label}.csv', index=False)
        near = np.stack([temporal_counts(manifest, stamps) for stamps in boundary_stamps])
        comparison[label] = dict(samples=len(manifest), mean_active=float(counts.mean()),
            zero_active_observed=int(((counts == 0) & table.observed).sum()),
            zero_in_hours_00_and_23_observed=int(((counts == 0) & table.observed &
                table.timestamp.dt.hour.isin([0, 23])).sum()),
            consecutive_training_boundaries=len(near), boundary_zero_slots=int((near == 0).sum()),
            boundary_minimum_active=int(near.min()),
            boundary_midnight_mean=float(near[:,15].mean()))
        series[label] = table, near
    write_json(audit/'temporal_comparison.json', comparison)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout='constrained')
    for key, color, title in [('daily_windows', '#777777', f'Daily windows · {len(old):,} clips'),
                              ('continuous_windows', '#ed542f', f'Continuous windows · {len(frame):,} clips')]:
        table, near = series[key]
        mean = table.groupby('minute').active.mean()
        axes[0].plot(mean.index/60, mean, color=color, lw=1.7, label=title)
        axes[1].plot(np.arange(-75,76,5), near.mean(axis=0), color=color, lw=1.7, label=title)
        freq = table.active.value_counts(normalize=True).sort_index()*100
        axes[2].plot(freq.index, freq, color=color, marker='o', ms=3, label=title)
    axes[0].set(title='Temporal coverage on training dates', xlabel='Hour (UTC)',
        ylabel='Mean active clips', xticks=np.arange(0,25,4), xlim=(0,24))
    axes[1].set(title='Consecutive training dates', xlabel='Minutes from midnight (UTC)',
        ylabel='Mean active clips', xticks=[-75,-60,-30,0,30,60,75], xlim=(-75,75))
    axes[1].axvline(0, color='#aaaaaa', lw=.8)
    ceiling = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]: ax.set_ylim(0, ceiling)
    axes[2].set(title='Distribution of simultaneous clips', xlabel='Active clips',
        ylabel='Five-minute slots (%)', xticks=np.arange(0, max(table.active)+1,4))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'temporal_comparison.png', dpi=150)
    plt.close(fig)
    fig, origins = manifest_coverage_figure(frame, dataset/'sources',
        title=f'Continuous sampling · {len(frame):,} clips')
    fig.savefig(output/'coverage.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    origins.to_csv(audit/'origins.csv', index=False)
    print(json.dumps(dict(summary=summary, temporal=comparison), indent=2), flush=True)


def compare_fresh_selection(dataset, fresh_dataset, output):
    """Compare identical dates with a fresh selection and a retained parent."""
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((fresh_dataset/'preparation_plan.json').read_text())
    config = plan['config']
    if config.get('parent_dataset'):
        raise ValueError('The fresh control must start with an empty candidate pool')
    fresh = pd.read_csv(fresh_dataset/'training/dataset/train.csv')
    retained = pd.read_csv(dataset/'training/dataset/train.csv')
    parent = Path(json.loads((dataset/'preparation_plan.json').read_text())['config']['parent_dataset'])
    original = pd.read_csv(parent/'training/dataset/train.csv')
    days = sorted(plan['records'])
    observed = pd.DatetimeIndex(pd.to_datetime([t for r in plan['records'].values() for t in r['times']], utc=True))
    stamps = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=288, freq='5min', tz='UTC').asi8 for d in days])).tz_localize('UTC')
    midnight_spans = []
    for day in days:
        midnight = pd.Timestamp(day, tz='UTC')
        if (midnight-pd.Timedelta(days=1)).strftime('%Y-%m-%d') in days:
            span = pd.date_range(midnight-pd.Timedelta(minutes=75), midnight+pd.Timedelta(minutes=75), freq='5min')
            if span.isin(observed).all(): midnight_spans.append(span)
    checks = []
    for day in days:
        report = audit_added_pairs(event_array(temporal_context(fresh, day)), 0)
        if report['maximum_active_iou'] > .2+1e-12 or report['maximum_new_pair_shared_fraction'] > .2+1e-12:
            raise ValueError('Fresh control violates overlap limits')
        checks.append(report)
    assert_day_separation(fresh, config['reserved_days'], config['temporal_separation_hours'])
    figures, stats = plt.subplots(1, 2, figsize=(12,4.5), layout='constrained'), {}
    fig, axes = figures
    for name, manifest, color, title in [
            ('daily_windows', original, '#777777', 'Daily windows'),
            ('retained_continuous', retained, '#ed542f', 'Continuous, previous clips retained'),
            ('fresh_continuous', fresh, '#2166ac', 'Continuous, fresh selection')]:
        count = temporal_counts(manifest, stamps)
        near = np.stack([temporal_counts(manifest, span) for span in midnight_spans])
        minutes = stamps.hour*60+stamps.minute
        mean = pd.Series(count).groupby(minutes).mean()
        n = int(manifest.day.isin(days).sum())
        label = f'{title} · {n:,} clips'
        axes[0].plot(mean.index/60, mean, color=color, label=label, lw=1.7)
        axes[1].plot(np.arange(-75,76,5), near.mean(axis=0), color=color, label=label, lw=1.7)
        stats[name] = dict(samples_in_period=n, mean_active=float(count.mean()),
            boundary_midnight_mean=float(near[:,15].mean()), boundary_zero_slots=int((near == 0).sum()),
            boundary_minimum_active=int(near.min()))
    axes[0].set(title='Matched observation dates', xlabel='Hour (UTC)', ylabel='Mean active clips',
        xticks=np.arange(0,25,4), xlim=(0,24))
    axes[1].set(title='Coverage across midnight', xlabel='Minutes from midnight (UTC)',
        ylabel='Mean active clips', xticks=[-75,-60,-30,0,30,60,75], xlim=(-75,75))
    ceiling = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0,ceiling)
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.savefig(output/'fresh_week_comparison.png', dpi=150)
    plt.close(fig)
    starts = pd.to_datetime(fresh.start_time, utc=True)
    report = dict(start_date=days[0], end_date=days[-1], consecutive_training_boundaries=len(midnight_spans),
        fresh_samples=len(fresh), cross_midnight_clips=int((starts.dt.normalize() !=
            (starts+pd.Timedelta(minutes=75)).dt.normalize()).sum()),
        maximum_active_iou=max(r['maximum_active_iou'] for r in checks),
        maximum_shared_fraction=max(r['maximum_new_pair_shared_fraction'] for r in checks), comparisons=stats)
    write_json(output/'audit/fresh_week_comparison.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh-dataset', type=Path, help='Compare a fresh control on the same observation dates')
    args = parser.parse_args()
    if args.fresh_dataset:
        compare_fresh_selection(args.dataset, args.fresh_dataset, args.output)
    else:
        analyze(args.dataset, args.output)
