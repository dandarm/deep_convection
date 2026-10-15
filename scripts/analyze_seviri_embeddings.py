#!/usr/bin/env python3
"""Extract primitive descriptors and compare frozen V2 encoders by clustering.

No forecasting, teacher, or trained readout. All artifacts stay under results.
Reruns reuse fingerprint-verified descriptor/embedding caches, never raw clips.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from emma_gpm.atmospheric_descriptors import describe_clip
from emma_gpm.embedding_clustering import (
    pool_encoder_tokens, unit_embeddings, hierarchy, explained_variance,
    cross_day_neighbor_gain, day_mean_interval, stability_scores, silhouette,
)
from emma_gpm.seviri_clips import SeviriZarrClipReader, SeviriVideoMAEDataset
from emma_gpm.pretraining_validation import assert_temporal_holdout
from pretrain_videomae import initialize_loader_worker, load_training_stats
from run_seviri_masking_experiments import write_json


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save_arrays(path, **values):
    temporary = path.with_suffix('.part')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def prepare_holdout(queue, output, root, workers, count, seed):
    """Freeze earliest seven verified complete days beyond all existing splits."""
    from build_random_seviri_clip_manifest import complete_start_times, parallel_candidates, utc_iso
    from run_seviri_v2_experiments import spaced_starts
    import xarray as xr
    path = output / 'test_diagnostic.csv'
    if path.exists():
        reservation = json.loads((output / 'reserved_days.json').read_text())
        if digest(path) != reservation['manifest_sha256'] or reservation['samples'] != count:
            raise ValueError('Frozen diagnostic holdout changed')
        return path
    validation = pd.read_csv(queue / 'validation.csv')
    boundary = pd.to_datetime(validation.start_time, utc=True).max().tz_localize(None) + pd.Timedelta(minutes=75, hours=24)
    times, selected_days, grid = [], [], None
    for marker in sorted((root / '_acquisition/2020/completed_days').glob('*.json')):
        day = pd.Timestamp(json.loads(marker.read_text())['day'])
        if day <= boundary.normalize():
            continue
        # Never inspect acquisition staging stores or the currently changing day.
        store = root / f'{day:%Y}' / f'{day:%Y%m%d}.zarr'
        with xr.open_zarr(store, consolidated=True, chunks=None) as data:
            available = pd.DatetimeIndex(data.time.values)
            starts = complete_start_times(available, num_frames=16, cadence_minutes=5)
            if len(starts) == 0:
                continue
            shape = (data.sizes['y'], data.sizes['x'])
            if grid is not None and grid != shape:
                raise ValueError('Holdout grid mismatch')
            grid = shape
            times.extend(starts.tolist())
            selected_days.append(day.strftime('%Y-%m-%d'))
        if len(selected_days) == 7:
            break
    if len(selected_days) < 7:
        raise RuntimeError('Wait for seven verified complete days beyond existing validation')
    rng = np.random.default_rng(seed)
    candidates = []
    height, width = grid
    for stamp in spaced_starts(pd.DatetimeIndex(times)):
        dy = int(rng.integers(height % 224 + 1))
        dx = int(rng.integers(width % 224 + 1))
        for y in range(height // 224):
            for x in range(width // 224):
                candidates.append((utc_iso(stamp), dy + 224 * y, dx + 224 * x))
    rng.shuffle(candidates)
    accepted, rejected = [], 0
    stream = parallel_candidates(iter(candidates), workers, (root, 16, 5, 224, False))
    try:
        for result in stream:
            if result is None:
                rejected += 1
                continue
            (stamp, y, x), _ = result
            accepted.append(dict(clip_id=f'diagnostic_{len(accepted):06d}', start_time=stamp,
                                 origin_y=y, origin_x=x))
            if len(accepted) == count:
                break
    finally:
        stream.close()
    if len(accepted) != count:
        raise ValueError('Insufficient finite diagnostic clips')
    frame = pd.DataFrame(accepted)
    assert_temporal_holdout(pd.read_csv(queue / 'train8000.csv'), frame)
    assert_temporal_holdout(validation, frame)
    temporary = path.with_suffix('.part')
    frame.to_csv(temporary, index=False)
    reservation = dict(days=selected_days, samples=count, seed=seed, rejected_candidates=rejected,
        manifest_sha256=digest(temporary), role='Exploratory diagnostic holdout, not a final untouched test',
        policy='Exclude these entire days from any subsequent SSL expansion and final test selection',
        embargo_hours=24, zarr_root=str(root), grid=list(grid))
    write_json(output / 'reserved_days.json', reservation)
    temporary.replace(path)
    write_json(queue / 'clustering_reserved_days.json', reservation)
    return path


def descriptor_worker_init(root):
    global _reader
    import zarr
    from numcodecs import blosc
    zarr.config.set({'async.concurrency': 4, 'threading.max_workers': 2})
    blosc.set_nthreads(1)
    _reader = SeviriZarrClipReader(root)


def descriptor_worker(row):
    values, schema = describe_clip(_reader.load(row['start_time'], origin_y=int(row['origin_y']),
                                               origin_x=int(row['origin_x'])))
    return values, schema


def extract_descriptors(manifest, destination, root, workers, identity, status):
    cache = destination / 'descriptors.csv'
    metadata = destination / 'descriptors_identity.json'
    if cache.exists() and metadata.exists():
        recorded = json.loads(metadata.read_text())
        if recorded['identity'] != identity or recorded['csv_sha256'] != digest(cache):
            raise ValueError('Descriptor cache does not match source/protocol')
        return pd.read_csv(cache), json.loads((destination / 'descriptor_schema.json').read_text())
    frame = pd.read_csv(manifest)
    rows, schema = [], None
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=descriptor_worker_init, initargs=(root,)) as executor:
        for index, (values, schema) in enumerate(executor.map(descriptor_worker, frame.to_dict('records'), chunksize=8)):
            rows.append(values)
            if (index + 1) % 128 == 0:
                status('descriptors', split=destination.name, completed=index + 1, total=len(frame))
    descriptors = pd.DataFrame(rows)
    temporary = cache.with_suffix('.part')
    descriptors.to_csv(temporary, index=False)
    temporary.replace(cache)
    write_json(destination / 'descriptor_schema.json', schema)
    write_json(metadata, dict(identity=identity, csv_sha256=digest(cache)))
    return descriptors, schema


def extract_embeddings(manifest, destination, run, root, model_path, stats, device,
                       workers, batch_size, identity, status, seed):
    import torch
    from emma_gpm.videomae_v2 import VideoMAEV2ForPreTraining
    from transformers import VideoMAEConfig
    cache = destination / f'{run}_embeddings.npz'
    metadata = destination / f'{run}_embeddings_identity.json'
    if cache.exists() and metadata.exists():
        recorded = json.loads(metadata.read_text())
        if recorded['identity'] != identity or recorded['npz_sha256'] != digest(cache):
            raise ValueError('Embedding cache does not match source/protocol')
        with np.load(cache, allow_pickle=False) as arrays:
            return {name: arrays[name].copy() for name in arrays.files}
    torch.manual_seed(seed)
    model = (VideoMAEV2ForPreTraining(VideoMAEConfig.from_pretrained(model_path)) if run == 'random'
             else VideoMAEV2ForPreTraining.from_pretrained(model_path))
    encoder = model.videomae.eval().requires_grad_(False).to(device)
    del model
    reader = SeviriZarrClipReader(root)
    dataset = SeviriVideoMAEDataset(manifest, reader, global_mean=stats['global_mean_k'],
                                   global_std=stats['global_std_k'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == 'cuda',
        **(dict(multiprocessing_context='spawn', prefetch_factor=1,
                worker_init_fn=initialize_loader_worker) if workers else {}))
    pooled = {'mean': [], 'mean_slope': []}
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            pixels = batch['pixel_values'].to(device, non_blocking=loader.pin_memory)
            # All encoder tokens, FP32, no decoder and no stochastic masking.
            tokens = encoder(pixel_values=pixels, bool_masked_pos=None).last_hidden_state
            for name, values in pool_encoder_tokens(tokens).items():
                pooled[name].append(values.cpu().numpy())
            completed += len(pixels)
            if completed % 128 == 0 or completed == len(dataset):
                status('embeddings', split=destination.name, run=run, completed=completed, total=len(dataset))
    arrays = {name: np.concatenate(values) for name, values in pooled.items()}
    save_arrays(cache, **arrays)
    write_json(metadata, dict(identity=identity, npz_sha256=digest(cache)))
    del loader, encoder
    reader.close()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return arrays


def analyze_one(values, descriptors, frame, schema, destination, run, pooling, method, counts, seed):
    prefix = f'{run}_{pooling}_{method}'
    days = pd.to_datetime(frame.start_time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    tree, distances = hierarchy(values, method)
    np.save(destination / f'{prefix}_linkage.npy', tree, allow_pickle=False)
    partitions = {count: cut_tree(tree, n_clusters=count).ravel() for count in counts}
    stability = stability_scores(values, days, method, counts, partitions, seed=seed)
    matrix = descriptors.to_numpy(dtype=np.float64)
    gains, neighbors = cross_day_neighbor_gain(distances, matrix, days)
    mean, low, high = day_mean_interval(gains, days, seed)
    neighbor_rows = [dict(run=run, pooling=pooling, method=method, descriptor=name,
        group=schema[name]['group'], gain=mean[index], ci_low=low[index], ci_high=high[index])
        for index, name in enumerate(descriptors)]
    save_arrays(destination / f'{prefix}_neighbors.npz', indices=neighbors, gain_by_clip=gains)
    metrics, physical, representatives = [], [], []
    from sklearn.metrics import adjusted_mutual_info_score
    stamps = pd.to_datetime(frame.start_time, utc=True)
    hours = stamps.dt.hour.to_numpy() + stamps.dt.minute.to_numpy() / 60
    nuisance = pd.DataFrame(dict(native_origin_y=frame.origin_y.to_numpy(),
        native_origin_x=frame.origin_x.to_numpy(), utc_hour_sin=np.sin(2 * np.pi * hours / 24),
        utc_hour_cos=np.cos(2 * np.pi * hours / 24)))
    assignments = frame.copy()
    for count, labels in partitions.items():
        scores = explained_variance(matrix, labels)
        assignments[f'k{count}'] = labels
        sizes = np.bincount(labels)
        metrics.append(dict(run=run, pooling=pooling, method=method, clusters=count,
            silhouette=silhouette(distances, labels, seed), stability_ari=stability[count],
            utc_day_adjusted_mutual_information=float(adjusted_mutual_info_score(days, labels)),
            smallest_cluster=int(sizes.min()), largest_cluster=int(sizes.max()),
            singleton_fraction=float(np.mean(sizes == 1))))
        for index, name in enumerate(descriptors):
            physical.append(dict(run=run, pooling=pooling, method=method, clusters=count,
                descriptor=name, group=schema[name]['group'], adjusted_eta2=scores[index]))
        for name, score in zip(nuisance, explained_variance(nuisance.to_numpy(), labels)):
            physical.append(dict(run=run, pooling=pooling, method=method, clusters=count,
                descriptor=name, group='nuisance_context', adjusted_eta2=score))
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            within = distances[np.ix_(members, members)].mean(1)
            medoid = members[np.argmin(within)]
            edge = members[np.argmax(distances[medoid, members])]
            rng = np.random.default_rng(seed + count * 1000 + int(label))
            for role, index in [('medoid', medoid), ('edge', edge),
                                ('random', int(rng.choice(members)))]:
                representatives.append(dict(run=run, pooling=pooling, method=method,
                    clusters=count, cluster=int(label), role=role, index=int(index),
                    **frame.iloc[index].to_dict()))
    assignments.to_csv(destination / f'{prefix}_assignments.csv', index=False)
    pd.DataFrame(representatives).to_csv(destination / f'{prefix}_representatives.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 4))
    dendrogram(tree, truncate_mode='lastp', p=24, ax=axis, show_leaf_counts=True)
    axis.set(title=f'{run} / {pooling} / {method} — {destination.name}',
             xlabel='Aggregated branches (counts in parentheses)', ylabel='Linkage distance')
    fig.tight_layout(); fig.savefig(destination / f'{prefix}_dendrogram.png', dpi=150); plt.close(fig)
    return metrics, physical, neighbor_rows


def comparison_plots(metrics, physical, neighbors, output):
    import matplotlib.pyplot as plt
    for split in metrics.split.unique():
        selected = metrics[(metrics.split == split) & (metrics.pooling == 'mean_slope') & (metrics.method == 'average')]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for run, rows in selected.groupby('run', sort=False):
            axes[0].plot(rows.clusters, rows.silhouette, '-o', label=run)
            axes[1].plot(rows.clusters, rows.stability_ari, '-o', label=run)
        for axis, label in zip(axes, ['Silhouette (geometry only)', 'Day subsampling stability (ARI)']):
            axis.set(xlabel='Number of clusters', ylabel=label); axis.grid(alpha=.2); axis.legend()
        fig.suptitle(f'{split} — mean + temporal tendency / average cosine')
        fig.tight_layout(); fig.savefig(output / f'{split}_geometry.png', dpi=150); plt.close(fig)
        selected = physical[(physical.split == split) & (physical.pooling == 'mean_slope') & (physical.method == 'average')]
        groups = ['thermal', 'spatial', 'spectral', 'dynamic']
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        for axis, group in zip(axes, groups):
            subset = selected[selected.group == group].groupby(['run', 'clusters']).adjusted_eta2.mean().reset_index()
            for run, rows in subset.groupby('run', sort=False):
                axis.plot(rows.clusters, rows.adjusted_eta2, '-o', label=run)
            axis.set(title=group, xlabel='Number of clusters'); axis.grid(alpha=.2)
        axes[0].set_ylabel('Mean adjusted explained variance'); axes[-1].legend()
        fig.suptitle(f'{split} — physical agreement, higher is better (no single winner selected)')
        fig.tight_layout(); fig.savefig(output / f'{split}_physical.png', dpi=150); plt.close(fig)
        selected = neighbors[(neighbors.split == split) & (neighbors.pooling == 'mean_slope') & (neighbors.method == 'average')]
        table = selected.pivot(index='descriptor', columns='run', values='gain')
        fig, axis = plt.subplots(figsize=(8, max(8, .23 * len(table))))
        shown = axis.imshow(table.to_numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
        axis.set_xticks(range(len(table.columns)), table.columns)
        axis.set_yticks(range(len(table.index)), table.index, fontsize=7)
        axis.set_title(f'{split} — cross-day neighbor gain; higher is better')
        fig.colorbar(shown, ax=axis, label='1 − physical error / random cross-day error')
        fig.tight_layout(); fig.savefig(output / f'{split}_neighbors.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', type=Path, default=ROOT / 'results/videomae_v2_small_scaling')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/seviri_embedding_clustering')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--test-count', type=int, default=256)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--clusters', type=int, nargs='+', default=[4, 8, 16, 32])
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.test_count < 64 or min(args.clusters) < 2:
        parser.error('Invalid workers/batch size/test count/cluster counts')
    import matplotlib
    matplotlib.use('Agg')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    queue, output = args.queue_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = 20261005
    root = Path(json.loads((queue / 'plan.json').read_text())['zarr_root']).resolve()
    def status(stage, **details):
        value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **details)
        write_json(output / 'progress.json', value)
        print(json.dumps(value), flush=True)
    with (output / 'analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Read-only analysis starts after all three identical-budget runs.
            stats = {}
            for run in ('mask50', 'mask75', 'mask90'):
                summary = json.loads((queue / run / 'smoke_summary.json').read_text())
                if (summary['epochs'], summary['samples'], summary['model_version'], summary['status']) != (150, 2000, 'v2', 'completed'):
                    raise ValueError('Expected completed 150-epoch V2 runs on 2000 clips')
                if summary['mask_ratio_requested'] != int(run[4:]) / 100:
                    raise ValueError('Unexpected masking ratio for run')
                stats[run] = load_training_stats(queue / run / 'channel_stats.json', queue / 'train2000.csv', root)
            if len({(value['global_mean_k'], value['global_std_k']) for value in stats.values()}) != 1:
                raise ValueError('The three models must share the same input normalization')
            status('preparing_diagnostic_holdout')
            test = prepare_holdout(queue, output, root, args.workers, args.test_count, seed)
            manifests = {'train': queue / 'train2000.csv', 'validation': queue / 'validation.csv', 'test_diagnostic': test}
            for path in manifests.values():
                if max(args.clusters) >= len(pd.read_csv(path)):
                    raise ValueError('Too many clusters for split')
            source_hashes = {name: digest(ROOT / 'src/emma_gpm' / name) for name in
                ('atmospheric_descriptors.py', 'embedding_clustering.py', 'seviri_clips.py', 'videomae_v2.py', 'videomae.py')}
            protocol = dict(seed=seed, manifests={name: digest(path) for name, path in manifests.items()},
                checkpoint='final150 exported encoder, never reconstruction-selected best',
                source_hashes=source_hashes, clusters=args.clusters, models=['mask50', 'mask75', 'mask90', 'random'],
                precision='fp32, no encoder/decoder masking; original global training transform',
                linkage={'average': 'cosine on unit embeddings', 'ward': 'Euclidean on unit embeddings'},
                poolings=['mean', 'mean_slope'], test_role='exploratory diagnostic holdout',
                stability='8 resamples of 80% whole days; adjusted Rand on common clips',
                neighbors='10 neighbors from different UTC start days; equal-day mean; day bootstrap 95% descriptive CI',
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                script_sha256=digest(__file__), input_stats=stats['mask50'])
            write_json(output / 'protocol.json', protocol)
            descriptions, frames, schema = {}, {}, None
            for split, manifest in manifests.items():
                destination = output / split; destination.mkdir(exist_ok=True)
                frames[split] = pd.read_csv(manifest)
                identity = dict(manifest_sha256=digest(manifest), root=str(root),
                                descriptors_source=source_hashes['atmospheric_descriptors.py'],
                                loader_source=source_hashes['seviri_clips.py'])
                descriptions[split], schema = extract_descriptors(manifest, destination, root, args.workers, identity, status)
                frames[split].to_csv(destination / 'manifest.csv', index=False)
            # Train-only descriptor moments are for diagnostic plots/metadata,
            # never applied to pixels fed into the encoder.
            write_json(output / 'descriptor_training_moments.json', dict(
                mean=descriptions['train'].mean().to_dict(), std=descriptions['train'].std(ddof=0).to_dict(),
                scope='Diagnostic descriptors only, fitted on training; encoder pixels unchanged'))
            all_metrics, all_physical, all_neighbors = [], [], []
            for run in protocol['models']:
                model_path = queue / ('mask50' if run == 'random' else run) / 'model'
                model_hash = (dict(random_seed=seed, config=digest(model_path / 'config.json')) if run == 'random' else
                    {path.name: digest(path) for path in sorted(model_path.glob('*')) if path.is_file()})
                for split, manifest in manifests.items():
                    destination = output / split
                    identity = dict(manifest_sha256=digest(manifest), model=model_hash, stats=stats['mask50'],
                                    source_hashes=source_hashes, precision='fp32', encoder_mask=None)
                    embeddings = extract_embeddings(manifest, destination, run, root, model_path, stats['mask50'],
                        device, args.workers, args.batch_size, identity, status, seed)
                    for pooling, values in embeddings.items():
                        for method in ('average', 'ward'):
                            status('clustering', split=split, run=run, pooling=pooling, method=method)
                            metrics, physical, neighbors = analyze_one(values, descriptions[split], frames[split],
                                schema, destination, run, pooling, method, args.clusters, seed)
                            for rows, target in ((metrics, all_metrics), (physical, all_physical), (neighbors, all_neighbors)):
                                target.extend(dict(split=split, **row) for row in rows)
                    # Incremental CSVs survive a later model failure.
                    for name, rows in [('cluster_metrics', all_metrics), ('descriptor_agreement', all_physical),
                                       ('neighbor_agreement', all_neighbors)]:
                        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
            comparison_plots(pd.DataFrame(all_metrics), pd.DataFrame(all_physical), pd.DataFrame(all_neighbors), output)
            status('completed', samples={name: len(frame) for name, frame in frames.items()}, descriptors=len(schema))
        except BaseException as error:
            status('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
