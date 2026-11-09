"""Aligned CMA/CT/RDT volumes, causal validity and inter-teacher acceptance.

Labels are produced on the complete satellite domain before spatial sampling.
The teachers' native class definitions and thresholds remain unchanged.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cloud_probe_data import check_split_days
from .rdt.storage import DomainFrameReader, atomic_json


CLASS_NAMES = {
    'cma': ['Clear', 'Cloud'],
    'ct': ['Clear', 'Low cloud', 'Mid cloud', 'High opaque', 'Cirrus / multilayer'],
    'rdt': ['Background', 'Cold non-convective', 'Triggering / growing',
            'Mature / decaying', 'Persistent overshooting top'],
}
CT_MAPPING = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 10: 1, 7: 2,
              8: 3, 9: 3, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4}
RDT_MAPPING = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4}


def map_classes(values, mapping):
    values = np.asarray(values)
    if not np.isin(values, [*mapping, 0, 255]).all():
        raise ValueError('Unexpected native teacher class')
    result = np.full(values.shape, 255, dtype=np.uint8)
    for native, target in mapping.items():
        result[values == native] = target
    return result


def valid_history(observations, fields, track_histories):
    """Background uses four valid observations; cells also need tracked history.

    Native temporal flags describe one-step parent continuity and domain-border
    contact. The downstream target instead requires an actual four-observation
    trajectory, preventing the teacher's short-history tendency fallback from
    becoming a fifteen-minute target.
    """
    current = np.asarray(fields['valid_observation'], bool)
    if len(observations) < 4:
        return np.zeros_like(current)
    history = np.logical_and.reduce(observations)
    tracked = np.zeros_like(current)
    expected = pd.date_range(pd.Timestamp(fields['time'])-pd.Timedelta(minutes=15), periods=4, freq='5min')
    for cell in fields['cells'].values():
        trajectory = track_histories.get(int(cell['track_id']), [])
        if len(trajectory) < 4:
            continue
        actual = pd.DatetimeIndex([entry['timestamp'] for entry in trajectory[-4:]])
        # This native field is finite only when the teacher actually used its
        # fifteen-minute reference; the fallback rate alone cannot prove that.
        has_real_tendency = np.isfinite(cell.get('cooling_15m_k', np.nan))
        if has_real_tendency and np.array_equal(actual.asi8, expected.asi8):
            tracked[cell['bbox']] |= cell['mask']
    return history & current & ((fields['cell_id_map'] == 0) | tracked)


def gate_counts(cma, ct, rdt, valid):
    convective = valid & (rdt >= 2) & (rdt <= 4)
    return dict(convective=int(convective.sum()),
                inside_cma=int((convective & (cma == 1)).sum()),
                inside_high_opaque=int((convective & (ct == 3)).sum()),
                inside_clear_low=int((convective & np.isin(ct, [0, 1])).sum()))


def quality_gate(counts):
    n = counts['convective']
    fractions = {key: counts[key] / n if n else None for key in
                 ('inside_cma', 'inside_high_opaque', 'inside_clear_low')}
    passed = bool(n and fractions['inside_cma'] >= .995
                  and fractions['inside_high_opaque'] >= .90
                  and fractions['inside_clear_low'] < .001)
    return dict(counts=counts, fractions=fractions, passed=passed,
                thresholds=dict(inside_cma_min=.995, inside_high_opaque_min=.90,
                                inside_clear_low_max_exclusive=.001))


def teacher_acceptance(report, policy='strict'):
    """Keep the original diagnostic separate from the approved acceptance policy."""
    if policy not in ('strict', 'ct_high_opaque_diagnostic'):
        raise ValueError('Unknown teacher acceptance policy: ' + policy)
    original = quality_gate(report['counts'])
    fractions = original['fractions']
    cloud_checks = bool(original['counts']['convective']
        and fractions['inside_cma'] >= .995 and fractions['inside_clear_low'] < .001)
    return dict(policy=policy,
        accepted=original['passed'] if policy == 'strict' else cloud_checks,
        original_gate_passed=original['passed'],
        high_opaque_fraction=fractions['inside_high_opaque'],
        high_opaque_role='required' if policy == 'strict' else 'diagnostic',
        required_inside_cma_min=.995, required_inside_clear_low_max_exclusive=.001)


def validate_manifest_acceptance(manifest):
    policy = manifest.get('teacher_acceptance_policy', 'strict')
    reports = manifest['teacher_quality_gates']
    if set(reports) != {'train', 'validation', 'test'}:
        raise ValueError('Teacher acceptance requires all three dataset splits')
    accepted = {split: teacher_acceptance(report, policy) for split, report in reports.items()}
    if not all(value['accepted'] for value in accepted.values()):
        raise ValueError('Training requires accepted full-domain teacher periods')
    if 'teacher_acceptance' in manifest and accepted != manifest['teacher_acceptance']:
        raise ValueError('Persisted teacher acceptance differs from its recorded policy')
    return accepted


def pretraining_days(manifests):
    """Include every day touched by a full clip, including midnight crossings."""
    days = set()
    for filename in manifests:
        frame = pd.read_csv(filename)
        starts = pd.to_datetime(frame['start_time'])
        days.update(starts.dt.strftime('%Y-%m-%d'))
        days.update((starts + pd.Timedelta(minutes=75)).dt.strftime('%Y-%m-%d'))
    return sorted(days)


def prepare_teacher_period(root, output, split, plan, geometry_function):
    """Stream a whole split; reset state only at genuine temporal discontinuities.

    A completed split is immutable. An interrupted split is recomputed causally
    from its first observation; diagnostic files are kept even if the gate fails.
    No checksum audits are performed inside training or RAM staging.
    """
    from CT_original.ct_nwcsaf.pipeline import NwcSafCmaCtTracker
    from CT_original.ct_nwcsaf.config import NwcSafCmaCtConfig
    from .rdt.pure_gen_utls import PureGenUtlsTracker
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = output / split
    folder.mkdir(exist_ok=True)
    excluded = pretraining_days(plan['pretraining_manifests'])
    check_split_days(plan['splits'], excluded)
    cfg = NwcSafCmaCtConfig.full_domain_pure_ir()
    ct_tracker = NwcSafCmaCtTracker(cfg)
    rdt_tracker = PureGenUtlsTracker(separate_observation_support=plan.get('rdt_separate_observation_support', True))
    protocol = dict(version=2, split=split, days=plan['splits'][split],
                    source_root=str(root.resolve()), excluded_pretraining_days=excluded,
                    ct_config=asdict(cfg), rdt_config_hash=rdt_tracker.config_hash,
                    ct_mapping=CT_MAPPING, rdt_mapping=RDT_MAPPING,
                    valid_history_minutes=15,
                    cell_history='four exact five-minute track timestamps and finite native cooling_15m_k',
                    teacher_domain='full archive rectangle',
                    orientation='north-up, west-left')
    # JSON canonicalization also normalizes integer mapping keys on resume.
    protocol = json.loads(json.dumps(protocol))
    previous_protocol = folder / 'protocol.json'
    if previous_protocol.exists():
        previous = json.loads(previous_protocol.read_text())
        # Additional pretraining manifests change the split audit, while the
        # physical teacher inputs, dates and configuration remain identical.
        physical = lambda record: {k:v for k,v in record.items() if k != 'excluded_pretraining_days'}
        if physical(previous) != physical(protocol):
            raise ValueError('Teacher protocol changed: choose a new output directory')
    else:
        atomic_json(previous_protocol, protocol)
    atomic_json(folder/'split_audit.json', dict(excluded_pretraining_days=excluded,
                                               manifests=plan['pretraining_manifests']))
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'complete.json').read_text())
    reader = DomainFrameReader(root)
    observations = deque(maxlen=4)
    previous_time, geometry = None, None
    counts = dict.fromkeys(['convective', 'inside_cma', 'inside_high_opaque', 'inside_clear_low'], 0)
    eligible_counts = dict(counts)
    native_confusion = np.zeros((7,5), dtype=np.int64)
    eligible_confusion = np.zeros((5,5), dtype=np.int64)
    frames, resets = [], []
    timings = dict(read=0., ct=0., rdt=0., save=0.)
    started = time.monotonic()
    try:
        for day in plan['splits'][split]:
            key = pd.Timestamp(day).strftime('%Y%m%d')
            source = root / key[:4] / (key + '.zarr')
            dataset = reader.reader._open_store(source)
            times = pd.DatetimeIndex(dataset.time.values)
            if times.hasnans or times.has_duplicates:
                raise ValueError('Invalid source timestamps')
            if geometry is None:
                geometry = geometry_function(source)
                with (folder / 'geometry.npz').open('wb') as stream:
                    np.savez_compressed(stream, **geometry)
            for stamp in times:
                if previous_time is None or stamp - previous_time != pd.Timedelta(minutes=5):
                    if previous_time is not None and stamp <= previous_time:
                        raise ValueError('Teacher observations must advance in time')
                    ct_tracker.reset(); rdt_tracker.reset(); observations.clear()
                    resets.append(stamp.isoformat())
                tick = time.monotonic()
                frame = reader.load(stamp)
                timings['read'] += time.monotonic() - tick
                observation = (np.isfinite(frame).all(0) & (frame >= 100).all(0)
                               & (frame <= 400).all(0))
                observations.append(observation)
                frame[:, ~observation] = np.nan
                tick = time.monotonic()
                ct_result = ct_tracker.step(frame, lat_deg=geometry['latitude'],
                                            lon_deg=geometry['longitude'])
                timings['ct'] += time.monotonic() - tick
                tick = time.monotonic()
                rdt_result = rdt_tracker.step_frame(frame, stamp, geometry['area_km2'],
                    lat_deg=geometry['latitude'], lon_deg=geometry['longitude'])
                timings['rdt'] += time.monotonic() - tick
                cma = np.asarray(ct_result.cma, dtype=np.uint8).copy()
                ct = map_classes(ct_result.ct, CT_MAPPING)
                rdt = map_classes(rdt_result['semantic_label'], RDT_MAPPING)
                # Gate uses all observed convective pixels after warm-up, before
                # cell-history filtering, so exclusions cannot manufacture a pass.
                if len(observations) == 4:
                    local = gate_counts(cma, ct, rdt, rdt_result['valid_observation'])
                    counts = {key: counts[key] + local[key] for key in counts}
                    observed = rdt_result['valid_observation'] & (ct < 5)
                    native_confusion += np.bincount(5*rdt_result['semantic_label'][observed].astype(np.int64)
                                                   + ct[observed], minlength=35).reshape(7,5)
                native_cma = cma.copy()
                valid = valid_history(observations, rdt_result, rdt_tracker.track_histories) & (cma < 2) & (ct < 5)
                local = gate_counts(cma, ct, rdt, valid)
                eligible_counts = {key: eligible_counts[key] + local[key] for key in eligible_counts}
                eligible_confusion += np.bincount(5*rdt[valid].astype(np.int64)+ct[valid], minlength=25).reshape(5,5)
                for labels in (cma, ct, rdt):
                    labels[~valid] = 255
                tick = time.monotonic()
                destination = folder / f'{stamp:%Y%m%dT%H%M}.npz'
                temporary = destination.with_suffix('.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, cma=cma, ct=ct, rdt=rdt, valid=valid,
                        quality=rdt_result['quality_mask'], native_cma=native_cma, native_ct=ct_result.ct,
                        native_rdt=rdt_result['semantic_label'])
                temporary.replace(destination)
                timings['save'] += time.monotonic() - tick
                frames.append(dict(time=stamp.isoformat(), file=destination.name,
                    valid_pixels=int(valid.sum()), rdt_support=np.bincount(rdt[valid], minlength=5).tolist()))
                previous_time = stamp
                atomic_json(folder / 'progress.json', dict(status='labeling', split=split,
                    frames=len(frames), last_time=stamp.isoformat(), timings_seconds=timings,
                    elapsed_seconds=time.monotonic()-started, quality_gate=quality_gate(counts)))
        report = dict(status='complete', frames=frames, resets=resets, timings_seconds=timings,
                      quality_gate=quality_gate(counts), eligible_target_gate=quality_gate(eligible_counts),
                      native_rdt_by_ct_confusion=native_confusion.tolist(),
                      eligible_rdt5_by_ct_confusion=eligible_confusion.tolist(),
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(folder / 'complete.json', report)
        atomic_json(folder / 'progress.json', {**report, 'frames': len(frames)})
        return report
    finally:
        reader.close()


def valid_origins(latitude, longitude, size=224, bounds=(-20., 40., 30., 70.)):
    """All pixel origins whose entire crop lies in the agreed geographic domain."""
    west, east, south, north = bounds
    inside = (np.isfinite(latitude) & np.isfinite(longitude) & (longitude >= west)
              & (longitude <= east) & (latitude >= south) & (latitude <= north))
    integral = np.pad(inside.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return np.argwhere(counts == size * size)


def bucket_from_summary(has_convection, high, cold, *, cold_centre=False,
                        policy='strict', high_fraction=.25, cold_fraction=.01):
    if policy not in ('strict', 'cold_cloud_centre'):
        raise ValueError('Unknown sampling policy: ' + policy)
    rich = high >= high_fraction and cold >= cold_fraction
    if rich and (not has_convection or (policy == 'cold_cloud_centre' and cold_centre)):
        return 'B'
    if has_convection:
        return 'A'
    return 'C' if high < .5 else None


def clip_bucket(ct, rdt, high_fraction=.25, cold_fraction=.01, policy='strict'):
    valid = (ct != 255) & (rdt != 255)
    if not valid.any():
        return None, False
    convective = bool(((rdt >= 2) & (rdt <= 4) & valid).any())
    high = (np.isin(ct, [3, 4]) & valid).sum() / valid.sum()
    cold = ((rdt == 1) & valid).sum() / valid.sum()
    final_ct, final_rdt = (ct[-1], rdt[-1]) if ct.ndim == 3 else (ct, rdt)
    centre = tuple(length // 2 for length in final_ct.shape)
    cold_centre = bool(final_rdt[centre] == 1 and final_ct[centre] in (3, 4))
    bucket = bucket_from_summary(convective, high, cold, cold_centre=cold_centre,
                                 policy=policy, high_fraction=high_fraction, cold_fraction=cold_fraction)
    return bucket, bool(((rdt == 4) & valid).any())


def sample_weights(samples):
    """50/30/20 probability mass; OT clips have 3x weight within bucket A."""
    weights = np.zeros(len(samples), dtype=np.float64)
    for bucket, mass in [('A', .5), ('B', .3), ('C', .2)]:
        indices = [i for i, row in enumerate(samples) if row['bucket'] == bucket]
        if not indices:
            raise ValueError('Missing training stratum: ' + bucket)
        factors = np.array([3. if bucket == 'A' and samples[i]['contains_ot'] else 1. for i in indices])
        weights[indices] = mass * factors / factors.sum()
    return weights


def _crop_sums(values, origins, size):
    integral = np.pad(values.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y, x = origins.T
    return integral[y+size, x+size] - integral[y, x+size] - integral[y+size, x] + integral[y, x]


def teacher_period_directory(plan, split):
    """Reuse immutable full-domain periods across downstream split assignments."""
    references = plan.get('teacher_periods')
    if references is not None:
        if set(references) != set(plan['splits']):
            raise ValueError('Explicit teacher periods must cover every downstream split')
        return Path(references[split])
    return Path(plan['teacher_output']) / split


def materialize_dataset(plan):
    """Publish after every period satisfies the explicitly recorded policy."""
    from .seviri_clips import SeviriZarrClipReader, IncompleteSeviriClipError
    from .file_lock import exclusive_file_lock
    output = Path(plan['dataset_output'])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(output / 'build.lock'):
        excluded = pretraining_days(plan['pretraining_manifests'])
        check_split_days(plan['splits'], excluded)
        atomic_json(output/'temporal_split_audit.json', dict(excluded_pretraining_days=excluded,
            manifests=plan['pretraining_manifests'], splits=plan['splits']))
        protocol = {**plan, 'version': 2, 'classes': CLASS_NAMES,
                    'normalization': 'one scalar mean/std fitted on downstream training clips',
                    'validation_sampling': 'uniform space/time candidates, without label stratification',
                    'warmup_frames': 3, 'ignore_index': 255}
        if (output / 'protocol.json').exists():
            if json.loads((output / 'protocol.json').read_text()) != protocol:
                raise ValueError('Dataset protocol changed')
        else:
            atomic_json(output / 'protocol.json', protocol)
        if (output / 'manifest.json').exists():
            return json.loads((output / 'manifest.json').read_text())
        folders = {split: teacher_period_directory(plan, split) for split in plan['splits']}
        reports = {split: json.loads((folders[split] / 'complete.json').read_text())
                   for split in plan['splits']}
        teacher_signature = None
        for split in reports:
            prepared = json.loads((folders[split]/'protocol.json').read_text())
            if prepared.get('version') != 2:
                raise ValueError('Teacher targets require the explicit real fifteen-minute history protocol (version 2)')
            if prepared['days'] != plan['splits'][split] or prepared['source_root'] != str(Path(plan['source_root']).resolve()):
                raise ValueError('Prepared teacher periods differ from the requested dataset')
            signature = {key: prepared[key] for key in (
                'ct_config', 'rdt_config_hash', 'ct_mapping', 'rdt_mapping',
                'valid_history_minutes', 'cell_history', 'teacher_domain', 'orientation')}
            if teacher_signature is not None and signature != teacher_signature:
                raise ValueError('Prepared periods use different teacher or target-validity definitions')
            teacher_signature = signature
        acceptance_policy = plan.get('teacher_acceptance_policy', 'strict')
        acceptance = {split: teacher_acceptance(report['quality_gate'], acceptance_policy)
                      for split, report in reports.items()}
        failed = [split for split, report in acceptance.items() if not report['accepted']]
        if failed:
            raise ValueError('Inter-teacher quality gate failed: ' + ', '.join(failed))
        rng = np.random.default_rng(plan['seed'])
        size, length = plan['crop_size'], plan['clip_frames']
        if (size, length) != (224, 16):
            raise ValueError('Real-data three-head clips require 16 frames at 224px')
        samples, rejected = [], []
        sums, squares, pixels = 0., 0., 0
        support = {split: {head: np.zeros(len(names), dtype=np.int64)
                          for head, names in CLASS_NAMES.items()} for split in reports}
        reader = SeviriZarrClipReader(plan['source_root'])
        try:
            for split, report in reports.items():
                folder = folders[split]
                with np.load(folder / 'geometry.npz') as geometry:
                    origins = valid_origins(geometry['latitude'], geometry['longitude'], size,
                                            plan['geographic_bounds'])
                    source_x, source_y = geometry['source_x'], geometry['source_y']
                if not len(origins):
                    raise ValueError('No crop fits inside geographic domain')
                frames = report['frames']
                times = pd.DatetimeIndex([row['time'] for row in frames])
                # Full target windows are immutable and shared by many crops.
                # At this dataset size, caching one split requires about 4 GiB.
                target_windows = {}
                def target_window(start):
                    if start not in target_windows:
                        arrays = {key: [] for key in (*CLASS_NAMES, 'valid', 'quality')}
                        for row in frames[start:start+length]:
                            with np.load(folder / row['file']) as archive:
                                for key in arrays:
                                    arrays[key].append(archive[key])
                        target_windows[start] = {key: np.stack(values) for key, values in arrays.items()}
                    return target_windows[start]
                candidates = []
                for start in range(3, len(frames)-length+1, plan['temporal_stride_frames']):
                    if not (np.diff(times[start:start+length].asi8) == pd.Timedelta(minutes=5).value).all():
                        continue
                    selected_origins = origins[rng.choice(len(origins), size=min(len(origins),
                                                    plan['candidate_origins_per_window']), replace=False)]
                    arrays = target_window(start)
                    valid = arrays['valid']
                    rdt, ct = arrays['rdt'], arrays['ct']
                    counts = {key: _crop_sums(values.sum(0), selected_origins, size)
                              for key, values in dict(valid=valid,
                                  conv=(rdt >= 2) & (rdt <= 4) & valid,
                                  ot=(rdt == 4) & valid, cold=(rdt == 1) & valid,
                                  high=np.isin(ct, [3, 4]) & valid).items()}
                    cy, cx = (selected_origins + size // 2).T
                    cold_centres = (rdt[-1, cy, cx] == 1) & np.isin(ct[-1, cy, cx], [3, 4])
                    for i, (y, x) in enumerate(selected_origins):
                        if not counts['valid'][i]:
                            continue
                        high = counts['high'][i] / counts['valid'][i]
                        cold = counts['cold'][i] / counts['valid'][i]
                        bucket = bucket_from_summary(bool(counts['conv'][i]), high, cold,
                            cold_centre=bool(cold_centres[i]), policy=plan.get('sampling_policy', 'strict'),
                            high_fraction=plan['bucket_b_min_high_fraction'],
                            cold_fraction=plan['bucket_b_min_cold_fraction'])
                        candidates.append(dict(split=split, start_index=start, start_time=times[start].isoformat(),
                            end_time=times[start+length-1].isoformat(), canonical_y=int(y), canonical_x=int(x),
                            bucket=bucket, contains_ot=bool(counts['ot'][i]),
                            cold_cloud_centre=bool(cold_centres[i]), high_cloud_fraction=float(high),
                            cold_cloud_fraction=float(cold), convective_fraction=float(counts['conv'][i]/counts['valid'][i])))
                rng.shuffle(candidates)
                required = plan['training_clips' if split == 'train' else split+'_clips']
                if split == 'train':
                    quotas = dict(A=round(required*.5), B=round(required*.3))
                    quotas['C'] = required - sum(quotas.values())
                else:
                    quotas = {'all': required}
                saved = dict.fromkeys(quotas, 0)
                for row in candidates:
                    bucket = row['bucket'] if split == 'train' else 'all'
                    if bucket not in quotas or saved[bucket] >= quotas[bucket]:
                        continue
                    y, x = row['canonical_y'], row['canonical_x']
                    native_y = len(source_y)-y-size if source_y[0] < source_y[-1] else y
                    native_x = len(source_x)-x-size if source_x[0] > source_x[-1] else x
                    try:
                        clip = reader.load(row['start_time'], origin_y=native_y, origin_x=native_x)
                    except IncompleteSeviriClipError as error:
                        rejected.append({**row, 'reason': str(error)})
                        continue
                    if (clip < 100).any() or (clip > 400).any():
                        rejected.append({**row, 'reason': 'Radiometry outside 100..400 K'})
                        continue
                    native = clip.astype(np.float16)
                    if not np.array_equal(native.astype(np.float32), clip):
                        raise ValueError('Raw radiometry cannot be stored losslessly as float16')
                    targets = {key: values[:, y:y+size, x:x+size]
                               for key, values in target_window(row['start_index']).items()}
                    identifier = f"{pd.Timestamp(row['start_time']):%Y%m%dT%H%M}_{y}_{x}"
                    destination = output / split
                    destination.mkdir(exist_ok=True)
                    row.update(id=identifier, clip=f'{split}/{identifier}.npy',
                               targets=f'{split}/{identifier}_targets.npz')
                    for name, data in [('clip', native), ('targets', targets)]:
                        path = output / row[name]
                        with path.with_suffix('.part').open('wb') as stream:
                            if name == 'clip':
                                np.save(stream, data)
                            else:
                                np.savez_compressed(stream, **data)
                        path.with_suffix('.part').replace(path)
                    if split == 'train':
                        sums += clip.sum(dtype=np.float64)
                        squares += np.square(clip, dtype=np.float64).sum()
                        pixels += clip.size
                    for head in CLASS_NAMES:
                        labels = targets[head]
                        support[split][head] += np.bincount(labels[labels != 255], minlength=len(CLASS_NAMES[head]))
                    samples.append(row); saved[bucket] += 1
                    atomic_json(output / 'progress.json', dict(status='materializing', split=split,
                        saved=saved, quotas=quotas, total_clips=len(samples), rejected=len(rejected)))
                    if saved == quotas:
                        break
                if saved != quotas:
                    atomic_json(output / 'sampling_failure.json', dict(split=split, saved=saved, required=quotas,
                        candidate_counts={str(k): sum(row['bucket'] == k for row in candidates) for k in ('A','B','C',None)}))
                    raise ValueError(f'Insufficient candidate crops for {split}: {saved} / {quotas}')
            mean = sums / pixels
            result = dict(version=2, samples=samples, rejected=rejected,
                statistics=dict(global_mean_k=mean, global_std_k=float(np.sqrt(squares/pixels-mean*mean)),
                                values=pixels, clips=plan['training_clips']),
                support={split: {head: values.tolist() for head, values in heads.items()}
                         for split, heads in support.items()},
                teacher_quality_gates={split: report['quality_gate'] for split, report in reports.items()},
                teacher_acceptance_policy=acceptance_policy, teacher_acceptance=acceptance,
                sampling_policy=plan.get('sampling_policy', 'strict'),
                protocol_sha256=hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest())
            atomic_json(output / 'manifest.json', result)
            atomic_json(output / 'progress.json', dict(status='completed', clips=len(samples)))
            return result
        finally:
            reader.close()
