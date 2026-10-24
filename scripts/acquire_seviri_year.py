#!/usr/bin/env python3
"""Resume a SEVIRI RSS year, keeping verified seven-channel daily Zarr stores.

New native products use a private daily staging buffer and are removed only
when the processed store has been verified and published. Existing Zarr data
and native files outside this buffer are never removed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import signal
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from emma_gpm.seviri_conversion import ordered_native_conversions
from emma_gpm.file_lock import exclusive_file_lock
from download_msg_seviri import credentials, download_interval, DEFAULT_COLLECTION
from crop_msg_seviri_native import append_batch, existing_times
from emma_gpm.seviri import geographic_bbox_to_geos_bounds,load_native_crop,native_slot_start_from_name,THERMAL_CHANNELS

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def update_phase(work,phase,**values):
    path=work/'progress.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    state.update(status='running',phase=phase,updated_unix=time.time(),**values)
    atomic_json(path,state)


def frames_per_second(frame_count,elapsed_seconds):
    """Return a finite startup rate when the platform timer has not advanced."""
    return frame_count/elapsed_seconds if elapsed_seconds>0 else 0.0


def product_slot(product):
    value=product.sensing_start
    if value.tzinfo is not None:
        value=value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(minute=(value.minute//5)*5,second=0,microsecond=0)


def product_satellite(product):
    match=re.match(r'^(MSG[1-4])-SEVI-',str(product))
    return match.group(1) if match else None


def selection_policy(preferred_satellite=None,satellite_priority=None):
    if satellite_priority is not None:
        priority=list(satellite_priority)
        if (not priority or len(set(priority))!=len(priority)
                or any(s not in {'MSG1','MSG2','MSG3','MSG4'} for s in priority)
                or preferred_satellite is not None):
            raise ValueError('Satellite priority must contain distinct MSG satellites and cannot accompany preferred-satellite')
        return dict(version='rss-satellite-overlap-v3',satellite_priority=priority,
            maximum_sensing_duration_seconds=300,
            unique_slot='retain available RSS satellite',same_satellite_duplicates='error')
    return dict(version='rss-satellite-overlap-v1',preferred_satellite=preferred_satellite,
        unique_slot='retain available RSS satellite',same_satellite_duplicates='error')


def pin_selection_policy(work,preferred_satellite=None,satellite_priority=None,allow_update=False):
    path=work/'selection_policy.json'
    policy=selection_policy(preferred_satellite,satellite_priority)
    if path.exists() and json.loads(path.read_text())!=policy:
        if not allow_update:
            raise ValueError('Satellite selection policy changed; resume with the recorded flags or explicitly use --update-selection-policy')
        previous=json.loads(path.read_text())
        transition=dict(previous_policy=previous,new_policy=policy,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            completed_day_reports={p.name:native_digest(p) for p in sorted((work/'completed_days').glob('*.json'))},
            existing_frames='preserved; new priority applies only to missing observations')
        digest=hashlib.sha256(json.dumps([previous,policy],sort_keys=True).encode()).hexdigest()
        atomic_json(work/'selection_policy_history'/f'{digest}.json',transition)
    atomic_json(path,policy)


def select_slot_products(products,preferred_satellite=None,satellite_priority=None):
    """Resolve cross-satellite overlap only; never drop a unique RSS slot."""
    selection_policy(preferred_satellite,satellite_priority)
    priority=list(satellite_priority) if satellite_priority is not None else [preferred_satellite]
    grouped=defaultdict(list)
    for product in products:
        grouped[product_slot(product)].append(product)
    selected,decisions=[],[]
    for slot,group in sorted(grouped.items()):
        if len(group)==1:
            selected.extend(group)
            continue
        satellites=[product_satellite(product) for product in group]
        chosen_satellite=next((s for s in priority if s is not None and s in satellites),None)
        if (chosen_satellite is None or None in satellites
                or len(set(satellites))!=len(satellites)
                ):
            raise ValueError(f'Multiple provider products for the same RSS slot {slot.isoformat()}; '
                f'cannot uniquely select satellite from priority {priority}: {[str(p) for p in group]}')
        chosen=group[satellites.index(chosen_satellite)]
        selected.append(chosen)
        decisions.append(dict(slot=slot.isoformat(),selected_product=str(chosen),
            selected_satellite=chosen_satellite,
            unselected_products=sorted(str(p) for p in group if p is not chosen),
            reason='explicit satellite priority at cross-satellite overlap'))
    return selected,decisions


def rss_eligible_products(products):
    """Reject longer scans before satellite ranking, preserving their provenance."""
    eligible,excluded=[],[]
    for product in products:
        end=getattr(product,'sensing_end',None)
        if end is None:
            if product_satellite(product) is not None:
                raise ValueError(f'Missing sensing end for RSS product {product}')
            # Legacy references without satellite metadata receive native validation.
            eligible.append(product)
            continue
        seconds=(end-product.sensing_start).total_seconds()
        if seconds<=0:
            raise ValueError(f'Invalid sensing interval for RSS product {product}')
        if seconds>300:
            excluded.append(dict(id=str(product),slot=product_slot(product).isoformat(),
                sensing_start=product.sensing_start.isoformat(),sensing_end=end.isoformat(),
                sensing_duration_seconds=seconds,
                reason='scan exceeds five-minute RSS interval; excluded before satellite priority'))
        else:
            eligible.append(product)
    return eligible,excluded


def retain_unselected_native(native,day,work):
    """Keep cached unselected bytes outside the disposable daily buffer."""
    destination=work/'unselected_native'/f'{day:%Y%m%d}'/native.name
    digest=native_digest(native)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if native_digest(destination)!=digest:
            raise ValueError('Previously retained unselected native has different bytes')
        native.unlink()  # Identical owned buffer copy; retained original remains.
    else:
        native.replace(destination)
    return dict(file=native.name,path=str(destination),sha256=digest,size_bytes=destination.stat().st_size)


def native_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check_native_header(path):
    # Independent of our crop/conversion code: fail only on the known reader
    # buffer error. Other parser errors must remain visible and blocking.
    from satpy import Scene
    try:
        Scene(reader='seviri_l1b_native',filenames=[str(path)])
    except ValueError as error:
        if str(error)=='buffer is smaller than requested size':
            return False
        raise
    return True


def recover_native(native,slot,products,bounds,args,work,key,secret,error):
    if not getattr(args,'recover_native_read_errors',False) or not isinstance(error,ValueError) or str(error)!='buffer is smaller than requested size':
        raise error
    if check_native_header(native):
        raise error  # The error is not independently reproducible in the file.
    matches=[p for p in products if product_slot(p)==slot]
    if len(matches)!=1:
        raise ValueError('Cannot uniquely identify provider product for recovery')
    directory=work/'recovery'/native.stem
    directory.mkdir(parents=True,exist_ok=True)
    original=directory/'original.nat'
    digest=native_digest(native)
    if original.exists() and native_digest(original)!=digest:
        raise ValueError('Recovery original differs from current native')
    if not original.exists():
        shutil.copy2(native,original)
    class SingleProduct:
        def search(self,**kwargs):return matches
    print(f'recovery redownload native={native.name}',flush=True)
    download_interval(SingleProduct(),slot,slot+timedelta(minutes=5),directory/'redownload',False,
        collection_id=DEFAULT_COLLECTION,key=key,secret=secret,workers=1,retries=2,read_timeout=300)
    candidates=list((directory/'redownload').rglob('*.nat'))
    if len(candidates)!=1 or candidates[0].name!=native.name:
        raise ValueError('Unexpected recovery payload')
    fresh=candidates[0]
    fresh_digest=native_digest(fresh)
    readable=check_native_header(fresh)
    report=dict(file=native.name,filename_slot=slot.isoformat(),sha256=digest,
        redownload_sha256=fresh_digest,size_bytes=native.stat().st_size,
        redownload_size_bytes=fresh.stat().st_size,original=str(original),
        redownload=str(fresh),error_type=type(error).__name__,error=str(error))
    if readable:
        crop=load_native_crop(fresh,bounds,calibration='brightness_temperature')
        # Keep the original for audit and publish repaired bytes atomically.
        temporary=native.with_suffix('.nat.recovered.part')
        shutil.copy2(fresh,temporary);temporary.replace(native)
        report['status']='recovered_by_redownload'
        atomic_json(directory/'report.json',report)
        print(f'recovery successful native={native.name}',flush=True)
        return crop,None
    if digest!=fresh_digest:
        report['status']='different_unreadable_payloads'
        atomic_json(directory/'report.json',report)
        raise ValueError('Recovery payload differs but remains unreadable; refusing automatic exclusion')
    quarantine=work/'quarantine'/f'{slot:%Y%m%d}'/native.name
    quarantine.parent.mkdir(parents=True,exist_ok=True)
    if quarantine.exists() and native_digest(quarantine)!=digest:
        raise ValueError('Quarantine differs from recovery original')
    if not quarantine.exists():shutil.copy2(native,quarantine)
    report.update(status='quarantined_identical_unreadable_payload',quarantine=str(quarantine),
        reason='independent header failure reproduced after identical redownload; frame excluded, no replacement')
    atomic_json(directory/'report.json',report)
    print(f'recovery quarantined native={native.name} slot={slot}',flush=True)
    return None,report


def verify_store(store,expected_times):
    with xr.open_zarr(store,consolidated=True,chunks=None) as ds:
        times=pd.DatetimeIndex(ds.time.values)
        if not times.is_unique or set(times) != set(pd.DatetimeIndex(expected_times)):
            raise ValueError('Processed timestamps differ from expected unique timestamps')
        if tuple(str(c) for c in ds.channel.values)!=THERMAL_CHANNELS:
            raise ValueError('Unexpected channel order')
        if ds.attrs.get('calibration') != 'brightness_temperature' or ds.attrs.get('units') != 'K':
            raise ValueError('Unexpected radiometry')
        if ds.seviri.dtype != np.float16 or (ds.sizes['y'],ds.sizes['x']) != (763,1762):
            raise ValueError('Unexpected domain or dtype')
        if not (np.all(np.diff(ds.x.values)>0) or np.all(np.diff(ds.x.values)<0)) or not (np.all(np.diff(ds.y.values)>0) or np.all(np.diff(ds.y.values)<0)):
            raise ValueError('Non-monotonic coordinates')
        # Require usable data, allowing the existing out-of-view border NaNs.
        for index in {0,len(times)//2,len(times)-1}:
            pixels=ds.seviri.isel(time=index).values
            finite=np.isfinite(pixels)
            if not finite.any() or np.isinf(pixels).any():
                raise ValueError('Unusable processed frame')
        return len(times)


def acquire_day(day,collection,args,work,key,secret):
    following=day+timedelta(days=1)
    store=args.zarr_root/f'{day:%Y}'/f'{day:%Y%m%d}.zarr'
    marker=work/'completed_days'/f'{day:%Y%m%d}.json'
    if marker.exists():
        update_phase(work,'verifying_completed_day')
        previous=json.loads(marker.read_text())
        if previous['stored_frames']:
            verify_store(store,pd.to_datetime(previous['stored_times']))
        print(f'skip verified day={day:%Y-%m-%d}',flush=True)
        # Recover a buffer left after successful publication but before cleanup.
        shutil.rmtree(work/'native_buffer'/f'{day:%Y%m%d}',ignore_errors=True)
        return previous
    if shutil.disk_usage(args.zarr_root).free < 50*2**30:
        raise RuntimeError('Less than 50 GiB free; acquisition paused without deleting archive data')
    staging=work/'processing'/f'{day:%Y%m%d}.zarr.part'
    backup=work/'backups'/f'{day:%Y%m%d}.zarr.backup'
    if backup.exists():
        if not store.exists():
            backup.replace(store)
        else:
            verify_store(store,existing_times(store))
            shutil.rmtree(backup)
    provider_products=[p for p in collection.search(dtstart=day,dtend=following,set='brief') if day<=product_slot(p)<following]
    rss_products,excluded_provider=rss_eligible_products(provider_products)
    preferred=getattr(args,'preferred_satellite',None)
    priority=getattr(args,'satellite_priority',None)
    products,decisions=select_slot_products(rss_products,preferred,priority)
    before=existing_times(store)
    expected_slots={pd.Timestamp(product_slot(p)) for p in products}
    selection_path=work/'product_selection'/f'{day:%Y%m%d}.json'
    retained_directory=work/'unselected_native'/f'{day:%Y%m%d}'
    retained=[dict(file=p.name,path=str(p),sha256=native_digest(p),size_bytes=p.stat().st_size)
              for p in sorted(retained_directory.glob('*.nat'))]
    selection=dict(day=f'{day:%Y-%m-%d}',policy=selection_policy(preferred,priority),
        provider_products=len(provider_products),selected_provider_products=len(products),
        excluded_non_rss_products=excluded_provider,
        selected_satellites=dict(Counter(product_satellite(p) or 'unknown' for p in products)),
        products=[dict(id=str(p),slot=product_slot(p).isoformat(),satellite=product_satellite(p),
            existing_frame_preserved=pd.Timestamp(product_slot(p)) in before) for p in products],
        decisions=decisions,retained_unselected_native=retained)
    atomic_json(selection_path,selection)
    resolved={pd.Timestamp(d['slot']):d for d in decisions}
    for decision in decisions:
        print(f'selected satellite={decision["selected_satellite"]} slot={decision["slot"]} product={decision["selected_product"]}',flush=True)
    missing=[p for p in products if pd.Timestamp(product_slot(p)) not in before]
    buffer=work/'native_buffer'/f'{day:%Y%m%d}'
    if missing:
        update_phase(work,'downloading',download_workers=args.workers,products_to_download=len(missing))
        class SelectedCollection:
            def search(self,**kwargs):
                return missing
        download_interval(SelectedCollection(),day,following,buffer,False,
            collection_id=DEFAULT_COLLECTION,key=key,secret=secret,
            workers=args.workers,retries=5,read_timeout=300)
    shutil.rmtree(staging,ignore_errors=True)
    if store.exists():
        shutil.copytree(store,staging)
    bounds=geographic_bbox_to_geos_bounds(-20.,40.,30.,70.)
    pending=[]; converted=set(); rejected=[]; scheduled=set()
    excluded_ids={row['id'] for row in excluded_provider}
    def eligible_natives():
        for native in sorted(buffer.rglob('*.nat'),key=lambda p:(native_slot_start_from_name(p),p.name)):
            if native.stem in excluded_ids:
                retained.append(retain_unselected_native(native,day,work))
                atomic_json(selection_path,selection)
                print(f'retained non-RSS native={native.name}; excluded before conversion',flush=True)
                continue
            slot=native_slot_start_from_name(native)
            if not day<=slot<following:
                raise ValueError('Native belongs to a different requested day')
            decision=resolved.get(pd.Timestamp(slot))
            if decision and native.stem!=decision['selected_product']:
                if native.stem not in decision['unselected_products']:
                    raise ValueError('Unexpected cached native at resolved satellite overlap')
                retained.append(retain_unselected_native(native,day,work))
                atomic_json(selection_path,selection)
                continue
            if pd.Timestamp(slot) in before or pd.Timestamp(slot) in converted:
                continue
            if native.name in getattr(args,'exclude_native',[]):
                # Explicit, file-specific exclusion after diagnosis; preserve bytes
                # outside the disposable buffer and expose the missing RSS slot.
                quarantine=work/'quarantine'/f'{day:%Y%m%d}'/native.name
                quarantine.parent.mkdir(parents=True,exist_ok=True)
                with native.open('rb') as stream:
                    digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if quarantine.exists():
                    with quarantine.open('rb') as stream:
                        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                            raise ValueError('Quarantine file differs from excluded native')
                else:
                    shutil.copy2(native,quarantine)
                rejected.append(dict(file=native.name,filename_slot=slot.isoformat(),
                    reason='explicit exclusion of diagnosed unreadable native; no replacement',
                    size_bytes=native.stat().st_size,sha256=digest,quarantine=str(quarantine)))
                atomic_json(work/'rejected_products'/f'{day:%Y%m%d}.json',rejected)
                print(f'excluded native={native.name} slot={slot} quarantine={quarantine}',flush=True)
                continue
            if slot in scheduled:
                raise ValueError(f'Multiple cached natives for the same RSS slot {slot}: {native.name}')
            scheduled.add(slot)
            yield native
    conversion_started=time.monotonic()
    conversion_workers=getattr(args,'conversion_workers',0)
    stream=ordered_native_conversions(eligible_natives(),bounds,workers=conversion_workers,
        threads=getattr(args,'conversion_threads',2),prefetch=getattr(args,'conversion_prefetch',2),
        serial_loader=load_native_crop)
    def conversion_progress():
        elapsed=time.monotonic()-conversion_started
        update_phase(work,'converting',current_day=f'{day:%Y-%m-%d}',
            converted_frames=len(converted),frames_to_convert=len(missing),
            conversion_workers=conversion_workers,conversion_threads=getattr(args,'conversion_threads',2),
            conversion_prefetch=getattr(args,'conversion_prefetch',2),conversion_batch_size=args.crop_batch_size,
            conversion_elapsed_seconds=elapsed,
            conversion_frames_per_second=frames_per_second(len(converted),elapsed))
    conversion_progress()
    try:
        for native,read_crop in stream:
            slot=native_slot_start_from_name(native)
            try:
                crop=read_crop()
            except Exception as error:
                try:
                    crop,rejection=recover_native(native,slot,products,bounds,args,work,key,secret,error)
                except Exception as recovery_error:
                    raise RuntimeError(f'Cannot read native {native.name}: {type(recovery_error).__name__}: {recovery_error}') from recovery_error
                if rejection is not None:
                    rejected.append(rejection)
                    atomic_json(work/'rejected_products'/f'{day:%Y%m%d}.json',rejected)
                    continue
            if crop.channels!=THERMAL_CHANNELS or crop.units!='K':
                raise ValueError('Native spectral/radiometric metadata mismatch')
            if getattr(crop,'end_time',crop.start_time+timedelta(minutes=5))-crop.start_time>timedelta(minutes=5):
                raise ValueError(f'Native scan exceeds five-minute RSS interval: {native.name}')
            if crop.start_time!=slot:
                rejected.append(dict(file=native.name,filename_slot=slot.isoformat(),
                    metadata_start=crop.start_time.isoformat(),metadata_end=crop.end_time.isoformat(),
                    reason='filename/Satpy timestamp mismatch; frame excluded, no replacement'))
                atomic_json(work/'rejected_products'/f'{day:%Y%m%d}.json',rejected)
                print(f'rejected native={native.name} filename_slot={slot} metadata_start={crop.start_time}',flush=True)
                continue
            if crop.data.shape!=(7,763,1762):
                raise ValueError('Native crop shape mismatch')
            pending.append(crop);converted.add(pd.Timestamp(slot))
            if len(pending)>=args.crop_batch_size:
                append_batch(staging,pending,'float16');pending=[]
                conversion_progress()
                print(f'converted day={day:%Y-%m-%d} frames={len(converted)}/{len(missing)}',flush=True)
    finally:
        stream.close()
    if pending:
        append_batch(staging,pending,'float16')
    conversion_seconds=time.monotonic()-conversion_started
    rejected_slots={pd.Timestamp(row['filename_slot']) for row in rejected}
    if not (expected_slots-rejected_slots).issubset(before | converted):
        raise ValueError('Some provider products were not converted')
    all_times=before | converted
    if all_times:
        update_phase(work,'verifying_and_publishing')
        verify_store(staging,all_times)
        store.parent.mkdir(parents=True,exist_ok=True)
        if store.exists():
            backup.parent.mkdir(parents=True,exist_ok=True);store.replace(backup)
        staging.replace(store)
        verify_store(store,all_times)
        shutil.rmtree(backup,ignore_errors=True)
    expected_nominal=set(pd.date_range(day,following-timedelta(minutes=5),freq='5min'))
    report=dict(day=f'{day:%Y-%m-%d}',provider_products=len(provider_products),
        selected_provider_products=len(products),selected_satellites=selection['selected_satellites'],
        satellite_selection_policy=selection['policy'],satellite_selection_decisions=decisions,
        product_selection_report=str(selection_path),previous_frames=len(before),
        conversion_workers=conversion_workers,conversion_threads=getattr(args,'conversion_threads',2),
        conversion_prefetch=getattr(args,'conversion_prefetch',2),conversion_batch_size=args.crop_batch_size,
        conversion_seconds=conversion_seconds,
        conversion_frames_per_second=frames_per_second(len(converted),conversion_seconds),
        new_frames=len(converted),stored_frames=len(all_times),stored_times=[t.isoformat() for t in sorted(all_times)],
        missing_nominal_slots=[t.isoformat() for t in sorted(expected_nominal-all_times)],
        rejected_products=rejected,
        empty_provider_day=not bool(products),store=str(store))
    atomic_json(marker,report)
    shutil.rmtree(buffer,ignore_errors=True)
    print(f'completed day={day:%Y-%m-%d} stored={len(all_times)} gaps={len(report["missing_nominal_slots"])}',flush=True)
    return report


def main():
    def stop_on_signal(signum,frame):
        raise KeyboardInterrupt(f'Acquisition interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop_on_signal)
    import eumdac
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--year',type=int,default=2020)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--conversion-workers',type=int,default=0,help='Independent native readers/calibrators; one parent writes Zarr')
    parser.add_argument('--conversion-threads',type=int,default=2,help='Dask threads per conversion process')
    parser.add_argument('--conversion-prefetch',type=int,default=2,help='Bound outstanding native conversions per process')
    parser.add_argument('--crop-batch-size',type=int,default=4)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    satellites=parser.add_mutually_exclusive_group()
    satellites.add_argument('--preferred-satellite',choices=['MSG1','MSG2','MSG3','MSG4'],
        help='Choose this satellite only at cross-satellite overlaps; unique RSS products are retained')
    satellites.add_argument('--satellite-priority',nargs='+',choices=['MSG1','MSG2','MSG3','MSG4'],
        help='Ordered satellite preference at overlaps, for example MSG4 MSG3 MSG2')
    parser.add_argument('--update-selection-policy',action='store_true',
        help='Explicitly archive and update pinned policy; completed days and existing frames are preserved')
    parser.add_argument('--recover-native-read-errors',action='store_true',
        help='Redownload independently unreadable native files; quarantine only identical payloads with the same reader buffer error')
    parser.add_argument('--exclude-native',action='append',default=[],metavar='FILENAME',
        help='Explicitly exclude a diagnosed native filename, retain it in quarantine and report its slot as missing; repeatable')
    args=parser.parse_args()
    try:
        selection_policy(args.preferred_satellite,args.satellite_priority)
    except ValueError as error:
        parser.error(str(error))
    if (args.workers<1 or args.crop_batch_size<1 or args.conversion_workers<0
            or args.conversion_threads<1 or args.conversion_prefetch<1):
        parser.error('Worker and batch counts must be positive')
    if any(Path(name).name!=name or not name.endswith('.nat') for name in args.exclude_native):
        parser.error('exclude-native must be an exact .nat filename without a path')
    args.zarr_root=args.zarr_root.resolve();args.zarr_root.mkdir(parents=True,exist_ok=True)
    work=args.zarr_root/'_acquisition'/str(args.year)
    work.mkdir(parents=True,exist_ok=True)
    from download_msg_seviri import load_dotenv
    load_dotenv(args.env_file)
    with exclusive_file_lock(work/'acquisition.lock'):
        pin_selection_policy(work,args.preferred_satellite,args.satellite_priority,args.update_selection_policy)
        key,secret=credentials()
        collection=eumdac.DataStore(eumdac.AccessToken((key,secret))).get_collection(DEFAULT_COLLECTION)
        day=datetime(args.year,1,1);stop=datetime(args.year+1,1,1)
        completed=0
        while day<stop:
            atomic_json(work/'progress.json',dict(status='running',phase='checking_or_searching',current_day=f'{day:%Y-%m-%d}',completed_days=completed,
                requested_days=(stop-datetime(args.year,1,1)).days,updated_unix=time.time()))
            succeeded=False
            for attempt in range(1,6):
                try:
                    report=acquire_day(day,collection,args,work,key,secret)
                    succeeded=True;break
                except Exception as error:
                    message=str(error).replace(key,'[redacted]').replace(secret,'[redacted]')
                    atomic_json(work/'last_error.json',dict(day=f'{day:%Y-%m-%d}',attempt=attempt,error_type=type(error).__name__,message=message))
                    print(f'day retry={attempt}/5 day={day:%Y-%m-%d} error_type={type(error).__name__}',flush=True)
                    if attempt<5:
                        time.sleep(min(60,attempt*10))
            if not succeeded:
                atomic_json(work/'progress.json',dict(status='failed',current_day=f'{day:%Y-%m-%d}',completed_days=completed))
                raise SystemExit('Acquisition stopped; see last_error.json, then resume same command')
            completed+=1;day+=timedelta(days=1)
        atomic_json(work/'progress.json',dict(status='completed',year=args.year,completed_days=completed))


if __name__=='__main__':
    main()
