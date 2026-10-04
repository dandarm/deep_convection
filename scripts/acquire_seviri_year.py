#!/usr/bin/env python3
"""Resume a SEVIRI RSS year, keeping verified seven-channel daily Zarr stores.

New native products use a private daily staging buffer and are removed only
when the processed store has been verified and published. Existing Zarr data
and native files outside this buffer are never removed.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
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
from download_msg_seviri import credentials, download_interval, DEFAULT_COLLECTION
from crop_msg_seviri_native import append_batch, existing_times
from emma_gpm.seviri import geographic_bbox_to_geos_bounds,load_native_crop,native_slot_start_from_name,THERMAL_CHANNELS


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def product_slot(product):
    value=product.sensing_start
    if value.tzinfo is not None:
        value=value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(minute=(value.minute//5)*5,second=0,microsecond=0)


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
    products=[p for p in collection.search(dtstart=day,dtend=following,set='brief') if day<=product_slot(p)<following]
    before=existing_times(store)
    expected_slots={pd.Timestamp(product_slot(p)) for p in products}
    if len(products) != len(expected_slots):
        raise ValueError('Multiple provider products for the same RSS slot')
    missing=[p for p in products if pd.Timestamp(product_slot(p)) not in before]
    buffer=work/'native_buffer'/f'{day:%Y%m%d}'
    if missing:
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
    pending=[]; converted=set(); rejected=[]
    for native in sorted(buffer.rglob('*.nat')):
        slot=native_slot_start_from_name(native)
        if not day<=slot<following:
            raise ValueError('Native belongs to a different requested day')
        if pd.Timestamp(slot) in before or pd.Timestamp(slot) in converted:
            continue
        crop=load_native_crop(native,bounds,calibration='brightness_temperature')
        if crop.channels!=THERMAL_CHANNELS or crop.units!='K':
            raise ValueError('Native spectral/radiometric metadata mismatch')
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
            print(f'converted day={day:%Y-%m-%d} frames={len(converted)}/{len(missing)}',flush=True)
    if pending:
        append_batch(staging,pending,'float16')
    rejected_slots={pd.Timestamp(row['filename_slot']) for row in rejected}
    if not (expected_slots-rejected_slots).issubset(before | converted):
        raise ValueError('Some provider products were not converted')
    all_times=before | converted
    if all_times:
        verify_store(staging,all_times)
        store.parent.mkdir(parents=True,exist_ok=True)
        if store.exists():
            backup.parent.mkdir(parents=True,exist_ok=True);store.replace(backup)
        staging.replace(store)
        verify_store(store,all_times)
        shutil.rmtree(backup,ignore_errors=True)
    expected_nominal=set(pd.date_range(day,following-timedelta(minutes=5),freq='5min'))
    report=dict(day=f'{day:%Y-%m-%d}',provider_products=len(products),previous_frames=len(before),
        new_frames=len(converted),stored_frames=len(all_times),stored_times=[t.isoformat() for t in sorted(all_times)],
        missing_nominal_slots=[t.isoformat() for t in sorted(expected_nominal-all_times)],
        rejected_products=rejected,
        empty_provider_day=not bool(products),store=str(store))
    atomic_json(marker,report)
    shutil.rmtree(buffer,ignore_errors=True)
    print(f'completed day={day:%Y-%m-%d} stored={len(all_times)} gaps={len(report["missing_nominal_slots"])}',flush=True)
    return report


def main():
    import eumdac
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr-root',type=Path,required=True)
    parser.add_argument('--year',type=int,default=2020)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--crop-batch-size',type=int,default=4)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    args=parser.parse_args()
    if args.workers<1 or args.crop_batch_size<1:
        parser.error('Worker and batch counts must be positive')
    args.zarr_root=args.zarr_root.resolve();args.zarr_root.mkdir(parents=True,exist_ok=True)
    work=args.zarr_root/'_acquisition'/str(args.year)
    work.mkdir(parents=True,exist_ok=True)
    from download_msg_seviri import load_dotenv
    load_dotenv(args.env_file)
    with (work/'acquisition.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,secret=credentials()
        collection=eumdac.DataStore(eumdac.AccessToken((key,secret))).get_collection(DEFAULT_COLLECTION)
        day=datetime(args.year,1,1);stop=datetime(args.year+1,1,1)
        completed=0
        while day<stop:
            atomic_json(work/'progress.json',dict(status='running',current_day=f'{day:%Y-%m-%d}',completed_days=completed,
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
