#!/usr/bin/env python3
"""Acquire specific SEVIRI RSS days into E:\\Datasets\\MSG-SEVIRI\\<year>\\<date>.zarr."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emma_gpm.file_lock import exclusive_file_lock
from acquire_seviri_year import acquire_day, pin_selection_policy
from download_msg_seviri import credentials, load_dotenv, DEFAULT_COLLECTION
import eumdac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", required=True, help="Days to acquire (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--zarr-root", type=Path, default=Path(r"E:\Datasets\MSG-SEVIRI"))
    parser.add_argument("--workers", type=int, default=6, help="Download worker threads")
    parser.add_argument("--crop-batch-size", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key, secret = credentials()
    collection = eumdac.DataStore(eumdac.AccessToken((key, secret))).get_collection(DEFAULT_COLLECTION)

    for day_str in args.days:
        clean_str = day_str.replace("-", "")
        day = datetime.strptime(clean_str, "%Y%m%d")
        year = day.year
        work = args.zarr_root / "_acquisition" / str(year)
        work.mkdir(parents=True, exist_ok=True)

        day_args = argparse.Namespace(
            zarr_root=args.zarr_root,
            workers=args.workers,
            crop_batch_size=args.crop_batch_size,
            conversion_workers=0,
            conversion_threads=2,
            conversion_prefetch=2,
            preferred_satellite=None,
            satellite_priority=["MSG4", "MSG3", "MSG2", "MSG1"],
            update_selection_policy=False,
            recover_native_read_errors=True,
            exclude_native=[],
        )

        with exclusive_file_lock(work / "acquisition.lock"):
            pin_selection_policy(work, day_args.preferred_satellite, day_args.satellite_priority, day_args.update_selection_policy)
            print(f"\n==========================================", flush=True)
            print(f"Starting acquisition of {day:%Y-%m-%d}...", flush=True)
            print(f"==========================================", flush=True)
            succeeded = False
            for attempt in range(1, 6):
                try:
                    report = acquire_day(day, collection, day_args, work, key, secret)
                    succeeded = True
                    print(f"Acquisition succeeded for {day:%Y-%m-%d}: {report['stored_frames']} frames stored.", flush=True)
                    break
                except Exception as e:
                    print(f"Retry {attempt}/5 for {day:%Y-%m-%d}: {type(e).__name__}: {e}", flush=True)
                    if attempt < 5:
                        time.sleep(min(60, attempt * 10))
            if not succeeded:
                print(f"FAILED to acquire {day:%Y-%m-%d}!", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
