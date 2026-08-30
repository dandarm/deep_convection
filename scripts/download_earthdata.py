#!/usr/bin/env python3
"""Download selected CMR granules using an Earthdata bearer token.

Set EARTHDATA_TOKEN in the environment. The script deliberately never accepts a
token on the command line, keeping it out of shell history and process listings.
"""
from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise SystemExit("EARTHDATA_TOKEN is not set; create a free Earthdata Login/token first")
    table = pd.read_parquet(args.inventory)
    table = table[table.collection.eq(args.collection)]
    if args.limit:
        table = table.head(args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    for row in table.itertuples():
        destination = args.output / row.producer_granule_id
        partial = destination.with_suffix(destination.suffix + ".part")
        with requests.get(row.download_url, headers=headers, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(destination)
        print(destination)


if __name__ == "__main__":
    main()
