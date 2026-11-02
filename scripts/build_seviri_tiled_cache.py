#!/usr/bin/env python3
"""Create verified lossless spatial tiles from an existing native-frame cache."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.file_lock import exclusive_file_lock
from emma_gpm.seviri_tiled_cache import build_tiled_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tile',type=int,default=16)
    args=parser.parse_args()
    with exclusive_file_lock(args.output/'.conversion.lock'):
        build_tiled_cache(args.source,args.output,tile=args.tile)


if __name__=='__main__':main()
