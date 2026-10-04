#!/usr/bin/env python3
"""Measure exact pixel/frame reuse in validated SEVIRI manifests."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from emma_gpm.spatial_sampling import pixel_frame_coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reference-manifest',type=Path,required=True)
    parser.add_argument('--height',type=int,required=True)
    parser.add_argument('--width',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.height<224 or args.width<224:
        parser.error('Domain smaller than crop')
    comparison={}
    for name,path in [('spaced',args.manifest),('reference',args.reference_manifest)]:
        comparison[name]=dict(manifest=str(path.resolve()),
            **pixel_frame_coverage(pd.read_csv(path),args.height,args.width))
    comparison['interpretation']='Exact repeated pixel/frame observations, not an estimate of statistical or meteorological independence.'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'overlap_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=['reference','spaced']
    labels=[f"Casuali ({comparison['reference']['samples']})",f"Separati ({comparison['spaced']['samples']})"]
    unique=[comparison[name]['unique_pixel_frames']/1e6 for name in names]
    repeated=[(comparison[name]['sampled_pixel_frames']-comparison[name]['unique_pixel_frames'])/1e6 for name in names]
    axes[0].bar(labels,unique,label='Osservazioni uniche')
    axes[0].bar(labels,repeated,bottom=unique,label='Riutilizzi',alpha=.6)
    axes[0].set_ylabel('Milioni di posizioni pixel/frame (un canale)')
    axes[0].set_title('Contributi alle clip: unici e ripetuti');axes[0].legend()
    axes[1].bar(labels,unique)
    axes[1].set_ylabel('Milioni di posizioni pixel/frame (un canale)')
    axes[1].set_title('Copertura unica effettiva')
    fig.tight_layout();fig.savefig(args.output_dir/'overlap_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
