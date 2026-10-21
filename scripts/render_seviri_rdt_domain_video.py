#!/usr/bin/env python3
"""Export committed full-domain RDT maps without recomputing scientific labels."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap,Normalize,TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import CRS,Transformer
from emma_gpm.rdt.storage import DomainFrameReader,atomic_json,open_fields,frame_key,sha256
from emma_gpm.rdt_teacher import CLASS_COLORS


def build_renderer(protocol,extent,graticule,width,height):
    figure,panels=plt.subplots(3,2,figsize=(width/100,height/100),dpi=100,
        gridspec_kw={'width_ratios':[1,.023]})
    figure.subplots_adjust(left=.055,right=.945,bottom=.025,top=.958,hspace=.24,wspace=.035)
    axes,caxes=panels[:,0],panels[:,1]
    empty=np.ma.masked_array(np.zeros(protocol['shape'],np.float32),mask=True)
    btd=axes[0].imshow(empty,extent=extent,origin='upper',interpolation='nearest',
        cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-80,vcenter=0,vmax=10))
    figure.colorbar(btd,cax=caxes[0],label='Differenza BT (K)',extend='both')
    satellite=axes[1].imshow(empty,extent=extent,origin='upper',interpolation='nearest',cmap='gray_r',vmin=180,vmax=310)
    classes=axes[1].imshow(empty,extent=extent,origin='upper',interpolation='nearest',
        cmap=ListedColormap(CLASS_COLORS),norm=BoundaryNorm(np.arange(-.5,5.5),5),alpha=.85)
    caxes[1].set_visible(False)
    axes[1].legend(handles=[Patch(color=c,label=n) for c,n in zip(CLASS_COLORS[1:],
        ['Nube / non confermata','Sviluppo','Maturità / decadimento','OT candidato persistente'])],
        loc='upper center',bbox_to_anchor=(.5,-.1),ncol=4,frameon=False,fontsize=10)
    depth=axes[2].imshow(empty,extent=extent,origin='upper',interpolation='nearest',cmap='viridis',norm=Normalize(0,2))
    figure.colorbar(depth,cax=caxes[2],label='Profondità relativa D',extend='both')
    foreground=[]
    for axis,title in zip(axes,['WV 6.2 − IR 10.8','Classi RDT-CW-inspired su IR 10.8','Profondità D — proxy UTLS/incudine']):
        axis.set_title(title,fontsize=14)
        axis.set_facecolor('#d9d9d9')
        axis.set(xlabel='x proiettata (km)',ylabel='y proiettata (km)')
        xx,yy,lon,lat=graticule
        for values,levels in [(lon,range(-60,91,10)),(lat,range(20,91,10))]:
            lines=axis.contour(xx,yy,values,levels=levels,colors='#444444',linewidths=.45,alpha=.5)
            labels=axis.clabel(lines,inline=True,fontsize=8,fmt=lambda value:f'{value:g}°')
            foreground.extend((axis,artist) for artist in [lines,*labels])
    clock=figure.suptitle('',fontsize=17,y=.985)
    for artist in [btd,satellite,classes,depth,clock,*[artist for _,artist in foreground]]:
        artist.set_animated(True)
    figure.canvas.draw()
    if figure.canvas.get_width_height()!=(width,height):raise ValueError('Unexpected canvas size')
    background=figure.canvas.copy_from_bbox(figure.bbox)
    def draw(stamp,entry):
        valid=entry['valid_observation'];label=entry['semantic_label']
        btd.set_data(np.ma.masked_where(~valid,entry['btd']))
        satellite.set_data(np.ma.masked_where(~valid,entry['ir']))
        classes.set_data(np.ma.masked_where(~valid|(label==0),label))
        depth.set_data(np.ma.masked_where(~valid|(label==0)|~np.isfinite(entry['thermal_depth']),entry['thermal_depth']))
        clock.set_text(f'{stamp:%Y-%m-%d %H:%M} UTC  ·  SEVIRI / RDT-CW-inspired')
        figure.canvas.restore_region(background)
        for axis,artist in [(axes[0],btd),(axes[1],satellite),(axes[1],classes),(axes[2],depth),*foreground]:
            axis.draw_artist(artist)
        figure.draw_artist(clock)
        return figure.canvas.buffer_rgba().tobytes()
    return draw


def initialize(root,labels,protocol,geometry):
    import zarr
    from numcodecs import blosc
    global READER,LABELS,PROTOCOL,DRAW
    zarr.config.set({'async.concurrency':2,'threading.max_workers':1})
    blosc.set_nthreads(1)
    READER,LABELS,PROTOCOL=DomainFrameReader(root),Path(labels),protocol
    extent,bounds,width,height=geometry
    xx,yy=np.meshgrid(np.linspace(bounds[0],bounds[1],128),np.linspace(bounds[2],bounds[3],64))
    lon,lat=Transformer.from_crs(CRS.from_user_input(protocol['projection_wkt']),'EPSG:4326',always_xy=True).transform(xx*1000,yy*1000)
    graticule=(xx,yy,np.ma.masked_invalid(lon),np.ma.masked_invalid(lat))
    DRAW=build_renderer(protocol,extent,graticule,width,height)


def load_frame(stamp):
    raw=READER.load(stamp)
    x,y,projection=READER.geometry
    if hashlib.sha256(x.tobytes()+y.tobytes()+projection.encode()).hexdigest()!=PROTOCOL['geometry_sha256']:
        raise ValueError('Raw geometry differs from committed labels')
    directory=LABELS/'frames'/frame_key(stamp)
    record=json.loads((directory/'record.json').read_text())
    if hashlib.sha256(raw.tobytes()).hexdigest()!=record['source_frame_sha256']:
        raise ValueError(f'Raw frame changed since teacher calculation: {stamp}')
    with open_fields(directory) as group:
        if group.attrs['config_hash']!=PROTOCOL['config_hash'] or pd.Timestamp(group.attrs['time'])!=stamp:
            raise ValueError('Label timestamp/configuration mismatch')
        fields={name:np.asarray(group[name][:]) for name in ['semantic_label','valid_observation','thermal_depth']}
    if any(a.shape!=raw.shape[1:] for a in fields.values()) or tuple(raw.shape[1:])!=tuple(PROTOCOL['shape']):
        raise ValueError('Raw/label shapes differ')
    if np.any((fields['semantic_label']<0)|(fields['semantic_label']>4)):
        raise ValueError('Unknown semantic class')
    return dict(ir=raw[4].copy(),btd=raw[0]-raw[4],**fields)


def render_frame(stamp):
    return DRAW(stamp,load_frame(stamp))


def ordered_frames(times,root,labels,protocol,workers,geometry):
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(root,labels,protocol,geometry)) as pool:
        pending,remaining=deque(),iter(times)
        def enqueue():
            stamp=next(remaining,None)
            if stamp is not None:pending.append((stamp,pool.submit(render_frame,stamp)))
        try:
            for _ in range(2*workers):enqueue()
            while pending:
                stamp,future=pending.popleft()
                yield stamp,future.result()
                enqueue()
        finally:
            for _,future in pending:future.cancel()


def find_ffmpeg(explicit):
    if explicit:return explicit
    binary=shutil.which('ffmpeg')
    if binary:return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError('Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg') from error


def read_geometry(root,stamp,protocol):
    """Notebook geometry without scanning the whole annual viewer catalogue."""
    reader=DomainFrameReader(root)
    try:
        reader.load(stamp)
        x,y,projection=reader.geometry
        fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()+projection.encode()).hexdigest()
        if fingerprint!=protocol['geometry_sha256']:raise ValueError('Archive geometry changed')
        x=x if x[0]<x[-1] else x[::-1]
        y=y if y[0]>y[-1] else y[::-1]
        dx,dy=x[1]-x[0],y[1]-y[0]
        extent=((x[0]-dx/2)/1000,(x[-1]+dx/2)/1000,(y[-1]+dy/2)/1000,(y[0]-dy/2)/1000)
        xx,yy=np.meshgrid(np.linspace(x[0],x[-1],128),np.linspace(y[0],y[-1],64))
        lon,lat=Transformer.from_crs(CRS.from_user_input(projection),'EPSG:4326',always_xy=True).transform(xx,yy)
        return extent,(xx/1000,yy/1000,np.ma.masked_invalid(lon),np.ma.masked_invalid(lat))
    finally:reader.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-root',type=Path,default=ROOT/'results/rdt_preview/domain_utls_2020_parallel')
    parser.add_argument('--zarr-root',type=Path)
    parser.add_argument('--start-date',default='2020-09-12')
    parser.add_argument('--end-date',default='2020-09-20',help='Inclusive UTC date')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=float,default=12)
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--max-frames',type=int,help='Smoke export only; metadata reports the shortened interval')
    args=parser.parse_args()
    if args.fps<1 or args.width<640 or args.workers<1 or (args.max_frames is not None and args.max_frames<1):
        parser.error('Invalid export dimensions/rates/frame limit')
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads((args.label_root/'protocol.json').read_text())
    root=args.zarr_root or Path(protocol['root'])
    start,end=pd.Timestamp(args.start_date).normalize(),pd.Timestamp(args.end_date).normalize()+pd.Timedelta(days=1)
    if end<=start:parser.error('End must follow start')
    times=pd.date_range(start,end,freq='5min',inclusive='left')
    if args.max_frames:times=times[:args.max_frames]
    missing=[stamp for stamp in times if not (args.label_root/'frames'/frame_key(stamp)/'record.json').exists()]
    if missing:raise ValueError(f'Missing {len(missing)} committed frames; no frames repeated/interpolated: {missing[:5]}')
    # Reuse the notebook's native projection, north-up orientation and graticule.
    extent,graticule=read_geometry(root,times[0],protocol)
    height=int(np.ceil((args.width*.92*protocol['shape'][0]/protocol['shape'][1]*3+270)/2)*2)
    width=args.width+(args.width%2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_name(args.output.stem+'.part.mp4')
    progress=args.output.with_suffix('.progress.json')
    metadata=args.output.with_suffix('.json')
    configuration=dict(label_root=str(args.label_root.resolve()),zarr_root=str(root.resolve()),
        start_utc=times[0].isoformat(),last_utc=times[-1].isoformat(),frames=len(times),fps=args.fps,
        duration_seconds=len(times)/args.fps,width=width,height=height,cadence_minutes=5,
        btd_limits_k=[-80,0,10],depth_limits=[0,2],protocol_sha256=sha256(args.label_root/'protocol.json'),
        raw_frame_hashes_checked=True,scientific_labels_recomputed=False,renderer_sha256=sha256(Path(__file__)))
    atomic_json(metadata,configuration)
    command=[find_ffmpeg(args.ffmpeg),'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}',
        '-r',str(args.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
        '-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-f','mp4',str(partial)]
    started=time.monotonic();processed=0
    with args.output.with_suffix('.ffmpeg.log').open('w') as log:
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        bounds=(float(graticule[0][0,0]),float(graticule[0][0,-1]),float(graticule[1][0,0]),float(graticule[1][-1,0]))
        stream=ordered_frames(times,root,args.label_root,protocol,args.workers,(extent,bounds,width,height))
        try:
            for stamp,entry in stream:
                if not processed:Image.frombytes('RGBA',(width,height),entry).save(args.output.with_suffix('.preview.png'))
                encoder.stdin.write(entry)
                processed+=1
                if processed%12==0 or processed==len(times):
                    elapsed=time.monotonic()-started
                    state=dict(status='rendering',processed=processed,requested=len(times),last_time=stamp.isoformat(),
                        elapsed_seconds=elapsed,frames_per_second=processed/elapsed)
                    atomic_json(progress,state);print(json.dumps(state),flush=True)
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError(f'FFmpeg failed; see {args.output.with_suffix(".ffmpeg.log")}')
            partial.replace(args.output)
            atomic_json(progress,dict(state,status='complete',output=str(args.output.resolve()),bytes=args.output.stat().st_size))
        except BaseException as error:
            if encoder.poll() is None:encoder.kill();encoder.wait()
            atomic_json(progress,dict(status='error',processed=processed,requested=len(times),error=str(error)))
            raise
        finally:
            stream.close()


if __name__=='__main__':main()
