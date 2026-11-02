"""Lossless spatial tiling of daily arrays for arbitrary pixel-resolution crops."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .seviri_frame_cache import SeviriFrameCacheReader, file_digest, source_stamp
from .seviri_clips import (SeviriZarrClipReader, VIDEO_MAE_CHANNELS,
    IncompleteSeviriClipError, normalize_seviri_clip)

LAYOUT='tiled time,yblock,xblock,channel,y,x'


def storage_shape(shape,tile):
    t,c,h,w=shape
    return (t,(h+tile-1)//tile,(w+tile-1)//tile,c,tile,tile)


def build_tiled_cache(source,output,*,tile=16,frames_per_read=8):
    """Rearrange original float16 bits; verify every stored native pixel."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if tile<1 or frames_per_read<1:raise ValueError('Positive tile and read sizes required')
    original=json.loads((source/'metadata.json').read_text())
    if original.get('layout')!='native time,channel,y,x' or original.get('lossless') is not True:
        raise ValueError('Expected a verified native-frame cache')
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(source=str(source),source_metadata_sha256=file_digest(source/'metadata.json'),tile=tile,
        converter_sha256=file_digest(Path(__file__)))
    plan=output/'conversion_plan.json'
    if plan.exists() and json.loads(plan.read_text())!=identity:
        raise ValueError('Tiled cache conversion identity changed')
    plan.write_text(json.dumps(identity,indent=2)+'\n')
    archive=output/'source_archive';archive.mkdir(exist_ok=True)
    (archive/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records={}
    for key,entry in original['stores'].items():
        src=source/entry['file'];target=output/entry['file'];record=output/(Path(entry['file']).stem+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if saved['source_sha256']!=entry['sha256'] or file_digest(target)!=saved['sha256']:
                raise ValueError(f'Completed tiled file changed: {key}')
            records[key]=saved
            continue
        stamp=(src.stat().st_size,src.stat().st_mtime_ns,src.stat().st_ino)
        if file_digest(src)!=entry['sha256']:
            raise ValueError(f'Native cache changed: {src}')
        raw=np.load(src,mmap_mode='r',allow_pickle=False)
        shape=storage_shape(entry['shape'],tile)
        if raw.dtype!=np.float16 or list(raw.shape)!=entry['shape']:
            raise ValueError('Unexpected native source shape or dtype')
        temporary=target.with_suffix('.npy.part')
        tiled=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=shape)
        t,c,h,w=raw.shape;hp,wp=shape[1]*tile,shape[2]*tile
        for first in range(0,t,frames_per_read):
            last=min(t,first+frames_per_read)
            # Padding belongs only to storage beyond the native image bounds.
            # Reader bounds always use the original height and width.
            padded=np.zeros((last-first,c,hp,wp),dtype=np.float16)
            padded[:,:,:h,:w]=raw[first:last]
            tiled[first:last]=padded.reshape(last-first,c,shape[1],tile,shape[2],tile).transpose(0,2,4,1,3,5)
            restored=tiled[first:last].transpose(0,3,1,4,2,5).reshape(last-first,c,hp,wp)[:,:,:h,:w]
            if not np.array_equal(restored.view(np.uint16),raw[first:last].view(np.uint16)):
                raise ValueError(f'Lossless tiled conversion failed: {key}')
        tiled.flush();del tiled,raw,padded,restored
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        checksum=file_digest(temporary)
        if stamp!=(src.stat().st_size,src.stat().st_mtime_ns,src.stat().st_ino):
            raise ValueError('Native source changed during tiling')
        temporary.replace(target)
        saved=dict(entry,bytes=target.stat().st_size,sha256=checksum,
            source_sha256=entry['sha256'],storage_shape=list(shape),native_bits_verified=True)
        temp_record=record.with_suffix('.json.part');temp_record.write_text(json.dumps(saved,indent=2)+'\n');temp_record.replace(record)
        records[key]=saved
        print(f'Tiled {key}: {target.stat().st_size/2**30:.1f} GiB; native bits verified',flush=True)
    metadata=dict(original,layout=LAYOUT,tile_size=tile,stores=records,converted_from=identity,
        total_bytes=sum(r['bytes'] for r in records.values()))
    temporary=output/'metadata.json.part';temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(output/'metadata.json')
    return metadata


class TiledFrameCacheReader(SeviriFrameCacheReader):
    """Use the native reader's checks and orientation with tiled pixel access."""
    def __init__(self,root,cache_root,**kwargs):
        SeviriZarrClipReader.__init__(self,root,**kwargs)
        self.cache_root=Path(cache_root)
        metadata=json.loads((self.cache_root/'metadata.json').read_text())
        if (metadata.get('version')!=1 or metadata.get('units')!='K'
                or metadata.get('layout')!=LAYOUT or metadata.get('lossless') is not True
                or Path(metadata['zarr_root']).resolve()!=self.root.resolve()):
            raise ValueError('Unexpected tiled-frame cache metadata')
        self.tile=metadata['tile_size'];self.entries=metadata['stores'];self._tiles={}
        if type(self.tile) is not int or self.tile<1:raise ValueError('Invalid tile size')

    def _open_store(self,path):
        path=Path(path)
        if path not in self._stores:
            key=str(path.relative_to(self.root))
            if key not in self.entries:raise IncompleteSeviriClipError(f'Day missing from tiled cache: {key}')
            entry=self.entries[key];cached=self.cache_root/entry['file']
            if (tuple(entry['channel'])!=VIDEO_MAE_CHANNELS or entry['file']!=f'{path.stem}.npy'
                    or cached.stat().st_size!=entry['bytes'] or source_stamp(path)!=entry['source_stamp']):
                raise ValueError('Tiled cache source, channels or file size changed')
            values=np.load(cached,mmap_mode='r',allow_pickle=False)
            if values.dtype!=np.float16 or values.shape!=storage_shape(entry['shape'],self.tile) or list(values.shape)!=entry['storage_shape']:
                raise ValueError('Unexpected tiled storage shape or dtype')
            if [len(entry[a]) for a in ('time','channel','y','x')]!=entry['shape']:
                raise ValueError('Tiled cache coordinate lengths differ from native shape')
            ds=xr.Dataset(coords=dict(time=pd.to_datetime(entry['time']),channel=entry['channel'],y=entry['y'],x=entry['x']))
            ds.set_close(values._mmap.close)
            self._stores[path]=ds;self._tiles[path]=values
        return self._stores[path]

    def _native_piece(self,path,ds,selection,y0,y1,x0,x1):
        tile=self.tile;by0,bx0=y0//tile,x0//tile;by1,bx1=(y1+tile-1)//tile,(x1+tile-1)//tile
        block=self._tiles[path][selection,by0:by1,bx0:bx1]
        native=block.transpose(0,3,1,4,2,5).reshape(len(block),len(self.channels),(by1-by0)*tile,(bx1-bx0)*tile)
        return native[:,:,y0-by0*tile:y1-by0*tile,x0-bx0*tile:x1-bx0*tile]

    def load(self,start_time,*,origin_y,origin_x,global_mean=None,global_std=None):
        clip=self.load_native(start_time,origin_y=origin_y,origin_x=origin_x).astype(np.float32)
        if (global_mean is None)!=(global_std is None):raise ValueError('Provide both global mean and standard deviation')
        return normalize_seviri_clip(clip,global_mean,global_std) if global_mean is not None else clip

    def close(self):
        super().close();self._tiles.clear()
