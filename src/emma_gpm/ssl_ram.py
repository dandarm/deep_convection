"""Shared RAM copies of immutable native arrays verified at dataset ingestion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from .file_lock import exclusive_file_lock


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def cache_files(source,metadata_sha256,*,verify=False):
    source=Path(source)
    content=(source/'metadata.json').read_bytes()
    if verify and hashlib.sha256(content).hexdigest()!=metadata_sha256:
        raise ValueError('Frame metadata checksum differs')
    metadata=json.loads(content)
    is_clips = ('clips' in metadata and metadata.get('shape') == [16, 7, 224, 224]
                and metadata.get('dtype') == 'float16' and metadata.get('units') == 'K')
    if (metadata.get('layout')!='native time,channel,y,x' and not is_clips) or metadata.get('lossless') is not True:
        raise ValueError('A lossless native frame cache is required')
    files={}
    entries = metadata['clips'].items() if is_clips else ((entry['file'], entry) for entry in metadata['stores'].values())
    suffix = '.blosc' if is_clips else '.npy'
    for name, entry in entries:
        if Path(name).name!=name or not name.endswith(suffix) or name in files:
            raise ValueError('Invalid or repeated frame filename')
        files[name]=dict(bytes=int(entry['bytes']),sha256=entry['sha256'])
    files['metadata.json']=dict(bytes=len(content),sha256=metadata_sha256)
    return files


def stage_frame_cache(source,destination,metadata_sha256,*,reserve_bytes=8*1024**3,progress=None,verify=False):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination==source or destination.is_relative_to(source):
        raise ValueError('Staging destination must be separate from the source')
    files=cache_files(source,metadata_sha256,verify=verify)
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    started=time.perf_counter();copied=0
    with exclusive_file_lock(destination/'.staging.lock'):
        needed=sum(record['bytes'] for name,record in files.items() if not (destination/name).exists())
        if shutil.disk_usage(destination).free<needed+reserve_bytes:
            raise ValueError('Insufficient space for verified RAM staging and its reserve')
        def copy_one(name):
            record=files[name];copied_bytes=0
            target=destination/name
            if target.exists():
                if verify and (target.stat().st_size!=record['bytes'] or digest(target)!=record['sha256']):
                    raise ValueError('Existing staged file checksum differs: '+name)
            else:
                original=source/name;before=original.stat()
                temporary=target.with_suffix(target.suffix+'.part')
                if verify:
                    checksum=hashlib.sha256()
                    with original.open('rb') as reader,temporary.open('wb') as writer:
                        for block in iter(lambda:reader.read(8*1024**2),b''):
                            writer.write(block);checksum.update(block)
                        writer.flush();os.fsync(writer.fileno())
                    after=original.stat()
                    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                        raise ValueError('Source changed during RAM staging: '+name)
                    if before.st_size!=record['bytes'] or checksum.hexdigest()!=record['sha256'] or digest(temporary)!=record['sha256']:
                        raise ValueError('Copied frame checksum differs: '+name)
                else:
                    shutil.copyfile(original,temporary)
                temporary.replace(target);copied_bytes=record['bytes']
            return name,copied_bytes
        arrays=[name for name in files if name!='metadata.json']
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name,size in pool.map(copy_one,arrays):
                copied+=size
                if progress is not None:progress(name)
        name,size=copy_one('metadata.json');copied+=size
        if progress is not None:progress(name)
    return dict(source=str(source),destination=str(destination),metadata_sha256=metadata_sha256,
                total_bytes=sum(row['bytes'] for row in files.values()),copied_bytes=copied,
                elapsed_seconds=time.perf_counter()-started,files=len(files),
                status='verified' if verify else 'ready',integrity_audit=verify)


_MEMFD_FILES = {}


def stage_memfd_cache(source, destination, metadata_sha256, *, progress=None):
    """Keep sealed RAM files in the controller, independent of /dev/shm capacity.

    Array paths point to the controller's open descriptors. Its DDP children
    read these paths as ordinary mmap files. Memory is released when the
    controller and readers close their descriptors; no byte audit is repeated.
    """
    import fcntl
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('RAM destination must be separate from the source')
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    files = cache_files(source, metadata_sha256)
    started = time.perf_counter()
    def copy_one(name):
        target = destination/name
        if str(target) in _MEMFD_FILES:
            return name, 0
        fd = os.memfd_create('seviri_'+name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            with (source/name).open('rb') as reader, os.fdopen(os.dup(fd), 'wb') as writer:
                shutil.copyfileobj(reader, writer, length=8*1024**2)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            temporary = target.with_suffix(target.suffix+'.part')
            if temporary.is_symlink():temporary.unlink()
            temporary.symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
            temporary.replace(target)
            _MEMFD_FILES[str(target)] = fd
            return name, files[name]['bytes']
        except BaseException:
            os.close(fd)
            raise
    copied = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name, size in pool.map(copy_one, files):
            copied += size
            if progress is not None:
                progress(name)
    return dict(status='ready', backend='sealed_memfd', source=str(source), destination=str(destination),
        total_bytes=sum(row['bytes'] for row in files.values()), copied_bytes=copied,
        files=len(files), controller_pid=os.getpid(), elapsed_seconds=time.perf_counter()-started,
        integrity_audit=False)
