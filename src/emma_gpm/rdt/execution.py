"""Bounded independent detection, causal tracking, ordered atomic publication."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import pickle
import shutil
import time

from .pipeline import prepare_detection
from .storage import DomainFrameReader, atomic_json, save_frame


def _initialize(root, area, config):
    import zarr
    from numcodecs import blosc
    global _reader, _area, _config
    zarr.config.set({"async.concurrency": 2, "threading.max_workers": 1})
    blosc.set_nthreads(1)
    _reader, _area, _config = DomainFrameReader(root), area, config


def _prepare(stamp):
    frame = _reader.load(stamp)
    return frame, prepare_detection(frame, _area, _config)


def prepared_frames(root, times, area, config, workers=0):
    """FIFO delivery, at most two tasks per worker; no tracking in children."""
    if workers < 0:
        raise ValueError("Detection worker count must be nonnegative")
    if not workers:
        reader = DomainFrameReader(root)
        try:
            for stamp in times:
                yield stamp, reader.load(stamp), None
        finally:
            reader.close()
        return
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(root, area, config)) as pool:
        pending, stamps = deque(), iter(times)
        def enqueue():
            stamp = next(stamps, None)
            if stamp is not None:
                pending.append((stamp, pool.submit(_prepare, stamp)))
        for _ in range(2*workers):
            enqueue()
        try:
            while pending:
                stamp, future = pending.popleft()
                frame, detection = future.result()
                enqueue()
                yield stamp, frame, detection
        finally:
            for _, future in pending:
                future.cancel()


def _save(output, result, snapshot, metadata, packed, started):
    # snapshot is serialized BEFORE any subsequent mutation of the live teacher.
    teacher = pickle.loads(snapshot) if isinstance(snapshot, bytes) else snapshot
    record = save_frame(output, result, teacher, metadata, packed=packed)
    record["frame_seconds"] = time.monotonic()-started
    atomic_json(output/"frames"/f"{result.time:%Y%m%dT%H%M}"/"record.json", record)
    return record


def process_frames(teacher, frames, output, metadata, on_commit, *, packed=False, async_save=False):
    """Only main process advances state; a single writer preserves commit order.

    At most one result is being saved while the next is computed. Failed future
    detections never suppress publication of an already successful earlier frame.
    """
    pool = ThreadPoolExecutor(max_workers=1) if async_save else None
    pending = None
    def finish():
        nonlocal pending
        if pending is not None:
            future, pending = pending, None
            on_commit(future.result())
    try:
        for stamp, frame, detection in frames:
            if shutil.disk_usage(output).free < 50*2**30:
                raise RuntimeError("Less than 50 GiB free; stopping without deleting data")
            started = time.monotonic()
            result = teacher.step(frame, stamp, allow_invalid=True, prepared=detection)
            values = metadata(frame, result, started)
            if pool is None:
                on_commit(_save(output, result, teacher, values, packed, started))
            else:
                snapshot = pickle.dumps(teacher, protocol=5)
                finish()
                pending = pool.submit(_save, output, result, snapshot, values, packed, started)
    finally:
        try:
            finish()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            frames.close()
