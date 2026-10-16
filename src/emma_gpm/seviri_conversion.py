"""Bounded native-file conversion with FIFO delivery to one Zarr writer."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

from .seviri import load_native_crop


def _initialize(threads):
    import dask
    from numcodecs import blosc
    dask.config.set(scheduler='threads',num_workers=threads)
    blosc.set_nthreads(1)


def _convert(path,bounds):
    return load_native_crop(path,bounds,calibration='brightness_temperature')


def ordered_native_conversions(paths,bounds,*,workers=0,threads=2,prefetch=2,serial_loader=None):
    """Yield (path, result callable), preserving exceptions for parent recovery.

    Only children read/calibrate/crop. The caller alone validates, handles native
    recovery and writes Zarr. Prefetch bounds outstanding tasks per worker.
    """
    if workers<0 or threads<1 or prefetch<1:
        raise ValueError('Require nonnegative conversion workers and positive threads')
    if not workers:
        loader=serial_loader or load_native_crop
        for path in paths:
            yield path,partial(loader,path,bounds,calibration='brightness_temperature')
        return
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize,initargs=(threads,)) as pool:
        pending,remaining=deque(),iter(paths)
        def enqueue():
            path=next(remaining,None)
            if path is not None:
                pending.append((path,pool.submit(_convert,path,bounds)))
        try:
            for _ in range(prefetch*workers):enqueue()
            while pending:
                path,future=pending.popleft()
                yield path,future.result
                enqueue()
        finally:
            for _,future in pending:future.cancel()
