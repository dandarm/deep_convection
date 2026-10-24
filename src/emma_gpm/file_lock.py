"""Small cross-platform helpers for non-blocking process locks."""

from contextlib import contextmanager
from pathlib import Path
import sys


def _lock_stream(stream, path):
    if sys.platform == "win32":
        import msvcrt

        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(f"Process lock is already held: {path}") from error
    else:
        import fcntl

        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Process lock is already held: {path}") from error


def _unlock_stream(stream):
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_UN)


def acquire_file_lock(path):
    """Acquire a process lock and return its open binary stream.

    Keeping the returned stream alive keeps the lock alive. Closing it, or
    terminating the process, releases the operating-system lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        _lock_stream(stream, path)
    except Exception:
        stream.close()
        raise
    return stream


@contextmanager
def exclusive_file_lock(path):
    """Hold a non-blocking process lock for the duration of a context."""
    stream = acquire_file_lock(path)
    try:
        yield stream
    finally:
        _unlock_stream(stream)
        stream.close()
