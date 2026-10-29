"""Ordered clip access to verified daily arrays for the pinned batch loader."""
from pathlib import Path

import pandas as pd

from .seviri_frame_cache import SeviriFrameCacheReader, file_digest, manifest_days


class FrameFileDataset:
    """Share read-only daily arrays across threads; normalize after GPU transfer.

    Every daily file is SHA256-verified on construction. File metadata is checked
    on each read; temporal, spectral and spatial checks remain in the reader.
    ForecastWindow.split validates every pixel on the GPU before model use.
    """
    def __init__(self, root, manifest, *, zarr_root):
        self.root = Path(root)
        self.manifest = pd.read_csv(manifest)
        self.metadata_sha256 = file_digest(self.root/'metadata.json')
        self._reader = SeviriFrameCacheReader(zarr_root, self.root, require_finite=False)
        self.records = list(self.manifest.itertuples(index=False))
        self._files = {}
        for day in manifest_days(self.manifest):
            path = self._reader._path_for_time(pd.Timestamp(day))
            entry = self._reader.entries[str(path.relative_to(self._reader.root))]
            cached = self.root/entry['file']
            if file_digest(cached) != entry['sha256']:
                raise ValueError(f'Cached frame checksum differs: {cached}')
            self._reader._open_store(path)
            self._files[day] = (cached, self._stamp(cached))

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def __len__(self):
        return len(self.records)

    def decode_into(self, index, out):
        row = self.records[index]
        start = pd.Timestamp(row.start_time)
        if start.tzinfo is not None:
            start = start.tz_convert('UTC').tz_localize(None)
        end = start + pd.Timedelta(minutes=75)
        for day in {start.strftime('%Y%m%d'), end.strftime('%Y%m%d')}:
            path, stamp = self._files[day]
            if self._stamp(path) != stamp:
                raise ValueError(f'Cached frame file changed during use: {path}')
        return self._reader.load_native_into(row.start_time, origin_y=row.origin_y,
                                             origin_x=row.origin_x, out=out)

    def close(self):
        self._reader.close()
