"""Read prevalidated immutable clip files from disk or shared node RAM."""
import numpy as np

from .seviri_clip_files import ClipFileDataset, SHAPE, codec
from .storage_roots import storage_root


class ImmutableClipFileDataset(ClipFileDataset):
    """Decode clips verified when materialized and transferred to HPC.

    Dataset construction checks manifest binding, metadata and file sizes.
    Epoch reads only decode the precomputed Kelvin tensors. Payload checksums
    belong to ingestion, before these files are made available to training.
    """

    def __init__(self, root, manifest, *, zarr_root):
        super().__init__(storage_root(root), manifest, zarr_root=zarr_root)

    def decode_into(self, index, out):
        if out.shape != SHAPE or out.dtype != np.float16 or not out.flags.c_contiguous:
            raise ValueError('Clip destination must be contiguous canonical float16')
        name, _ = self.records[index]
        codec().decode((self.root/name).read_bytes(), out=out)
        return out
