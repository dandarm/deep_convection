"""Explicit physical locations for data roots whose provenance stays immutable."""
import hashlib
import json
import os
from pathlib import Path


def storage_mapping():
    filename = os.environ.get("SEVIRI_STORAGE_ROOTS")
    if not filename:
        return {}, None
    content = Path(filename).read_bytes()
    document = json.loads(content)
    if set(document) != {"version", "roots"} or document["version"] != 1:
        raise ValueError("Unsupported SEVIRI storage-root mapping")
    if not isinstance(document["roots"], dict):
        raise ValueError("Storage roots must be a mapping")
    roots = {}
    for logical, physical in document["roots"].items():
        if not isinstance(physical, str) or not Path(logical).is_absolute() or not Path(physical).is_absolute():
            raise ValueError("Storage roots must be absolute paths")
        key = str(Path(logical).resolve())
        if key in roots:
            raise ValueError("Duplicate canonical storage root")
        roots[key] = Path(physical).resolve()
    return roots, hashlib.sha256(content).hexdigest()


def storage_root(path):
    """Resolve an exact root alias; all byte/provenance checks remain in readers."""
    roots, _ = storage_mapping()
    original = Path(path).resolve()
    return roots.get(str(original), original)


def storage_mapping_digest():
    return storage_mapping()[1]
