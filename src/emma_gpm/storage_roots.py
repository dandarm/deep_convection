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
    if not {"version", "roots"} <= set(document) or set(document) - {"version", "roots", "source_provenance"} or document["version"] != 1:
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


def source_provenance(path):
    """Canonical source times for a filesystem with coarser timestamp precision.

    Inventory bytes are bound to the storage mapping. Callers must verify each
    source file against its SHA256 before using the canonical timestamp.
    """
    filename = os.environ.get("SEVIRI_STORAGE_ROOTS")
    if not filename:
        return None
    document = json.loads(Path(filename).read_text())
    spec = document.get("source_provenance")
    if spec is None:
        return None
    if set(spec) != {"root", "snapshot", "snapshot_sha256", "timestamps", "timestamps_sha256"}:
        raise ValueError("Invalid source provenance mapping")
    root = Path(spec["root"]).resolve()
    if not Path(path).resolve().is_relative_to(root):
        return None
    records = []
    for key in ("snapshot", "timestamps"):
        content = Path(spec[key]).read_bytes()
        if hashlib.sha256(content).hexdigest() != spec[key + "_sha256"]:
            raise ValueError("Source provenance inventory checksum differs")
        records.append(json.loads(content))
    return root, records[0], records[1]
