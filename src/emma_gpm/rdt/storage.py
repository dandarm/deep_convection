"""Strict rectangular archive access and atomic, crop-readable frame records."""
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import zlib
import zipfile
from contextlib import contextmanager
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from ..seviri import THERMAL_CHANNELS
from ..seviri_clips import SeviriZarrClipReader, IncompleteSeviriClipError


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(directory):
    """Check all published compressed chunks, not just array metadata."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


class DomainFrameReader:
    """Use strict reader store validation, but read the complete rectangle.

    Invalid pixels are preserved for explicit masking by the domain teacher.
    Crop readers still refuse any incomplete/nonfinite input clip.
    """
    def __init__(self, root):
        self.reader = SeviriZarrClipReader(root, num_frames=1)
        self.geometry = None

    def close(self):
        self.reader.close()

    def load(self, time):
        stamp = pd.Timestamp(time)
        dataset = self.reader._open_store(self.reader._path_for_time(stamp))
        if dataset.attrs.get("units") != "K" or dataset.attrs.get("calibration") != "brightness_temperature":
            raise ValueError("Teacher requires raw Kelvin calibration metadata")
        x, y = dataset.x.values, dataset.y.values
        if not ((np.diff(x) > 0).all() or (np.diff(x) < 0).all()) or not ((np.diff(y) > 0).all() or (np.diff(y) < 0).all()):
            raise ValueError("Nonmonotonic projected coordinates")
        geometry = (x.copy(), y.copy(), dataset.attrs["projection_wkt"])
        if self.geometry is None:
            self.geometry = geometry
        elif not (np.array_equal(x, self.geometry[0]) and np.array_equal(y, self.geometry[1]) and geometry[2] == self.geometry[2]):
            raise ValueError("Domain geometry changes")
        times = pd.DatetimeIndex(dataset.time.values)
        if times.has_duplicates or times.hasnans:
            raise ValueError("Invalid archive timestamps")
        index = times.get_indexer([stamp])[0]
        if index < 0:
            raise IncompleteSeviriClipError(f"Missing RSS time {stamp}")
        values = np.asarray(dataset.seviri.isel(time=int(index)).transpose("channel", "y", "x").values, np.float32)
        if y[0] < y[-1]:
            values = values[:, ::-1, :]
        if x[0] > x[-1]:
            values = values[:, :, ::-1]
        # Bound open daily metadata during incremental multi-day scans.
        if len(self.reader._stores) > 2:
            for path in list(self.reader._stores)[:-2]:
                self.reader._stores.pop(path).close()
        return np.ascontiguousarray(values)


def frame_key(stamp):
    return f"{pd.Timestamp(stamp):%Y%m%dT%H%M}"


@contextmanager
def open_fields(directory):
    """Read legacy directory Zarr or lossless ZIP-contained Zarr chunks."""
    directory = Path(directory)
    packed = directory/"fields.zarr.zip"
    if packed.exists():
        with zarr.storage.ZipStore(packed, mode="r") as store:
            yield zarr.open_group(store, mode="r")
    else:
        yield zarr.open_group(str(directory/"fields.zarr"), mode="r")


_snapshot_paths = {}

def save_frame(output, result, teacher, metadata, *, packed=False):
    """One rename commits labels, object/lineage tables and resume state."""
    directory = Path(output)/"frames"/frame_key(result.time)
    temporary = directory.with_name(directory.name+".part")
    if directory.exists():
        raise FileExistsError(f"Already committed frame {directory}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    buffers = {}
    store = zarr.storage.MemoryStore(store_dict=buffers) if packed else str(temporary/"fields.zarr")
    group = zarr.open_group(store, mode="w", zarr_format=2)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for name, field in result.fields.items():
        group.create_array(name, data=field, chunks=(224, 224), compressor=compressor)
    group.attrs.update(time=result.time.isoformat(), orientation="north-up west-left",
                       teacher_version=teacher.config.version, config_hash=teacher.config.config_hash)
    zarr.consolidate_metadata(store)
    if packed:
        with zipfile.ZipFile(temporary/"fields.zarr.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            for name, buffer in sorted(buffers.items()):
                archive.writestr(name, buffer.to_bytes())
    result.objects.to_csv(temporary/"objects.csv.gz", index=False, compression="gzip")
    result.links.to_csv(temporary/"links.csv.gz", index=False, compression="gzip")
    state = zlib.compress(pickle.dumps(teacher, protocol=5), level=1)
    (temporary/"state.pkl.zlib").write_bytes(state)
    metadata = dict(metadata, time=result.time.isoformat(), checkpoint_sha256=hashlib.sha256(state).hexdigest(),
                    objects_sha256=sha256(temporary/"objects.csv.gz"), links_sha256=sha256(temporary/"links.csv.gz"),
                    fields_sha256=sha256(temporary/"fields.zarr.zip") if packed else tree_sha256(temporary/"fields.zarr"))
    atomic_json(temporary/"record.json", metadata)
    temporary.rename(directory)
    # Retain only the newest two local state snapshots, including the former
    # one until the new complete directory has been atomically published.
    key = str(Path(output).resolve())
    if key not in _snapshot_paths:
        _snapshot_paths[key] = sorted((Path(output)/"frames").glob("*/state.pkl.zlib"))
    elif directory/"state.pkl.zlib" not in _snapshot_paths[key]:
        _snapshot_paths[key].append(directory/"state.pkl.zlib")
    for old in _snapshot_paths[key][:-2]:
        old.unlink(missing_ok=True)
    _snapshot_paths[key] = _snapshot_paths[key][-2:]
    return metadata


def restore_latest(output, config_hash):
    directories = sorted(p for p in (Path(output)/"frames").glob("*") if p.is_dir() and not p.name.endswith(".part"))
    if not directories:
        return None, []
    records = [json.loads((p/"record.json").read_text()) for p in directories]
    last = directories[-1]
    if sha256(last/"state.pkl.zlib") != records[-1]["checkpoint_sha256"]:
        raise ValueError("Corrupt domain checkpoint")
    # Pickle is solely a local checkpoint produced by this process; never
    # deserialize a downloaded/user-supplied external checkpoint here.
    teacher = pickle.loads(zlib.decompress((last/"state.pkl.zlib").read_bytes()))
    if teacher.config.config_hash != config_hash:
        raise ValueError("Incompatible domain teacher config")
    for directory, record in zip(directories, records):
        if sha256(directory/"objects.csv.gz") != record["objects_sha256"] or sha256(directory/"links.csv.gz") != record["links_sha256"]:
            raise ValueError("Corrupt object/lineage table")
        packed = directory/"fields.zarr.zip"
        actual = sha256(packed) if packed.exists() else tree_sha256(directory/"fields.zarr")
        if actual != record["fields_sha256"]:
            raise ValueError("Corrupt published domain fields")
    return teacher, records


def read_table(path):
    try:
        # Hashes are opaque text. Numeric inference on a hexadecimal prefix
        # such as 7e252472 can crash the installed native pandas parser.
        return pd.read_csv(path, dtype={name: str for name in (
            "config_hash", "source_frame_sha256", "fields_sha256",
            "checkpoint_sha256", "objects_sha256", "links_sha256")})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def publish_domain_tables(output):
    """Materialize descriptive outgoing lineage after the causal scan.

    Per-frame objects remain immutable; this joined table supplies all children
    known in the scanned interval, including cross-day transitions.
    """
    output = Path(output)
    directories = sorted(p for p in (output/"frames").glob("*") if (p/"record.json").exists())
    object_tables = [read_table(p/"objects.csv.gz") for p in directories]
    link_tables = [read_table(p/"links.csv.gz") for p in directories]
    objects = pd.concat(object_tables, ignore_index=True) if object_tables else pd.DataFrame()
    links = pd.concat(link_tables, ignore_index=True) if link_tables else pd.DataFrame()
    if not objects.empty:
        if not links.empty:
            children = links.groupby(["parent_time", "parent_track"]).track_id.agg(lambda values: json.dumps(sorted(set(map(int, values)))))
            index = pd.MultiIndex.from_arrays([objects.time, objects.track_id])
            objects["child_track_ids"] = children.reindex(index).to_numpy()
            objects["child_track_ids"] = objects.child_track_ids.fillna("[]")
        available_times = pd.DatetimeIndex([json.loads((p/"record.json").read_text())["time"] for p in directories])
        objects["children_observed_next_frame"] = (pd.to_datetime(objects.time)+pd.Timedelta(minutes=5)).isin(available_times)
    for name, table in (("objects.csv.gz", objects), ("lineage.csv.gz", links)):
        temporary = output/(name+".part")
        table.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(output/name)


def load_domain_crop(root, output, start_time, origin_y, origin_x, *, crop_size=224):
    """Read labels cropped AFTER full-domain detection/tracking; never rerun it."""
    output = Path(output)
    protocol = json.loads((output/"protocol.json").read_text())
    if str(Path(root).resolve()) != protocol["root"]:
        raise ValueError("Label archive belongs to a different SEVIRI root")
    with SeviriZarrClipReader(root, num_frames=16, crop_size=crop_size) as reader:
        clip = reader.load(start_time, origin_y=origin_y, origin_x=origin_x)
        dataset = next(iter(reader._stores.values()))
        height, width = dataset.sizes["y"], dataset.sizes["x"]
        cy = height-origin_y-crop_size if dataset.y.values[0] < dataset.y.values[-1] else origin_y
        cx = width-origin_x-crop_size if dataset.x.values[0] > dataset.x.values[-1] else origin_x
        native_x = dataset.x.values[origin_x:origin_x+crop_size]
        native_y = dataset.y.values[origin_y:origin_y+crop_size]
    roi = (slice(cy, cy+crop_size), slice(cx, cx+crop_size))
    times = pd.date_range(start_time, periods=16, freq="5min")
    fields, tables, link_tables = {}, [], []
    visible_ids = set()
    for t, stamp in enumerate(times):
        directory = output/"frames"/frame_key(stamp)
        if not (directory/"record.json").exists():
            raise IncompleteSeviriClipError(f"Full-domain labels not available for {stamp}")
        with open_fields(directory) as group:
            for name in group.array_keys():
                fields.setdefault(name, []).append(np.asarray(group[name][roi]))
            if "invalid_radiometry" not in group:
                # Before the audited quality-mask upgrade every committed frame
                # passed the global range guard: this diagnostic flag is known
                # false, not an inferred/repaired satellite observation.
                fields.setdefault("invalid_radiometry", []).append(np.zeros((crop_size,crop_size),bool))
        table = read_table(directory/"objects.csv.gz")
        ids = np.unique(fields["track_id"][-1]); ids = ids[ids > 0]
        visible_ids.update(ids.tolist())
        if not table.empty:
            table = table[table.track_id.isin(ids)].copy()
            table["frame"] = t
            tables.append(table)
        links = read_table(directory/"links.csv.gz")
        if not links.empty:
            links["frame"] = t
            link_tables.append(links)
    fields = {key: np.stack(value) for key, value in fields.items()}
    if not fields["valid_observation"].all():
        raise IncompleteSeviriClipError("Crop contains invalid full-domain observations/geometry")
    objects = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["frame", "track_id", "class_id", "ot_pixels"])
    links = pd.concat(link_tables, ignore_index=True) if link_tables else pd.DataFrame()
    if not objects.empty:
        # Outgoing edges are descriptive lineage only; they never alter the
        # stored causal phase/proxy/OT decisions.
        for index, row in objects.iterrows():
            children = links[(links.parent_time == row.time) & (links.parent_track == row.track_id)].track_id.unique().tolist() if not links.empty else []
            objects.at[index, "child_track_ids"] = json.dumps(children)
    area = np.load(output/"pixel_area_north_up.npy", mmap_mode="r")[roi]
    from pyproj import CRS, Transformer
    lon, lat = Transformer.from_crs(CRS.from_user_input(protocol["projection_wkt"]), "EPSG:4326", always_xy=True).transform(float(native_x.mean()), float(native_y.mean()))
    result = dict(fields, classes=fields["semantic_label"], tracks=fields["track_id"],
        uncertain=fields["uncertainty"], phase_uncertain=fields["phase_uncertainty"], objects=objects, links=links)
    return dict(clip=clip, times=times, result=result, area=area, longitude=lon, latitude=lat,
        start_time=pd.Timestamp(start_time).isoformat(), origin_y=origin_y, origin_x=origin_x,
        teacher_version="multispectral_utls_v0", warmup_frames=0)
