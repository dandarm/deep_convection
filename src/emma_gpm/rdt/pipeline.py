"""Causal streaming teacher; the scientific state belongs to the full domain."""
from dataclasses import asdict, dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy import ndimage
from ..seviri import THERMAL_CHANNELS
from ..rdt_teacher import projected_pixel_areas
from .config import RDTTeacherConfig
from .features import spectral_features, FEATURE_NAMES
from .detection import detect_cells
from .tracking import associate, primary_lineage, update_motion
from .utls_proxy import estimate_proxy, thermal_depth
from .lifecycle import lifecycle, ot_candidates


@dataclass(frozen=True)
class FrameResult:
    fields: dict
    objects: pd.DataFrame
    links: pd.DataFrame
    time: pd.Timestamp
    reset_reason: str | None


@dataclass
class PreparedDetection:
    """Single-use, current-frame-only work; never contains temporal state."""
    frame_sha256: str
    area_sha256: str
    config_hash: str
    labels: np.ndarray
    cells: dict
    consumed: bool = False


def prepare_detection(frame, area, config):
    frame = np.asarray(frame, dtype=np.float32)
    area = np.asarray(area, dtype=float)
    if frame.shape != (7, *area.shape):
        raise ValueError("Invalid seven-channel frame/coordinates")
    valid = (np.isfinite(frame).all(axis=0) & np.isfinite(area)
             & ((frame >= 100) & (frame <= 400)).all(axis=0))
    labels, cells = detect_cells(frame[4], spectral_features(frame), area, valid, config)
    return PreparedDetection(hashlib.sha256(frame.tobytes()).hexdigest(),
        hashlib.sha256(area.tobytes()).hexdigest(), config.config_hash, labels, cells)


class RDTDomainTeacher:
    def __init__(self, area, config=None, *, channels=THERMAL_CHANNELS, track_id_offset=0):
        if tuple(channels) != THERMAL_CHANNELS:
            raise ValueError("Require canonical seven-channel order")
        self.area = np.asarray(area, dtype=float)
        if self.area.ndim != 2 or np.any(np.isfinite(self.area) & (self.area <= 0)):
            raise ValueError("Require physical pixel areas; NaN only denotes unavailable geometry")
        self.config = config or RDTTeacherConfig()
        self.previous = {}
        self.previous_ir = None
        self.last_time = None
        self.track_id_offset = int(track_id_offset)
        if self.track_id_offset < 0 or self.track_id_offset > (1 << 63)-(1 << 32) or self.track_id_offset % (1 << 32):
            raise ValueError("Track namespace must be a nonnegative multiple of 2**32")
        self.next_track_id = self.track_id_offset + 1
        self.frame_index = 0

    def step(self, frame, time, *, channels=THERMAL_CHANNELS, allow_invalid=False, prepared=None):
        frame = np.asarray(frame, dtype=np.float32)
        stamp = pd.Timestamp(time)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        if tuple(channels) != THERMAL_CHANNELS or frame.shape != (7, *self.area.shape) or pd.isna(stamp):
            raise ValueError("Invalid seven-channel frame/coordinates/time")
        if self.last_time is not None and stamp <= self.last_time:
            raise ValueError("Time must increase; no reordering or repeating frames")
        finite = np.isfinite(frame).all(axis=0)
        observed = finite & np.isfinite(self.area)
        kelvin_range = ((frame >= 100) & (frame <= 400)).all(axis=0)
        invalid_radiometry = observed & ~kelvin_range
        valid = observed & kelvin_range
        # Domain analysis already permits explicitly missing observations.
        # Apply the SAME Kelvin sanity limits per pixel, never clip or replace
        # its radiances. Strict clip mode still rejects any invalid sample.
        if invalid_radiometry.any() and (not allow_invalid or not valid.any()):
            values = frame[:, observed]
            raise ValueError(f"Raw Kelvin outside [100,400]: {int(invalid_radiometry.sum())} pixels; "
                             f"observed range [{float(values.min()):g},{float(values.max()):g}]")
        if not allow_invalid and not valid.all():
            raise ValueError("Invalid observations; clips must be complete")
        if prepared is not None:
            if (prepared.consumed or prepared.config_hash != self.config.config_hash
                    or prepared.labels.shape != self.area.shape
                    or prepared.frame_sha256 != hashlib.sha256(frame.tobytes()).hexdigest()
                    or prepared.area_sha256 != hashlib.sha256(self.area.tobytes()).hexdigest()):
                raise ValueError("Prepared detection is consumed or differs from input/config/geometry")
            prepared.consumed = True
        reset = None
        if self.last_time is not None and stamp-self.last_time != pd.Timedelta(minutes=5):
            self.previous, self.previous_ir = {}, None
            reset = "observation_gap"
        cfg = self.config
        ir = frame[4]
        features = spectral_features(frame)
        if prepared is None:
            labels, current = detect_cells(ir, features, self.area, valid, cfg)
        else:
            labels, current = prepared.labels, prepared.cells
        parents, children, edges = associate(self.previous, current, labels, self.area, cfg)
        primary, events = primary_lineage(self.previous, current, parents, children)
        shape = self.area.shape
        fields = dict(semantic_label=np.zeros(shape, np.uint8), cell_id=labels,
            track_id=np.zeros(shape, np.int64 if getattr(self, "track_id_offset", 0) else np.int32), thermal_depth=np.full(shape, np.nan, np.float32),
            t_edge_k=np.full(shape, np.nan, np.float32), t_utls_proxy_k=np.full(shape, np.nan, np.float32),
            t_utls_instant_k=np.full(shape, np.nan, np.float32), proxy_confidence=np.zeros(shape, np.float32),
            plateau_mask=np.zeros(shape, bool), ot_mask=np.zeros(shape, bool), provisional_ot=np.zeros(shape, bool),
            uncertainty=np.zeros(shape, bool), phase_uncertainty=np.zeros(shape, bool), valid_observation=valid,
            invalid_radiometry=invalid_radiometry)
        fields.update({name: values.astype(np.float32) for name, values in zip(FEATURE_NAMES, features)})
        rows, links = [], []
        config_hash = cfg.config_hash
        for c, now in current.items():
            p = primary.get(c)
            before = self.previous.get(p)
            event = events[c]
            now["event"] = event
            now["track_id"] = before["track_id"] if before else self.next_track_id
            if before is None:
                if getattr(self, "track_id_offset", 0) and self.next_track_id-self.track_id_offset >= (1 << 32):
                    raise OverflowError("Sequence exhausted its track ID namespace")
                self.next_track_id += 1
            history = before.get("history", []) if before else []
            now.update(lifecycle(now, before, history, event, cfg))
            bbox, mask = now["bbox"], now["mask"]
            local_ir, local_features, local_area = ir[bbox], features[:, bbox[0], bbox[1]], self.area[bbox]
            # Never fill invalid raw observations. Derivative diagnostics use
            # the same domain pixels; movement is used only in association.
            tendency = local_ir-self.previous_ir[bbox] if self.previous_ir is not None else None
            proxy_history = before.get("proxy_history", []) if before else []
            proxy = estimate_proxy(local_ir, local_features, mask, local_area, now["t_edge_k"], cfg,
                history=proxy_history, tendency=tendency, support_confidence=now["support_confidence"])
            depth = thermal_depth(local_ir, mask, now["t_edge_k"], proxy.filtered_k, cfg)
            candidate = ot_candidates(local_ir, local_features, mask, local_area, depth, proxy, cfg, convective=now["convective"])
            counts = np.zeros(mask.shape, np.uint16)
            if before is not None and event == "continuation" and "ot_count" in before:
                # OT persistence follows the tracked cell's integer displacement;
                # only the previous diagnostic mask is shifted, never radiances.
                displacement = np.rint(now["centroid"]-before["centroid"]).astype(int)
                old = before["ot_count"]
                by, bx = before["bbox"]
                yy, xx = np.where(old > 0)
                ry, rx = yy+by.start+displacement[0]-bbox[0].start, xx+bx.start+displacement[1]-bbox[1].start
                inside = (ry >= 0) & (rx >= 0) & (ry < mask.shape[0]) & (rx < mask.shape[1])
                counts[ry[inside], rx[inside]] = old[yy[inside], xx[inside]]
                counts = ndimage.maximum_filter(counts, size=3, mode="constant")
            counts = np.where(candidate, np.minimum(counts.astype(np.uint32)+1, cfg.ot_persistence_frames), 0).astype(np.uint16)
            confirmed = candidate & (counts >= cfg.ot_persistence_frames)
            fields["semantic_label"][bbox][mask] = now["class_id"]
            fields["semantic_label"][bbox][confirmed] = 4
            fields["track_id"][bbox][mask] = now["track_id"]
            for key, value in (("thermal_depth", depth), ("t_edge_k", now["t_edge_k"]),
                    ("t_utls_proxy_k", proxy.filtered_k), ("t_utls_instant_k", proxy.instant_k), ("proxy_confidence", proxy.confidence)):
                fields[key][bbox][mask] = value[mask] if isinstance(value, np.ndarray) else value
            fields["plateau_mask"][bbox] |= proxy.plateau
            fields["provisional_ot"][bbox] |= candidate & ~confirmed
            fields["ot_mask"][bbox] |= confirmed
            fields["uncertainty"][bbox][mask] = not now["temporal_valid"] or now["border"]
            fields["phase_uncertainty"][bbox][mask] = not now["phase_valid"]
            parent_tracks = [int(self.previous[pp]["track_id"]) for pp in parents[c]]
            row = dict(frame=self.frame_index, time=stamp.isoformat(), component=c, cell_id=c, track_id=int(now["track_id"]),
                parents=parent_tracks, parent_track_ids=parent_tracks, child_track_ids=[], event=event,
                primary_parent=int(before["track_id"]) if before else None, area_km2=now["area_km2"],
                min_k=now["min_k"], mean_k=now["mean_k"], t_min_k=now["min_k"], t_mean_k=now["mean_k"],
                t_edge_k=now["t_edge_k"], edge_method=now["edge_method"], support_confidence=now["support_confidence"],
                t_utls_instant_k=proxy.instant_k, t_utls_proxy_k=proxy.filtered_k, proxy_confidence=proxy.confidence,
                proxy_reason=proxy.reason, plateau_area_km2=proxy.area_km2, plateau_dispersion_k=proxy.dispersion_k,
                contrast_k=now["contrast_k"], cooling_k_15min=now["cooling_k_15min"], growth_km2_15min=now["growth_km2_15min"],
                cooling_15min_k=now["cooling_k_15min"], area_change_15min_km2=now["growth_km2_15min"],
                aor_from_parent=edges[p,c]["aor"] if p is not None else np.nan,
                temporal_valid=now["temporal_valid"], growth_valid=now["growth_valid"], phase_valid=now["phase_valid"],
                border=now["border"], class_id=now["class_id"], lifecycle_state=now["class_id"], reason=now["reason"],
                ot_pixels=int(confirmed.sum()), provisional_ot_pixels=int((candidate & ~confirmed).sum()), config_hash=config_hash)
            samples = local_features[:, mask]
            medians = np.median(samples, axis=1)
            mads = np.median(np.abs(samples-medians[:, None]), axis=1)
            for name, median, mad in zip(FEATURE_NAMES, medians, mads):
                row[name+"_median_k"] = float(median)
                row[name+"_mad_k"] = float(mad)
            rows.append(row)
            for pp in parents[c]:
                linked = self.previous[pp]
                links.append(dict(time=stamp.isoformat(), frame=self.frame_index, parent_time=self.last_time.isoformat(),
                    parent_track=int(linked["track_id"]), track_id=int(now["track_id"]), component=c,
                    parent_component=pp, primary=pp == p, event=event, **edges[pp, c]))
            now["history"] = [*history, dict(min_k=now["min_k"], area_km2=now["area_km2"], border=now["border"], event=event)][-3:]
            # Missing instantaneous estimates break smoothing history explicitly.
            now["proxy_history"] = [*proxy_history, proxy.instant_k][-cfg.proxy_window_frames:] if np.isfinite(proxy.instant_k) else []
            now["motion"] = update_motion(before, now, cfg) if before else np.zeros(2)
            now["ot_count"] = counts
        fields["uncertainty"][~valid] = True
        fields["phase_uncertainty"][~valid] = True
        # All outgoing children are represented on the parent-time edges. The
        # storage layer builds child_track_ids from these, without future data
        # entering any causal classification/proxy calculation.
        self.previous, self.previous_ir, self.last_time = current, ir.copy(), stamp
        self.frame_index += 1
        return FrameResult(fields, pd.DataFrame(rows), pd.DataFrame(links), stamp, reset)


def build_rdt_teacher_labels(clip_tchw, *, times, x=None, y=None, channels=THERMAL_CHANNELS,
                             config=None, pixel_area_km2=None, projection_wkt=None, mode="multispectral_utls_v0"):
    clip = np.asarray(clip_tchw)
    times = pd.DatetimeIndex(times)
    if (clip.ndim != 4 or clip.shape[1] != 7 or len(times) != len(clip) or not len(times) or times.hasnans
            or not np.all(np.diff(times.asi8) == 5*60*1_000_000_000) or tuple(channels) != THERMAL_CHANNELS):
        raise ValueError("Require complete chronological five-minute seven-channel clip")
    area = pixel_area_km2
    if area is None:
        if x is None or y is None or projection_wkt is None:
            raise ValueError("Physical pixel areas or projected x/y and CRS required")
        area = projected_pixel_areas(x, y, projection_wkt)
    if mode == "legacy_fixed_2ch":
        from ..rdt_teacher import label_rdt_clip
        return label_rdt_clip(clip, times, area)
    if mode != "multispectral_utls_v0":
        raise ValueError("Unknown teacher mode")
    teacher = RDTDomainTeacher(area, config, channels=channels)
    results = [teacher.step(frame, stamp) for frame, stamp in zip(clip, times)]
    return dict(**{key: np.stack([result.fields[key] for result in results]) for key in results[0].fields},
        objects=pd.concat([r.objects for r in results], ignore_index=True),
        links=pd.concat([r.links for r in results], ignore_index=True), config=asdict(teacher.config))
