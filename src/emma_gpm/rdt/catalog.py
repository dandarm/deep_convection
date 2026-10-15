"""Diagnostic crop manifest derived only after complete domain labels exist."""
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from .storage import atomic_json, frame_key, open_fields


def publish_crop_catalog(output, start, times, area, native_x, native_y):
    """Append nonoverlapping 16-frame/224-pixel lattice windows incrementally."""
    output = Path(output)
    path = output/"catalog.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    seen = set(previous.start_time) if len(previous) else set()
    complete = {p.name for p in (output/"frames").glob("*") if (p/"record.json").exists()}
    height, width = area.shape
    starts = pd.date_range(start, pd.Timestamp(times[-1]), freq="80min")
    rows = []
    for stamp in starts:
        window = pd.date_range(stamp, periods=16, freq="5min")
        if stamp.isoformat() in seen or not all(frame_key(t) in complete for t in window):
            continue
        crops = []
        for y in range(0, height-223, 224):
            for x in range(0, width-223, 224):
                cy = height-y-224 if native_y[0] < native_y[-1] else y
                cx = width-x-224 if native_x[0] > native_x[-1] else x
                crops.append(dict(y=y, x=x, roi=(slice(cy, cy+224), slice(cx, cx+224)),
                                  conv=[], ot=[], cold=0, unknown=0, known=0, ids=set(), valid=True))
        for time in window:
            with open_fields(output/"frames"/frame_key(time)) as group:
                labels, tracks, uncertain, valid = [np.asarray(group[name][:]) for name in ("semantic_label", "track_id", "uncertainty", "valid_observation")]
            for crop in crops:
                roi = crop["roi"]
                c = labels[roi]
                crop["valid"] &= bool(valid[roi].all())
                crop["conv"].append(int(np.isin(c, [2, 3, 4]).sum()))
                crop["ot"].append(int((c == 4).sum()))
                crop["cold"] += int((c > 0).sum())
                crop["unknown"] += int(uncertain[roi].sum())
                crop["known"] += int(((c > 0) & ~uncertain[roi]).sum())
                crop["ids"].update(np.unique(tracks[roi]).tolist())
        from pyproj import CRS, Transformer
        import json
        projection = json.loads((output/"protocol.json").read_text())["projection_wkt"]
        transform = Transformer.from_crs(CRS.from_user_input(projection), "EPSG:4326", always_xy=True)
        for crop in crops:
            y, x = crop["y"], crop["x"]
            longitude, latitude = transform.transform(float(native_x[x:x+224].mean()), float(native_y[y:y+224].mean()))
            row = dict(clip_id=f"domain_{frame_key(stamp)}_{y}_{x}", start_time=stamp.isoformat(),
                origin_y=y, origin_x=x, month=stamp.month, longitude=longitude if np.isfinite(longitude) else np.nan,
                latitude=latitude if np.isfinite(latitude) else np.nan,
                status="ok" if crop["valid"] else "rejected", reason="" if crop["valid"] else "nonfinite_domain_crop",
                convective_pixels=sum(crop["conv"]), ot_pixels=sum(crop["ot"]), cold_pixels=crop["cold"],
                temporal_unknown_pixels=crop["unknown"], known_cold_pixels=crop["known"], tracked_objects=len(crop["ids"]-{0}),
                convective_fraction=sum(crop["conv"])/(16*224*224), convective_peak_fraction=max(crop["conv"])/(224*224),
                ot_fraction=sum(crop["ot"])/(16*224*224), ot_peak_fraction=max(crop["ot"])/(224*224),
                peak_frame=int(np.argmax(crop["conv"])), unknown_fraction=crop["unknown"]/(16*224*224),
                frames=16, pixels_per_frame=224*224, teacher_version="multispectral_utls_v0", label_root=str(output.resolve()))
            rows.append(row)
    if rows:
        catalog = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
        temporary = path.with_suffix(".csv.part")
        catalog.to_csv(temporary, index=False)
        temporary.replace(path)
        atomic_json(output/"catalog_status.json", dict(clips=len(catalog), valid=int((catalog.status == "ok").sum()),
            windows=int(catalog.start_time.nunique()), sampling="regular lattice, diagnostic only; no SSL manifests changed"))
    return len(rows)
