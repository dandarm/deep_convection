from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable

import requests


def stream_download(url: str, destination: str | Path, chunk_mb: int = 4) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(destination)
    return destination


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_emma_year(record_id: int, year: int, raw_dir: str | Path, extract_dir: str | Path) -> Path:
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    metadata = requests.get(metadata_url, timeout=(30, 120))
    metadata.raise_for_status()
    candidates = [item for item in metadata.json()["files"] if f"_{year}_" in item["key"]]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one EMMA archive for {year}, found {len(candidates)}")
    item = candidates[0]
    archive = Path(raw_dir) / item["key"]
    stream_download(item["links"]["self"], archive)
    algorithm, expected = item["checksum"].split(":", 1)
    if algorithm != "md5" or md5(archive) != expected:
        raise RuntimeError(f"Checksum verification failed for {archive}")
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extract_dir, filter="data")
    return archive


def cmr_granules(
    concept_id: str,
    start: str,
    end: str,
    bbox: tuple[float, float, float, float],
) -> tuple[list[dict], int | None]:
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    response = requests.get(
        url,
        params={
            "collection_concept_id": concept_id,
            "temporal": f"{start},{end}",
            "bounding_box": ",".join(map(str, bbox)),
            "page_size": 2000,
        },
        timeout=(30, 180),
    )
    response.raise_for_status()
    hits = response.headers.get("CMR-Hits")
    return response.json()["feed"].get("entry", []), int(hits) if hits else None


def choose_link(entry: dict, rel_suffix: str) -> str | None:
    for link in entry.get("links", []):
        if link.get("rel", "").endswith(rel_suffix) and not link.get("inherited", False):
            return link.get("href")
    return None


def polygon_bounds(polygons: Iterable[Iterable[str]]) -> tuple[float, float, float, float]:
    latitudes: list[float] = []
    longitudes: list[float] = []
    for polygon_group in polygons:
        for text in polygon_group:
            values = list(map(float, text.split()))
            latitudes.extend(values[0::2])
            longitudes.extend(values[1::2])
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def normalize_cmr_entries(entries: list[dict], collection: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        polygons = entry.get("polygons", [])
        bounds = polygon_bounds(polygons) if polygons else (None, None, None, None)
        rows.append(
            {
                "collection": collection,
                "granule_concept_id": entry.get("id"),
                "producer_granule_id": entry.get("producer_granule_id"),
                "time_start": entry.get("time_start"),
                "time_end": entry.get("time_end"),
                "granule_size_mb_reported": entry.get("granule_size"),
                "bbox_lon_min": bounds[0],
                "bbox_lat_min": bounds[1],
                "bbox_lon_max": bounds[2],
                "bbox_lat_max": bounds[3],
                "polygons_json": json.dumps(polygons, separators=(",", ":")),
                "download_url": choose_link(entry, "/data#"),
                "opendap_url": choose_link(entry, "/service#"),
                "s3_url": choose_link(entry, "/s3#"),
            }
        )
    return rows

