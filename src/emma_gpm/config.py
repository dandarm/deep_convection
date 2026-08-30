from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config/pilot_2020.json") -> dict[str, Any]:
    """Load the pilot configuration and leave paths relative to the project root."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    """Resolve a configured path, optionally relocating the ``data/`` tree.

    Set ``EMMA_GPM_DATA_ROOT`` to keep raw/intermediate datasets outside the
    Git repository.  Non-data outputs (catalogs, figures, results) remain below
    the project root, and absolute configured paths are preserved.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    external_root = os.environ.get("EMMA_GPM_DATA_ROOT")
    if external_root and parts and parts[0] == "data":
        return Path(external_root).expanduser() / Path(*parts[1:])
    return Path(project_root) / path


def ensure_output_dirs(config: dict[str, Any], project_root: str | Path = ".") -> None:
    for path in config["outputs"].values():
        resolve_project_path(project_root, path).mkdir(parents=True, exist_ok=True)
