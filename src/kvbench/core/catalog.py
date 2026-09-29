"""Read declarative profile catalogs without changing process-global registries."""

from pathlib import Path
from typing import Any

import yaml


def read_catalog(path: Path) -> dict[str, Any]:
    with path.open() as source:
        data = yaml.safe_load(source)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"{path}: expected a version: 1 profile catalog")
    unknown = data.keys() - {"version", "gpus", "models"}
    if unknown:
        raise ValueError(f"{path}: unknown catalog fields: {sorted(unknown)}")
    for kind in ("gpus", "models"):
        entries = data.get(kind, {})
        if not isinstance(entries, dict):
            raise ValueError(f"{path}: {kind} must be a mapping")
        for name, profile in entries.items():
            if not isinstance(name, str) or not isinstance(profile, dict):
                raise ValueError(f"{path}: invalid {kind} entry {name!r}")
    return data


BUILTIN_CATALOG = Path(__file__).with_name("profiles") / "catalog.yaml"
