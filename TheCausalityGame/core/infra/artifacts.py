"""Artifact helpers: run folders, JSON/JSONL writers, provenance snapshot."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from TheCausalityGame.core.infra.serialization import (
    dumps,
)  # your JSON dump (strict, ensure_ascii=False)


def ensure_run_dir(base: str | Path, run_id: str, agent_id: str) -> Path:
    """Create runs/<run_id>/<agent_id>/ and return its Path."""
    path = Path(base) / run_id / agent_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "datasets").mkdir(exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    """Write a JSON file atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = obj
    if is_dataclass(obj):
        data = asdict(obj)
    tmp.write_text(dumps(data), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (creates if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def snapshot_provenance(path: Path) -> None:
    """Record environment info to provenance.json."""
    # lightweight snapshot; extend as needed
    prov = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": _py_version(),
        "platform": _platform(),
        "pid": os.getpid(),
    }
    write_json(path / "provenance.json", prov)


def _py_version() -> str:
    import sys

    return sys.version.replace("\n", " ")


def _platform() -> dict[str, Any]:
    import platform

    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
