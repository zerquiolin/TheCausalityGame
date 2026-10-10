"""Source and manifest fingerprints for experiment isolation and resumption."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def source_fingerprint() -> str:
    """Fingerprint installed scientific Python/R sources, including uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".R"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest_fingerprint(manifest: dict) -> str:
    """Hash canonical full manifest content independently of JSON formatting."""
    return hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def dependency_versions() -> dict[str, str]:
    """Record actual versions; explicitly mark absent optional backends."""
    result = {}
    for name in (
        "thecausalitygame",
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "sympy",
        "scikit-learn",
        "pydantic",
        "causal-learn",
        "lingam",
        "notears",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "absent"
    return result
