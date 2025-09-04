"""Safe dynamic loading utilities with allow-list enforcement."""

from __future__ import annotations

import importlib
import importlib.metadata as im
from typing import Any

from ..contracts.errors import LoadError

_ALLOWLIST = ("TheCausalityGame.",)


def load_class(class_path: str) -> type[Any]:
    """Load a class with `module:Class` syntax and allow-list enforcement.

    Args:
      class_path: Fully qualified string.

    Returns
    -------
      A class object.

    Raises
    ------
      LoadError: If path is malformed, disallowed, or import fails.
    """
    try:
        pkg, cls = class_path.split(":")
    except ValueError as e:  # noqa: PERF203
        raise LoadError(
            f"invalid class path '{class_path}', expected 'module:Class'"
        ) from e
    if not any(pkg.startswith(a) for a in _ALLOWLIST):
        raise LoadError(f"class path not allowed by policy: {pkg}")
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, cls)
    except Exception as e:  # noqa: BLE001
        raise LoadError(f"failed to import '{class_path}': {e}") from e


def load_entry_point(group: str, name: str) -> Any:
    """Load a Python entry point."""
    eps = im.entry_points().select(group=group, name=name)
    if not eps:
        raise LoadError(f"entry point not found: {group}:{name}")
    return eps[0].load()


def build_from_spec(spec: dict) -> Any:
    """Instantiate from a spec dict.

    Spec shape:
      { "class": "TheCausalityGame.mod:Class", "config": {...} }
      or { "class": "...:Class", "params": {...} }
    """
    if "class" not in spec:
        raise LoadError("spec missing 'class'")
    cls = load_class(spec["class"])
    kwargs = spec.get("config") or spec.get("params") or {}
    return cls(**kwargs)
