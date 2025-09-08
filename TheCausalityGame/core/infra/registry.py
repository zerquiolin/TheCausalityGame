"""Safe dynamic loading utilities with allow-list enforcement."""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

from pydantic import BaseModel

from TheCausalityGame.core.contracts.errors import LoadError
from TheCausalityGame.core.contracts.serializable import Serializable

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
    except ValueError as e:
        raise LoadError(
            f"invalid class path '{class_path}', expected 'module:Class'"
        ) from e
    if not any(pkg.startswith(a) for a in _ALLOWLIST):
        raise LoadError(f"class path not allowed by policy: {pkg}")
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, cls)
    except Exception as e:
        raise LoadError(f"failed to import '{class_path}': {e}") from e


def get_class_path(obj_or_class: Any) -> str:
    """Return 'module:Class' for a class or instance."""
    if isinstance(obj_or_class, str):
        if ":" in obj_or_class:
            return obj_or_class
        raise ValueError("String must be in 'module:Class' form, e.g. 'pkg.mod:Type'.")
    cls = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    module = getattr(cls, "__module__", None)
    name = getattr(cls, "__name__", None)
    if not module or not name:
        raise TypeError(f"Cannot derive class path from {obj_or_class!r}")
    if module.startswith("pathlib._"):
        module = "pathlib"
    return f"{module}:{name}"


def build_from_spec(spec: BaseModel | dict[str, Any] | str):
    """Instantiate an object from a spec mapping or JSON string.

    The spec must include a class path under 'class_' (preferred) or 'class'.
    """
    # Check for an spec instance class
    if isinstance(spec, BaseModel):
        cls = load_class(spec.class_)
        return cls.from_spec(spec)

    # Check for a JSON string
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise LoadError(f"failed to parse spec: {e}") from e

    # Check for a dictionary
    if not isinstance(spec, dict):
        raise LoadError("spec is not a valid format")

    class_path = spec.get("class_") or spec.get("class")
    spec_path = spec.get("spec_") or spec.get("spec")

    if not class_path:
        raise LoadError("spec must contain a 'class_' (or 'class') key")

    if not spec_path:
        raise LoadError("spec must contain a 'spec_' (or 'spec') key")

    cls = load_class(class_path)  # validate class
    spc = load_class(spec_path)  # validate spec

    from_spec = getattr(cls, "from_spec", None)
    if not callable(from_spec):
        raise LoadError(f"'{class_path}' does not implement a 'from_spec' classmethod")

    # Create Spec class
    spec = spc(**spec)

    return from_spec(spec)
