"""Safe dynamic loading utilities with allow-list enforcement."""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel

from TheCausalityGame.core.infrastructure.serialization import loads
from TheCausalityGame.core.lib.errors.registry import (
    ClassPathError,
    DeriveClassPathError,
    InvalidSpecFormatError,
    LoadError,
    MissingAttributeError,
    MissingMethodError,
    NotAllowedByPolicyError,
    PathFormatError,
)

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
        raise ClassPathError(class_path) from e
    if not any(pkg.startswith(a) for a in _ALLOWLIST):
        raise NotAllowedByPolicyError(class_path)
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, cls)
    except Exception as e:
        raise LoadError(class_path) from e


def get_class_path(obj_or_class: Any) -> str:
    """Return 'module:Class' for a class or instance."""
    if isinstance(obj_or_class, str):
        if ":" in obj_or_class:
            return obj_or_class
        raise PathFormatError(obj_or_class)
    cls = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    module = getattr(cls, "__module__", None)
    name = getattr(cls, "__name__", None)
    if not module or not name:
        raise DeriveClassPathError(obj_or_class)
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
        spec = loads(spec)

    # Check for a dictionary
    if not isinstance(spec, dict):
        raise InvalidSpecFormatError()

    class_path = spec.get("class_") or spec.get("class")
    spec_path = spec.get("spec_") or spec.get("spec")

    if not class_path:
        raise MissingAttributeError("class_")

    if not spec_path:
        raise MissingAttributeError("spec_")

    cls = load_class(class_path)  # validate class
    spc = load_class(spec_path)  # validate spec

    from_spec = getattr(cls, "from_spec", None)
    if not callable(from_spec):
        raise MissingMethodError("from_spec")

    # Create Spec class
    spec = spc(**spec)

    return from_spec(spec)
