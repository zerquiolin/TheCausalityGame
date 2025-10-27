"""The Causality Game - Registry Infrastructure."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

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

# Only modules starting with this prefix may be dynamically loaded
_ALLOWLIST = ("TheCausalityGame.",)


def load_class(class_path: str) -> type[Any]:
    """
    Dynamically import a class using 'module:Class' syntax.

    Enforces an allow-list to prevent unsafe imports.

    Parameters
    ----------
    class_path : str
        Fully qualified class path in the form `'module:Class'`.

    Returns
    -------
    type
        The imported class type.

    Raises
    ------
    ClassPathError
        If the class path format is invalid.
    NotAllowedByPolicyError
        If the module is not allow-listed.
    LoadError
        If the module or class could not be loaded.
    """
    try:
        module_path, class_name = class_path.split(":")
    except ValueError as e:
        raise ClassPathError(class_path) from e

    if not any(module_path.startswith(prefix) for prefix in _ALLOWLIST):
        raise NotAllowedByPolicyError(class_path)

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise LoadError(class_path) from e


def get_class_path(obj_or_class: Any) -> str:  # noqa: ANN401
    """
    Derive the 'module:Class' path from a class or object instance.

    Parameters
    ----------
    obj_or_class : Any
        A class object, instance, or string.

    Returns
    -------
    str
        The module-qualified class path in the format `'module:Class'`.

    Raises
    ------
    PathFormatError
        If the string is not in a valid format.
    DeriveClassPathError
        If the path cannot be derived from the given object.
    """
    if isinstance(obj_or_class, str):
        if ":" in obj_or_class:
            return obj_or_class
        raise PathFormatError(obj_or_class)

    cls = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    module = getattr(cls, "__module__", None)
    name = getattr(cls, "__name__", None)

    if not module or not name:
        raise DeriveClassPathError(  # noqa: TRY003
            "Invalid object — missing __module__ or __name__."
        )

    if module.startswith("pathlib._"):
        module = "pathlib"

    return f"{module}:{name}"


def build_from_spec(spec: BaseModel | dict[str, Any] | str) -> Any:  # noqa: ANN401
    """
    Build an object instance from a specification.

    Parameters
    ----------
    spec : BaseModel | dict[str, Any] | str
        The spec object, dictionary, or serialized JSON string. Must contain:
        - `class_`: the class to instantiate.
        - `spec_`: the spec class used to validate/construct parameters.

    Returns
    -------
    Any
        The constructed object.

    Raises
    ------
    InvalidSpecFormatError
        If the input is not a recognized spec format.
    MissingAttributeError
        If required keys ('class_', 'spec_') are missing.
    MissingMethodError
        If the target class lacks a callable `from_spec` method.
    LoadError
        If dynamic loading fails.
    """
    # Handle pydantic BaseModel input
    if isinstance(spec, BaseModel):
        cls = load_class(spec.class_)  # type: ignore[attr-defined]
        return cls.from_spec(spec)

    # Handle JSON string input
    if isinstance(spec, str):
        spec = loads(spec)

    # Validate dictionary input
    if not isinstance(spec, dict):
        raise InvalidSpecFormatError()

    class_path = spec.get("class_") or spec.get("class")
    spec_path = spec.get("spec_") or spec.get("spec")

    if not class_path:
        raise MissingAttributeError("class_")
    if not spec_path:
        raise MissingAttributeError("spec_")

    cls = load_class(class_path)
    spec_cls = load_class(spec_path)

    from_spec = getattr(cls, "from_spec", None)
    if not callable(from_spec):
        raise MissingMethodError("from_spec")

    # Construct typed spec
    typed_spec = spec_cls(**spec)
    return from_spec(typed_spec)


T = TypeVar("T")


def load_subclasses_from_path(base_class: type[T], root_path: str) -> list[type[T]]:
    """
    Load subclasses from path.

    Recursively loads all subclasses of `base_class` from .py files
    located under `root_path` (including subfolders).
    """
    root = Path(root_path)
    subclasses: list[type[T]] = []

    for file in root.rglob("*.py"):
        # Skip __init__.py or other irrelevant files if you want
        module = _import_module_from_file(file)
        if not module:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            module_name = module.__name__
            if getattr(obj, "__module__", "") != module_name:
                continue
            if issubclass(obj, base_class) and obj is not base_class:
                subclasses.append(obj)

    return subclasses


def _import_module_from_file(path: Path) -> ModuleType | None:
    """Dynamically import a module directly from a file path."""
    module_name = path.with_suffix("").as_posix().replace("/", ".")
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None
