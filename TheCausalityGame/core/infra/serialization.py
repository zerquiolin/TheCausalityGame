from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SerializationError(TypeError):
    """Raised when an object cannot be serialized to strict JSON."""


def _is_dataclass_instance(obj: Any) -> bool:
    return _dc.is_dataclass(obj) and not isinstance(obj, type)


def _default_encoder(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump()
    if _is_dataclass_instance(o):
        return _dc.asdict(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
        if isinstance(o, _dt.datetime) and o.tzinfo is None:
            o = o.replace(tzinfo=_dt.timezone.utc)
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, (bytes, bytearray, memoryview)):
        raise SerializationError("bytes-like objects are not allowed in strict JSON")
    raise SerializationError(
        f"Object of type {type(o).__name__} is not JSON-serializable"
    )


def dumps(obj: Any, *, indent: int | None = None) -> str:
    try:
        return json.dumps(
            obj,
            default=_default_encoder,
            allow_nan=False,
            indent=indent,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as e:
        raise SerializationError(str(e)) from e


def dump(obj: Any, path: Path, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj, indent=indent))


def loads(s: str) -> Any:
    return json.loads(s)


def jsonl_write(path: Path, record: Mapping[str, Any] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps(record))
        f.write("\n")


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
