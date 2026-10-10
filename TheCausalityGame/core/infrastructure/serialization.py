"""The Causality Game - Serialization Infrastructure."""

from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import os
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from TheCausalityGame.core.lib.errors.serialization import (
    ObjectNotDeserializableError,
    ObjectNotSerializableError,
)


def is_serializable(obj: Any) -> bool:  # noqa :ANN401
    """
    Check if an object is serializable to JSON.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if the object is serializable, False otherwise.
    """
    try:
        dumps(obj)
    except (ObjectNotSerializableError, TypeError, ValueError, RecursionError):
        return False
    else:
        return True


def _is_dataclass_instance(obj: Any) -> bool:  # noqa :ANN401
    """Check if the object is a dataclass instance (not a class itself)."""
    return _dc.is_dataclass(obj) and not isinstance(obj, type)


def _default_encoder(o: Any) -> Any:  # noqa :ANN401
    """
    Convert objects to JSON-serializable formats.

    Parameters
    ----------
    o : Any
        Object to be encoded.

    Returns
    -------
    Any
        JSON-serializable representation of the input.

    Raises
    ------
    SerializationError
        If the object cannot be serialized to strict JSON.
    """
    if isinstance(o, BaseModel):
        return o.model_dump()
    if _is_dataclass_instance(o):
        return _dc.asdict(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, _dt.datetime | _dt.date | _dt.time):
        if isinstance(o, _dt.datetime) and o.tzinfo is None:
            o = o.replace(tzinfo=_dt.UTC)
        return o.isoformat()
    if isinstance(o, set | tuple):
        return list(o)  # type: ignore

    raise ObjectNotSerializableError(o)


def dumps(obj: Any, *, indent: int | None = None) -> str:  # noqa :ANN401
    """Convert an object to a JSON string with strict rules.

    Parameters
    ----------
    obj : Any
        The object to serialize.
    indent : int, optional
        If specified, pretty-prints the JSON with indentation.

    Returns
    -------
    str
        JSON string representation of the object.

    Raises
    ------
    SerializationError
        If serialization fails.
    """
    try:
        return json.dumps(
            obj,
            default=_default_encoder,
            allow_nan=False,
            indent=indent,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as e:
        raise ObjectNotSerializableError(obj) from e


def dump(obj: Any, path: Path, *, indent: int | None = 2) -> None:  # noqa :ANN401
    """Serialize an object to a JSON file.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    path : Path
        Destination file path.
    indent : int, optional
        Indentation level for pretty-printed JSON (default is 2).
    """
    temporary_path: Path | None = None
    try:
        serialized = dumps(obj, indent=indent)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except (TypeError, ValueError) as e:
        raise ObjectNotSerializableError(obj) from e
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _reject_nonfinite_constant(value: str) -> None:
    """Reject JavaScript numeric extensions that are not valid JSON."""
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def loads(s: str) -> Any:  # noqa :ANN401
    """Deserialize a JSON string to a Python object.

    Parameters
    ----------
    s : str
        JSON string.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    try:
        return json.loads(s, parse_constant=_reject_nonfinite_constant)
    except (ValueError, RecursionError) as e:
        raise ObjectNotDeserializableError(s) from e


def jsonl_write(path: Path, record: Mapping[str, Any]) -> None:
    """Append a record to a JSON Lines (JSONL) file.

    Parameters
    ----------
    path : Path
        File to write to (will be created if it doesn't exist).
    record : Mapping
        Single record (typically a dictionary) to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps(record))
        f.write("\n")
