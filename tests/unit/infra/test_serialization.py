"""The Causality Game - Infrastructure Serialization Tests."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from TheCausalityGame.core.infrastructure.serialization import dumps, loads


@dataclass
class DC:
    x: int
    y: list[int]


class E(Enum):
    A = "alpha"
    B = "beta"


def test_dumps_loads_roundtrip() -> None:
    payload: dict[str, Any] = {
        "a": 1,
        "b": [1, 2, 3],
        "c": {"d": "x"},
        "path": Path("foo/bar"),
        "dc": DC(5, [7, 8]),
        "enum": E.A,
        "set": {9, 10},
        "tuple": (1, 2),
    }
    s = dumps(payload)
    out = loads(s)
    # Basic structural checks
    assert out["a"] == 1
    assert out["b"] == [1, 2, 3]
    assert out["c"] == {"d": "x"}
    assert out["path"] == "foo/bar"
    assert out["dc"] == {"x": 5, "y": [7, 8]}
    assert out["enum"] == "alpha"
    assert sorted(out["set"]) == [9, 10]
    assert out["tuple"] == [1, 2]
