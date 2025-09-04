from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


class Samples(BaseModel):
    kind: str
    data: Mapping[str, list[Any]]
    n: int
    interventions: Mapping[str, Any] | None = None
    seed: int
    key: str


class SamplesBatch(BaseModel):
    items: list[Samples]
