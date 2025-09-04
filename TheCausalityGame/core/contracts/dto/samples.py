from __future__ import annotations
from typing import Any, List, Mapping, Optional
from pydantic import BaseModel, Field


class Samples(BaseModel):
    kind: str
    data: Mapping[str, List[Any]]
    n: int
    interventions: Optional[Mapping[str, Any]] = None
    seed: int
    key: str


class SamplesBatch(BaseModel):
    items: List[Samples]
