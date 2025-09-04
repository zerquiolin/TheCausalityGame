from __future__ import annotations
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# TODO: This might work?
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["experiment", "submit"]
    payload: Dict[str, Any] = Field(default_factory=dict)


# TODO: I don't know the use case for this yet.
class Observation(BaseModel):
    kind: str
    payload: Dict[str, Any] = Field(default_factory=dict)
