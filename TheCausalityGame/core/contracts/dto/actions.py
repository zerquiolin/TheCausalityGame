from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# TODO: This might work?
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["experiment", "submit"]
    payload: dict[str, Any] = Field(default_factory=dict)


# TODO: I don't know the use case for this yet.
class Observation(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
