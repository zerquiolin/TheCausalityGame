from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExperimentSpace(BaseModel):
    variables: dict[str, list[Any]]
    max_n: int | None = None


class AvailableActions(BaseModel):
    experiment: ExperimentSpace
    submit: dict[str, Any] = Field(default_factory=dict)
