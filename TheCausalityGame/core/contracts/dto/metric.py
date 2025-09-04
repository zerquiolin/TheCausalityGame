"""The Causality Game - Metric DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# TODO: Check usage of this class
class MetricScore(BaseModel):
    """Metric score."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)
