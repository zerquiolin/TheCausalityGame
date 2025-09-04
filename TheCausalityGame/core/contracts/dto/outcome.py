from __future__ import annotations

from typing import Any

from contracts.types.common import JsonDict
from pydantic import BaseModel, ConfigDict, Field

from .metric import MetricScore
from .samples import SamplesBatch


class Feedback(BaseModel):
    deliverable: Any | None = None
    final: bool | None = None
    message: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ActionOutcome(BaseModel):
    action_kind: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    samples: SamplesBatch | None = None
    feedback: Feedback | None = None


# TODO: Is this still used anywhere?
class ScoreReport(BaseModel):
    """Aggregated scores for a run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    per_metric: list[MetricScore]
    aggregates: JsonDict
