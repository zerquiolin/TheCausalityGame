from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from .samples import SamplesBatch
from .metric import MetricScore
from contracts.types.common import JsonDict


class Feedback(BaseModel):
    deliverable: Optional[Any] = None
    final: Optional[bool] = None
    message: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class ActionOutcome(BaseModel):
    action_kind: str
    action_payload: Dict[str, Any] = Field(default_factory=dict)
    samples: Optional[SamplesBatch] = None
    feedback: Optional[Feedback] = None


# TODO: Is this still used anywhere?
class ScoreReport(BaseModel):
    """Aggregated scores for a run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    per_metric: list[MetricScore]
    aggregates: JsonDict
