from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field, ConfigDict

from .actions import Action, Observation


class TranscriptEntry(BaseModel):
    """Canonical transcript unit written by the Environment per step.

    A 'step' is one atomic record in the round loop (e.g., status, dataset batch,
    experiment action, final submission). We attach run-scoped metadata and a
    single Step (kind + payload).
    """

    round_index: int
    step_index: int
    agent_id: str
    mission_id: str
    step: StepRecord
    done: bool = False
    # TODO: This requires to have the feedback in the transcript to know the reward at that point of time.


class StepRecord(BaseModel):
    """One micro-step record in the run transcript."""

    model_config = ConfigDict(extra="forbid")

    action: Action | None = None
    observation: Observation | None = None

    budgets_consumed: dict[str, float | int] = {}
    error: str | None = None


class Transcript(BaseModel):
    """Full transcript of a run, including metadata and all steps."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    agent_id: str
    mission_id: str

    metadata: Dict[str, Any] = Field(default_factory=dict)

    steps: list[TranscriptEntry] = Field(default_factory=list)
