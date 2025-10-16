from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from TheCausalityGame.core.contracts.dto.common import CommonDTO
from TheCausalityGame.core.contracts.dto.environment import (
    BudgetSnapshot,
    Feedback,
    SamplesCollection,
)
from TheCausalityGame.core.infrastructure.decisions import Decision


class TranscriptEntry(CommonDTO):
    """Canonical transcript unit written by the Environment per step."""

    model_config = ConfigDict(extra="allow", frozen=False, arbitrary_types_allowed=True)

    round: int

    decision: Decision | None = None
    result: Any | None = None
    samples_collection: SamplesCollection | None = None

    budget_snapshot: BudgetSnapshot | None = None
    feedback: Feedback | None = None


class Transcript(CommonDTO):
    """Full transcript of a run, including metadata and all steps."""

    agent_id: str
    mission_id: str
    manifest_id: str

    entries: list[TranscriptEntry] = Field(default=list)
