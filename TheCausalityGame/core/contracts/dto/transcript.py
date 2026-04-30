"""The Causality Game - Run Transcript DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from TheCausalityGame.core.contracts.dto.common import CommonDTO
from TheCausalityGame.core.contracts.dto.environment import (
    BudgetSnapshot,
    Feedback,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.decisions import Decision


class TranscriptEntry(CommonDTO):
    """
    Canonical transcript unit representing a single step in the game.

    Captures all agent-environment interactions per round, including:
      - the round index,
      - the agent's decision,
      - resulting outcome or estimate,
      - feedback from the environment,
      - and budget snapshot after action execution.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow dynamic/extra fields (e.g., future extensions)
        frozen=False,  # Mutable if runtime updates are needed
        arbitrary_types_allowed=True,  # Permit complex types like Decision
    )

    round: int
    decision: Decision | None = None
    result: Any | None = None
    samples_collection: SamplesCollection | None = None
    budget_snapshot: BudgetSnapshot | None = None
    feedback: Feedback | None = None

    # Custom attributes for extensibility
    custom_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes for custom extensibility and hooks integration.",
    )


class Transcript(CommonDTO):
    """
    Full transcript for a single agent's run.

    Stores the complete sequence of interactions along with metadata identifying the run.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=False,
        arbitrary_types_allowed=True,
    )

    agent_id: str
    mission_id: str
    manifest_id: str

    entries: list[TranscriptEntry] = Field(
        default_factory=list, description="Chronological list of transcript entries."
    )
    budget: BudgetSpec
    invalidated: bool = Field(
        default=False,
        description="Whether the run ended because a budget or runtime error invalidated it.",
    )
    invalidation_reason: str | None = Field(
        default=None,
        description="Error type and message explaining why the run was invalidated.",
    )

    def invalidate(self, error: Exception) -> None:
        """Mark this transcript as invalidated by a runtime or budget error."""
        self.invalidated = True
        self.invalidation_reason = f"{type(error).__name__}: {error}"
