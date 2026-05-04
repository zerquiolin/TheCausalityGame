"""Agent-specific DTOs used by inferers, deciders, and policies."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from TheCausalityGame.core.contracts.dto.common import CommonDTO
from TheCausalityGame.core.contracts.dto.environment import RoundInfo, SamplesCollection
from TheCausalityGame.core.infrastructure.decisions import Decision


class BeliefSnapshot(CommonDTO):
    """Typed handoff from an inferer to a decider."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    estimate: Any = None
    summary: dict[str, Any] = Field(default_factory=dict)
    capabilities: tuple[str, ...] = Field(default_factory=tuple)


class RoundObservation(CommonDTO):
    """Single-round observation passed into inferers, deciders, and policies."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    round_info: RoundInfo
    decision: Decision
    samples: SamplesCollection
