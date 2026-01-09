"""The Causality Game - Environment DTOs."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import ConfigDict, Field

from TheCausalityGame.core.contracts.dto.common import CommonDTO


# === Round Info ===
class RoundInfo(CommonDTO):
    """Information about the current round state."""

    round: int
    budget_snapshot: BudgetSnapshot | None = None


# === Availability of Actions ===
class ExperimentVariable(CommonDTO):
    """Describes a single experiment variable (intervention target)."""

    name: str
    domain: list[int | float | str]


class AvailableActions(CommonDTO):
    """Actions available to the agent in a given round."""

    experiments: list[ExperimentVariable]
    answer: Literal["submit"] = "submit"  # Currently hardcoded


class Experiment(CommonDTO):
    """Describes an experiment."""

    treatment: dict[str, int | float | str] | None
    n: int = Field(..., gt=0)

    @property
    def is_observational(self) -> bool:
        """Return whether the experiment is observational."""
        return self.treatment is None


# === Samples ===
class Samples(CommonDTO):
    """A batch of observational or interventional data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str  # Either "observational" or "interventional"
    n: int
    data: pd.DataFrame
    interventions: dict[str, Any] | None = None


class SamplesCollection(list[Samples]):
    """A collection of samples gathered across multiple rounds."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def total_n(self) -> int:
        """Return the total number of samples collected."""
        return sum(s.n for s in self)

    def total_bytes(self) -> float:
        """Return the total memory usage (in bytes) of all sample data."""
        return sum(s.data.memory_usage(deep=True).sum() for s in self)


# === Metric Feedback ===
class Feedback(CommonDTO):
    """Metric scores returned by the environment after each round."""

    model_config = ConfigDict(
        extra="forbid",  # No extra fields allowed
        frozen=False,  # Mutable if sscore updates are needed
        arbitrary_types_allowed=False,  # Only standard types allowed
    )

    result: float | None = None
    behavior: float | None = None
    custom_metrics: dict[str, float] | None = None


# === Budgets ===
class BudgetSnapshot(CommonDTO):
    """Snapshot of resource budgets remaining in the current round."""

    rounds_left: int | None = None
    time_s_left: float | None = None
    samples_left: int | None = None
    memory_mb_left: float | None = None
