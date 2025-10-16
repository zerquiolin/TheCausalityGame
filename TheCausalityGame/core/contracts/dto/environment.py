from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# TODO: Create a common base class for DTOs


# === Round Info ===
class RoundInfo(BaseModel):
    round: int
    budget_snapshot: BudgetSnapshot | None = None


# === Availability of Actions ===
class ExperimentVariable(BaseModel):
    name: str
    domain: list[int | float]


class AvailableActions(BaseModel):
    experiments: list[ExperimentVariable]
    answer: Literal["submit"] = "submit"  # TODO: Hardcoded for now, only one action


# === Samples ===
class Samples(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: str  # "observational" | "interventional"
    n: int
    data: pd.DataFrame
    interventions: dict[str, Any] | None = None


class SamplesCollection(list[Samples]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def total_n(self) -> int:
        return sum(s.n for s in self)

    def total_bytes(self) -> float:
        return sum(s.data.memory_usage(deep=True).sum() for s in self)  # Bytes


# === Metric Scores ===
class Feedback(BaseModel):
    result: float | None = None
    behavior: float | None = None
    custom_metrics: dict[str, float] | None = Field(default=None)


# === Budgets ===
class BudgetSnapshot(BaseModel):
    rounds_left: int | None = None
    time_s_left: float | None = None
    samples_left: int | None = None
    memory_mb_left: float | None = None
