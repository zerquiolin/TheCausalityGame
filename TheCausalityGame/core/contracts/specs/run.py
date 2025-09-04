from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from .budget import BudgetSpec


class RunPlan(BaseModel):
    """Execution policy for agent runs.

    Runs are performed in isolated environments per agent. Execution can be
    sequential (one agent after the other) or parallel (concurrently).

    Attributes:
        rounds: Number of rounds per agent.
        execution: 'sequential' or 'parallel'.
        parallel_backend: 'thread' for I/O-bound or 'process' for CPU-bound runs.
        max_workers: Optional cap on parallel workers; None → auto.
        budgets: Resource budgets enforced per agent run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rounds: int = Field(ge=1)
    execution: Literal["sequential", "parallel"] = "sequential"
    parallel_backend: Literal["thread", "process"] = "thread"
    max_workers: int | None = Field(default=None, ge=1)
    budgets: BudgetSpec = Field(default_factory=BudgetSpec)
