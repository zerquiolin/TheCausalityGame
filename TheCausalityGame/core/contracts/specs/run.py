from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.contracts.specs.plot import PlotSpec
from TheCausalityGame.core.lib.enum.runplan import (
    RunPlanExecution,
    RunPlanParallelBackEnd,
)

from .budget import BudgetSpec


class RunPlanSpec(BaseModel):
    """Execution policy for agent runs.

    Runs are performed in isolated environments per agent. Execution can be
    sequential (one agent after the other) or parallel (concurrently).

    Attributes
    ----------
        rounds: Number of rounds per agent.
        execution: 'sequential' or 'parallel'.
        parallel_backend: 'thread' for I/O-bound or 'process' for CPU-bound runs.
        max_workers: Optional cap on parallel workers; None → auto.
        budgets: Resource budgets enforced per agent run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution: RunPlanExecution = Field(
        default=RunPlanExecution.PARALLEL,
        description="Execution strategy for multiple agents.",
    )
    parallel_backend: RunPlanParallelBackEnd = Field(
        default=RunPlanParallelBackEnd.THREAD,
        description="Parallel backend for parallel execution strategy.",
    )
    max_workers: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of parallel workers; None for auto.",
    )
    budget: BudgetSpec = Field(
        default_factory=BudgetSpec, description="Resource budgets per agent run."
    )
    hook_plan: list[HookSpec] = Field(
        default_factory=list,
        description="Hook subscriptions for lifecycle events.",
    )
    plot_plan: list[PlotSpec] = Field(
        default_factory=list,
        description="Plot specifications for visualizations.",
    )
