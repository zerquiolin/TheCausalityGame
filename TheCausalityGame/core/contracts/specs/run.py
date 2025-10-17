"""The Causality Game - Run Plan Specification."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.contracts.specs.plot import PlotSpec
from TheCausalityGame.core.lib.enum.runplan import (
    RunPlanExecution,
    RunPlanParallelBackEnd,
)


class RunPlanSpec(BaseModel):
    """
    Execution policy for agent runs.

    Defines how agents are executed during a run—either sequentially or in parallel—
    including resource budgets, hook integrations, and plotting options.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    execution : RunPlanExecution
        Execution mode: either 'sequential' or 'parallel'.
    parallel_backend : RunPlanParallelBackEnd
        Backend for parallel execution: 'thread' (for I/O-bound tasks) or
        'process' (for CPU-bound tasks).
    max_workers : int or None
        Maximum number of parallel workers. `None` means auto-detect.
    budget : BudgetSpec
        Resource constraints per agent run (e.g., time, memory, samples).
    hook_plan : list[HookSpec]
        Optional hooks to trigger on lifecycle events.
    plot_plan : list[PlotSpec]
        Optional plot definitions for runtime visualization.
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
        default_factory=BudgetSpec,
        description="Resource budgets per agent run.",
    )
    hook_plan: list[HookSpec] = Field(
        default_factory=list,
        description="Hook subscriptions for lifecycle events.",
    )
    plot_plan: list[PlotSpec] = Field(
        default_factory=list,
        description="Plot specifications for visualizations.",
    )
