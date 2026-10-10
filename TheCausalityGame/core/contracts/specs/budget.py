"""The Causality Game - Budget Specification."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.specs.acquisition import (
    AcquisitionPricing,
    StopCondition,
)


class BudgetSpec(BaseModel):
    """
    Resource budgets applied to each agent run.

    These limits are enforced by the runtime system to control execution behavior.

    Attributes
    ----------
    rounds : int | None, optional
        Maximum number of interaction rounds the agent is allowed to run. Must be non-negative.
    time_s : float | None, optional
        Maximum allowed wall-clock time in seconds for the run. Must be non-negative.
    samples : int | None, optional
        Maximum number of samples the agent is permitted to request. Must be non-negative.
    memory_mb : float | None, optional
        Advisory memory usage limit in megabytes. Must be non-negative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    experiments: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    pricing: AcquisitionPricing = Field(default_factory=AcquisitionPricing)
    stopping: StopCondition | None = None

    rounds: int = Field(default=100, ge=0, description="Maximum number of rounds.")
    time_s: float | None = Field(
        default=None, ge=0, description="Time budget in seconds."
    )
    samples: int | None = Field(default=None, ge=0, description="Sample request limit.")
    memory_mb: float | None = Field(
        default=None, ge=0, description="Memory budget in MB."
    )
