from pydantic import BaseModel, ConfigDict, Field


class BudgetSpec(BaseModel):
    """Resource budgets applied per agent run.

    Attributes:
        time_s: Optional wall-clock time budget in seconds.
        samples: Optional maximum number of samples the agent may request.
        memory_mb: Optional advisory memory budget in megabytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_s: float | None = Field(default=None, ge=0)
    samples: int | None = Field(default=None, ge=0)
    memory_mb: int | None = Field(default=None, ge=0)
