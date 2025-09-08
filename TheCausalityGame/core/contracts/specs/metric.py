from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.types.common import JsonDict


class MetricSpec(BaseModel):
    """Specification for constructing a metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_: str = Field(alias="class")
    params: JsonDict = Field(
        default_factory=dict,
        description="Optional metric configuration payload.",
    )


class MetricSpecs(BaseModel):
    """Canonical metric pair for each mission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior: MetricSpec
    result: MetricSpec
    custom: tuple[MetricSpec, ...] = Field(
        default_factory=tuple,
        description="Optional custom metrics.",
    )
