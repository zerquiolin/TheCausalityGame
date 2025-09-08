"""The Causality Game - Agent Spec."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.specs.metric import MetricSpec


class MissionSpec(BaseModel):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_: str = Field(alias="class")
    behavior_metric: MetricSpec
    result_metric: MetricSpec
