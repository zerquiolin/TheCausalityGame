"""The Causality Game - Mission Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec


class MissionSpec(CommonSpec):
    """
    Specification for constructing a mission.

    A mission defines the evaluation criteria and validation logic for a game run.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path (aliased from 'class' in JSON).
    spec_ : str | None
        Optional override for the spec class path.
    params : dict
        Optional mission-specific configuration payload.
    id : str
        Unique identifier for the mission.
    behavior_metric : MetricSpec
        Metric used to evaluate agent behavior (e.g., efficiency, cost).
    result_metric : MetricSpec
        Metric used to score the quality of the final result.
    result_validator : ResultValidatorSpec
        Validator used to determine whether the agent's final result is valid.
    """

    id: str = Field(..., description="Unique identifier for the mission.")
    behavior_metric: MetricSpec
    result_metric: MetricSpec
    result_validator: ResultValidatorSpec
