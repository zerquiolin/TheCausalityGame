"""The Causality Game - Agent Spec."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec


class MissionSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """

    behavior_metric: MetricSpec
    result_metric: MetricSpec
    result_validator: ResultValidatorSpec
