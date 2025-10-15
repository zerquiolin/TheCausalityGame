"""The Causality Game - Agent Spec."""

from TheCausalityGame.core.specs.common import CommonSpec
from TheCausalityGame.core.specs.metric import MetricSpec
from TheCausalityGame.core.specs.result_validator import ResultValidatorSpec


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
