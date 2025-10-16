"""The Causality Game - Agent Spec."""

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class NoiseDistributionSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """
