"""The Causality Game - Result Validator contract."""

from TheCausalityGame.core.specs.common import CommonSpec


class ResultValidatorSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        spec_: Import path 'module:Class' (aliased from 'spec' in JSON).
        params: Optional agent configuration payload.
    """
