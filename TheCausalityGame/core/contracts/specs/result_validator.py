"""The Causality Game - Result Validator Specification."""

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class ResultValidatorSpec(CommonSpec):
    """
    Specification for constructing a result validator.

    A result validator checks the correctness or format of agent outputs
    before evaluation and scoring.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Import path in the form 'module:Class' used to construct the validator.
        This field is aliased from 'class' in JSON/YAML configs.
    params : dict, optional
        Optional configuration parameters required by the validator.
    """

    pass
