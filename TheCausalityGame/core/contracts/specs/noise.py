"""The Causality Game - Noise Distribution Specification."""

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class NoiseDistributionSpec(CommonSpec):
    """
    Specification for constructing a noise distribution.

    This specification is used to instantiate a noise generator
    for SCM node sampling during data generation.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path (aliased from 'class' in JSON).
    spec_ : str | None
        Optional override for the spec class path.
    params : dict
        Optional noise distribution configuration payload.
    """

    pass
