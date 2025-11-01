"""The Causality Game - Metric Specification."""

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class MetricSpec(CommonSpec):
    """
    Specification for constructing a metric.

    This spec supports both behavior and result metrics used to evaluate agent performance.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path in the format 'module:Class'. (Aliased from 'class' in JSON.)
    params : dict
        Optional configuration parameters specific to the metric.
    """

    pass
