"""Compatibility exports for the project-specific empirical SCM inferer."""

from TheCausalityGame.agent.inferers.empirical.scm import (  # noqa: F401
    DiscreteCPTMechanism,
    EstimatedANMSCM,
    NumericANMMechanism,
    SCMDiscoveryInferer as _EmpiricalSCMDiscoveryInferer,
    fit_anm_mechanism_for_node,
)


class SCMDiscoveryInferer(_EmpiricalSCMDiscoveryInferer):
    """Compatibility path for the custom empirical SCM-discovery composite."""
