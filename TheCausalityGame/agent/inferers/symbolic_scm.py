"""Compatibility exports for the project-specific symbolic SCM inferer."""

from TheCausalityGame.agent.inferers.empirical.symbolic_scm import (  # noqa: F401
    EstimatedSymbolicSCM,
    SparseSymbolicSCMDiscoveryInferer as _EmpiricalSymbolicInferer,
    SymbolicMechanism,
    fit_sparse_symbolic_mechanism_for_node,
)


class SparseSymbolicSCMDiscoveryInferer(_EmpiricalSymbolicInferer):
    """Compatibility path for the custom empirical symbolic-SCM composite."""
