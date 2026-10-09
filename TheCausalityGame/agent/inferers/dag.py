"""Compatibility exports for the project-specific empirical DAG inferer."""

from TheCausalityGame.agent.inferers.empirical.dag import (  # noqa: F401
    DAGDiscoveryInferer as _EmpiricalDAGDiscoveryInferer,
    apply_meek_r1,
    ci_test_continuous_nonlinear,
    ci_test_discrete_chi2_stratified,
    ci_test_gaussian,
    learn_dag_from_samples,
    orient_v_structures,
)


class DAGDiscoveryInferer(_EmpiricalDAGDiscoveryInferer):
    """Compatibility path for the custom empirical DAG-discovery composite."""
