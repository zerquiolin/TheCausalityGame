"""The Causality Game - SCM Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.dag import DAGSpec
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec


class SCMSpec(CommonSpec):
    """
    Specification for constructing a Structural Causal Model (SCM).

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Import path in the format 'module:Class' (aliased from 'class' in JSON).
    spec_ : str or None
        Path to the specification class. Automatically filled if not provided.
    params : dict
        Optional additional parameters for instantiating the SCM.

    vars : list of SCMNodeSpec
        List of node specifications defining the variables in the SCM.
    dag : DAGSpec
        Specification of the directed acyclic graph (DAG) structure of dependencies.
    random_state : str or None
        Serialized random state for reproducibility. If None, a default is used.
    """

    vars: list[SCMNodeSpec]
    dag: DAGSpec
    random_state: str | None = Field(
        default=None,
        description=(
            "Serialized NumPy random state. If None, a new random state will be created "
            "using a fixed seed for reproducibility."
        ),
    )
