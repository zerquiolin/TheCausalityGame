"""The Causality Game - Agent Spec."""

from pydantic import Field

from TheCausalityGame.core.specs.common import CommonSpec
from TheCausalityGame.core.specs.dag import DAGSpec
from TheCausalityGame.core.specs.scm_node import SCMNodeSpec


class SCMSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """

    vars: list[SCMNodeSpec]
    dag: DAGSpec
    random_state: str | None = Field(
        default=None,
        description="Random state for the node. If None, a new random state will be created using a fixed seed for reproducibility.",
    )
