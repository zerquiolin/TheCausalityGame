"""The Causality Game - SCM Node Specification."""

from typing import Any

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.lib.constants.nodes import ACCESSIBILITY_CONTROLLABLE


class SCMNodeSpec(CommonSpec):
    """
    Specification for constructing a Structural Causal Model (SCM) node.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    name : str
        Name of the node in the causal graph.
    accessibility : str
        Accessibility of the node. Must be one of: "controllable", "observable", or "latent".
    domain : list[float | str] | tuple[float, int]
        Domain of values the node can take. Can be categorical (list of str) or numerical.
    parents : list[str] or None
        List of parent node names. `None` if the node has no parents.
    parent_mappings : dict[str, int | float] or None
        Optional mapping for categorical parents to index values. Useful for encoding.
    random_state : str or None
        JSON-encoded random state for reproducibility. If `None`, a new default state is used.
    equation : dict[str, Any] or str or None
        Optional expression or function definition to compute node values.
    cdfs : dict[str, list[float]] or None
        Optional cumulative distribution functions (CDFs) for categorical outputs.
    domain_distribution : dict[str, float] or None
        Optional domain probability distribution for sampling categorical variables.
    noise_distribution : NoiseDistributionSpec or None
        Optional noise distribution to use during value generation.
    """

    name: str
    accessibility: str = Field(
        default=ACCESSIBILITY_CONTROLLABLE,
        description=(
            "Accessibility of the node. Must be one of: 'controllable', 'observable', or 'latent'."
        ),
    )
    domain: list[float | str] | tuple[float, int] = Field(
        description="Set of possible values this node can take.",
    )
    parents: list[str] | None = Field(
        default=None,
        description="List of parent node names. `None` if no parents.",
    )
    parent_mappings: dict[str, int | float] | None = Field(
        default=None,
        description=(
            "Optional index mapping for categorical parent values. "
            "Useful when deriving integer mappings from string categories."
        ),
    )
    random_state: str | None = Field(
        default=None,
        description="Serialized random state. Used for reproducibility.",
    )
    equation: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Optional computation rule or expression. If absent, generation relies on noise."
        ),
    )
    cdfs: dict[str, list[float]] | None = Field(
        default=None,
        description="Optional CDFs for categorical variable generation.",
    )
    domain_distribution: dict[str, float] | None = Field(
        default=None,
        description="Optional domain-level probability distribution.",
    )
    noise_distribution: NoiseDistributionSpec | None = Field(
        default=None,
        description="Optional noise distribution specification.",
    )
