"""The Causality Game - Agent Spec."""

from typing import Any

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.lib.constants.nodes import ACCESSIBILITY_CONTROLLABLE


class SCMNodeSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """

    # Base Attributes
    name: str
    accessibility: str = Field(
        default=ACCESSIBILITY_CONTROLLABLE,
        description="Accessibility of the node. Can be either 'controllable', 'observable' or 'latent'.",
    )
    domain: list[float | str] | tuple[float | int] = Field(
        description="Domain of the node. Can be either a list of floats or strings.",
    )
    parents: list[str] | None = Field(
        default=None,
        description="List of parent nodes. Can be either a list of strings or None.",
    )
    parent_mappings: dict[str, int | float] | None = Field(
        default=None,
        description="Mapping from parent names to their fixed values. If a parent is not in this mapping, it is assumed to be variable.",
    )
    random_state: str | None = Field(
        default=None,
        description="Random state for the node. If None, a new random state will be created using a fixed seed for reproducibility.",
    )
    # Specific Attributes
    equation: dict[str, Any] | str | None = Field(
        default=None,
        description="Equation for the node. If None, the node will be generated using a noise distribution.",
    )
    cdfs: dict[str, list[float]] | None = Field(
        default=None,
        description="Cumulative distribution functions for categorical nodes. If None, the node will be generated using a noise distribution.",
    )
    domain_distribution: dict[str, float] | None = Field(
        default=None,
        description="Domain distribution for categorical nodes. If None, the node will be generated using a noise distribution.",
    )
    noise_distribution: NoiseDistributionSpec | None = Field(
        default=None,
        description="Noise distribution for the node. If None, the node will be generated using an equation.",
    )
