"""The Causality Game - Dag Spec."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.types.common import JsonDict

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class DAGSpec(CommonSpec):
    """Specification for constructing a DAG.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        nodes: List of node identifiers.
        edges: List of (source, target) edge tuples.
    """

    nodes: list[str] = Field(
        default_factory=list,
        description="List of node identifiers.",
    )
    edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="List of (source, target) edge tuples.",
    )
