"""The Causality Game - DAG Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class DAGSpec(CommonSpec):
    """
    Specification for constructing a Directed Acyclic Graph (DAG).

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path in the format 'module:Class'. (Aliased from 'class' in JSON.)
    spec_ : str | None
        Optional override for the spec class path. Defaults to this class path if not provided.
    params : dict
        Optional parameters passed during DAG instantiation.

    nodes : list of str
        List of node identifiers in the DAG.
    edges : list of tuple of (str, str)
        List of directed edges represented as (source, target) tuples.
    """

    nodes: list[str] = Field(
        default_factory=list,
        description="List of node identifiers in the DAG.",
    )
    edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Directed edges as (source, target) node pairs.",
    )
