"""Tests for the symbolic SCM result validator."""

from __future__ import annotations

import networkx as nx
import pytest
import sympy as sp

from TheCausalityGame.agent.inferers.symbolic_scm import (
    EstimatedSymbolicSCM,
    SymbolicMechanism,
)
from TheCausalityGame.core.lib.errors.result_validator import InvalidResultTypeError
from TheCausalityGame.metric.result.result_validator.symbolic_scm_validator import (
    SymbolicSCMResultValidator,
)


def test_symbolic_scm_validator_accepts_symbolic_scm() -> None:
    """The validator accepts the symbolic SCM result object."""
    graph = nx.DiGraph()
    graph.add_edge("a", "F")
    result = EstimatedSymbolicSCM(
        graph,
        {"F": SymbolicMechanism("F", ["a"], sp.Symbol("a"))},
    )

    assert SymbolicSCMResultValidator().validate(result) is result


def test_symbolic_scm_validator_rejects_wrong_type() -> None:
    """The validator rejects outputs with the wrong top-level type."""
    with pytest.raises(InvalidResultTypeError):
        SymbolicSCMResultValidator().validate({"F": "a"})


def test_symbolic_scm_validator_rejects_invalid_expression() -> None:
    """The validator rejects mechanism expressions SymPy cannot parse."""
    graph = nx.DiGraph()
    graph.add_node("F")
    result = EstimatedSymbolicSCM(
        graph,
        {"F": SymbolicMechanism("F", [], object())},  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidResultTypeError):
        SymbolicSCMResultValidator().validate(result)
