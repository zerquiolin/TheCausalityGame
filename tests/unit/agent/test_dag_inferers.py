"""Tests for DAG discovery inferer implementations."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.lingam import LiNGAMInferer
from TheCausalityGame.agent.inferers.notears import NOTEARSInferer
from TheCausalityGame.agent.inferers.pc import PCInferer
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import RoundInfo, Samples, SamplesCollection
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import build_from_spec

EDGE_XY = ("X", "Y")
EDGE_YZ = ("Y", "Z")


def _observation(data: pd.DataFrame) -> RoundObservation:
    """Build a single-round observational update for DAG inferer tests."""
    sample = Samples(
        kind="observational",
        n=len(data),
        data=data,
        interventions=None,
    )
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=None, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _chain_data(n: int = 400, seed: int = 911) -> pd.DataFrame:
    """Generate a simple linear chain X -> Y -> Z."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.3, size=n)
    z = -2.0 * y + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.mark.parametrize(
    "inferer",
    [
        PCInferer(),
        NOTEARSInferer(),
        LiNGAMInferer(),
    ],
)
def test_dag_inferer_empty_answer_returns_graph(inferer: Any) -> None:  # noqa: ANN401
    """DAG inferers return an empty directed graph before seeing data."""
    result = inferer.answer()

    assert isinstance(result, nx.DiGraph)
    assert result.number_of_edges() == 0


@pytest.mark.parametrize(
    "inferer",
    [
        PCInferer(),
        NOTEARSInferer(),
        LiNGAMInferer(),
    ],
)
def test_dag_inferer_spec_roundtrip(inferer: Any) -> None:  # noqa: ANN401
    """DAG inferer specs rebuild to equivalent inferers."""
    rebuilt = build_from_spec(inferer.to_spec())

    assert rebuilt.to_dict() == inferer.to_dict()


def test_pc_inferer_recovers_chain_skeleton() -> None:
    """PC inferer keeps the two local chain edges on simple linear data."""
    inferer = PCInferer(is_numerical=True, alpha=0.01)
    inferer.update(_observation(_chain_data()))

    result = inferer.answer()

    skeleton = {frozenset(edge) for edge in result.edges()}
    assert frozenset(EDGE_XY) in skeleton
    assert frozenset(EDGE_YZ) in skeleton


def test_notears_inferer_recovers_chain_orientation() -> None:
    """NOTEARS recovers the chain skeleton and returns a DAG."""
    inferer = NOTEARSInferer(lambda_l2=0.01, w_threshold=0.15, max_iter=25)
    inferer.update(_observation(_chain_data()))

    result = inferer.answer()

    skeleton = {frozenset(edge) for edge in result.edges()}
    assert frozenset(EDGE_XY) in skeleton
    assert frozenset(EDGE_YZ) in skeleton
    assert nx.is_directed_acyclic_graph(result)


def test_lingam_inferer_recovers_chain_orientation() -> None:
    """LiNGAM recovers the chain skeleton on non-Gaussian data."""
    data = _chain_data()
    data["X"] = np.sign(data["X"]) * np.square(data["X"])
    inferer = LiNGAMInferer(ridge=1e-3, coef_threshold=0.1)
    inferer.update(_observation(data))

    result = inferer.answer()

    skeleton = {frozenset(edge) for edge in result.edges()}
    assert frozenset(EDGE_XY) in skeleton
    assert frozenset(EDGE_YZ) in skeleton
    assert nx.is_directed_acyclic_graph(result)
