"""Tests for symbolic SCM discovery inferer utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.symbolic_scm import (
    EstimatedSymbolicSCM,
    SparseSymbolicSCMDiscoveryInferer,
    fit_sparse_symbolic_mechanism_for_node,
)
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import RoundInfo, Samples, SamplesCollection
from TheCausalityGame.core.infrastructure.decisions import Decision


def _observation(
    data: pd.DataFrame,
    *,
    kind: str = "observational",
    interventions: dict[str, float] | None = None,
) -> RoundObservation:
    """Build a round observation for symbolic inferer tests."""
    sample = Samples(
        kind=kind,
        n=len(data),
        data=data,
        interventions=interventions,
    )
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=interventions, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _mechanism_data(n: int = 300) -> pd.DataFrame:
    """Generate noiseless product-mechanism data."""
    rng = np.random.default_rng(911)
    a = rng.uniform(1.0, 3.0, size=n)
    m = rng.uniform(2.0, 4.0, size=n)
    return pd.DataFrame({"a": a, "m": m, "F": a * m})


def test_sparse_symbolic_mechanism_recovers_product() -> None:
    """Sparse symbolic fitting recovers a noiseless product mechanism."""
    mechanism = fit_sparse_symbolic_mechanism_for_node(
        node="F",
        parents=["a", "m"],
        df=_mechanism_data(),
        alpha=1e-8,
        coefficient_threshold=1e-4,
        reciprocal_eps=1e-6,
        min_reciprocal_valid_fraction=0.95,
    )

    assert mechanism is not None
    predictions = mechanism.evaluate(_mechanism_data()[["a", "m"]])
    assert float(np.mean((predictions - _mechanism_data()["F"].to_numpy()) ** 2)) < 1e-4


def test_symbolic_inferer_excludes_target_intervened_rows() -> None:
    """Target-intervened rows are excluded for that target's mechanism fit."""
    inferer = SparseSymbolicSCMDiscoveryInferer()
    obs = _mechanism_data(5)
    target_intervened = obs.assign(F=999.0)
    other_intervened = obs.assign(a=5.0, F=obs["F"])

    inferer.update(_observation(obs))
    inferer.update(_observation(target_intervened, kind="interventional", interventions={"F": 1.0}))
    inferer.update(_observation(other_intervened, kind="interventional", interventions={"a": 5.0}))

    training_df = inferer._training_df_for_node("F")

    assert 999.0 not in set(training_df["F"])
    assert 5.0 in set(training_df["a"])


def test_symbolic_inferer_empty_answer_returns_symbolic_scm() -> None:
    """An empty inferer still returns a valid symbolic SCM result object."""
    inferer = SparseSymbolicSCMDiscoveryInferer()

    result = inferer.answer()

    assert isinstance(result, EstimatedSymbolicSCM)
    assert result.dag.number_of_nodes() == 0


def test_symbolic_inferer_uses_learned_dag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inferer fits symbolic mechanisms against the learned DAG parents."""
    def fake_learn_dag_from_samples(
        obs_df: pd.DataFrame,
        interventional_batches: dict[str, list[pd.DataFrame]],
        is_numerical: bool,
        alpha: float,
        seed: int,
    ) -> nx.DiGraph:
        """Return a deterministic product DAG."""
        _ = (obs_df, interventional_batches, is_numerical, alpha, seed)
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "F"), ("m", "F")])
        return graph

    monkeypatch.setattr(
        "TheCausalityGame.agent.inferers.symbolic_scm.learn_dag_from_samples",
        fake_learn_dag_from_samples,
    )
    inferer = SparseSymbolicSCMDiscoveryInferer(lasso_alpha=1e-8, coefficient_threshold=1e-4)
    inferer.update(_observation(_mechanism_data()))

    result = inferer.answer()

    assert "F" in result.mechanisms
    predictions = result.evaluate_mechanism("F", _mechanism_data()[["a", "m"]])
    assert float(np.mean((predictions - _mechanism_data()["F"].to_numpy()) ** 2)) < 1e-4
