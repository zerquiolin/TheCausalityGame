"""Real integration checks against the pinned authoritative research backends."""

from __future__ import annotations

import shutil

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.causal_tree import HonestCausalTreeInferer
from TheCausalityGame.agent.inferers.lingam import LiNGAMInferer
from TheCausalityGame.agent.inferers.notears import NOTEARSInferer
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer
from TheCausalityGame.agent.inferers.pc import PCInferer
from TheCausalityGame.agent.inferers.transformed_outcome import (
    TransformedOutcomeInferer,
)
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.core.contracts.dto.environment import (
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.infrastructure.decisions import Decision

pytestmark = pytest.mark.authoritative


def _observation(data: pd.DataFrame) -> RoundObservation:
    sample = Samples(kind="observational", n=len(data), data=data, interventions=None)
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=None, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _linear_data(*, non_gaussian: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1204)
    size = 1000
    noise = rng.laplace(size=(size, 3)) if non_gaussian else rng.normal(size=(size, 3))
    x = noise[:, 0]
    y = 1.8 * x + noise[:, 1]
    z = 1.5 * y + noise[:, 2]
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _cate_data() -> pd.DataFrame:
    rng = np.random.default_rng(144)
    z = rng.uniform(-1, 1, 600)
    treatment = rng.binomial(1, 0.5, 600)
    outcome = z + treatment * (1.5 + z) + rng.normal(0, 0.2, 600)
    return pd.DataFrame({"T": treatment, "Z": z, "Y": outcome})


def test_pc_returns_chain_cpdag() -> None:
    pytest.importorskip("causallearn")
    inferer = PCInferer(alpha=0.01)
    inferer.update(_observation(_linear_data()))

    result = inferer.answer()

    assert isinstance(result, PartiallyDirectedGraph)
    assert {frozenset(edge) for edge in result.undirected_edges} == {
        frozenset(("X", "Y")),
        frozenset(("Y", "Z")),
    }


def test_notears_returns_acyclic_graph() -> None:
    pytest.importorskip("notears")
    inferer = NOTEARSInferer(lambda1=0.05)
    inferer.update(_observation(_linear_data()))

    result = inferer.answer()

    assert nx.is_directed_acyclic_graph(result)
    assert result.number_of_edges() >= 1


def test_direct_lingam_fits_real_backend() -> None:
    pytest.importorskip("lingam")
    inferer = LiNGAMInferer(random_state=42)
    inferer.update(_observation(_linear_data(non_gaussian=True)))

    assert inferer.answer().number_of_edges() >= 1


@pytest.mark.parametrize(
    "inferer_class",
    [
        OutcomeRegressionInferer,
        TransformedOutcomeInferer,
        HonestCausalTreeInferer,
    ],
)
def test_r_cate_backend_fits_and_predicts(inferer_class) -> None:  # noqa: ANN001
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not installed")
    inferer = inferer_class(seed=3)
    inferer.update(_observation(_cate_data()))
    control = pd.DataFrame({"T": [0, 0], "Z": [-0.5, 0.5]})
    treated = pd.DataFrame({"T": [1, 1], "Z": [-0.5, 0.5]})

    result = inferer.answer()(["Z"], "T", "Y", (control, treated))

    assert result.shape == (2, 1)
    assert np.isfinite(result.to_numpy()).all()
    backend = inferer.snapshot().summary["backend"]
    assert backend["source_commit"] == backend["pinned_commit"]
    assert backend["compatibility_patch"]
    if isinstance(inferer, OutcomeRegressionInferer):
        nested = backend["dependencies"]["Rforestry"]
        assert nested["source_commit"] == "92b396d77d7ce2bbaba1c6d97c8861ea43ae1566"
