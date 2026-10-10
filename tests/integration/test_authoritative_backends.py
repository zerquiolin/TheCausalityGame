"""Real integration checks against the pinned authoritative research backends."""

from __future__ import annotations

import os
import shutil
import subprocess

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


def _require_r_packages_or_skip(packages: tuple[str, ...]) -> None:
    """Skip live-R cases when their explicitly provisioned libraries are absent."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    package_vector = ", ".join(f'"{package}"' for package in packages)
    expression = (
        f"packages <- c({package_vector}); "
        "available <- vapply(packages, requireNamespace, logical(1), quietly=TRUE); "
        "quit(status=if (all(available)) 0 else 1)"
    )
    env = os.environ.copy()
    r_library = env.get("TCG_R_LIBS")
    if r_library:
        env["R_LIBS_USER"] = r_library
    check = subprocess.run(
        [rscript, "-e", expression],
        capture_output=True,
        check=False,
        timeout=30,
        env=env,
    )
    if check.returncode != 0:
        pytest.skip(
            "Required pinned R backend packages are not installed; "
            "run scripts/comparisons/bootstrap_r.R"
        )


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
    """PC recovers the undirected chain equivalence class on Gaussian data."""
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
    """NOTEARS returns a DAG from a synthetic linear chain."""
    pytest.importorskip("notears")
    inferer = NOTEARSInferer(lambda1=0.05)
    inferer.update(_observation(_linear_data()))

    result = inferer.answer()

    assert nx.is_directed_acyclic_graph(result)
    assert result.number_of_edges() >= 1


def test_direct_lingam_fits_real_backend() -> None:
    """DirectLiNGAM runs on non-Gaussian chain data."""
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
    """Pinned R CATE backends return finite predictions and source metadata."""
    if inferer_class is OutcomeRegressionInferer:
        _require_r_packages_or_skip(("jsonlite", "causalToolbox", "Rforestry"))
    else:
        _require_r_packages_or_skip(("jsonlite", "causalTree", "rpart"))
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


def test_honest_tree_waits_for_two_rows_per_arm_then_fits() -> None:
    """The honest tree waits for adequate arms before fitting its backend."""
    _require_r_packages_or_skip(("jsonlite", "causalTree", "rpart"))
    inferer = HonestCausalTreeInferer(seed=3)
    first = pd.DataFrame(
        {"T": [0, 0, 0, 1], "Z": [-1.0, 0.0, 1.0, 0.5], "Y": [0.0, 0.0, 0.0, 2.0]}
    )
    query = (pd.DataFrame({"T": [0], "Z": [0.0]}), pd.DataFrame({"T": [1], "Z": [0.0]}))
    inferer.update(_observation(first))
    assert inferer.answer()(["Z"], "T", "Y", query).iloc[0, 0] == 0
    assert inferer.snapshot().summary["backend"]["fit_status"] == "insufficient_data"
    inferer.update(_observation(_cate_data()))
    result = inferer.answer()(["Z"], "T", "Y", query)
    assert np.isfinite(result.to_numpy()).all()
    assert inferer.snapshot().summary["backend"]["fit_status"] == "fitted"
