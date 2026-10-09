"""Unit tests for authoritative inferer adapter boundaries."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.causal_tree import HonestCausalTreeInferer
from TheCausalityGame.agent.inferers.lingam import LiNGAMInferer
from TheCausalityGame.agent.inferers.notears import NOTEARSInferer
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer
from TheCausalityGame.agent.inferers.pc import PCInferer
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.core.contracts.dto.environment import (
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.lib.errors.inferer import (
    InfererDependencyError,
    ObsoleteInfererParameterError,
    UnsupportedInfererInputError,
)


def _observation(data: pd.DataFrame) -> RoundObservation:
    sample = Samples(kind="observational", n=len(data), data=data, interventions=None)
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=None, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _chain() -> pd.DataFrame:
    return pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [0.0, 2.0, 4.0]})


def test_pc_adapter_preserves_unresolved_edge(monkeypatch) -> None:  # noqa: ANN001
    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_name(self) -> str:
            return self.name

    class Edge:
        def get_node1(self):  # noqa: ANN201
            return Node("X")

        def get_node2(self):  # noqa: ANN201
            return Node("Y")

        def get_endpoint1(self):  # noqa: ANN201
            return SimpleNamespace(name="TAIL")

        def get_endpoint2(self):  # noqa: ANN201
            return SimpleNamespace(name="TAIL")

    pc_module = ModuleType("causallearn.search.ConstraintBased.PC")
    pc_module.pc = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        G=SimpleNamespace(get_graph_edges=lambda: [Edge()])
    )
    monkeypatch.setitem(sys.modules, "causallearn", ModuleType("causallearn"))
    monkeypatch.setitem(
        sys.modules, "causallearn.search", ModuleType("causallearn.search")
    )
    monkeypatch.setitem(
        sys.modules,
        "causallearn.search.ConstraintBased",
        ModuleType("causallearn.search.ConstraintBased"),
    )
    monkeypatch.setitem(sys.modules, "causallearn.search.ConstraintBased.PC", pc_module)

    inferer = PCInferer(show_progress=False)
    inferer.update(_observation(_chain()))
    result = inferer.answer()

    assert isinstance(result, PartiallyDirectedGraph)
    assert result.undirected_edges == (("X", "Y"),)


def test_notears_adapter_calls_official_function(monkeypatch) -> None:  # noqa: ANN001
    called: dict[str, object] = {}

    def fake_notears(data, **kwargs):  # noqa: ANN001, ANN202
        called.update(kwargs)
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    module = ModuleType("notears.linear")
    module.notears_linear = fake_notears  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "notears", ModuleType("notears"))
    monkeypatch.setitem(sys.modules, "notears.linear", module)
    inferer = NOTEARSInferer(lambda1=0.2)
    inferer.update(_observation(_chain()))

    result = inferer.answer()

    assert isinstance(result, nx.DiGraph)
    assert set(result.edges) == {("X", "Y")}
    assert called["lambda1"] == 0.2


def test_lingam_adapter_transposes_child_parent_matrix(
    monkeypatch,
) -> None:  # noqa: ANN001
    class DirectLiNGAM:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def fit(self, data) -> None:  # noqa: ANN001
            self.adjacency_matrix_ = np.array([[0.0, 0.0], [2.0, 0.0]])

    module = ModuleType("lingam")
    module.DirectLiNGAM = DirectLiNGAM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lingam", module)
    inferer = LiNGAMInferer()
    inferer.update(_observation(_chain()))

    assert set(inferer.answer().edges) == {("X", "Y")}


def test_r_cate_adapter_uses_checked_boundary(monkeypatch) -> None:  # noqa: ANN001
    calls: list[dict[str, object]] = []

    def fake_backend(payload):  # noqa: ANN001, ANN202
        calls.append(payload)
        return {
            "predictions": [2.0, 3.0],
            "backend": {"package_version": "test"},
        }

    monkeypatch.setattr(
        "TheCausalityGame.agent.inferers._r_cate.run_r_cate_backend", fake_backend
    )
    data = pd.DataFrame(
        {"T": [0, 1, 0, 1], "Z": [0.0, 0.0, 1.0, 1.0], "Y": [0, 2, 0, 3]}
    )
    inferer = OutcomeRegressionInferer()
    inferer.update(_observation(data))
    control = pd.DataFrame({"T": [0, 0], "Z": [0.0, 1.0]})
    treated = pd.DataFrame({"T": [1, 1], "Z": [0.0, 1.0]})

    result = inferer.answer()(["Z"], "T", "Y", (control, treated))

    assert result["treatment_effect"].tolist() == [2.0, 3.0]
    assert calls[0]["method"] == "s_learner"
    assert (
        inferer.snapshot().summary["backend"]["implementation_kind"] == "authoritative"
    )


def test_authoritative_empty_answers_do_not_require_backends() -> None:
    assert PCInferer().answer() == PartiallyDirectedGraph()
    assert NOTEARSInferer().answer().number_of_nodes() == 0
    assert LiNGAMInferer().answer().number_of_nodes() == 0
    assert callable(HonestCausalTreeInferer().answer())


def test_missing_authoritative_dependency_is_explicit(
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "notears.linear", None)
    inferer = NOTEARSInferer()
    inferer.update(_observation(_chain()))

    with pytest.raises(InfererDependencyError, match="authoritative extra"):
        inferer.answer()


def test_obsolete_parameter_has_migration_error() -> None:
    spec = InfererSpec(
        class_="TheCausalityGame.agent.inferers.pc:PCInferer",
        params={"is_numerical": True},
    )

    with pytest.raises(ObsoleteInfererParameterError, match="indep_test"):
        PCInferer.from_spec(spec)


def test_native_parameters_round_trip() -> None:
    inferer = NOTEARSInferer(lambda1=0.25, loss_type="poisson", max_iter=7)

    assert NOTEARSInferer.from_spec(inferer.to_spec()).to_spec() == inferer.to_spec()


def test_r_cate_rejects_nonbinary_training_treatment() -> None:
    data = pd.DataFrame({"T": [0, 1, 2], "Z": [0.0, 1.0, 2.0], "Y": [0.0, 1.0, 2.0]})
    inferer = OutcomeRegressionInferer()
    inferer.update(_observation(data))
    control = pd.DataFrame({"T": [0], "Z": [0.0]})
    treated = pd.DataFrame({"T": [1], "Z": [0.0]})

    with pytest.raises(UnsupportedInfererInputError, match="binary treatments"):
        inferer.answer()(["Z"], "T", "Y", (control, treated))
