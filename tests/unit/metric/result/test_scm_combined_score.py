"""Scientific regression checks for directed-SHD SCM scoring."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.scm import EstimatedANMSCM, NumericANMMechanism
from TheCausalityGame.metric.result.scm_combined_score import (
    SCMCombinedScoreResultMetric,
)

STRUCTURAL_SCORE_WEIGHT = 0.25
MOUNT_TRIAL_COUNT = 4


class _ConstantModel:
    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.zeros(len(rows))


class _TwoNodeTruth:
    def __init__(self) -> None:
        self.dag = nx.DiGraph([("X", "Y")])
        self.vars = ["X", "Y"]
        self.controllable_vars = ["X"]
        self.nodes = {"X": SimpleNamespace(domain=[-1.0, 1.0])}

    def generate_samples(
        self,
        num_samples: int,
        interventions: dict[str, object] | None,
        *,
        cancel_noise: bool = False,
        random_state: object | None = None,
    ) -> pd.DataFrame:
        del cancel_noise, random_state
        x = (interventions or {}).get("X", 0.0)
        return pd.DataFrame({"X": [x] * num_samples, "Y": [0.0] * num_samples})


def _metric() -> SCMCombinedScoreResultMetric:
    truth = _TwoNodeTruth()
    metric = SCMCombinedScoreResultMetric()
    metric.scm = truth
    metric.is_mounted = True
    metric._cached_num_samples = 32
    metric._trial_specs = [
        {
            "intervention_var": "X",
            "value": 0.25,
            "true_samples": truth.generate_samples(32, {"X": 0.25}),
        }
    ]
    metric._true_edges = {("X", "Y")}
    return metric


def _estimate(edges: list[tuple[str, str]]) -> EstimatedANMSCM:
    graph = nx.DiGraph()
    graph.add_nodes_from(["X", "Y", *(node for edge in edges for node in edge)])
    graph.add_edges_from(edges)
    mechanisms = {}
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        mechanisms[node] = NumericANMMechanism(
            node=node,
            parents=parents,
            model=_ConstantModel() if parents else None,
            feature_info={"num_parents": parents, "cat_parents": [], "cat_levels": {}},
            residuals=np.array([0.0]),
        )
    return EstimatedANMSCM(graph, mechanisms)


def test_perfect_directed_graph_and_distribution_score_is_zero() -> None:
    """An exact graph and matching intervention distribution score zero."""
    assert _metric().evaluate("SCM", _estimate([("X", "Y")])) == 0.0


def test_reversing_an_edge_adds_one_directed_shd_unit() -> None:
    """A reversed edge is one structural edit under the configured metric."""
    metric = _metric()
    correct = metric.evaluate("SCM", _estimate([("X", "Y")]))
    reversed_edge = metric.evaluate("SCM", _estimate([("Y", "X")]))

    assert reversed_edge - correct == STRUCTURAL_SCORE_WEIGHT


def test_missing_and_extra_directed_edges_are_counted_independently() -> None:
    """Missing and extra directed edges each contribute one structural unit."""
    metric = _metric()
    missing_score = metric.evaluate("SCM", _estimate([]))
    extra_score = metric.evaluate("SCM", _estimate([("X", "Y"), ("Y", "Z")]))
    assert missing_score == STRUCTURAL_SCORE_WEIGHT
    assert extra_score == STRUCTURAL_SCORE_WEIGHT


def test_metric_mount_samples_string_category_as_intervention() -> None:
    """Categorical controllable domains must use a label, not numeric bounds."""
    truth = _TwoNodeTruth()
    truth.nodes["X"].domain = ["low", "medium", "high"]
    metric = SCMCombinedScoreResultMetric()
    metric.num_trials = MOUNT_TRIAL_COUNT
    metric.num_samples = 8

    metric.mount(truth)  # type: ignore[arg-type]

    assert len(metric._trial_specs) == MOUNT_TRIAL_COUNT
    assert {trial["value"] for trial in metric._trial_specs}.issubset(
        {"low", "medium", "high"}
    )


def test_metric_uses_observational_trials_when_no_controls_exist() -> None:
    """Uncontrollable SCMs get a finite observational distribution score."""
    truth = _TwoNodeTruth()
    truth.controllable_vars = []
    metric = SCMCombinedScoreResultMetric()
    metric.num_trials = 2
    metric.num_samples = 8

    metric.mount(truth)  # type: ignore[arg-type]

    assert all(trial["intervention_var"] is None for trial in metric._trial_specs)
    assert metric.evaluate("SCM", _estimate([("X", "Y")])) == 0.0


def test_metric_keeps_finite_shd_when_no_nonintervened_variables_are_scoreable() -> None:
    """A one-node SCM has no distributional target after intervening on its node."""
    truth = _TwoNodeTruth()
    truth.vars = ["X"]
    truth.dag = nx.DiGraph()
    truth.dag.add_node("X")
    metric = SCMCombinedScoreResultMetric()
    metric.num_trials = 2
    metric.num_samples = 8
    metric.mount(truth)  # type: ignore[arg-type]

    estimate = _estimate([])
    estimate.dag = nx.DiGraph()
    estimate.dag.add_node("X")
    assert metric.evaluate("SCM", estimate) == 0.0


def test_combined_score_handles_numeric_and_string_category_descendants() -> None:
    """Distribution comparison covers numeric moments and categorical mass."""
    truth = _TwoNodeTruth()
    truth.vars = ["X", "Y", "Z"]
    truth.dag = nx.DiGraph([("X", "Y"), ("X", "Z")])
    metric = SCMCombinedScoreResultMetric()
    metric.scm = truth
    metric.is_mounted = True
    metric._cached_num_samples = 8
    metric._trial_specs = [
        {
            "intervention_var": "X",
            "value": 0.25,
            "true_samples": pd.DataFrame(
                {"X": [0.25] * 8, "Y": [2.0] * 8, "Z": ["case"] * 8}
            ),
        }
    ]
    metric._true_edges = {("X", "Y"), ("X", "Z")}
    estimate = _estimate([("X", "Y"), ("X", "Z")])
    estimate.generate_samples = (  # type: ignore[method-assign]
        lambda **_kwargs: pd.DataFrame(
            {"X": [0.25] * 8, "Y": [2.0] * 8, "Z": ["case"] * 8}
        )
    )

    assert metric.evaluate("SCM", estimate) == 0.0


def test_combined_score_penalizes_missing_generated_descendant() -> None:
    """A model missing a scoreable descendant gets the documented hard penalty."""
    metric = _metric()
    estimate = _estimate([("X", "Y")])
    estimate.generate_samples = (  # type: ignore[method-assign]
        lambda **_kwargs: pd.DataFrame({"X": [0.25] * 32})
    )

    assert metric.evaluate("SCM", estimate) == 5.0 * 0.75


@pytest.mark.parametrize("domain", [[], [0.0], [0.0, 1.0, 2.0]])
def test_metric_mount_rejects_malformed_controllable_domains(
    domain: list[object],
) -> None:
    """Reject empty categorical and malformed numeric intervention domains."""
    truth = _TwoNodeTruth()
    truth.nodes["X"].domain = domain
    with pytest.raises(ValueError, match=r"domain|bounds"):
        SCMCombinedScoreResultMetric().mount(truth)  # type: ignore[arg-type]


@pytest.mark.parametrize("reference_output", [None, pd.DataFrame({"X": [0.0]})])
def test_metric_rejects_invalid_reference_sample_output(reference_output) -> None:  # noqa: ANN001
    """Invalid ground-truth samples fail during metric setup with a clear error."""
    truth = _TwoNodeTruth()
    truth.generate_samples = (
        lambda *_args, **_kwargs: reference_output
    )  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="reference sampling"):
        SCMCombinedScoreResultMetric().mount(truth)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_output", [None, pd.DataFrame(columns=["X", "Y"])])
def test_metric_penalizes_invalid_estimated_sample_outputs(bad_output) -> None:  # noqa: ANN001
    """A missing or wrong-sized generated table receives a finite hard penalty."""
    metric = _metric()
    estimate = _estimate([("X", "Y")])
    estimate.generate_samples = (
        lambda **_kwargs: bad_output
    )  # type: ignore[method-assign]

    assert metric.evaluate("SCM", estimate) == 5.0 * 0.75
