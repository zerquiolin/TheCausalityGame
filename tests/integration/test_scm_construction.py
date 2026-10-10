"""Analytic and sampling regressions for required publication construction paths."""

import itertools
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.utils.seeding import intervention_identity
from TheCausalityGame.generators import (
    SyntheticSCMGenerator,
    dataset_scm,
    load_bif,
    migrate_legacy_scm,
)
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.no_noise import NoNoiseDistribution


@pytest.mark.parametrize("noise", ["uniform", "gaussian", "laplace", "dirac"])
@pytest.mark.parametrize("categorical_fraction", [0, 0.5, 1])
def test_synthetic_mixed_type_bounds_roundtrip_and_chunking(
    noise, categorical_fraction
):
    generator = SyntheticSCMGenerator(
        n_nodes=10, categorical_fraction=categorical_fraction, noise=noise, seed=23
    )
    spec = generator.generate()
    assert spec == generator.generate()
    scm = build_from_spec(spec)
    assert nx.is_directed_acyclic_graph(scm.dag.graph)
    assert max(dict(scm.dag.graph.in_degree()).values()) <= generator.max_indegree
    assert nx.dag_longest_path_length(scm.dag.graph) <= generator.max_depth
    rs = scm.prepare_new_random_state_structure(np.random.RandomState(57))
    samples = scm.generate_samples(num_samples=30, random_state=rs)
    restored = build_from_spec(scm.to_spec())
    whole = restored.generate_samples(
        num_samples=30,
        random_state=restored.prepare_new_random_state_structure(
            np.random.RandomState(57)
        ),
    )
    pd.testing.assert_frame_equal(samples, whole)
    streams = scm.prepare_new_random_state_structure(np.random.RandomState(57))
    chunks = [
        scm.generate_samples(num_samples=n, random_state=streams) for n in (13, 17)
    ]
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), whole)
    for node in scm.nodes.values():
        if all(isinstance(value, str) for value in node.domain):
            assert set(samples[node.name]).issubset(node.domain)
        else:
            assert samples[node.name].between(*node.domain).all()


def test_constant_and_root_equations_broadcast_and_validate():
    import sympy as sp

    node = EquationBasedNumericalSCMNode(
        name="Y",
        evaluation=sp.Integer(3),
        domain=[0, 5],
        parents=["X"],
        noise_distribution=NoNoiseDistribution(),
    )
    np.testing.assert_array_equal(
        node.generate_values(pd.DataFrame({"X": [1, 2, 3]})), [3, 3, 3]
    )
    root = EquationBasedNumericalSCMNode(
        name="R",
        evaluation=sp.Integer(0),
        domain=[0, 5],
        noise_distribution=NoNoiseDistribution(),
    )
    np.testing.assert_array_equal(
        root.generate_values(pd.DataFrame(index=range(3))), [0, 0, 0]
    )
    assert build_from_spec(root.to_spec()).evaluation == 0


def test_bif_parent_order_and_normalized_cpt(tmp_path):
    path = tmp_path / "ordered.bif"
    path.write_text("""network tiny {} variable A { type discrete [2] { no, yes }; }
variable B { type discrete [2] { low, high }; }
variable Y { type discrete [2] { off, on }; }
probability (A) { table 0.4, 0.6; }
probability (B) { table 0.3, 0.7; }
probability (Y | B, A) {
(low, no) 1, 0; (low, yes) 0, 1;
(high, no) 0, 1; (high, yes) 1, 0;
}""")
    scm = build_from_spec(load_bif(path))
    assert scm.nodes["Y"].parents == ["B", "A"]
    for a, b in itertools.product(["no", "yes"], ["low", "high"]):
        frame = scm.generate_samples({"A": a, "B": b}, num_samples=20)
        assert set(frame.Y) == (
            {"off"} if (a, b) in [("no", "low"), ("yes", "high")] else {"on"}
        )
    restored = build_from_spec(scm.to_spec())
    assert (
        restored.nodes["Y"].probability_distribution
        == scm.nodes["Y"].probability_distribution
    )


def test_bif_rejects_incomplete_cpt(tmp_path):
    path = tmp_path / "broken.bif"
    path.write_text(
        "variable A { type discrete [2] { no, yes }; } probability (A) { table 0.1, 0.1; }"
    )
    with pytest.raises(ValueError, match="probability"):
        load_bif(path)


def test_legacy_asia_migration_samples_and_roundtrips():
    from pathlib import Path

    asset = Path("scripts/data/scms/literature_cases/small/asia.json")
    scm = build_from_spec(migrate_legacy_scm(json.loads(asset.read_text())))
    assert len(scm.generate_samples(num_samples=10)) == 10
    assert scm.to_spec() == build_from_spec(scm.to_spec()).to_spec()


def test_dataset_controls_do_not_condition_other_controls_and_preserve_rows():
    data = pd.DataFrame(
        {
            "X": [0, 0, 1, 1],
            "T": [0, 1, 0, 1],
            "W": [0, 1, 1, 0],
            "Y": [1.0, 3.0, 2.0, 4.0],
        }
    )
    spec = dataset_scm(
        data,
        outcomes=["Y"],
        controllable=["T", "W"],
        categorical=["T", "W"],
        seed=41,
        factual_lookup=True,
    )
    scm = build_from_spec(spec)
    assert "T" not in (scm.nodes["W"].parents or [])
    assert "W" not in (scm.nodes["T"].parents or [])
    assert scm.nodes["T"].domain == ["0", "1"]
    assert scm.nodes["W"].domain == ["0", "1"]
    assert set(scm.nodes["T"].data["T"]) == {"0", "1"}
    frame = scm.generate_samples({"T": "1", "W": "1"}, num_samples=50)
    assert set(frame["T"]).issubset({"0", "1"})
    assert set(frame["W"]).issubset({"0", "1"})
    assert set(frame.X).issubset({0, 1})
    assert np.isfinite(frame.Y).all()
    restored = build_from_spec(scm.to_spec())
    first = scm.generate_samples(
        num_samples=15,
        random_state=scm.prepare_new_random_state_structure(np.random.RandomState(19)),
    )
    second = restored.generate_samples(
        num_samples=15,
        random_state=restored.prepare_new_random_state_structure(
            np.random.RandomState(19)
        ),
    )
    pd.testing.assert_frame_equal(first, second)


def test_dataset_categorical_stringification_rejects_label_collisions():
    data = pd.DataFrame({"T": pd.Series([1, "1"], dtype=object), "Y": [2.0, 3.0]})
    with pytest.raises(ValueError, match="collide"):
        dataset_scm(data, outcomes=["Y"], controllable=["T"], categorical=["T"])


def test_equation_categorical_nodes_reject_numeric_labels():
    from TheCausalityGame.scm.nodes.sympy import EquationBasedCategoricalSCMNode

    with pytest.raises(ValueError, match="string labels"):
        EquationBasedCategoricalSCMNode(
            name="T",
            evaluation=None,
            domain=[0, 1],
            noise_distribution=NoNoiseDistribution(),
        )


def test_typed_canonical_interventions():
    assert intervention_identity({"B": 1, "A": 0.0}) == intervention_identity(
        {"A": 0, "B": 1.0}
    )
    assert intervention_identity({"A": "1"}) != intervention_identity({"A": 1})
    assert intervention_identity({}) == intervention_identity(None)


def test_dataset_configurable_scikit_learn_outcome_matches_analytic_function():
    data = pd.DataFrame(
        {"X": [0, 0, 1, 1], "T": [0, 1, 0, 1], "Y": [1.0, 3.0, 2.0, 4.0]}
    )
    spec = dataset_scm(
        data,
        outcomes=["Y"],
        controllable=["T"],
        estimator={"class_": "sklearn.linear_model:LinearRegression"},
    )
    scm = build_from_spec(spec)
    generated = scm.generate_samples({"T": 1}, num_samples=30)
    np.testing.assert_allclose(generated.Y, 1 + generated.X + 2 * generated["T"])


def test_unknown_intervention_does_not_silently_become_observational():
    scm = build_from_spec(SyntheticSCMGenerator(n_nodes=3).generate())
    with pytest.raises(ValueError, match="Unknown intervention"):
        scm.generate_samples(interventions={"typo_variable": 1}, num_samples=4)
