"""Regression tests for the PEHE result metric."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from TheCausalityGame.core.contracts.specs.problem_instance import (
    ProblemInstanceSpec,
)
from TheCausalityGame.core.infrastructure.registry import build_from_spec


def test_pehe_excludes_post_treatment_mediators() -> None:
    """Conditioning on Z -> X -> Y's mediator must not erase Z's total effect."""
    manifest = Path(
        "scripts/problem_instances_rebuttal/treatment_effect/"
        "honest_causal_tree/hill_rs-47.json"
    )
    with manifest.open() as handle:
        spec = ProblemInstanceSpec(**json.load(handle))

    problem_instance = build_from_spec(spec)
    metric = problem_instance.mission.result_metric
    metric.mount(problem_instance.scm)

    assert metric.treatment_node == "Z"
    assert metric.te_node == "Y"
    assert metric.input_features == []
    assert np.isclose(metric.true_cate, 2.0)
