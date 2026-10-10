"""Smoke tests for deterministic comparison artifacts."""

import csv
import json

import numpy as np
import pandas as pd

from scripts.comparisons._common import (
    buffered_rounds,
    input_digest,
    observation,
    write_reports,
)
from scripts.comparisons.compare_cate_inferers import _hill_case, _query
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer


def test_comparison_reports_include_provenance(tmp_path) -> None:  # noqa: ANN001
    """Comparison exports retain backend provenance and aggregate metrics."""
    rows = [
        {
            "case": "chain",
            "method": "pc",
            "implementation": "authoritative",
            "status": "ok",
            "shd": 0.0,
            "runtime_s": 0.1,
            "backend": {"project": "py-why/causal-learn"},
        }
    ]

    write_reports(rows, tmp_path, "comparison")

    record = json.loads((tmp_path / "comparison.jsonl").read_text().strip())
    assert record["backend"]["project"] == "py-why/causal-learn"
    with (tmp_path / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        aggregate = next(csv.DictReader(handle))
    assert aggregate["mean_shd"] == "0.0"
    assert aggregate["successes"] == "1"
    assert (tmp_path / "comparison.md").exists()


def test_input_digest_is_stable_and_data_sensitive() -> None:
    """Paired-input evidence is reproducible and changes with input content."""
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    first = input_digest(data, ["query", 1])
    second = input_digest(data.copy(), ["query", 1])
    changed = input_digest(data.assign(x=[1.0, 9.0]), ["query", 1])

    assert first == second
    assert first != changed


def test_buffered_rounds_are_incremental_and_exhaustive() -> None:
    """Round chunks are disjoint while cumulative buffers contain all prior samples."""
    data = pd.DataFrame({"x": range(10)})

    rounds = buffered_rounds(data, 3)

    assert [number for number, _, _ in rounds] == [1, 2, 3]
    assert [len(chunk) for _, chunk, _ in rounds] == [4, 3, 3]
    assert [len(cumulative) for _, _, cumulative in rounds] == [4, 7, 10]
    assert rounds[-1][2]["x"].tolist() == list(range(10))


def test_hill_binary_coding_matches_queries_and_backend_payload() -> None:
    """Hill's categorical labels must denote the same arms in data and queries."""
    name, data, values, truth = _hill_case(seed=911, n=120)
    assert name == "benchmark_hill"
    assert set(data["T"].unique()) == {0, 1}
    assert pd.api.types.is_numeric_dtype(data["T"])
    np.testing.assert_array_equal(truth, np.full(len(values), 2.0))
    covariates, treatment, outcome, frames = _query(values, has_covariate=False)
    inferer = OutcomeRegressionInferer()
    inferer.update(observation(data))
    payload = inferer._payload(
        X=covariates, treatment=treatment, outcome=outcome, covariate_values=frames
    )
    assert payload is not None
    assert set(payload["treatment"]) == {0, 1}
    assert len(payload["query"][".tcg_intercept"]) == len(truth)
