"""Smoke tests for deterministic comparison artifacts."""

import csv
import json

import pandas as pd

from scripts.comparisons._common import buffered_rounds, input_digest, write_reports


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
