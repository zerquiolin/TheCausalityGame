"""Tests for empirical-versus-authoritative comparison post-processing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from post_processing.comparison_analysis import analyze_comparisons, audit_records


def _backend(kind: str) -> dict[str, str]:
    if kind == "empirical":
        return {"project": "TheCausalityGame", "implementation_kind": kind}
    return {
        "project": "upstream/project",
        "algorithm": "Published method",
        "version": "abc123",
        "source": "https://example.test/upstream",
        "citation": "Author et al. (2024)",
        "license": "MIT",
        "adapter_version": "1",
        "implementation_kind": kind,
    }


def _row(method: str, implementation: str, **values: object) -> dict[str, object]:
    return {
        "case": "chain",
        "method": method,
        "seed": 911,
        "round": 1,
        "implementation": implementation,
        "status": "ok",
        "runtime_s": 0.1 if implementation == "empirical" else 0.2,
        "input_digest": "same-input",
        "backend": _backend(implementation),
        **values,
    }


def test_analysis_exports_only_successful_pairs_and_validates_figures(
    tmp_path: Path,
) -> None:
    """Only successful pairs enter deltas and all rendered files pass validation."""
    run = tmp_path / "20260829T120000Z"
    run.mkdir()
    rows = [
        _row("pc", "empirical", round=1, shd=3.0, adjacency_f1=0.4),
        _row("pc", "authoritative", round=1, shd=4.0, adjacency_f1=0.3),
        _row("pc", "empirical", round=2, shd=2.0, adjacency_f1=0.5),
        _row("pc", "authoritative", round=2, shd=1.0, adjacency_f1=0.75),
        _row("notears", "empirical", shd=3.0, adjacency_f1=0.4),
        _row(
            "notears",
            "authoritative",
            status="error",
            error="backend unavailable",
        ),
    ]
    artifact = run / "combined_comparison.jsonl"
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "report"
    result = analyze_comparisons(
        [run], output, bootstrap_resamples=100, bootstrap_seed=17
    )

    summary = result["summary"].set_index(["method", "metric"])
    assert summary.loc[("pc", "shd"), "mean_improvement"] == pytest.approx(1.0)
    assert summary.loc[("pc", "adjacency_f1"), "mean_improvement"] == pytest.approx(
        0.25
    )
    assert set(result["differences"]["method"]) == {"pc"}
    assert set(result["differences"]["round"]) == {1, 2}
    assert "unavailable_pair" in set(result["audit"]["code"])
    expected_artifact_count = 26
    assert len(result["figure_artifacts"]) == expected_artifact_count
    assert result["figure_artifacts"]["valid"].all()
    assert (output / "analysis.md").is_file()
    assert (output / "figure_artifacts.csv").is_file()
    assert (output / "inferer_pc_side_by_side.png").is_file()
    assert (output / "inferer_notears_side_by_side.png").is_file()
    assert (output / "inferer_pc_chain_round_trajectories.png").is_file()
    assert (output / "inferer_pc_chain_round_vs_all_metrics.png").is_file()
    assert "Authoritative` means" in (output / "analysis.md").read_text()


def test_audit_detects_nonidentical_pair_inputs() -> None:
    """A pair is rejected when the two implementations saw different input data."""
    records = pd.DataFrame(
        [
            {
                **_row("pc", "empirical", shd=1.0),
                "run_id": "run",
                "input_digest": "first",
            },
            {
                **_row("pc", "authoritative", shd=1.0),
                "run_id": "run",
                "input_digest": "second",
            },
        ]
    )

    audit = audit_records(records)

    assert "input_digest_mismatch" in set(audit["code"])
