"""Validation tests for the targeted rebuttal experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.problem_instances_rebuttal import run_rebuttal
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_ROOT = (
    ROOT
    / "scripts"
    / "problem_instances_rebuttal"
    / "treatment_effect"
    / "honest_causal_tree"
)
EXPECTED_BUDGET = {
    "rounds": 100,
    "time_s": 400,
    "samples": 400,
    "memory_mb": 2,
}
EXPECTED_MANIFEST_COUNT = 21
EXPECTED_ACTIVE_AGENT_COUNT = 7


def test_rebuttal_suite_has_21_unique_valid_matched_seed_manifests() -> None:
    """Every checked-in rebuttal target has a valid unique matched seed."""
    paths = sorted(MANIFEST_ROOT.glob("hill_rs-*.json"))
    assert len(paths) == EXPECTED_MANIFEST_COUNT

    experiment_ids: set[str] = set()
    seeds: set[int] = set()
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ProblemInstanceSpec(**manifest)
        experiment_ids.add(manifest["id"])
        seeds.add(int(path.stem.rsplit("_rs-", 1)[1]))

        assert manifest["id"].startswith(
            "rebuttal/treatment_effect/honest_causal_tree/hill_rs-"
        )
        assert manifest["run_plan"]["budget"] == EXPECTED_BUDGET
        assert len(
            [agent for agent in manifest["agents"] if agent.get("active", True)]
        ) == EXPECTED_ACTIVE_AGENT_COUNT

    assert len(experiment_ids) == EXPECTED_MANIFEST_COUNT
    assert len(seeds) == EXPECTED_MANIFEST_COUNT


def _complete_transcript() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "agent_id": "agent-a",
        "manifest_id": "manifest-a",
        "mission_id": "mission-a",
        "invalidated": False,
        "termination_reason": "submitted",
        "forced_stop": False,
        "response_status": "provided",
        "final_result": {"type": "dag", "nodes": [], "edges": []},
        "final_feedback": {"result": 0.0},
        "final_score_status": "finite",
        "entries": [
            {
                "round": 1,
                "feedback": {"result": 0.0},
                "samples_collection": [],
            }
        ],
    }


def test_resume_classification_rejects_malformed_protocol_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must not trust score fields when the wire schema is invalid."""
    timestamp = tmp_path / "timestamp"
    agent_dir = timestamp / "agents" / "agent-a"
    agent_dir.mkdir(parents=True)
    provenance = {
        "source_fingerprint": "source",
        "manifest_fingerprint": "manifest",
        "protocol": "post-inform-v2",
    }
    (timestamp / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    transcript = _complete_transcript()
    transcript["schema_version"] = "unknown"
    (agent_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    monkeypatch.setattr(run_rebuttal, "source_fingerprint", lambda: "source")

    assert not run_rebuttal._timestamp_is_valid(timestamp, {"agent-a"}, "manifest")


def test_resume_classification_accepts_valid_current_protocol_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid, matching current-protocol run remains resumable as complete."""
    timestamp = tmp_path / "timestamp"
    agent_dir = timestamp / "agents" / "agent-a"
    agent_dir.mkdir(parents=True)
    provenance = {
        "source_fingerprint": "source",
        "manifest_fingerprint": "manifest",
        "protocol": "post-inform-v2",
    }
    (timestamp / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (agent_dir / "transcript.json").write_text(
        json.dumps(_complete_transcript()), encoding="utf-8"
    )
    monkeypatch.setattr(run_rebuttal, "source_fingerprint", lambda: "source")

    assert run_rebuttal._timestamp_is_valid(timestamp, {"agent-a"}, "manifest")


def test_resume_classification_rejects_duplicate_or_misplaced_agent_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate identities or path/payload mismatches are not complete runs."""
    timestamp = tmp_path / "timestamp"
    first = timestamp / "agents" / "agent-a"
    duplicate = timestamp / "agents" / "another-directory"
    first.mkdir(parents=True)
    duplicate.mkdir(parents=True)
    provenance = {
        "source_fingerprint": "source",
        "manifest_fingerprint": "manifest",
        "protocol": "post-inform-v2",
    }
    (timestamp / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    wire = json.dumps(_complete_transcript())
    (first / "transcript.json").write_text(wire, encoding="utf-8")
    (duplicate / "transcript.json").write_text(wire, encoding="utf-8")
    monkeypatch.setattr(run_rebuttal, "source_fingerprint", lambda: "source")

    assert not run_rebuttal._timestamp_is_valid(timestamp, {"agent-a"}, "manifest")

