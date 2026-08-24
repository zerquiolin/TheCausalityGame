"""Validation tests for the targeted rebuttal experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_rebuttal_suite_has_21_unique_valid_matched_seed_manifests() -> None:
    paths = sorted(MANIFEST_ROOT.glob("hill_rs-*.json"))
    assert len(paths) == 21

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
        ) == 7

    assert len(experiment_ids) == 21
    assert len(seeds) == 21

