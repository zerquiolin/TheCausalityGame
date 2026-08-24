"""Tests for the reviewer-facing round-level analysis utilities."""

from __future__ import annotations

import math

from post_processing.rebuttal_utils import (
    align_run_to_budgets,
    anytime_auc,
    bootstrap_mean_ci,
    interaction_difference,
    paired_effect,
    pareto_lower,
    trajectory_instability,
    transcript_to_round_records,
    valid_feedback_records,
)


def _manifest() -> dict:
    return {
        "run_plan": {
            "budget": {
                "rounds": 10,
                "samples": 40,
                "time_s": 20,
                "memory_mb": 2,
            }
        }
    }


def _transcript(*, invalidated: bool = False) -> dict:
    return {
        "manifest_id": "seeded/graph_discovery/pc/toy_rs-7",
        "agent_id": "agent",
        "invalidated": invalidated,
        "invalidation_reason": "test failure" if invalidated else None,
        "entries": [
            {
                "round": 1,
                "decision": {"kind": "experiment"},
                "samples_collection": [
                    {"kind": "observational", "n": 1},
                    {"kind": "interventional", "n": 3},
                ],
                "budget_snapshot": {
                    "rounds_left": 9,
                    "samples_left": 36,
                    "time_s_left": 18,
                    "memory_mb_left": 1.75,
                },
                "feedback": {"result": 5, "behavior": 10},
            },
            {
                "round": 2,
                "decision": {"kind": "experiment"},
                "samples_collection": [
                    {"kind": "observational", "n": 2},
                    {"kind": "interventional", "n": 2},
                ],
                "budget_snapshot": {
                    "rounds_left": 8,
                    "samples_left": 32,
                    "time_s_left": 16,
                    "memory_mb_left": 1.5,
                },
                "feedback": {"result": 3, "behavior": 20},
            },
            {
                "round": 3,
                "decision": {"kind": "answer"},
                "samples_collection": [],
                "budget_snapshot": None,
                "feedback": None,
            },
        ],
    }


def test_transcript_extraction_uses_collected_samples_and_raw_resources() -> None:
    records = transcript_to_round_records(
        _transcript(),
        transcript_path="/tmp/transcript.json",
        manifest=_manifest(),
    )

    assert len(records) == 3
    assert records[0]["observational_increment"] == 1
    assert records[0]["interventional_increment"] == 3
    assert records[1]["observational_cumulative"] == 3
    assert records[1]["interventional_cumulative"] == 5
    assert records[1]["total_cumulative"] == 8
    assert records[1]["time_used_s"] == 4
    assert records[1]["memory_used_mb"] == 0.5
    assert records[2]["decision_kind"] == "answer"
    assert records[2]["error"] is None


def test_invalidated_rounds_are_never_valid_feedback() -> None:
    records = transcript_to_round_records(
        _transcript(invalidated=True),
        transcript_path="/tmp/transcript.json",
        manifest=_manifest(),
    )
    assert valid_feedback_records(records) == []
    assert all(record["invalidation_reason"] == "test failure" for record in records)


def test_budget_alignment_is_left_continuous_and_does_not_look_ahead() -> None:
    records = transcript_to_round_records(
        _transcript(),
        transcript_path="/tmp/transcript.json",
        manifest=_manifest(),
    )
    assert align_run_to_budgets(records, [0, 3, 4, 7, 8, 20]) == [
        None,
        None,
        5.0,
        5.0,
        3.0,
        3.0,
    ]


def test_anytime_auc_and_instability() -> None:
    run = [
        {"total_cumulative": 4, "error": 5},
        {"total_cumulative": 8, "error": 3},
        {"total_cumulative": 12, "error": 4},
    ]
    assert anytime_auc(run, max_budget=12) == 4
    instability = trajectory_instability(run)
    assert instability["worsening_fraction"] == 0.5
    assert instability["max_rebound"] == 1


def test_bootstrap_and_paired_effect_are_deterministic() -> None:
    first = bootstrap_mean_ci([1, 2, 3], resamples=500, seed=911)
    second = bootstrap_mean_ci([1, 2, 3], resamples=500, seed=911)
    assert first == second
    assert first.estimate == 2

    effect = paired_effect({1: 2, 2: 4, 3: 6}, {2: 1, 3: 2, 4: 99}, resamples=500)
    assert effect.seeds == (2, 3)
    assert effect.estimate == 3.5
    assert effect.win_rate_left == 0


def test_interaction_uses_the_four_way_seed_intersection() -> None:
    effect = interaction_difference(
        {1: 5, 2: 6},
        {1: 2, 2: 2},
        {1: 4, 2: 4},
        {1: 3, 2: 3},
        resamples=500,
    )
    assert effect.seeds == (1, 2)
    assert effect.estimate == 2.5
    assert not math.isnan(effect.ci_low)


def test_pareto_lower_minimizes_every_dimension() -> None:
    assert pareto_lower([(1, 3), (2, 2), (3, 3), (1, 4), (2, 2)]) == [
        (1.0, 3.0),
        (2.0, 2.0),
    ]

