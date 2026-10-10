"""Mutate one valid wire message at a time to test independent invariants."""

from copy import deepcopy

import pytest

from TheCausalityGame.core.infrastructure.artifact_contract import (
    validate_publication_transcript,
)


@pytest.fixture
def payload():
    return {
        "schema_version": "2.0.0",
        "agent_id": "a",
        "manifest_id": "m",
        "mission_id": "q",
        "invalidated": False,
        "termination_reason": "samples",
        "forced_stop": True,
        "response_status": "provided",
        "final_result": {"type": "dag", "nodes": [], "edges": []},
        "final_feedback": {"result": 0.0},
        "final_score_status": "finite",
        "entries": [
            {
                "round": 1,
                "feedback": {"result": 0.0},
                "samples_collection": [
                    {"n": 1, "kind": "observational", "data": [{"X": 0}]}
                ],
            }
        ],
    }


def test_current_protocol_validates_without_mutating_input(payload):
    before = deepcopy(payload)
    validate_publication_transcript(payload, {"protocol": "post-inform-v2"})
    assert payload == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "3.0.0"),
        ("forced_stop", False),
        ("response_status", "missing"),
        ("final_score_status", "missing"),
        ("invalidated", True),
        ("entries", []),
        ("termination_reason", "unknown"),
    ],
)
def test_invalid_headers_are_rejected(payload, field, value):
    payload[field] = value
    with pytest.raises(ValueError):
        validate_publication_transcript(payload, {"protocol": "post-inform-v2"})


@pytest.mark.parametrize("protocol", [None, "pre-inform-v1", "post-inform-v3"])
def test_unknown_protocol_is_not_silently_pooled(payload, protocol):
    with pytest.raises(ValueError, match="protocol"):
        validate_publication_transcript(payload, {"protocol": protocol})


def test_sample_count_mismatch_is_rejected(payload):
    payload["entries"][0]["samples_collection"][0]["n"] = 2
    with pytest.raises(ValueError, match="count"):
        validate_publication_transcript(payload, {"protocol": "post-inform-v2"})


@pytest.mark.parametrize(
    ("data", "interventions", "kind"),
    [
        (["not-a-row-object"], None, "observational"),
        ([{"X": 0}], ["X", 1], "interventional"),
        ([{"X": 0}], {"X": 1}, "observational"),
        ([{"X": 0}], None, "interventional"),
    ],
)
def test_nested_sample_schema_rejects_invalid_rows_and_interventions(
    payload, data, interventions, kind
):
    batch = payload["entries"][0]["samples_collection"][0]
    batch.update(data=data, n=len(data), interventions=interventions, kind=kind)
    with pytest.raises(ValueError):
        validate_publication_transcript(payload, {"protocol": "post-inform-v2"})


def test_duplicate_round_is_rejected(payload):
    payload["entries"].append(deepcopy(payload["entries"][0]))
    with pytest.raises(ValueError, match="rounds"):
        validate_publication_transcript(payload, {"protocol": "post-inform-v2"})


def test_forced_stop_without_response_is_valid_but_unscored(payload):
    payload.update(
        response_status="missing",
        final_result=None,
        final_feedback={"result": None},
        final_score_status="missing",
    )
    payload["entries"][-1]["feedback"]["result"] = None
    validate_publication_transcript(payload, {"protocol": "post-inform-v2"})


@pytest.mark.parametrize(
    "field,value", [("samples_collection", {}), ("feedback", []), ("round", True)]
)
def test_malformed_entry_fields_raise_contract_errors(payload, field, value):
    payload["entries"][0][field] = value
    with pytest.raises(ValueError):
        validate_publication_transcript(payload, {"protocol": "post-inform-v2"})
