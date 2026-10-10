"""Version and integrity checks for current publication transcripts.

Legacy artifacts require a separate, explicit migration; a provenance label is
not a substitute for validating the data it describes.
"""

from collections.abc import Mapping
from math import isfinite
from typing import Any

PROTOCOL = "post-inform-v2"
TRANSCRIPT_SCHEMA_VERSION = "2.0.0"


def validate_publication_transcript(
    payload: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    """Reject incompatible or internally inconsistent completed-run artifacts."""
    if not isinstance(payload, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("Transcript and provenance must be objects")
    if provenance.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported or missing artifact protocol")
    if payload.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing transcript schema_version")
    for key in ("agent_id", "manifest_id", "mission_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Missing transcript identity: {key}")
    if payload.get("invalidated") is not False:
        raise ValueError("Publication transcript must not be invalidated")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Publication transcript requires entries")
    previous = -1
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Transcript entry must be an object")
        current = entry.get("round")
        if type(current) is not int or current <= previous:
            raise ValueError("Transcript rounds must be strictly increasing integers")
        previous = current
        batches = entry.get("samples_collection")
        if batches is not None and not isinstance(batches, list):
            raise ValueError("Sample collection must be a list or null")
        for batch in batches or []:
            if not isinstance(batch, dict):
                raise ValueError("Sample batch must be an object")
            n, data = batch.get("n"), batch.get("data")
            if (
                type(n) is not int
                or n <= 0
                or not isinstance(data, list)
                or len(data) != n
            ):
                raise ValueError("Sample count does not match serialized rows")
            if batch.get("kind") not in {"observational", "interventional"}:
                raise ValueError("Unknown sample kind")
            if any(not isinstance(row, dict) for row in data):
                raise ValueError("Sample rows must be objects")
            interventions = batch.get("interventions")
            if interventions is not None and not isinstance(interventions, dict):
                raise ValueError("Sample interventions must be an object or null")
            if batch["kind"] == "observational" and interventions is not None:
                raise ValueError("Observational samples cannot declare interventions")
            if batch["kind"] == "interventional" and not interventions:
                raise ValueError("Interventional samples require intervention values")
        feedback = entry.get("feedback")
        if feedback is not None and not isinstance(feedback, dict):
            raise ValueError("Feedback must be an object or null")
        feedback = feedback or {}
        score = feedback.get("result")
        if score is not None and (
            type(score) not in (int, float) or not isfinite(score)
        ):
            raise ValueError("Feedback score must be finite or null")
    reason = payload.get("termination_reason")
    forced_reasons = {
        "rounds",
        "samples",
        "time",
        "memory",
        "experiments",
        "cost",
        "stopping",
    }
    if reason not in forced_reasons | {"submitted"}:
        raise ValueError("Unknown successful termination reason")
    if payload.get("forced_stop") is not (reason in forced_reasons):
        raise ValueError("Forced-stop marker disagrees with termination reason")
    status = payload.get("response_status")
    feedback = payload.get("final_feedback")
    if feedback is not None and not isinstance(feedback, dict):
        raise ValueError("Final feedback must be an object or null")
    feedback = feedback or {}
    score = feedback.get("result")
    if status not in {"provided", "missing", "failed"}:
        raise ValueError("Unknown final response status")
    if status != "provided":
        if score is not None or payload.get("final_result") is not None:
            raise ValueError(
                "A missing or failed response cannot retain a result or score"
            )
        expected = "missing"
    elif score is None:
        expected = "missing"
    elif type(score) in (int, float) and isfinite(score):
        expected = "finite"
    else:
        raise ValueError("Final feedback score must be finite or null")
    if payload.get("final_score_status") != expected:
        raise ValueError("Final score status disagrees with feedback")
    last_feedback = entries[-1].get("feedback") or {}
    if score != last_feedback.get("result"):
        raise ValueError("Final score disagrees with terminal entry")
