"""Normalize typed AgentDecision objects into JSON-safe Actions for persistence."""

from __future__ import annotations

from typing import Any, Tuple

from TheCausalityGame.core.contracts.decisions import (
    AgentDecision,
    Intervene,
    SubmitFinal,
    SubmitPartial,
)
from TheCausalityGame.core.contracts.deliverables import (
    CallableDeliverable,
    CallableRef,
    DataDeliverable,
)
from TheCausalityGame.core.contracts.dto import Action
from TheCausalityGame.core.infra.persistence import action_payload_for_transcript


def _payload_from_deliverable(deliv: Any, *, trusted: bool) -> dict[str, Any]:
    """Build a raw (possibly non-JSON) payload from a deliverable."""
    if isinstance(deliv, DataDeliverable):
        return {
            "deliverable_type": deliv.schema_id,  # mission-defined schema id
            "path": deliv.path,  # artifact path under run dir
            "schema_version": deliv.schema_version,
        }
    if isinstance(deliv, CallableDeliverable):
        if not isinstance(deliv.ref, CallableRef):
            raise TypeError("CallableDeliverable.ref must be a CallableRef")
        raw = {
            "deliverable_type": deliv.protocol_id,  # mission-defined protocol id
            "callable_ref": {"class": deliv.ref.class_path, "config": deliv.ref.config},
        }
        if trusted:
            # Allow live object in memory; it will be sanitized for transcripts.
            raw["callable_obj"] = deliv.obj
        return raw
    raise TypeError(f"Unsupported deliverable type: {type(deliv).__name__}")


def decision_to_action(
    decision: AgentDecision, *, trusted: bool
) -> Tuple[Action, dict[str, Any]]:
    """Convert a typed AgentDecision to (JSON-safe Action, raw payload).

    The Action is what we persist (transcripts). The raw payload is passed to
    the mission for validation and to metrics for evaluation (may include live objects).
    """
    if isinstance(decision, Intervene):
        raw = {
            "interventions": decision.interventions or {},
            "n": decision.n,
            "seed": decision.seed,
        }
        act = Action(kind="intervene", payload=action_payload_for_transcript(raw))
        return act, raw

    if isinstance(decision, SubmitPartial):
        raw = _payload_from_deliverable(decision.deliverable, trusted=trusted)
        act = Action(kind="submit_partial", payload=action_payload_for_transcript(raw))
        return act, raw

    if isinstance(decision, SubmitFinal):
        raw = _payload_from_deliverable(decision.deliverable, trusted=trusted)
        act = Action(kind="submit_final", payload=action_payload_for_transcript(raw))
        return act, raw

    raise TypeError(f"Unknown decision type: {type(decision).__name__}")
