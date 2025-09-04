"""Utilities to sanitize actions/observations for JSON-only persistence."""

from __future__ import annotations

import json
from typing import Any

from ..contracts.deliverables import CallableRef, DeliverableHandle


def _is_serializable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def action_payload_for_transcript(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of an action payload.

    - Preserves JSON types.
    - Replaces DeliverableHandle with its JSON manifest only.
    - Replaces CallableRef with a {'class','config'} dict.
    - Drops non-serializable objects (writes a redacted marker).
    """
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, DeliverableHandle):
            out[k] = {"kind": v.kind, "manifest": v.manifest}
        elif isinstance(v, CallableRef):
            out[k] = {"class": v.class_path, "config": v.config}
        elif _is_serializable(v):
            out[k] = v
        else:
            out[k] = f"<non-serializable:{type(v).__name__}>"
    return out
