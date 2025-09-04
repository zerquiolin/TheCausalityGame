"""Generic deliverable contracts and helpers (mission-agnostic).

This module defines two foundational deliverable shapes:

- DataDeliverable: JSON/JSONL artifacts identified by a mission-defined `schema_id`.
- CallableDeliverable: a live Python object implementing a *mission-defined* protocol,
  identified by a `protocol_id`, plus a JSON-safe `CallableRef` for reproducibility.

Missions decide which schema_ids/protocol_ids they accept. The core remains generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallableRef:
    """JSON-safe reference to rebuild a callable later (no pickled code).

    Attributes:
      class_path: Fully-qualified 'module:Class' for reconstruction.
      config: JSON-serializable kwargs for the constructor.
    """

    class_path: str
    config: dict[str, Any]


@dataclass
class DataDeliverable:
    """Data deliverable (portable artifact).

    Attributes:
      schema_id: Mission-defined identifier for the JSON/JSONL schema (e.g., 'pred_grid_v1').
      path: Path to the artifact under the run directory.
      schema_version: Optional version string for the schema itself.
    """

    schema_id: str
    path: str
    schema_version: str = "1.0.0"


@dataclass
class CallableDeliverable:
    """Callable deliverable (dev/trusted mode only).

    Attributes:
      protocol_id: Mission-defined identifier for the protocol (e.g., 'predict_fn_v1').
      obj: Live Python object used NOW in trusted mode (implements that protocol).
      ref: JSON-safe reference for later reconstruction (no pickling).
    """

    protocol_id: str
    obj: Any
    ref: CallableRef


@dataclass
class DeliverableHandle:
    """Internal envelope after mission validation.

    Attributes:
      kind: Mission-defined deliverable kind identifier (schema_id or protocol_id).
      manifest: JSON-safe dict persisted to transcripts (e.g., {'deliverable_type': ..., ...}).
      in_memory: Optional live object usable at runtime (callables in trusted mode).
    """

    kind: str
    manifest: dict[str, Any]
    in_memory: Any | None = None


def ensure_callable_allowed(restricted_mode: bool) -> None:
    """Raise if callable deliverables are disallowed by policy."""
    from ..contracts.errors import SecurityViolation

    if restricted_mode:
        raise SecurityViolation(
            "Callable deliverables are disabled in restricted/benchmark mode. "
            "Submit a data deliverable instead."
        )
