from __future__ import annotations

import hashlib
from typing import Any

from .serialization import dumps

# personalization string to scope hashes to this framework + version
_PERSONALIZATION = b"TCG:intervention:v1"


def _blake2b_64(data: bytes, *, person: bytes = _PERSONALIZATION) -> bytes:
    """Return a 64-byte BLAKE2b digest with a personalization string."""
    h = hashlib.blake2b(digest_size=64, person=person)
    h.update(data)
    return h.digest()


def hash_intervention_key(
    *,
    base_seed: int | None,
    manifest_id: str,
    agent_id: str,
    round_index: int,
    interventions: dict[str, Any] | None,
    n: int | None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable 512-bit hex digest that identifies a sampling request.

    The digest encodes intent (who/what/when), not the resulting dataset.
    Uses canonical JSON to avoid dict-order nondeterminism and forbids NaN/Inf.

    Args:
        base_seed: Optional global seed (from RunPlan or user).
        manifest_id: ProblemInstance id.
        agent_id: Agent identifier.
        round_index: 0-based round index.
        interventions: do()-style mapping used to generate data.
        n: number of requested samples.
        extra: optional bag (e.g., mission/scm ids/versions).

    Returns
    -------
        128-character hex string (blake2b-512).
    """
    payload = {
        "v": 1,
        "base_seed": base_seed,
        "manifest_id": manifest_id,
        "agent_id": agent_id,
        "round_index": round_index,
        "interventions": interventions or {},
        "n": n,
        "extra": extra or {},
    }
    s = dumps(payload, ensure_ascii=True, indent=None)
    return _blake2b_64(s.encode("utf-8")).hex()


def make_intervention_seed(
    *,
    base_seed: int | None,
    manifest_id: str,
    agent_id: str,
    round_index: int,
    interventions: dict[str, Any] | None,
    n: int | None,
) -> int:
    """Derive a 32-bit RNG seed from the intervention key.

    Returns
    -------
        Integer in [0, 2**32-2] suitable for numpy/random seeding.
    """
    hex_digest = hash_intervention_key(
        base_seed=base_seed,
        manifest_id=manifest_id,
        agent_id=agent_id,
        round_index=round_index,
        interventions=interventions,
        n=n,
    )
    seed64 = int(hex_digest[:16], 16)
    return seed64 % (2**32 - 1)
