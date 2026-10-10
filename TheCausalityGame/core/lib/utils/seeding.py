"""Stable, typed identities for independent scientific random streams."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def intervention_identity(treatment: dict | None) -> str:
    """Canonicalize numeric arms while distinguishing categorical string labels."""
    items = []
    for name, value in sorted((treatment or {}).items()):
        if isinstance(value, (int, float, np.number)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Intervention values must be finite.")
            value = ["number", 0.0 if number == 0 else number]
        else:
            value = ["category", str(value)]
        items.append([name, value])
    return json.dumps(items, separators=(",", ":"), ensure_ascii=False)


def state_identity(random_state: np.random.RandomState) -> str:
    """Hash an RNG state without consuming its stream."""
    algorithm, keys, position, gaussian, cached = random_state.get_state()
    payload = (
        keys.astype("<u4").tobytes()
        + repr((algorithm, position, gaussian, cached)).encode()
    )
    return hashlib.sha256(payload).hexdigest()


def derive_seed(*parts: object) -> int:
    """Derive an unsigned 32-bit seed from an explicit namespace and identity."""
    payload = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def derive_random_state(*parts: object) -> np.random.RandomState:
    """Initialize an independent stream with the full SHA-256 key array."""
    payload = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return np.random.RandomState(
        np.frombuffer(hashlib.sha256(payload).digest(), dtype="<u4")
    )
