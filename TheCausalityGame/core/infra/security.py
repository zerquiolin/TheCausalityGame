from __future__ import annotations

from typing import Any

from TheCausalityGame.core.infra.serialization import SerializationError, dumps


class TrustError(RuntimeError):
    """Raised when an action is not allowed in restricted/trusted modes."""


def assert_no_callables(payload: Any) -> None:
    stack = [payload]
    while stack:
        cur = stack.pop()
        if callable(cur):
            raise TrustError("Callable values are not allowed in restricted mode")
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple, set)):
            stack.extend(cur)


def ensure_json_safe(payload: Any) -> None:
    try:
        dumps(payload)
    except SerializationError as e:
        raise TrustError(f"Payload is not JSON-safe: {e}") from e
