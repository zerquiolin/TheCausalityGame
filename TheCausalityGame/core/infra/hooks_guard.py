from __future__ import annotations

from typing import Any, Callable, Protocol

from TheCausalityGame.core.contracts.enums import HookEvent
from TheCausalityGame.core.infra.security import (
    TrustError,
    assert_no_callables,
    ensure_json_safe,
)


class HookEmit(Protocol):
    def __call__(self, event: HookEvent, payload: dict[str, Any]) -> None: ...


def wrap_hook_emit(base_emit: HookEmit, *, trusted: bool) -> HookEmit:
    """Wrap a hook emitter enforcing security in restricted mode.

    - In trusted mode: pass-through.
    - In restricted mode: forbid callables anywhere in payload and require strict JSON-serializability.

    If validation fails, a TrustError is raised (fail-fast: hooks must be safe).
    """
    if trusted:
        return base_emit

    def _safe_emit(event: HookEvent, payload: dict[str, Any]) -> None:
        assert_no_callables(payload)
        ensure_json_safe(payload)
        base_emit(event, payload)

    return _safe_emit
