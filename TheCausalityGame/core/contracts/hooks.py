"""Hook contracts and canonical event names."""

from __future__ import annotations

from typing import Any, Protocol

from core.contracts.enum.hooks import HookEvent


class Hook(Protocol):
    """A hook implementation that subscribes to canonical events.

    Implementers may expose methods named after the events, e.g.:
    `def on_round_start(self, **payload) -> None: ...`
    The runtime event bus discovers and calls matching methods.
    """

    id: str

    def handles(self) -> list[HookEvent]:
        """Return the list of events this hook wants to receive."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Receive configuration values after construction."""
        ...
