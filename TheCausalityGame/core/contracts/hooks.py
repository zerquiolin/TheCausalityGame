"""The Causality Game - Hook Contract."""

from __future__ import annotations

from typing import Any, Protocol

from TheCausalityGame.core.lib.enum.hooks import HookEvent


class Hook(Protocol):
    """
    Base protocol for hook implementations that subscribe to game events.

    Hook implementations must define which events they respond to, and may
    optionally implement methods named after the events (e.g., `on_round_start`).
    The runtime will automatically call matching methods when those events occur.

    Attributes
    ----------
    id : str
        Unique identifier for the hook instance.
    """

    id: str

    def handles(self) -> list[HookEvent]:
        """
        Return the list of events this hook should be notified about.

        Returns
        -------
        list of HookEvent
            The events this hook subscribes to.
        """
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """
        Receive configuration data after hook construction.

        Parameters
        ----------
        config : dict[str, Any]
            Key-value configuration dictionary.
        """
        ...
