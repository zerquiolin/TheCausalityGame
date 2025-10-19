"""The Causality Game - Hook Contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from TheCausalityGame.core.contracts.dto.transcript import TranscriptEntry
from TheCausalityGame.core.lib.enum.hook import HookEvent


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

    def run(self, hooks_dir: Path, context: TranscriptEntry | None) -> list[HookEvent]:
        """
        Return the list of events this hook should be notified about.

        Parameters
        ----------
        hooks_dir : Path
            Directory where hook-related artifacts can be stored.
        context : TranscriptEntry | None
            The current context of the game transcript.

        Returns
        -------
        list of HookEvent
            The events this hook subscribes to.
        """
        ...
