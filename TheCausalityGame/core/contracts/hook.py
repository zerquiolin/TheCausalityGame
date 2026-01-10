"""The Causality Game - Hook Contract."""

from __future__ import annotations

from pathlib import Path
from typing import override

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.hook import HookEvent


class Hook(Serializable):
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
    step: HookEvent
    _spec: str = "TheCausalityGame.core.contracts.specs.hook:HookSpec"

    def run(self, hooks_dir: Path, context: dict[str, Transcript] | TranscriptEntry | None) -> None:
        """
        Return the list of events this hook should be notified about.

        Parameters
        ----------
        hooks_dir : Path
            Directory where hook-related artifacts can be stored.
        context : TranscriptEntry | None
            The current context of the game transcript.
        """
        raise NotImplementedError

    @override
    def to_spec(self) -> HookSpec:
        return HookSpec(
            class_=get_class_path(self.__class__),
            id=self.id,
            step=self.step,
        )

    @classmethod
    @override
    def from_spec(cls, spec: HookSpec) -> Hook:
        return cls()
