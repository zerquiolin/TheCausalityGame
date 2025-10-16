"""Hook contracts and canonical event names."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.lib.enum.plots import PlotKind


class Plot(Serializable):
    """A plot hook.

    This is a hook that can be used to generate plots at different stages of the game.
    The `generate` method must be implemented by subclasses to create the desired plot.
    The `id` attribute is a unique identifier for the plot.
    """

    id: str
    kind: PlotKind

    @abstractmethod
    def generate(self, arg: Any) -> Any:
        """Return the figure object created by the plot."""
        raise NotImplementedError()
