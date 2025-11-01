"""The Causality Game - Plot Contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import matplotlib
import matplotlib.figure

from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.lib.enum.plots import PlotKind


class Plot(Serializable):
    """
    Base class for plot hooks in The Causality Game.

    Plot hooks are used to visualize different stages or outputs of the game.
    Each subclass must implement `generate`, which produces the figure.

    Attributes
    ----------
    id : str
        Unique identifier for the plot instance.
    kind : PlotKind
        Canonical type identifier (e.g., "round_end", "benchmark_end").
    """

    id: str
    kind: PlotKind
    _spec: str = "TheCausalityGame.core.contracts.specs.plot:PlotSpec"

    @abstractmethod
    def generate(self, arg: Any) -> matplotlib.figure.Figure:  # noqa: ANN401
        """
        Generate and return a figure based on game data.

        Parameters
        ----------
        arg : Any
            Input data for generating the plot (varies by plot kind).

        Returns
        -------
        Any
            The generated figure (typically a matplotlib or plotly object).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError()
