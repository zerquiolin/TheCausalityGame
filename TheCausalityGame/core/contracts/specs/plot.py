"""The Causality Game - Plot Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.lib.enum.plots import PlotKind


class PlotSpec(CommonSpec):
    """
    Specification for a plot hook.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path (aliased from 'class' in JSON).
    spec_ : str | None
        Optional override for the spec class path.
    params : dict
        Optional noise distribution configuration payload.
    id : str
        Unique identifier for the plot. Defaults to 'plot'.
    kind : PlotKind
        Stage of the game at which the plot should be generated.
        For example, 'game_end' or 'round_start'.
    """

    id: str = Field(default="plot", description="Unique identifier for the plot.")
    kind: PlotKind = Field(
        default=PlotKind.GAME_END,
        description="Stage of the game when the plot is triggered.",
    )
