from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.lib.enum.plots import PlotKind


class PlotSpec(CommonSpec):

    id: str = "plot"
    kind: PlotKind = "game_end"
