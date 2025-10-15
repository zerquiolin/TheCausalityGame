from typing import Literal

from TheCausalityGame.core.specs.common import CommonSpec


class PlotSpec(CommonSpec):

    trigger: Literal["game_end", "round_end", "benchmark_end"] = "game_end"
