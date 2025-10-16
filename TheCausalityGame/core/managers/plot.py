from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.plots import Plot
from TheCausalityGame.core.lib.enum.plots import PlotKind
from TheCausalityGame.core.infrastructure.registry import build_from_spec


class PlotManager:
    def __init__(self, plots: list[Plot]) -> None:
        self.end_plots = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.GAME_END
        ]
        self.round_plots = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.ROUND_END
        ]
        self.benchmark_plots = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.BENCHMARK_END
        ]

    def trigger_round(self, transcript_entry: TranscriptEntry) -> list:
        return [plot.generate(transcript_entry) for plot in self.round_plots]

    def trigger_end(self, transcript: Transcript) -> list:
        return [plot.generate(transcript) for plot in self.end_plots]

    def trigger_benchmark_end(self, transcript: dict[str, Transcript]) -> list:
        return [plot.generate(transcript) for plot in self.benchmark_plots]
