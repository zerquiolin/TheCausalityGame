from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.specs.plot import PlotSpec


class PlotManager:
    def __init__(self, plots: list[PlotSpec]) -> None:
        self.end_plots = [p for p in plots if p.trigger == "game_end"]
        self.round_plots = [p for p in plots if p.trigger == "round_end"]
        self.benchmark_plots = [p for p in plots if p.trigger == "benchmark_end"]

    def trigger_round(self, transcript_entry: TranscriptEntry) -> None:
        for plot in self.round_plots:
            plot.generate(transcript_entry)

    def trigger_end(self, transcript: Transcript) -> None:
        for plot in self.end_plots:
            plot.generate(transcript)

    def trigger_benchmark_end(self, transcript: dict[str, Transcript]) -> None:
        for plot in self.benchmark_plots:
            plot.generate(transcript)
