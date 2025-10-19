"""The Causality Game - Plot Manager."""

import matplotlib.figure

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.plot import Plot
from TheCausalityGame.core.contracts.specs.plot import PlotSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.plots import PlotKind


class PlotManager:
    """
    Manages and dispatches plot generation for various lifecycle phases.

    Attributes
    ----------
    end_plots : list[Plot]
        Plots triggered at the end of a single agent run.
    round_plots : list[Plot]
        Plots triggered after each round.
    benchmark_plots : list[Plot]
        Plots triggered after the benchmark (all agents).
    """

    def __init__(self, plots: list[PlotSpec]) -> None:
        """
        Initialize the plot manager with categorized plots.

        Parameters
        ----------
        plots : list[PlotSpec]
            List of plot specifications from the run plan.
        """
        self.end_plots: list[Plot] = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.GAME_END
        ]
        self.round_plots: list[Plot] = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.ROUND_END
        ]
        self.benchmark_plots: list[Plot] = [
            build_from_spec(p) for p in plots if p.kind == PlotKind.BENCHMARK_END
        ]

    def trigger_round(
        self, transcript_entry: TranscriptEntry
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        """
        Generate all plots configured for round-end.

        Parameters
        ----------
        transcript_entry : TranscriptEntry
            The transcript entry from the current round.

        Returns
        -------
        list[tuple[matplotlib.figure.Figure]]
            A tuple of plot id and the generated plot figure.
        """
        return [
            (plot.id, fig)
            for plot in self.round_plots
            if (fig := plot.generate(transcript_entry))
        ]

    def trigger_rounds(
        self, transcript: Transcript
    ) -> list[list[tuple[str, matplotlib.figure.Figure]]]:
        """
        Generate all plots configured for round-end over all rounds.

        Parameters
        ----------
        transcript : Transcript
            The full transcript of the agent run.

        Returns
        -------
        list[list[tuple[str, matplotlib.figure.Figure]]]
            A list of lists containing tuples of plot id and the generated plot figure for each round.
        """  # noqa: E501
        return [
            result
            for entry in transcript.entries
            if (result := self.trigger_round(entry))
        ]

    def trigger_end(
        self, transcript: Transcript
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        """
        Generate all plots configured for game-end.

        Parameters
        ----------
        transcript : Transcript
            The full transcript of the agent run.

        Returns
        -------
        list[tuple[str, matplotlib.figure.Figure]]
            A tuple of plot id and the generated plot figure.
        """
        return [
            (plot.id, fig)
            for plot in self.end_plots
            if (fig := plot.generate(transcript))
        ]

    def trigger_benchmark_end(
        self, transcript: dict[str, Transcript]
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        """
        Generate all plots configured for benchmark-end.

        Parameters
        ----------
        transcript : dict[str, Transcript]
            Mapping from agent ID to full run transcripts.

        Returns
        -------
        list[tuple[str, matplotlib.figure.Figure]]
            A tuple of plot id and the generated plot figure.
        """
        return [
            (plot.id, fig)
            for plot in self.benchmark_plots
            if (fig := plot.generate(transcript))
        ]
