import matplotlib.pyplot as plt

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.plots import Plot
from TheCausalityGame.core.contracts.specs.plot import PlotSpec
from TheCausalityGame.core.lib.enum.plots import PlotKind


class CorePlot(Plot):

    def __init__(self, id: str, kind: PlotKind) -> None:
        self.id = id
        self.kind = kind

    def _plot_scores(self, name: str, b_score: float, d_score: float, ax):
        """
        Scatter plot of final behavior vs. deliverable scores.

        Args:
            name (str): Agent name.
            b_score (float): Final behavior score.
            d_score (float): Final deliverable score.
            ax (Axes): Matplotlib axes object.
        """
        # Plot point with a specific facecolor (e.g., for auto-colors by label)
        point = ax.scatter(b_score, d_score, s=100, label=name)

        # Use the facecolor of the point for the label text
        ax.text(
            b_score,
            d_score,
            f"({b_score:.2f}, {d_score:.2f})",
            color=point.get_facecolor()[0],  # get the RGBA tuple
            fontsize=10,
            ha="left",
            va="center",
        )

    def _plot_time_series(self, name: str, scores: list[float], ylabel: str, ax):
        """
        Plot a time series of scores across rounds.

        Args:
            name (str): Agent name.
            scores (List[float]): Score trajectory.
            ylabel (str): Y-axis label.
            ax (Axes): Matplotlib axes object.
        """
        ax.plot(range(len(scores)), scores, label=name, alpha=0.7)
        ax.scatter(len(scores) - 1, scores[-1], s=100, edgecolor="black")
        # Plot final score as number with the same color as the line
        ax.text(
            len(scores) - 1,
            scores[-1],
            f"{scores[-1]:.2f}",
            color=ax.lines[-1].get_color(),
            fontsize=10,
            ha="left",
            va="center",
        )
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xscale("log")

    def generate(self, transcripts: dict[str, Transcript]):
        """
        Plot all agents' behavior vs deliverable, and their score trajectories.

        Raises:
            RuntimeError: If no results are available.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle("Agent Comparison Scores", fontsize=14)

        for id, transcript in transcripts.items():
            self._plot_scores(
                id,
                transcript.entries[-1].feedback.behavior,
                transcript.entries[-1].feedback.result,
                axes[0],
            )
            self._plot_time_series(
                id,
                [entry.feedback.behavior for entry in transcript.entries],
                "Behavior Score",
                axes[1],
            )

            self._plot_time_series(
                id,
                [entry.feedback.result for entry in transcript.entries],
                "Result Score",
                axes[2],
            )

        # Set common axis labels and formatting
        for ax, xlabel in zip(
            axes, ["Behavior Score", "Number of Rounds", "Number of Rounds"]
        ):
            ax.set_xlabel(xlabel)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Deduplicated legend
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(
            handles=unique.values(),
            labels=unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=11,
            fontsize="small",
        )

        plt.subplots_adjust(top=0.83)
        # plt.show()
        return fig

    def to_spec(self) -> PlotSpec:
        return PlotSpec(
            id=self.id,
            kind=PlotKind.BENCHMARK_END,
            params={},
        )

    @classmethod
    def from_spec(cls, spec: PlotSpec) -> Plot:
        return cls(id=spec.id, kind=spec.kind)
