from pathlib import Path
from typing import override

import matplotlib.pyplot as plt

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.hook import HookEvent


class AgentsComparisonPlotHook(Hook):
    id: str = "Agent Comparison"
    step: HookEvent = HookEvent.BENCHMARK_END

    def _plot_scores(self, name: str, b_score: float, d_score: float, ax):  # type: ignore # noqa: ANN001
        """
        Scatter plot of final behavior vs. deliverable scores.

        Args:
            name (str): Agent name.
            b_score (float): Final behavior score.
            d_score (float): Final deliverable score.
            ax (Axes): Matplotlib axes object.
        """
        # Plot point with a specific facecolor (e.g., for auto-colors by label)
        point = ax.scatter(b_score, d_score, s=100, label=name)  # type: ignore

        # Use the facecolor of the point for the label text
        ax.text(  # type: ignore
            b_score,
            d_score,
            f"({b_score:.2f}, {d_score:.2f})",
            color=point.get_facecolor()[0],  # get the RGBA tuple # type: ignore
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
        # ax.set_xscale("log")

    @override
    def run(self, hooks_dir: Path, context: dict[str, Transcript] | TranscriptEntry | None) -> None:
        if context is None or not isinstance(context, dict):
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle("Agent Comparison Scores", fontsize=14)

        for id, transcript in context.items():
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
        for ax, xlabel in zip(axes, ["Behavior Score", "Number of Rounds", "Number of Rounds"]):
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
        # Save the figure to a file
        plot_path = hooks_dir / "agent_comparison_plot.png"
        fig.savefig(plot_path)  # type: ignore

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
