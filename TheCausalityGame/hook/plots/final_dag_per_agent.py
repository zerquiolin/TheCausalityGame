from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.scm.dag.core import CoreDAG


def to_imshow_array(obj) -> np.ndarray:
    # Case 1: a Matplotlib Figure -> render to RGBA pixels
    if isinstance(obj, plt.Figure):
        canvas = FigureCanvas(obj)
        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba())
        plt.close(obj)  # optional: avoid leaking figures
        return arr

    # Case 2: PIL image / numpy array / list-of-lists, etc.
    arr = np.asarray(obj)

    # If numpy ends up with dtype=object, force a numeric conversion
    if arr.dtype == object:
        arr = np.array(obj, dtype=np.uint8)

    return arr


class FinalDAGPerAgent(Hook):
    id: str = "Agents DAGs"
    step: HookEvent = HookEvent.BENCHMARK_END

    @override
    def run(self, hooks_dir: Path, context: dict[str, Transcript] | TranscriptEntry | None) -> None:
        if context is None or not isinstance(context, dict):
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle("Agent Final DAGs", fontsize=14)

        for i, (id, transcript) in enumerate(context.items()):
            last_entry = transcript.entries[-1]
            result = last_entry.result
            dag = CoreDAG(result)

            figure_obj = dag.generate_figure()
            img = to_imshow_array(figure_obj)

            ax = axes[i]
            ax.set_title(f"Agent: {id}")
            ax.imshow(img)
            ax.axis("off")

        plt.subplots_adjust(top=0.83)
        # Save the figure to a file
        plot_path = hooks_dir / "agents_dags.png"
        fig.savefig(plot_path)  # type: ignore
