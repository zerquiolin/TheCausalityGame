import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from TheCausalityGame.core.contracts.dto.transcript import Transcript


class ArtifactWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

    def create_run_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.run_dir / ts
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(run_dir / "runs", exist_ok=True)
        os.makedirs(run_dir / "plots", exist_ok=True)
        os.makedirs(run_dir / "logs", exist_ok=True)
        os.makedirs(run_dir / "debug", exist_ok=True)
        return run_dir

    def write_plot(
        self, run_dir: str, fig: Any, plot_id: str, round_id: int | None = None
    ) -> None:
        subpath = f"plots/{plot_id}"
        if round_id is not None:
            subpath += f"_round{round_id}"
        path = os.path.join(run_dir, f"{subpath}.png")
        fig.savefig(path)

    def write_transcript(self, agent_id: str, transcript: Transcript) -> None:
        self.write_json(
            self.run_dir / "runs" / agent_id / "transcript.json",
            {"steps": 1},  # TODO: Fix this, hardcoded.
        )

    def write_debug(self, run_dir: str, name: str, content: Any) -> None:
        path = os.path.join(run_dir, "debug", f"{name}.json")
        self.write_json(path, content)

    def write_log(self, run_dir: str, name: str, text: str) -> None:
        path = os.path.join(run_dir, "logs", f"{name}.txt")
        self.write_text(path, text)
