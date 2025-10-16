from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from TheCausalityGame.core.contracts.dto.transcript import Transcript


class ArtifactWriter:
    def __init__(self, run_dir: Path, is_dev: bool):
        self.run_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runs_dir = self.run_dir / "runs"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"
        self.is_dev = is_dev

        self.create_dirs()

    def create_dirs(self) -> str:
        os.makedirs(self.plots_dir, exist_ok=True)
        if self.is_dev:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.runs_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)

    def create_agent_dirs(self, agent_id: str) -> None:
        os.makedirs(self.run_dir / agent_id, exist_ok=True)
        os.makedirs(self.run_dir / agent_id / "plots", exist_ok=True)
        if self.is_dev:
            os.makedirs(self.run_dir / agent_id / "logs", exist_ok=True)

    def write_plot(
        self, run_dir: str, fig: Any, plot_id: str, round_id: int | None = None
    ) -> None:
        subpath = f"plots/{plot_id}"
        if round_id is not None:
            subpath += f"_round{round_id}"
        path = os.path.join(run_dir, f"{subpath}.png")
        fig.savefig(path)

    def write_transcript(self, agent_id: str, transcript: Transcript) -> None:
        if self.is_dev:
            return

        self.write_json(
            self.run_dir / "runs" / agent_id / "transcript.json",
            {"steps": 1},  # TODO: Fix this, hardcoded.
        )

    def write_provenance(self) -> None:
        if self.is_dev:
            return

        self.write_json(
            self.run_dir / "provenance.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "env": {"TZ": os.environ.get("TZ", "UTC")},
            },
        )
