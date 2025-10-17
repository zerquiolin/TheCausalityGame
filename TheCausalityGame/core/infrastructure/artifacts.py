"""The Causality Game - Artifact Writer Infrastructure."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.figure

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.infrastructure.serialization import dump, is_serializable


class ArtifactWriter:
    """
    Handles the structured output of experiment artifacts during and after simulation runs.

    Attributes
    ----------
    run_dir : Path
        Root directory for writing artifacts.
    is_dev : bool
        Enables extended logging and artifact output in development mode.
    """

    def __init__(self, run_dir: Path, is_dev: bool) -> None:
        """
        Initialize the artifact writer and create required directories.

        Parameters
        ----------
        run_dir : Path
            Base directory where all artifacts will be stored.
        is_dev : bool
            Whether to enable development-only outputs (e.g., logs, transcripts).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir / timestamp
        self.runs_dir = self.run_dir / "runs"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"
        self.is_dev = is_dev

        self.create_dirs()

    def create_dirs(self) -> None:
        """Create base artifact directories."""
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        if self.is_dev:
            os.makedirs(self.logs_dir, exist_ok=True)

    def create_agent_dirs(self, agent_id: str) -> None:
        """
        Create directories for a specific agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier of the agent.
        """
        agent_dir = self.runs_dir / agent_id
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(agent_dir / "plots", exist_ok=True)

    def write_plot(
        self,
        run_dir: str,
        fig: matplotlib.figure.Figure,
        plot_id: str,
        round_id: int | None = None,
    ) -> None:
        """
        Save a matplotlib figure as PNG to the specified agent run directory.

        Parameters
        ----------
        run_dir : str
            Path to the agent's run directory.
        fig : matplotlib.figure.Figure
            Matplotlib-like figure object with `.savefig()`.
        plot_id : str
            ID of the plot (used as filename prefix).
        round_id : int, optional
            Optional round number to append to filename.
        """
        filename = f"plots/{plot_id}"
        if round_id is not None:
            filename += f"_round{round_id}"
        path = os.path.join(run_dir, f"{filename}.png")
        try:
            fig.savefig(path)  # type: ignore
        except OSError as e:
            print(f"[WARNING] Failed to save figure at {path}: {e}")

    def write_transcript(self, agent_id: str, transcript: Transcript) -> None:
        """
        Write a sanitized version of the transcript (removes unserializable fields).

        Parameters
        ----------
        agent_id : str
            Agent identifier.
        transcript : Transcript
            Full run transcript for the agent.
        """
        if not self.is_dev:
            return

        transcript_path = self.runs_dir / agent_id / "transcript.json"
        sanitized_entries: list[TranscriptEntry] = []

        for entry in transcript.entries:
            filtered_data = {
                key: value
                for key, value in entry.model_dump().items()
                if is_serializable(value)
            }
            sanitized_entry = TranscriptEntry.model_validate(filtered_data)
            sanitized_entries.append(sanitized_entry)

        clean_transcript = Transcript(
            agent_id=transcript.agent_id,
            mission_id=transcript.mission_id,
            manifest_id=transcript.manifest_id,
            entries=sanitized_entries,
        )

        dump(path=transcript_path, obj=clean_transcript)

    def write_provenance(self) -> None:
        """Save a JSON file with environment and system-level metadata."""
        if not self.is_dev:
            return

        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "env": {"TZ": os.environ.get("TZ", "UTC")},
        }

        dump(path=self.run_dir / "provenance.json", obj=metadata)
