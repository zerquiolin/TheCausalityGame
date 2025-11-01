"""The Causality Game - Artifact Writer Infrastructure."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.infrastructure.logger import Logger
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

    def __init__(
        self, run_dir: Path, is_dev: bool, logger: Logger | None = None
    ) -> None:
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
        self.agents_dir = self.run_dir / "agents"
        self.logs_dir = self.run_dir / "logs"
        self.is_dev = is_dev
        self.logger = logger

        # Post Init
        self.create_dirs()
        self.write_provenance()

    def set_logger(self, logger: Logger) -> None:
        """
        Set the logger for the artifact writer.

        Parameters
        ----------
        logger : Logger
            Logger instance to be used by the artifact writer.
        """
        self.logger = logger

    def create_dirs(self) -> None:
        """Create base artifact directories."""
        os.makedirs(self.run_dir, exist_ok=True)  # Core run directory
        if self.is_dev:
            os.makedirs(self.agents_dir, exist_ok=True)  # Directory for agents

            os.makedirs(self.logs_dir, exist_ok=True)  # Directory for logs

    def _clean_transcript_entry_for_serialization(
        self, entry: TranscriptEntry
    ) -> dict[str, Any]:
        """
        Clean a transcript entry.

        Extracts serializable attributes and removes unserializable objects.

        Parameters
        ----------
        entry : TranscriptEntry
            The transcript entry to clean.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the cleaned transcript entry.
        """
        clean_entry: dict[str, Any] = {}
        # Core attributes
        clean_entry["round"] = entry.round
        clean_entry["decision"] = entry.decision.to_dict() if entry.decision else None
        clean_entry["result"] = (
            entry.result if is_serializable(entry.result) else "Not Serializable"
        )
        clean_entry["samples_collection"] = (
            [
                {
                    "kind": sample.kind,
                    "n": sample.n,
                    "data": sample.data.to_dict(orient="records"),  # type: ignore
                    "interventions": sample.interventions,
                }
                for sample in entry.samples_collection
                if sample
            ]
            if entry.samples_collection
            else None
        )
        clean_entry["budget_snapshot"] = (
            entry.budget_snapshot.model_dump() if entry.budget_snapshot else None
        )
        clean_entry["feedback"] = (
            entry.feedback.model_dump() if entry.feedback else None
        )
        # Custom attributes
        for key, value in entry.custom_attributes.items():
            if is_serializable(value):
                clean_entry[key] = value
            else:
                clean_entry[key] = "Not Serializable"

        return clean_entry

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

        clean_transcript = {
            "agent_id": transcript.agent_id,
            "mission_id": transcript.mission_id,
            "manifest_id": transcript.manifest_id,
            "entries": [
                self._clean_transcript_entry_for_serialization(entry)
                for entry in transcript.entries
            ],
        }

        dump(path=self.agents_dir / agent_id / "transcript.json", obj=clean_transcript)

    def write_transcripts(self, transcripts: dict[str, Transcript]) -> None:
        """
        Write a sanitized version of the transcripts (removes unserializable fields).

        Parameters
        ----------
        transcripts : dict[str, Transcript]
            Full run transcripts for all agents.
        """
        if not self.is_dev:
            return

        for agent_id, transcript in transcripts.items():
            self.write_transcript(agent_id, transcript)

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
