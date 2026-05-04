"""The Causality Game - Artifact Writer Infrastructure."""

import os
import platform
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

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

    def __init__(self, run_dir: Path, is_dev: bool, logger: Logger | None = None) -> None:
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

    def _json_safe(self, value: Any) -> Any:  # noqa: ANN401
        """Convert common runtime values into strict JSON-safe values."""
        if isinstance(value, float):
            return value if isfinite(value) else None
        if isinstance(value, int | str | bool) or value is None:
            return value
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list | tuple | set):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "expressions") and callable(value.expressions):
            return self._json_safe(value.expressions())
        if is_serializable(value):
            return value
        return str(value)

    def _clean_transcript_entry_for_serialization(self, entry: TranscriptEntry) -> dict[str, Any]:
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
        clean_entry["decision"] = (
            self._json_safe(entry.decision.to_dict()) if entry.decision else None
        )
        clean_entry["result"] = self._json_safe(entry.result)
        clean_entry["samples_collection"] = (
            [
                {
                    "kind": self._json_safe(sample.kind),
                    "n": self._json_safe(sample.n),
                    "data": self._json_safe(sample.data.to_dict(orient="records")),
                    "interventions": self._json_safe(sample.interventions),
                }
                for sample in entry.samples_collection
                if sample
            ]
            if entry.samples_collection
            else None
        )
        clean_entry["budget_snapshot"] = (
            self._json_safe(entry.budget_snapshot.model_dump())
            if entry.budget_snapshot
            else None
        )
        clean_entry["feedback"] = (
            self._json_safe(entry.feedback.model_dump()) if entry.feedback else None
        )
        # Custom attributes
        for key, value in entry.custom_attributes.items():
            clean_entry[key] = self._json_safe(value)

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
            "invalidated": transcript.invalidated,
            "invalidation_reason": transcript.invalidation_reason,
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
