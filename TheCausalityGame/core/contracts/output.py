"""Output (figure/report/export) contract."""

from __future__ import annotations

from typing import Any, Protocol

from .serializable import JSONSerializable


class Output(JSONSerializable, Protocol):
    """Contract for pluggable outputs.

    Implementations typically subscribe via hooks (e.g., after_metric_eval)
    and render artifacts to the run directory.
    """

    id: str

    def render(
        self,
        run_dir: str,
        *,
        scores_path: str,
        transcripts_path: str,
        datasets_dir: str,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        """Render output artifacts.

        Args:
          run_dir: Path to the run directory.
          scores_path: Path to final scores JSON.
          transcripts_path: Path to transcripts JSONL.
          datasets_dir: Path to datasets directory.
          config: Optional output-specific configuration.

        Returns:
          A list of emitted file paths.
        """
        ...
