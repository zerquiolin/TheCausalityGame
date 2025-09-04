from __future__ import annotations

from pathlib import Path
from typing import Any

from TheCausalityGame.core.contracts.dto import ProblemInstance
from TheCausalityGame.core.engine.game import Game
from TheCausalityGame.core.engine.game_instance import GameInstance
from TheCausalityGame.core.infra.logging_ import configure_logging, get_logger
from TheCausalityGame.core.infra.serialization import loads
from TheCausalityGame.core.infra.settings import RuntimeSettings


class GameRunner:
    """High-level façade to run a manifest from code or CLI."""

    def __init__(
        self, *, run_dir: Path = "runs", settings: RuntimeSettings | None = None
    ) -> None:
        self.run_dir = Path(run_dir)
        self._settings = settings or RuntimeSettings.from_sources()
        configure_logging(debug=self._settings.debug)
        self._logger = get_logger("tcg.runner")

    def run_manifest_file(
        self, *, manifest_path: str | Path
    ) -> dict[str, dict[str, Any]]:
        """Load, validate, and run a ProblemInstance manifest from disk."""
        manifest_json = Path(manifest_path).read_text(encoding="utf-8")
        manifest = ProblemInstance.model_validate(loads(manifest_json))
        return self.run_manifest(manifest=manifest, runs_dir=self.run_dir)

    def run_manifest(self, *, manifest: ProblemInstance) -> dict[str, dict[str, Any]]:
        """Run an in-memory ProblemInstance."""
        instance = GameInstance.build(manifest=manifest, settings=self._settings)
        game = Game(instance=instance, runs_dir=self.run_dir)
        self._logger.info(
            "Starting run",
            extra={"run_id": instance.run_id, "mode": self._settings.mode},
        )
        results = game.run()
        self._logger.info("Run finished", extra={"run_id": instance.run_id})
        return results
