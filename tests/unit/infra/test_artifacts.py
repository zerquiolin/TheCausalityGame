"""Tests for artifact transcript serialization."""

from pathlib import Path

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.artifacts import ArtifactWriter
from TheCausalityGame.core.infrastructure.serialization import loads


def test_write_transcript_includes_invalidation_metadata(tmp_path: Path) -> None:
    """Serialized transcripts include validity metadata for failed agent runs."""
    writer = ArtifactWriter(run_dir=tmp_path, is_dev=True)
    transcript = Transcript(
        agent_id="agent",
        mission_id="mission",
        manifest_id="manifest",
        entries=[],
        budget=BudgetSpec(rounds=1),
    )
    transcript.invalidate(ValueError("invalid action"))

    writer.write_transcript("agent", transcript)

    payload = loads((writer.agents_dir / "agent" / "transcript.json").read_text())
    assert payload["invalidated"] is True
    assert payload["invalidation_reason"] == "ValueError: invalid action"
