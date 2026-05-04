"""Tests for transcript DTO behavior."""

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec


def test_transcript_invalidation_metadata_defaults_to_valid() -> None:
    """Transcripts are valid unless explicitly invalidated."""
    transcript = Transcript(
        agent_id="agent",
        mission_id="mission",
        manifest_id="manifest",
        entries=[],
        budget=BudgetSpec(rounds=1),
    )

    assert transcript.invalidated is False
    assert transcript.invalidation_reason is None


def test_transcript_can_be_marked_invalidated_with_error_reason() -> None:
    """Invalidation records both the boolean state and the failure reason."""
    transcript = Transcript(
        agent_id="agent",
        mission_id="mission",
        manifest_id="manifest",
        entries=[],
        budget=BudgetSpec(rounds=1),
    )

    transcript.invalidate("RuntimeError: boom")

    assert transcript.invalidated is True
    assert transcript.invalidation_reason == "RuntimeError: boom"
