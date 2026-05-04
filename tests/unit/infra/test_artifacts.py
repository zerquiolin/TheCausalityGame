"""Tests for artifact transcript serialization."""

from pathlib import Path

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.dto.environment import Samples, SamplesCollection
from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.dto.transcript import TranscriptEntry
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.artifacts import ArtifactWriter
from TheCausalityGame.core.infrastructure.decisions import Decision
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
    transcript.invalidate("ValueError: invalid action")

    writer.write_transcript("agent", transcript)

    payload = loads((writer.agents_dir / "agent" / "transcript.json").read_text())
    assert payload["invalidated"] is True
    assert payload["invalidation_reason"] == "ValueError: invalid action"


def test_write_transcript_converts_numpy_values(tmp_path: Path) -> None:
    """Serialized transcript entries convert NumPy scalar values to JSON primitives."""
    writer = ArtifactWriter(run_dir=tmp_path, is_dev=True)
    decision = Decision.experiment().add_experiment(
        treatment={"x": np.float64(1.5)},
        n=np.int64(2),
    )
    entry = TranscriptEntry(
        round=1,
        decision=decision,
        samples_collection=SamplesCollection([
            Samples(
                kind="interventional",
                n=np.int64(1),
                data=pd.DataFrame({"x": [np.float64(1.5)]}),
                interventions={"x": np.float64(1.5)},
            )
        ]),
    )
    transcript = Transcript(
        agent_id="agent",
        mission_id="mission",
        manifest_id="manifest",
        entries=[entry],
        budget=BudgetSpec(rounds=1),
    )

    writer.write_transcript("agent", transcript)

    payload = loads((writer.agents_dir / "agent" / "transcript.json").read_text())
    assert payload["entries"][0]["decision"]["experiments"][0]["treatment"]["x"] == 1.5
    assert payload["entries"][0]["samples_collection"][0]["interventions"]["x"] == 1.5


def test_write_transcript_converts_non_finite_floats(tmp_path: Path) -> None:
    """Serialized transcript entries convert inf/nan to strict JSON-safe nulls."""
    writer = ArtifactWriter(run_dir=tmp_path, is_dev=True)
    decision = Decision.experiment().add_experiment(treatment={"x": float("inf")}, n=1)
    entry = TranscriptEntry(
        round=1,
        decision=decision,
        result={"score": float("-inf"), "other": float("nan")},
    )
    transcript = Transcript(
        agent_id="agent",
        mission_id="mission",
        manifest_id="manifest",
        entries=[entry],
        budget=BudgetSpec(rounds=1),
    )

    writer.write_transcript("agent", transcript)

    payload = loads((writer.agents_dir / "agent" / "transcript.json").read_text())
    assert payload["entries"][0]["decision"]["experiments"][0]["treatment"]["x"] is None
    assert payload["entries"][0]["result"]["score"] is None
    assert payload["entries"][0]["result"]["other"] is None
