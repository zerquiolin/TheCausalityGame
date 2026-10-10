"""Contract-first regressions: strict wire data and independently owned BN state."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.core.contracts.dto.transcript import TranscriptEntry
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infrastructure import serialization
from TheCausalityGame.core.infrastructure.artifacts import ArtifactWriter
from TheCausalityGame.core.infrastructure.serialization import is_serializable, loads
from TheCausalityGame.core.lib.errors.serialization import ObjectNotDeserializableError
from TheCausalityGame.scm.nodes.bn import BayesianNetworkSCMNode


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_reader_rejects_non_json_numbers(token):
    with pytest.raises(ObjectNotDeserializableError):
        loads('{"score":' + token + "}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_serializable_predicate_rejects_nonfinite_values(value):
    assert not is_serializable({"score": value}), "Strict JSON cannot encode this score"


def test_serializable_predicate_handles_cycles():
    value = []
    value.append(value)
    assert not is_serializable(
        value
    ), "Cycles are invalid JSON, not a predicate exception"


def test_serializable_predicate_recognizes_supported_path_encoder():
    assert is_serializable(Path("result.json"))


def test_atomic_dump_replaces_complete_json(tmp_path):
    """Successful atomic replacement leaves one complete destination file."""
    target = tmp_path / "nested" / "artifact.json"
    target.parent.mkdir()
    target.write_text('{"old": true}', encoding="utf-8")

    serialization.dump({"new": [1, 2]}, target)

    assert loads(target.read_text(encoding="utf-8")) == {"new": [1, 2]}
    assert list(target.parent.iterdir()) == [target]


def test_atomic_dump_preserves_previous_artifact_when_replace_fails(
    tmp_path, monkeypatch
):
    """A failed replacement keeps the previous artifact and removes its temp."""
    target = tmp_path / "artifact.json"
    target.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated interrupted commit")

    monkeypatch.setattr(serialization.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        serialization.dump({"new": True}, target)

    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.iterdir()) == [target]


def test_bn_owns_constructor_configuration():
    domain, parents = ["low", "high"], ["P"]
    node = BayesianNetworkSCMNode("Y", parents, domain, {"0": [1, 0]})
    domain.reverse()
    parents.clear()
    assert node.domain == ["low", "high"], "Caller mutation changed category meaning"
    assert node.parents == ["P"], "Caller mutation removed a causal parent"
    assert node.generate_values(pd.DataFrame({"P": [0]})).tolist() == ["low"]


def test_bn_distribution_lookup_does_not_expose_mutable_cpt():
    node = BayesianNetworkSCMNode("Y", [], ["low", "high"], [1, 0])
    rows = node.get_distributions(pd.DataFrame(index=range(1)))
    rows[0][:] = [0, 1]
    assert node.generate_values(pd.DataFrame(index=range(4))).tolist() == ["low"] * 4


def test_bn_lookup_preserves_parent_scalar_types():
    node = BayesianNetworkSCMNode(
        "Y", ["integer", "real"], ["zero", "one"], {"0,1.5": [0, 1]}
    )
    parents = pd.DataFrame({"integer": [0], "real": [1.5]})
    assert node.generate_values(parents).tolist() == ["one"]


def test_bn_rejects_numeric_category_labels():
    with pytest.raises(ValueError, match="string labels"):
        BayesianNetworkSCMNode("Y", [], [0, 1], [0.5, 0.5])


def test_categorical_node_spec_rejects_numeric_labels_before_build():
    with pytest.raises(ValueError, match="string labels"):
        SCMNodeSpec(
            class_="TheCausalityGame.scm.nodes.bn:BayesianNetworkSCMNode",
            name="Y",
            domain=[0, 1],
        )


def test_bn_reconstruction_continues_rng_without_sharing_it():
    node = BayesianNetworkSCMNode(
        "Y", [], ["0", "1"], [0.5, 0.5], random_state=np.random.RandomState(17)
    )
    frame = pd.DataFrame(index=range(31))
    node.generate_values(frame)
    restored = BayesianNetworkSCMNode.from_spec(node.to_spec())
    np.testing.assert_array_equal(
        node.generate_values(frame), restored.generate_values(frame)
    )
    before = node.to_spec().random_state
    restored.generate_values(frame)
    assert node.to_spec().random_state == before


def test_artifact_extensions_cannot_replace_canonical_round(tmp_path):
    writer = ArtifactWriter(tmp_path, is_dev=False)
    entry = TranscriptEntry(round=1, custom_attributes={"round": 999})
    with pytest.raises(ValueError, match="reserved"):
        writer._clean_transcript_entry_for_serialization(entry)


def test_bn_serialization_preserves_string_category_labels():
    node = BayesianNetworkSCMNode("Y", [], ["0", "other"], [1, 0])
    restored = BayesianNetworkSCMNode.from_spec(node.to_spec())
    assert restored.domain == ["0", "other"]
    assert restored.generate_values(pd.DataFrame(index=range(3))).tolist() == [
        "0",
        "0",
        "0",
    ]
