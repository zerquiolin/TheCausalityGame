from __future__ import annotations

import json
from pathlib import Path

from TheCausalityGame.core.infra import (
    append_jsonl,
    ensure_run_dir,
    snapshot_provenance,
    write_json,
)


def test_ensure_run_dir_creates_structure(tmp_path: Path) -> None:
    run_dir = ensure_run_dir(tmp_path, "run_001", "agent_alpha")
    assert run_dir.exists() and run_dir.is_dir()
    # subfolders are created eagerly
    assert (run_dir / "datasets").exists()
    assert (run_dir / "plots").exists()


def test_write_json_creates_file_atomically(tmp_path: Path) -> None:
    run_dir = ensure_run_dir(tmp_path, "run_002", "agent_beta")
    target = run_dir / "manifest.json"

    payload = {"id": "run_002", "agent": "agent_beta", "params": {"k": 1}}
    write_json(target, payload)

    assert target.exists()
    read_back = json.loads(target.read_text(encoding="utf-8"))
    assert read_back == payload

    # Overwrite safely
    payload2 = {"id": "run_002", "agent": "agent_beta", "params": {"k": 2}}
    write_json(target, payload2)
    read_back2 = json.loads(target.read_text(encoding="utf-8"))
    assert read_back2 == payload2


def test_append_jsonl_creates_and_appends(tmp_path: Path) -> None:
    run_dir = ensure_run_dir(tmp_path, "run_003", "agent_gamma")
    jsonl = run_dir / "transcripts.jsonl"

    rows1 = [{"i": 0}, {"i": 1}]
    append_jsonl(jsonl, rows1)
    assert jsonl.exists()

    rows2 = [{"i": 2}]
    append_jsonl(jsonl, rows2)

    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"i": 0}
    assert json.loads(lines[1]) == {"i": 1}
    assert json.loads(lines[2]) == {"i": 2}


def test_snapshot_provenance_writes_expected_keys(tmp_path: Path) -> None:
    run_dir = ensure_run_dir(tmp_path, "run_004", "agent_delta")
    snapshot_provenance(run_dir)

    prov_path = run_dir / "provenance.json"
    assert prov_path.exists()

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    # Minimal set of expected keys
    assert "utc_start" in prov
    assert "python" in prov
    assert "platform" in prov and isinstance(prov["platform"], dict)
    assert "pid" in prov and isinstance(prov["pid"], int)
