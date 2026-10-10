"""The subprocess boundary must turn backend failures into typed errors."""

from __future__ import annotations

import subprocess

import pytest

from TheCausalityGame.agent.inferers import _r_bridge
from TheCausalityGame.core.lib.errors.inferer import (
    InfererBackendError,
    InfererDependencyError,
)

SINGLETON_PREDICTION = 1.25
COMMUNICATE_CALLS_AFTER_TIMEOUT = 2


def _payload() -> dict[str, object]:
    return {"method": "s_learner", "train": {"Y": [1.0]}, "query": {"X": [0.0]}}


class _FakeProcess:
    def __init__(
        self,
        command: list[str],
        *,
        timeout: bool = False,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.command = command
        self.pid = 91234
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.killed = False
        self.calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        self.calls += 1
        if self.timeout and self.calls == 1:
            raise subprocess.TimeoutExpired(self.command, timeout=3600)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True


def test_missing_rscript_is_a_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report missing system prerequisites as dependency failures."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: None)
    with pytest.raises(InfererDependencyError, match="Rscript"):
        _r_bridge.run_r_cate_backend(_payload())


def test_unknown_method_is_rejected_before_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never pass an untrusted backend method to Rscript."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        _r_bridge.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not invoke an unknown entrypoint"),
    )
    with pytest.raises(InfererBackendError, match=r"Unknown.*method"):
        _r_bridge.run_r_cate_backend({"method": "shell"})


def test_nonzero_exit_surfaces_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep actionable backend diagnostics when R exits unsuccessfully."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        _r_bridge.subprocess,
        "Popen",
        lambda command, **_kwargs: _FakeProcess(
            command, returncode=2, stderr="R package load failed"
        ),
    )
    with pytest.raises(InfererBackendError, match="R package load failed"):
        _r_bridge.run_r_cate_backend(_payload())


def test_timeout_is_converted_to_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalize subprocess deadlines to the inferer backend error type."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")

    process = _FakeProcess([], timeout=True)
    monkeypatch.setattr(_r_bridge.subprocess, "Popen", lambda *_args, **_kwargs: process)
    killed_groups: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _r_bridge.os,
        "killpg",
        lambda pid, signum: killed_groups.append((pid, signum)),
    )
    with pytest.raises(InfererBackendError, match="timed out"):
        _r_bridge.run_r_cate_backend(_payload())
    assert killed_groups == [(process.pid, _r_bridge.signal.SIGKILL)]
    assert process.calls == COMMUNICATE_CALLS_AFTER_TIMEOUT


def test_success_without_output_file_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a successful process that failed to produce its contract output."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        _r_bridge.subprocess,
        "Popen",
        lambda command, **_kwargs: _FakeProcess(command),
    )
    with pytest.raises(InfererBackendError, match="no output"):
        _r_bridge.run_r_cate_backend(_payload())


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        ("not json", "malformed JSON"),
        ('{"predictions": "2"}', "finite numbers"),
        ('{"predictions": [NaN]}', "finite numbers"),
        ('{"predictions": [Infinity]}', "finite numbers"),
        ('{"predictions": [1], "backend": []}', "metadata"),
    ],
)
def test_invalid_success_payloads_are_rejected(
    monkeypatch: pytest.MonkeyPatch, wire: str, message: str
) -> None:
    """Malformed JSON and invalid result schemas must not cross the bridge."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")

    def write_response(command: list[str]) -> None:
        output_path = command[-1]
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(wire)

    monkeypatch.setattr(
        _r_bridge.subprocess,
        "Popen",
        lambda command, **_kwargs: (
            write_response(command) or _FakeProcess(command)
        ),
    )
    with pytest.raises(InfererBackendError, match=message):
        _r_bridge.run_r_cate_backend(_payload())


def test_valid_response_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return well-formed finite predictions and backend provenance."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")

    def write_response(command: list[str], **_kwargs: object) -> _FakeProcess:
        with open(command[-1], "w", encoding="utf-8") as stream:
            stream.write('{"predictions": [1.25, -2], "backend": {"version": "1"}}')
        return _FakeProcess(command)

    monkeypatch.setattr(_r_bridge.subprocess, "Popen", write_response)
    result = _r_bridge.run_r_cate_backend(_payload())
    assert result["predictions"] == [1.25, -2]


def test_jsonlite_singleton_unboxing_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jsonlite auto_unbox emits a singleton prediction as a scalar."""
    monkeypatch.setattr(_r_bridge.shutil, "which", lambda _: "/usr/bin/Rscript")

    def write_response(command: list[str], **_kwargs: object) -> _FakeProcess:
        with open(command[-1], "w", encoding="utf-8") as stream:
            stream.write('{"predictions": 1.25, "backend": {"version": "1"}}')
        return _FakeProcess(command)

    monkeypatch.setattr(_r_bridge.subprocess, "Popen", write_response)
    assert (
        _r_bridge.run_r_cate_backend(_payload())["predictions"]
        == SINGLETON_PREDICTION
    )
