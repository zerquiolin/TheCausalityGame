"""Checked subprocess bridge to the pinned authoritative R backends."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from TheCausalityGame.core.lib.errors.inferer import (
    InfererBackendError,
    InfererDependencyError,
)


def _run_with_timeout(
    command: list[str], *, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a backend in its own process group so timeout cleanup reaps children."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise InfererBackendError(
            f"Authoritative R backend timed out after {exc.timeout} seconds."
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_r_cate_backend(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the packaged R CATE entrypoint using JSON files as the boundary."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise InfererDependencyError(
            "This authoritative CATE inferer requires Rscript and the locked R packages. "
            "Run scripts/comparisons/bootstrap_r.R first."
        )

    entrypoints = {
        "s_learner": "s_learner.R",
        "transformed_outcome": "transformed_outcome.R",
        "honest_causal_tree": "honest_causal_tree.R",
    }
    method = str(payload.get("method", ""))
    if method not in entrypoints:
        raise InfererBackendError(f"Unknown authoritative R backend method: {method}")
    backend = Path(__file__).with_name("r_backends") / entrypoints[method]
    with tempfile.TemporaryDirectory(prefix="tcg-r-cate-") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        input_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        env = os.environ.copy()
        r_library = env.get("TCG_R_LIBS")
        if r_library:
            env["R_LIBS_USER"] = r_library
        completed = _run_with_timeout(
            [rscript, str(backend), str(input_path), str(output_path)],
            env=env,
            timeout=3600,
        )
        if completed.returncode != 0:
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "unknown R error"
            )
            raise InfererBackendError(f"Authoritative R backend failed: {message}")
        if not output_path.exists():
            raise InfererBackendError(
                "Authoritative R backend produced no output file."
            )
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InfererBackendError(
                "Authoritative R backend returned malformed JSON."
            ) from exc
        if not isinstance(result, dict) or "predictions" not in result:
            raise InfererBackendError(
                "Authoritative R backend returned an invalid response."
            )
        predictions = result["predictions"]
        values = predictions if isinstance(predictions, list) else [predictions]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise InfererBackendError(
                "Authoritative R backend predictions must be finite numbers."
            )
        if "backend" in result and not isinstance(result["backend"], dict):
            raise InfererBackendError(
                "Authoritative R backend metadata must be an object."
            )
        return result
