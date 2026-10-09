"""Checked subprocess bridge to the pinned authoritative R backends."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from TheCausalityGame.core.lib.errors.inferer import (
    InfererBackendError,
    InfererDependencyError,
)


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
        try:
            completed = subprocess.run(
                [rscript, str(backend), str(input_path), str(output_path)],
                capture_output=True,
                check=False,
                env=env,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired as exc:
            raise InfererBackendError(
                f"Authoritative R backend timed out after {exc.timeout} seconds."
            ) from exc
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
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or "predictions" not in result:
            raise InfererBackendError(
                "Authoritative R backend returned an invalid response."
            )
        return result
