"""Build and verify the base-only wheel from a clean, external working directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    """Install constrained base dependencies, check resources and execute the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tcg-wheel-") as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / "environment")
        bindir = root / "environment" / ("Scripts" if os.name == "nt" else "bin")
        python = bindir / ("python.exe" if os.name == "nt" else "python")
        cli = bindir / ("tcg.exe" if os.name == "nt" else "tcg")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["MPLCONFIGDIR"] = str(root / "matplotlib")

        def run(arguments: list[str]) -> None:
            subprocess.run(arguments, cwd=root, env=environment, check=True)

        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-c",
                str(repo / "constraints-publication-py312.txt"),
                str(args.wheel.resolve()),
            ]
        )
        run([str(python), "-m", "pip", "check"])
        run(
            [
                str(python),
                "-c",
                "from importlib.resources import files; from importlib.util import find_spec; root=files('TheCausalityGame.agent.inferers'); assert root.joinpath('REFERENCES.md').is_file(); assert root.joinpath('r_backends/s_learner.R').is_file(); assert all(find_spec(p) is None for p in ['lingam','causallearn','notears']); from TheCausalityGame.generators import SyntheticSCMGenerator; from TheCausalityGame.core.infrastructure.registry import build_from_spec; assert len(build_from_spec(SyntheticSCMGenerator().generate()).generate_samples(num_samples=4)) == 4",
            ]
        )
        manifest = root / "example.json"
        shutil.copyfile(repo / "examples/publication_smoke.json", manifest)
        run([str(cli), "--help"])
        run([str(cli), "run", str(manifest), "--run-dir", str(root / "runs")])
    print("Clean base-wheel installation, resources, generator and CLI verified.")


if __name__ == "__main__":
    main()
