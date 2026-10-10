"""Run pending rebuttal problem instances without deleting prior artifacts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST_ROOT = (
    ROOT / "scripts" / "problem_instances_rebuttal" / "treatment_effect"
)
DEFAULT_RUNS_DIR = ROOT / "runs"
from TheCausalityGame.core.contracts.specs.problem_instance import (  # noqa: E402
    ProblemInstanceSpec,
)
from TheCausalityGame.core.infrastructure.artifact_contract import (  # noqa: E402
    validate_publication_transcript,
)
from TheCausalityGame.core.infrastructure.provenance import (  # noqa: E402
    manifest_fingerprint,
    source_fingerprint,
)

RunState = Literal["complete", "incomplete", "pending"]


@dataclass(frozen=True)
class Target:
    """One rebuttal manifest and its current resumable state."""

    path: Path
    manifest: dict[str, Any]
    seed: int
    state: RunState


def _seed(path: Path) -> int:
    return int(path.stem.rsplit("_rs-", 1)[1])


def _active_agents(manifest: dict[str, Any]) -> set[str]:
    return {
        str(agent["id"])
        for agent in manifest.get("agents", [])
        if agent.get("active", True)
    }


def _timestamp_is_valid(
    timestamp_dir: Path, expected_agents: set[str], expected_manifest: str
) -> bool:
    try:
        provenance = json.loads(
            (timestamp_dir / "provenance.json").read_text(encoding="utf-8")
        )
        if not isinstance(provenance, dict):
            return False
        if (
            provenance.get("source_fingerprint") != source_fingerprint()
            or provenance.get("manifest_fingerprint") != expected_manifest
        ):
            return False
    except (OSError, json.JSONDecodeError):
        return False
    agents_dir = timestamp_dir / "agents"
    if not agents_dir.is_dir():
        return False
    observed: set[str] = set()
    for transcript_path in agents_dir.glob("*/transcript.json"):
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            if not isinstance(transcript, dict):
                return False
            validate_publication_transcript(transcript, provenance)
        except (json.JSONDecodeError, OSError):
            return False
        except (TypeError, ValueError):
            return False
        if (
            transcript.get("invalidated", False)
            or transcript.get("final_score_status") != "finite"
        ):
            return False
        agent_id = str(transcript.get("agent_id"))
        if transcript_path.parent.name != agent_id or agent_id in observed:
            return False
        observed.add(agent_id)
    return observed == expected_agents


def classify(manifest: dict[str, Any], runs_dir: Path) -> RunState:
    """Classify a manifest as complete, incomplete, or not yet run."""
    run_root = runs_dir / str(manifest["id"])
    if not run_root.exists():
        return "pending"
    timestamp_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    if any(
        _timestamp_is_valid(
            path,
            _active_agents(manifest),
            manifest_fingerprint(
                ProblemInstanceSpec.model_validate(manifest).model_dump(mode="json")
            ),
        )
        for path in timestamp_dirs
    ):
        return "complete"
    return "incomplete"


def scan(manifest_root: Path, runs_dir: Path, seeds: set[int]) -> list[Target]:
    """Find selected rebuttal manifests and their run states."""
    targets: list[Target] = []
    for path in sorted(manifest_root.rglob("hill_rs-*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        seed = _seed(path)
        if seeds and seed not in seeds:
            continue
        targets.append(
            Target(
                path=path,
                manifest=manifest,
                seed=seed,
                state=classify(manifest, runs_dir),
            )
        )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="List work without running it."
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800,
        help="Per-manifest subprocess deadline; preserves partial artifacts.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="Stop scheduling new manifests after this elapsed time.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run selected manifests again while preserving prior timestamped artifacts.",
    )
    parser.add_argument("--manifests-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser.parse_args()


def main() -> int:
    """Run selected incomplete manifests and return a shell status."""
    args = parse_args()
    targets = scan(args.manifests_root, args.runs_dir, set(args.seed))
    counts = {
        state: sum(target.state == state for target in targets)
        for state in ("complete", "incomplete", "pending")
    }
    print(
        f"Selected {len(targets)} manifests: "
        f"{counts['complete']} complete, {counts['incomplete']} incomplete, "
        f"{counts['pending']} pending."
    )

    runnable = (
        targets
        if args.force
        else [target for target in targets if target.state != "complete"]
    )
    if args.limit is not None:
        runnable = runnable[: args.limit]
    for target in runnable:
        print(f"{target.state:10} seed={target.seed:3} {target.path.relative_to(ROOT)}")

    if args.dry_run or not runnable:
        return 0

    if args.timeout_s <= 0 or (
        args.max_duration_s is not None and args.max_duration_s <= 0
    ):
        raise ValueError("Scheduling deadlines must be positive.")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.runs_dir / "campaign_progress.json"
    started = time.monotonic()
    failures, completed = 0, []
    if progress_path.exists():
        prior = json.loads(progress_path.read_text())
        if prior.get("source_fingerprint") == source_fingerprint():
            completed = prior.get("results", [])
    for index, target in enumerate(runnable, start=1):
        if (
            args.max_duration_s is not None
            and time.monotonic() - started >= args.max_duration_s
        ):
            print(
                "Campaign scheduling deadline reached; remaining manifests can be resumed."
            )
            break
        print(f"Running [{index}/{len(runnable)}] seed={target.seed}", flush=True)
        before = time.monotonic()
        log_path = args.runs_dir / f"seed-{target.seed}-{time.time_ns()}.log"
        status = "failed"
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "TheCausalityGame.core.cli.main",
                        "run",
                        str(target.path),
                        "--run-dir",
                        str(args.runs_dir),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    returncode = process.wait(timeout=args.timeout_s)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            check=False,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                    raise
            if (
                returncode == 0
                and classify(target.manifest, args.runs_dir) == "complete"
            ):
                status = "complete"
            else:
                failures += 1
        except subprocess.TimeoutExpired:
            status = "timeout"
            failures += 1
        elapsed = time.monotonic() - before
        completed.append(
            {
                "seed": target.seed,
                "status": status,
                "elapsed_s": elapsed,
                "log": str(log_path),
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "source_fingerprint": source_fingerprint(),
                    "elapsed_s": time.monotonic() - started,
                    "results": completed,
                    "pending": len(runnable) - index,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"seed={target.seed}: {status}, elapsed={elapsed:.1f}s, log={log_path}",
            flush=True,
        )
    print(f"Finished with {failures} failed manifest(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
