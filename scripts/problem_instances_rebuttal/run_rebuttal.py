"""Run pending rebuttal problem instances without deleting prior artifacts."""

from __future__ import annotations

import argparse
import json
import sys
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


def _timestamp_is_valid(timestamp_dir: Path, expected_agents: set[str]) -> bool:
    agents_dir = timestamp_dir / "agents"
    if not agents_dir.is_dir():
        return False
    observed: set[str] = set()
    for transcript_path in agents_dir.glob("*/transcript.json"):
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if transcript.get("invalidated", False):
            return False
        observed.add(str(transcript.get("agent_id")))
    return observed == expected_agents


def classify(manifest: dict[str, Any], runs_dir: Path) -> RunState:
    """Classify a manifest as complete, incomplete, or not yet run."""
    run_root = runs_dir / str(manifest["id"])
    if not run_root.exists():
        return "pending"
    timestamp_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    if any(
        _timestamp_is_valid(path, _active_agents(manifest))
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
    parser.add_argument("--dry-run", action="store_true", help="List work without running it.")
    parser.add_argument("--seed", type=int, action="append", default=[])
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

    from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
    from TheCausalityGame.core.runtime.runner import Runner

    failures = 0
    for index, target in enumerate(runnable, start=1):
        print(f"Running [{index}/{len(runnable)}] seed={target.seed}")
        try:
            spec = ProblemInstanceSpec(**target.manifest)
            Runner(run_dir=args.runs_dir, problem_instance=spec).run()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAILED seed={target.seed}: {type(error).__name__}: {error}")

    print(f"Finished with {failures} failed manifest(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
