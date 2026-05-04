"""Run pending seeded problem instances one at a time."""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Edit this configuration to choose which seeded manifests are eligible to run.
# Folder and file values are relative to scripts/problem_instances/seeded.
# ---------------------------------------------------------------------------

SEEDED_ROOT = Path("scripts/problem_instances/seeded")
RUN_DIR = Path("runs")

INCLUDE_FOLDERS: set[str] = set()
EXCLUDE_FOLDERS: set[str] = set()

INCLUDE_FILES: set[str] = set()
EXCLUDE_FILES: set[str] = set()

INCLUDE_SEEDS: set[int] = set()
EXCLUDE_SEEDS: set[int] = set()

DRY_RUN = False
LIMIT: int | None = None
REMOVE_FAILED_RUN_FOLDER = True


SEED_PATTERN = re.compile(r"_rs-(\d+)$")
RunState = Literal["complete", "incomplete", "pending"]


@dataclass(frozen=True)
class ProblemInstanceTarget:
    """A seeded manifest and its current run state."""

    path: Path
    relative_file: str
    folder: str
    seed: int
    problem_instance_json: dict[str, Any]
    state: RunState

    @property
    def problem_instance_id(self) -> str:
        return str(self.problem_instance_json["id"])

    @property
    def run_root(self) -> Path:
        return RUN_DIR / self.problem_instance_id


@dataclass(frozen=True)
class RunFailure:
    """A failed manifest execution captured for final reporting."""

    path: Path
    error: str
    removed_paths: tuple[Path, ...]


def _normalize_path(value: str) -> str:
    return value.strip().strip("/")


def _normalized_set(values: set[str]) -> set[str]:
    return {_normalize_path(value) for value in values if _normalize_path(value)}


def _matches_prefix(path: str, prefixes: set[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def _seed_from_path(path: Path) -> int:
    seed_match = SEED_PATTERN.search(path.stem)
    if seed_match is None:
        raise ValueError(f"Seeded problem instance filename does not end in '_rs-<seed>': {path}")
    return int(seed_match.group(1))


def _active_agent_ids(problem_instance_json: dict[str, Any]) -> list[str]:
    return [
        str(agent["id"])
        for agent in problem_instance_json.get("agents", [])
        if agent.get("active", True)
    ]


def _timestamp_is_complete(timestamp_dir: Path, agent_ids: list[str]) -> bool:
    if not timestamp_dir.is_dir():
        return False
    return all(
        (timestamp_dir / "agents" / agent_id / "transcript.json").is_file()
        for agent_id in agent_ids
    )


def _classify_run(run_root: Path, agent_ids: list[str]) -> RunState:
    if not run_root.exists():
        return "pending"

    timestamp_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    if any(_timestamp_is_complete(path, agent_ids) for path in timestamp_dirs):
        return "complete"
    return "incomplete"


def _timestamp_dirs(run_root: Path) -> set[Path]:
    if not run_root.exists():
        return set()
    return {path for path in run_root.iterdir() if path.is_dir()}


def _target_from_path(root: Path, path: Path) -> ProblemInstanceTarget:
    relative = path.relative_to(root)
    with path.open(encoding="utf-8") as f:
        problem_instance_json = json.load(f)

    agent_ids = _active_agent_ids(problem_instance_json)
    run_root = RUN_DIR / str(problem_instance_json["id"])
    return ProblemInstanceTarget(
        path=path,
        relative_file=relative.as_posix(),
        folder=relative.parent.as_posix(),
        seed=_seed_from_path(path),
        problem_instance_json=problem_instance_json,
        state=_classify_run(run_root, agent_ids),
    )


def _passes_config(target: ProblemInstanceTarget) -> bool:
    include_folders = _normalized_set(INCLUDE_FOLDERS)
    exclude_folders = _normalized_set(EXCLUDE_FOLDERS)
    include_files = _normalized_set(INCLUDE_FILES)
    exclude_files = _normalized_set(EXCLUDE_FILES)

    if include_files and target.relative_file not in include_files:
        return False
    if target.relative_file in exclude_files:
        return False

    if include_folders and not _matches_prefix(target.folder, include_folders):
        return False
    if exclude_folders and _matches_prefix(target.folder, exclude_folders):
        return False

    if INCLUDE_SEEDS and target.seed not in INCLUDE_SEEDS:
        return False
    return target.seed not in EXCLUDE_SEEDS


def _scan_targets(root: Path) -> list[ProblemInstanceTarget]:
    targets = [_target_from_path(root, path) for path in sorted(root.rglob("*.json"))]
    return [target for target in targets if _passes_config(target)]


def _print_scan_summary(targets: list[ProblemInstanceTarget]) -> None:
    counts = {
        "complete": sum(target.state == "complete" for target in targets),
        "incomplete": sum(target.state == "incomplete" for target in targets),
        "pending": sum(target.state == "pending" for target in targets),
    }
    print(
        "Selected "
        f"{len(targets)} manifests: "
        f"{counts['complete']} complete, "
        f"{counts['incomplete']} incomplete, "
        f"{counts['pending']} pending."
    )

    incomplete = [target.relative_file for target in targets if target.state == "incomplete"]
    if incomplete:
        print("Incomplete manifests:")
        for relative_file in incomplete:
            print(f"  - {relative_file}")


def _cleanup_new_timestamp_dirs(run_root: Path, before: set[Path]) -> tuple[Path, ...]:
    if not REMOVE_FAILED_RUN_FOLDER:
        return ()

    after = _timestamp_dirs(run_root)
    new_dirs = tuple(sorted(after - before))
    for path in new_dirs:
        shutil.rmtree(path)
    return new_dirs


def _run_target(target: ProblemInstanceTarget) -> RunFailure | None:
    from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
    from TheCausalityGame.core.runtime.runner import Runner

    before = _timestamp_dirs(target.run_root)
    try:
        problem_instance_spec = ProblemInstanceSpec(**target.problem_instance_json)
        Runner(run_dir=RUN_DIR, problem_instance=problem_instance_spec).run()
    except Exception as error:  # noqa: BLE001
        removed = _cleanup_new_timestamp_dirs(target.run_root, before)
        print(f"FAILED {target.relative_file}: {type(error).__name__}: {error}")
        if removed:
            print("Removed failed run folders:")
            for path in removed:
                print(f"  - {path}")
        return RunFailure(
            path=target.path,
            error=f"{type(error).__name__}: {error}",
            removed_paths=removed,
        )
    return None


def main() -> None:
    try:
        if not SEEDED_ROOT.exists():
            raise FileNotFoundError(
                f"No seeded problem instances found under {SEEDED_ROOT}. Run root main.py first."
            )

        targets = _scan_targets(SEEDED_ROOT)
        _print_scan_summary(targets)

        runnable = [target for target in targets if target.state in {"pending", "incomplete"}]
        if LIMIT is not None:
            runnable = runnable[:LIMIT]

        if not runnable:
            print("No pending or incomplete manifests to run.")
            return

        print(f"Runnable manifests: {len(runnable)}")
        for index, target in enumerate(runnable, start=1):
            print(f"[{index}/{len(runnable)}] {target.relative_file} ({target.state})")

        if DRY_RUN:
            print("Dry run only; no manifests were executed.")
            return

        failures: list[RunFailure] = []
        for index, target in enumerate(runnable, start=1):
            print(f"Running [{index}/{len(runnable)}] {target.relative_file}")
            failure = _run_target(target)
            if failure is not None:
                failures.append(failure)

        print(f"Finished {len(runnable) - len(failures)} of {len(runnable)} manifests.")
        if failures:
            print("Failures:")
            for failure in failures:
                print(f"  - {failure.path}: {failure.error}")

    except Exception as error:  # noqa: BLE001
        print(f"Error during execution: {type(error).__name__}: {error}")
        # Re run main to attempt to run remaining targets, if any
        main()


if __name__ == "__main__":
    main()
