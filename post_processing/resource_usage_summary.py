"""Summarize maximum resource usage by mission and inferer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_processing.plot_utils import matches_prefix, parse_manifest_id


RUNS_ROOT = Path("runs/seeded")
PROBLEM_INSTANCES_ROOT = Path("scripts/problem_instances")
OUTPUT_PATH = Path("post_processing/resource_usage_summary.json")

# Folder prefixes relative to RUNS_ROOT. Empty allowlist means all folders.
ALLOW_FOLDERS: set[str] = set()
BLOCK_FOLDERS: set[str] = set()

SKIP_INVALIDATED = True


@dataclass(frozen=True)
class ResourceUsage:
    """Resource usage derived from one transcript entry budget snapshot."""

    samples: float
    time_s: float
    memory_mb: float


def _latest_complete_timestamp(instance_dir: Path) -> Path | None:
    timestamps = sorted(
        [path for path in instance_dir.iterdir() if path.is_dir()],
        reverse=True,
    )
    for timestamp_dir in timestamps:
        agents_dir = timestamp_dir / "agents"
        if not agents_dir.is_dir():
            continue

        agent_dirs = [path for path in agents_dir.iterdir() if path.is_dir()]
        if agent_dirs and all((path / "transcript.json").is_file() for path in agent_dirs):
            return timestamp_dir
    return None


def _manifest_path(manifest_id: str) -> Path:
    return PROBLEM_INSTANCES_ROOT / f"{manifest_id}.json"


def _load_budget(manifest_id: str) -> dict[str, float]:
    path = _manifest_path(manifest_id)
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("run_plan", {}).get("budget", {})


def _used_from_snapshot(
    budget: dict[str, float],
    snapshot: dict[str, float | int | None],
) -> ResourceUsage:
    samples_left = snapshot.get("samples_left")
    time_s_left = snapshot.get("time_s_left")
    memory_mb_left = snapshot.get("memory_mb_left")

    samples = (
        float(budget["samples"]) - float(samples_left)
        if budget.get("samples") is not None and samples_left is not None
        else 0.0
    )
    time_s = (
        float(budget["time_s"]) - float(time_s_left)
        if budget.get("time_s") is not None and time_s_left is not None
        else 0.0
    )
    memory_mb = (
        float(budget["memory_mb"]) - float(memory_mb_left)
        if budget.get("memory_mb") is not None and memory_mb_left is not None
        else 0.0
    )

    return ResourceUsage(
        samples=max(0.0, samples),
        time_s=max(0.0, time_s),
        memory_mb=max(0.0, memory_mb),
    )


def _transcript_max_usage(transcript: dict[str, Any], budget: dict[str, float]) -> ResourceUsage:
    max_usage = ResourceUsage(samples=0.0, time_s=0.0, memory_mb=0.0)
    for entry in transcript.get("entries", []):
        snapshot = entry.get("budget_snapshot")
        if not snapshot:
            continue

        usage = _used_from_snapshot(budget, snapshot)
        max_usage = ResourceUsage(
            samples=max(max_usage.samples, usage.samples),
            time_s=max(max_usage.time_s, usage.time_s),
            memory_mb=max(max_usage.memory_mb, usage.memory_mb),
        )
    return max_usage


def _replace_if_larger(
    current: dict[str, Any],
    metric: str,
    value: float,
    *,
    experiment_id: str,
    agent_id: str,
    run_path: Path,
) -> None:
    if value <= current[metric]["value"]:
        return

    current[metric] = {
        "value": value,
        "experiment_id": experiment_id,
        "agent_id": agent_id,
        "run_path": run_path.as_posix(),
    }


def summarize_resource_usage() -> dict[str, Any]:
    """Return maximum samples, time, and memory usage by mission and inferer."""
    summary: dict[str, Any] = {}
    processed_transcripts = 0

    for instance_dir in sorted(path for path in RUNS_ROOT.rglob("*") if path.is_dir()):
        timestamp_dir = _latest_complete_timestamp(instance_dir)
        if timestamp_dir is None:
            continue

        transcript_paths = sorted((timestamp_dir / "agents").glob("*/transcript.json"))
        if not transcript_paths:
            continue

        first_transcript = json.loads(transcript_paths[0].read_text(encoding="utf-8"))
        manifest_id = str(first_transcript["manifest_id"])
        parts = parse_manifest_id(manifest_id)
        folder = f"{parts.mission}/{parts.inferer}"

        if ALLOW_FOLDERS and not matches_prefix(folder, ALLOW_FOLDERS):
            continue
        if BLOCK_FOLDERS and matches_prefix(folder, BLOCK_FOLDERS):
            continue

        budget = _load_budget(manifest_id)
        inferer_summary = summary.setdefault(parts.mission, {}).setdefault(
            parts.inferer,
            {
                "max_samples": {"value": 0.0},
                "max_time_s": {"value": 0.0},
                "max_memory_mb": {"value": 0.0},
            },
        )

        for transcript_path in transcript_paths:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            if SKIP_INVALIDATED and transcript.get("invalidated", False):
                continue

            usage = _transcript_max_usage(transcript, budget)
            agent_id = str(transcript["agent_id"])
            _replace_if_larger(
                inferer_summary,
                "max_samples",
                usage.samples,
                experiment_id=manifest_id,
                agent_id=agent_id,
                run_path=timestamp_dir,
            )
            _replace_if_larger(
                inferer_summary,
                "max_time_s",
                usage.time_s,
                experiment_id=manifest_id,
                agent_id=agent_id,
                run_path=timestamp_dir,
            )
            _replace_if_larger(
                inferer_summary,
                "max_memory_mb",
                usage.memory_mb,
                experiment_id=manifest_id,
                agent_id=agent_id,
                run_path=timestamp_dir,
            )
            processed_transcripts += 1

    return {
        "runs_root": RUNS_ROOT.as_posix(),
        "problem_instances_root": PROBLEM_INSTANCES_ROOT.as_posix(),
        "processed_transcripts": processed_transcripts,
        "missions": summary,
    }


def main() -> None:
    summary = summarize_resource_usage()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH} with {summary['processed_transcripts']} transcripts.")


if __name__ == "__main__":
    main()
