"""Round-level analysis utilities for the NeurIPS rebuttal experiments.

The functions in this module deliberately treat result metrics as errors and
resource measurements as costs: lower values are better throughout.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CACHE_SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 911

MISSION_ERROR_LABELS = {
    "graph_discovery": "Structural Hamming Distance (SHD)",
    "scm_discovery": "SCM distributional error",
    "symbolic_scm_discovery": "Symbolic mechanism error",
    "treatment_effect": "PEHE",
}

DISPLAY_NAMES = {
    "Cho2016ActiveGBN": "ActiveGBN-inspired",
    "Tigas2022CBED": "CBED-inspired",
    "Annadani2024CAASLOnline": "CAASLOnline-inspired",
    "HeGeng2008Minimax": "Minimax-inspired",
    "GIES": "GIES-inspired",
    "ABCI_Fast": "ABCI-inspired (Fast)",
    "ABCI_Balanced": "ABCI-inspired (Balanced)",
    "OptimalEffectDesign_Fast": "OptimalEffectDesign-inspired (Fast)",
    "OptimalEffectDesign_Balanced": "OptimalEffectDesign-inspired (Balanced)",
    "OptimalEffectDesign_PathFocused": "OptimalEffectDesign-inspired (Path-focused)",
    "OptimalEffectDesign_DisagreementHeavy": (
        "OptimalEffectDesign-inspired (Disagreement-heavy)"
    ),
    "OptimalEffectDesign_InterventionHeavy": (
        "OptimalEffectDesign-inspired (Intervention-heavy)"
    ),
}


@dataclass(frozen=True)
class BootstrapEstimate:
    """A point estimate and percentile bootstrap confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class PairedEstimate:
    """A paired effect where positive values mean the left side has more error."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int
    win_rate_left: float
    seeds: tuple[int, ...]


def display_name(agent_id: str) -> str:
    """Return a claim-safe display name for an implemented agent."""
    return DISPLAY_NAMES.get(agent_id, f"{agent_id} (implemented agent)")


def parse_experiment_id(experiment_id: str) -> dict[str, Any]:
    """Parse ``collection/mission/inferer/problem_rs-SEED`` identifiers."""
    parts = experiment_id.split("/")
    if len(parts) < 4:
        raise ValueError(f"Malformed experiment id: {experiment_id!r}")
    problem_seed = "/".join(parts[3:])
    if "_rs-" not in problem_seed:
        raise ValueError(f"Experiment id has no '_rs-' seed suffix: {experiment_id!r}")
    problem, seed_text = problem_seed.rsplit("_rs-", 1)
    return {
        "collection": parts[0],
        "mission": parts[1],
        "inferer": parts[2],
        "problem": problem,
        "seed": int(seed_text),
    }


def index_manifests(manifest_roots: Iterable[Path | str]) -> dict[str, dict[str, Any]]:
    """Load problem-instance manifests keyed by their declared experiment id."""
    manifests: dict[str, dict[str, Any]] = {}
    for root_value in manifest_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            experiment_id = manifest.get("id")
            if not isinstance(experiment_id, str):
                continue
            manifest["_manifest_path"] = path.as_posix()
            manifests[experiment_id] = manifest
    return manifests


def latest_transcript_paths(run_roots: Iterable[Path | str]) -> list[Path]:
    """Return the latest transcript for every experiment/agent pair."""
    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for root_value in run_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.glob("**/agents/*/transcript.json"):
            try:
                transcript = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            experiment_id = transcript.get("manifest_id")
            agent_id = transcript.get("agent_id")
            if not isinstance(experiment_id, str) or not isinstance(agent_id, str):
                continue
            timestamp = path.parents[2].name
            key = (experiment_id, agent_id)
            if key not in latest or timestamp > latest[key][0]:
                latest[key] = (timestamp, path)
    return sorted(value[1] for value in latest.values())


def _resource_used(
    budget: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    total_key: str,
    remaining_key: str,
) -> float | None:
    if snapshot is None:
        return None
    total = budget.get(total_key)
    remaining = snapshot.get(remaining_key)
    if total is None or remaining is None:
        return None
    return max(0.0, float(total) - float(remaining))


def transcript_to_round_records(
    transcript: Mapping[str, Any],
    *,
    transcript_path: Path | str,
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Reduce a transcript to auditable round-level numerical records."""
    experiment_id = str(transcript["manifest_id"])
    parts = parse_experiment_id(experiment_id)
    budget = ((manifest or {}).get("run_plan") or {}).get("budget") or {}
    valid = not bool(transcript.get("invalidated", False))
    invalidation_reason = transcript.get("invalidation_reason")
    cumulative_observational = 0
    cumulative_interventional = 0
    records: list[dict[str, Any]] = []

    for entry in transcript.get("entries", []):
        observational_increment = 0
        interventional_increment = 0
        for collection in entry.get("samples_collection") or []:
            count = int(collection.get("n") or 0)
            if collection.get("kind") == "observational":
                observational_increment += count
            elif collection.get("kind") == "interventional":
                interventional_increment += count

        cumulative_observational += observational_increment
        cumulative_interventional += interventional_increment
        feedback = entry.get("feedback") or {}
        snapshot = entry.get("budget_snapshot")
        decision = entry.get("decision") or {}
        result = feedback.get("result")
        behavior = feedback.get("behavior")
        round_number = int(entry.get("round") or (len(records) + 1))

        records.append(
            {
                "experiment_id": experiment_id,
                "collection": parts["collection"],
                "mission": parts["mission"],
                "inferer": parts["inferer"],
                "problem": parts["problem"],
                "seed": parts["seed"],
                "agent": str(transcript["agent_id"]),
                "round": round_number,
                "valid": valid,
                "invalidation_reason": invalidation_reason,
                "decision_kind": decision.get("kind"),
                "error": float(result) if result is not None else None,
                "behavior_cost": float(behavior) if behavior is not None else None,
                "observational_increment": observational_increment,
                "interventional_increment": interventional_increment,
                "total_increment": observational_increment + interventional_increment,
                "observational_cumulative": cumulative_observational,
                "interventional_cumulative": cumulative_interventional,
                "total_cumulative": (
                    cumulative_observational + cumulative_interventional
                ),
                "rounds_used": _resource_used(
                    budget, snapshot, "rounds", "rounds_left"
                ),
                "samples_used": _resource_used(
                    budget, snapshot, "samples", "samples_left"
                ),
                "time_used_s": _resource_used(
                    budget, snapshot, "time_s", "time_s_left"
                ),
                "memory_used_mb": _resource_used(
                    budget, snapshot, "memory_mb", "memory_mb_left"
                ),
            }
        )
    return records


def scan_round_records(
    run_roots: Iterable[Path | str],
    manifest_roots: Iterable[Path | str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan latest transcripts and return round records plus a run-level QC table."""
    manifests = index_manifests(manifest_roots)
    records: list[dict[str, Any]] = []
    run_qc: list[dict[str, Any]] = []
    for path in latest_transcript_paths(run_roots):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        experiment_id = str(transcript["manifest_id"])
        manifest = manifests.get(experiment_id)
        round_records = transcript_to_round_records(
            transcript,
            transcript_path=path,
            manifest=manifest,
        )
        if not transcript.get("invalidated", False):
            records.extend(round_records)
        complete_feedback = sum(record["error"] is not None for record in round_records)
        run_qc.append(
            {
                "experiment_id": experiment_id,
                "agent": str(transcript["agent_id"]),
                "valid": not bool(transcript.get("invalidated", False)),
                "invalidation_reason": transcript.get("invalidation_reason"),
                "rounds_recorded": len(round_records),
                "rounds_with_feedback": complete_feedback,
                "has_manifest": manifest is not None,
                "transcript_path": path.as_posix(),
            }
        )
    return records, run_qc


def write_round_cache(
    path: Path | str,
    records: Sequence[Mapping[str, Any]],
    run_qc: Sequence[Mapping[str, Any]],
) -> None:
    """Write the versioned rebuttal analysis cache."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "records": list(records),
        "run_qc": list(run_qc),
    }
    output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_round_cache(
    path: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate a rebuttal analysis cache."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cache schema {payload.get('schema_version')!r}; "
            f"expected {CACHE_SCHEMA_VERSION}."
        )
    return list(payload.get("records", [])), list(payload.get("run_qc", []))


def valid_feedback_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only valid rounds with a result error."""
    return [
        dict(record)
        for record in records
        if record.get("valid") is True and record.get("error") is not None
    ]


def group_runs(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    """Group valid feedback by mission, inferer, agent, and independent seed."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in valid_feedback_records(records):
        key = (
            str(record["mission"]),
            str(record["inferer"]),
            str(record["agent"]),
            int(record["seed"]),
        )
        grouped[key].append(record)
    for run in grouped.values():
        run.sort(key=lambda item: int(item["round"]))
    return dict(grouped)


def align_run_to_budgets(
    run: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    *,
    x_key: str = "total_cumulative",
    y_key: str = "error",
) -> list[float | None]:
    """Align a run to common budgets using only evidence available by each budget."""
    observed = sorted(
        (
            (float(record[x_key]), float(record[y_key]))
            for record in run
            if record.get(x_key) is not None and record.get(y_key) is not None
        ),
        key=lambda pair: pair[0],
    )
    x_values = [pair[0] for pair in observed]
    y_values = [pair[1] for pair in observed]
    aligned: list[float | None] = []
    for budget in budgets:
        index = bisect.bisect_right(x_values, float(budget)) - 1
        aligned.append(y_values[index] if index >= 0 else None)
    return aligned


def anytime_auc(
    run: Sequence[Mapping[str, Any]],
    *,
    x_key: str = "total_cumulative",
    y_key: str = "error",
    max_budget: float | None = None,
) -> float:
    """Compute normalized left-continuous error area over the observed budget range."""
    observed = sorted(
        (
            (float(record[x_key]), float(record[y_key]))
            for record in run
            if record.get(x_key) is not None and record.get(y_key) is not None
        ),
        key=lambda pair: pair[0],
    )
    if not observed:
        return math.nan
    start = observed[0][0]
    stop = float(max_budget) if max_budget is not None else observed[-1][0]
    if stop < start:
        return math.nan
    if stop == start:
        return observed[0][1]

    area = 0.0
    current_x, current_y = observed[0]
    for next_x, next_y in observed[1:]:
        if next_x > stop:
            break
        area += (next_x - current_x) * current_y
        current_x, current_y = next_x, next_y
    area += (stop - current_x) * current_y
    return area / (stop - start)


def trajectory_instability(
    run: Sequence[Mapping[str, Any]],
    *,
    y_key: str = "error",
) -> dict[str, float]:
    """Measure how often and how severely more evidence worsens the estimate."""
    values = [
        float(record[y_key]) for record in run if record.get(y_key) is not None
    ]
    if len(values) < 2:
        return {"worsening_fraction": math.nan, "max_rebound": math.nan}
    changes = [right - left for left, right in zip(values, values[1:])]
    worsening = [change for change in changes if change > 0]
    return {
        "worsening_fraction": len(worsening) / len(changes),
        "max_rebound": max(worsening, default=0.0),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Return a deterministic percentile bootstrap interval for a mean."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return BootstrapEstimate(math.nan, math.nan, math.nan, 0)
    rng = random.Random(seed)
    size = len(clean)
    bootstrap = [
        statistics.fmean(clean[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    ]
    bootstrap.sort()
    alpha = (1.0 - confidence) / 2.0
    return BootstrapEstimate(
        estimate=statistics.fmean(clean),
        ci_low=_percentile(bootstrap, alpha),
        ci_high=_percentile(bootstrap, 1.0 - alpha),
        n=size,
    )


def paired_effect(
    left_by_seed: Mapping[int, float],
    right_by_seed: Mapping[int, float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedEstimate:
    """Estimate ``left - right`` using only the intersection of matched seeds."""
    seeds = tuple(sorted(set(left_by_seed) & set(right_by_seed)))
    differences = [
        float(left_by_seed[item]) - float(right_by_seed[item]) for item in seeds
    ]
    estimate = bootstrap_mean_ci(differences, resamples=resamples, seed=seed)
    return PairedEstimate(
        estimate=estimate.estimate,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        n=estimate.n,
        win_rate_left=(
            sum(difference < 0 for difference in differences) / len(differences)
            if differences
            else math.nan
        ),
        seeds=seeds,
    )


def interaction_difference(
    left_first: Mapping[int, float],
    right_first: Mapping[int, float],
    left_second: Mapping[int, float],
    right_second: Mapping[int, float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedEstimate:
    """Estimate ``(left-right)_first - (left-right)_second`` by seed."""
    seeds = tuple(
        sorted(
            set(left_first)
            & set(right_first)
            & set(left_second)
            & set(right_second)
        )
    )
    first = {
        item: float(left_first[item]) - float(right_first[item]) for item in seeds
    }
    second = {
        item: float(left_second[item]) - float(right_second[item]) for item in seeds
    }
    return paired_effect(first, second, resamples=resamples, seed=seed)


def pareto_lower(points: Iterable[Sequence[float]]) -> list[tuple[float, ...]]:
    """Return points not dominated when every dimension is minimized."""
    finite = [
        tuple(float(value) for value in point)
        for point in points
        if all(math.isfinite(float(value)) for value in point)
    ]
    frontier: list[tuple[float, ...]] = []
    for index, point in enumerate(finite):
        dominated = False
        for other_index, other in enumerate(finite):
            if index == other_index:
                continue
            no_worse = all(a <= b for a, b in zip(other, point))
            strictly_better = any(a < b for a, b in zip(other, point))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated and point not in frontier:
            frontier.append(point)
    return sorted(frontier)
