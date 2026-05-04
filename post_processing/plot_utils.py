"""Utilities for extracting seeded run results and plotting paper figures."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeAlias

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))

import matplotlib.pyplot as plt
import numpy as np

MetricRuns: TypeAlias = tuple[list[list[float]], list[list[float]]]
AgentsDict: TypeAlias = dict[str, MetricRuns]
MissionInfererAgentRuns: TypeAlias = dict[str, dict[str, AgentsDict]]
MissionAgentInfererRuns: TypeAlias = dict[str, dict[str, AgentsDict]]

SEED_RE = re.compile(r"^(?P<problem>.+)_rs-(?P<seed>\d+)$")


@dataclass(frozen=True)
class SummarySeries:
    """Mean/std summary for metric runs with potentially unequal lengths."""

    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class ManifestParts:
    """Parsed fields from a seeded manifest id."""

    manifest_id: str
    mission: str
    inferer: str
    problem: str
    seed: int


@dataclass(frozen=True)
class LineStyle:
    """Style values used consistently across all three plot panels."""

    color: str | None = None
    linestyle: str = "-"
    marker: str = "o"


def normalize_prefix(prefix: str) -> str:
    """Normalize a folder/file prefix to slash-separated relative form."""
    return prefix.strip().strip("/")


def matches_prefix(path: str, prefixes: Iterable[str]) -> bool:
    """Return whether path equals or lives below any prefix."""
    normalized = normalize_prefix(path)
    return any(
        normalized == normalize_prefix(prefix)
        or normalized.startswith(f"{normalize_prefix(prefix)}/")
        for prefix in prefixes
        if normalize_prefix(prefix)
    )


def parse_manifest_id(manifest_id: str) -> ManifestParts:
    """Parse ids like seeded/graph_discovery/pc/rc_circuit_rs-150."""
    parts = manifest_id.split("/")
    if len(parts) < 4 or parts[0] != "seeded":
        raise ValueError(f"Expected seeded manifest id, got {manifest_id!r}")

    mission = parts[1]
    inferer = parts[2]
    problem_seed = "/".join(parts[3:])
    match = SEED_RE.match(problem_seed)
    if match is None:
        raise ValueError(f"Could not parse problem/seed from {manifest_id!r}")

    return ManifestParts(
        manifest_id=manifest_id,
        mission=mission,
        inferer=inferer,
        problem=match.group("problem"),
        seed=int(match.group("seed")),
    )


def summarize_runs(runs: list[list[float]]) -> SummarySeries:
    """Summarize variable-length metric runs using nan padding."""
    if not runs:
        raise ValueError("Expected at least one run, got 0.")

    max_len = max(len(run) for run in runs)
    if max_len == 0:
        raise ValueError("Runs contain empty sequences.")

    arr = np.full((len(runs), max_len), np.nan, dtype=float)
    for i, run in enumerate(runs):
        if run:
            arr[i, : len(run)] = np.asarray(run, dtype=float)

    return SummarySeries(
        x=np.arange(max_len),
        mean=np.nanmean(arr, axis=0),
        std=np.nanstd(arr, axis=0, ddof=0),
    )


def final_stats(runs: list[list[float]]) -> tuple[float, float]:
    """Return mean/std of the last value of each non-empty run."""
    finals = [run[-1] for run in runs if run]
    if not finals:
        return np.nan, np.nan
    finals_arr = np.asarray(finals, dtype=float)
    return float(np.mean(finals_arr)), float(np.std(finals_arr, ddof=0))


def pareto_frontier(points: np.ndarray, maximize_y: bool = True) -> np.ndarray:
    """2D Pareto frontier where x is minimized and y is optionally maximized."""
    pts = points[np.isfinite(points).all(axis=1)]
    if pts.shape[0] == 0:
        return pts

    order = (
        np.lexsort((-pts[:, 1], pts[:, 0]))
        if maximize_y
        else np.lexsort((pts[:, 1], pts[:, 0]))
    )
    pts = pts[order]

    frontier: list[tuple[float, float]] = []
    best_y = -np.inf if maximize_y else np.inf
    for x, y in pts:
        if (maximize_y and y > best_y) or (not maximize_y and y < best_y):
            frontier.append((float(x), float(y)))
            best_y = y

    return np.asarray(frontier, dtype=float)


def _extract_metric_series(transcript: dict[str, Any]) -> tuple[list[float], list[float]]:
    behavior: list[float] = []
    result: list[float] = []
    for entry in transcript.get("entries", []):
        feedback = entry.get("feedback") or {}
        if "behavior" not in feedback or "result" not in feedback:
            continue
        behavior.append(float(feedback["behavior"]))
        result.append(float(feedback["result"]))
    return behavior, result


def _timestamp_is_complete(timestamp_dir: Path) -> bool:
    agents_dir = timestamp_dir / "agents"
    if not agents_dir.is_dir():
        return False

    agent_dirs = [path for path in agents_dir.iterdir() if path.is_dir()]
    return bool(agent_dirs) and all((path / "transcript.json").is_file() for path in agent_dirs)


def _latest_complete_timestamp(instance_dir: Path) -> Path | None:
    timestamps = sorted(
        [path for path in instance_dir.iterdir() if path.is_dir()],
        reverse=True,
    )
    for timestamp_dir in timestamps:
        if _timestamp_is_complete(timestamp_dir):
            return timestamp_dir
    return None


def scan_experiments(
    runs_root: Path | str,
    *,
    allow_folders: Iterable[str] = (),
    block_folders: Iterable[str] = (),
    skip_invalidated: bool = True,
    skip_empty: bool = True,
) -> list[dict[str, Any]]:
    """Scan seeded runs and return JSON-serializable experiment results."""
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {root}")

    experiments: list[dict[str, Any]] = []
    for instance_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
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

        if allow_folders and not matches_prefix(folder, allow_folders):
            continue
        if block_folders and matches_prefix(folder, block_folders):
            continue

        agent_results: dict[str, dict[str, Any]] = {}
        mission_id = str(first_transcript["mission_id"])
        for transcript_path in transcript_paths:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            behavior, result = _extract_metric_series(transcript)
            invalidated = bool(transcript.get("invalidated", False))

            if skip_invalidated and invalidated:
                continue
            if skip_empty and (not behavior or not result):
                continue

            agent_results[str(transcript["agent_id"])] = {
                "behavior": behavior,
                "result": result,
                "invalidated": invalidated,
            }

        if not agent_results:
            continue

        experiments.append(
            {
                "id": parts.manifest_id,
                "mission": parts.mission,
                "mission_id": mission_id,
                "inferer": parts.inferer,
                "problem": parts.problem,
                "seed": parts.seed,
                "run_path": timestamp_dir.as_posix(),
                "agent_results": agent_results,
            }
        )

    experiments.sort(key=lambda item: (item["mission"], item["inferer"], item["problem"], item["seed"]))
    return experiments


def write_results_cache(experiments: list[dict[str, Any]], cache_path: Path | str) -> None:
    """Write experiment results using the paper plotting cache schema."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"experiments": experiments}, f, indent=2)
        f.write("\n")


def load_results_cache(cache_path: Path | str) -> list[dict[str, Any]]:
    """Load experiment results from a JSON cache."""
    with Path(cache_path).open(encoding="utf-8") as f:
        return list(json.load(f).get("experiments", []))


def build_agent_comparisons(
    experiments: list[dict[str, Any]],
) -> MissionInfererAgentRuns:
    """Build mission -> inferer -> agent -> metric runs."""
    grouped: MissionInfererAgentRuns = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))  # type: ignore[assignment]
    for experiment in experiments:
        mission = str(experiment["mission"])
        inferer = str(experiment["inferer"])
        for agent, values in experiment["agent_results"].items():
            behavior_runs, result_runs = grouped[mission][inferer][agent]
            behavior_runs.append(list(values["behavior"]))
            result_runs.append(list(values["result"]))
    return _to_plain_nested_dict(grouped)


def build_inferer_comparisons(
    experiments: list[dict[str, Any]],
) -> MissionAgentInfererRuns:
    """Build mission -> agent -> inferer -> metric runs."""
    grouped: MissionAgentInfererRuns = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))  # type: ignore[assignment]
    for experiment in experiments:
        mission = str(experiment["mission"])
        inferer = str(experiment["inferer"])
        for agent, values in experiment["agent_results"].items():
            behavior_runs, result_runs = grouped[mission][agent][inferer]
            behavior_runs.append(list(values["behavior"]))
            result_runs.append(list(values["result"]))
    return _to_plain_nested_dict(grouped)


def inferer_comparison_agents(
    by_mission_agent_inferer: MissionAgentInfererRuns,
    mission: str,
    agent: str,
) -> AgentsDict:
    """Return plot-ready labels for one agent compared across inferers."""
    inferer_runs = by_mission_agent_inferer[mission][agent]
    return {
        f"{agent} @ {inferer}": runs
        for inferer, runs in sorted(inferer_runs.items())
    }


def _to_plain_nested_dict(value: Any) -> Any:
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {key: _to_plain_nested_dict(inner) for key, inner in value.items()}
    return value


def default_style_key(label: str) -> tuple[str, str]:
    """Use text before/after '@' as base/variant for colors and line forms."""
    if " @ " in label:
        base, variant = label.split(" @ ", 1)
        return base, variant
    return label, ""


def resolve_styles(
    labels: Iterable[str],
    *,
    style_key: Callable[[str], tuple[str, str]] = default_style_key,
) -> dict[str, LineStyle]:
    """Assign colors by base label and linestyles/markers by variant."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]

    base_colors: dict[str, str] = {}
    variant_index: dict[tuple[str, str], int] = {}
    styles: dict[str, LineStyle] = {}

    for label in labels:
        base, variant = style_key(label)
        if base not in base_colors:
            base_colors[base] = color_cycle[len(base_colors) % len(color_cycle)] if color_cycle else None
        variant_key = (base, variant)
        if variant_key not in variant_index:
            variant_index[variant_key] = len([key for key in variant_index if key[0] == base])
        idx = variant_index[variant_key]
        styles[label] = LineStyle(
            color=base_colors[base],
            linestyle=linestyles[idx % len(linestyles)],
            marker=markers[idx % len(markers)],
        )

    return styles


def plot_agents_mean_std(
    agents: AgentsDict,
    title: str = "Agent Comparison Scores",
    *,
    show_final_errorbars: bool = True,
    shade_alpha: float = 0.18,
    line_alpha: float = 0.85,
    linewidth: float = 2.0,
    behavior_ylabel: str = "Behavior Score",
    result_ylabel: str = "Result Score",
    result_ylog: bool = True,
    result_xlog: bool = False,
    grid: bool = True,
    legend_ncol: int | None = None,
    figsize: tuple[int, int] = (18, 4),
    plot_pareto: bool = True,
    pareto_label: str = "Pareto frontier (mean finals)",
    pareto_color: str = "black",
    pareto_linestyle: Any = (0, (5, 2, 1, 2)),
    pareto_linewidth: float = 2.5,
    pareto_alpha: float = 0.95,
    pareto_zorder: int = 4,
    styles: dict[str, LineStyle] | None = None,
    show_score_labels: bool = True,
    score_label_fontsize: float = 8.0,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot final score scatter plus behavior/result mean-std series."""
    if not agents:
        raise ValueError("agents dict is empty.")

    styles = styles or resolve_styles(agents.keys())
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    ax_scatter, ax_beh, ax_res = axes
    mean_final_points: list[tuple[float, float]] = []

    for name, (behavior_runs, result_runs) in agents.items():
        style = styles.get(name, LineStyle())
        beh_summary = summarize_runs(behavior_runs)
        res_summary = summarize_runs(result_runs)

        (line_beh,) = ax_beh.plot(
            beh_summary.x,
            beh_summary.mean,
            label=name,
            alpha=line_alpha,
            linewidth=linewidth,
            color=style.color,
            linestyle=style.linestyle,
            marker=None,
        )
        color = line_beh.get_color()
        ax_beh.fill_between(
            beh_summary.x,
            beh_summary.mean - beh_summary.std,
            beh_summary.mean + beh_summary.std,
            color=color,
            alpha=shade_alpha,
            linewidth=0,
        )

        x_last_b = int(beh_summary.x[~np.isnan(beh_summary.mean)][-1])
        y_last_b = float(beh_summary.mean[x_last_b])
        ax_beh.scatter(
            x_last_b,
            y_last_b,
            s=90,
            marker=style.marker,
            edgecolor="black",
            linewidth=0.8,
            color=color,
            zorder=3,
        )
        if show_score_labels:
            ax_beh.annotate(
                f"({x_last_b}, {y_last_b:.2g})",
                xy=(x_last_b, y_last_b),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=score_label_fontsize,
                ha="left",
                va="center",
            )

        (line_res,) = ax_res.plot(
            res_summary.x,
            res_summary.mean,
            label=name,
            alpha=line_alpha,
            linewidth=linewidth,
            color=color,
            linestyle=style.linestyle,
            marker=None,
        )
        color_r = line_res.get_color()
        ax_res.fill_between(
            res_summary.x,
            np.maximum(0.0, res_summary.mean - res_summary.std),
            res_summary.mean + res_summary.std,
            color=color_r,
            alpha=shade_alpha,
            linewidth=0,
        )

        x_last_r = int(res_summary.x[~np.isnan(res_summary.mean)][-1])
        y_last_r = float(res_summary.mean[x_last_r])
        ax_res.scatter(
            x_last_r,
            y_last_r,
            s=90,
            marker=style.marker,
            edgecolor="black",
            linewidth=0.8,
            color=color_r,
            zorder=3,
        )
        if show_score_labels:
            ax_res.annotate(
                f"({x_last_r}, {y_last_r:.2g})",
                xy=(x_last_r, y_last_r),
                xytext=(5, 0),
                textcoords="offset points",
                color=color_r,
                fontsize=score_label_fontsize,
                ha="left",
                va="center",
            )

        beh_final_mean, beh_final_std = final_stats(behavior_runs)
        res_final_mean, res_final_std = final_stats(result_runs)
        mean_final_points.append((beh_final_mean, res_final_mean))

        if show_final_errorbars:
            yerr = np.array(
                [[min(res_final_std, max(0.0, res_final_mean))], [res_final_std]],
                dtype=float,
            )
            ax_scatter.errorbar(
                beh_final_mean,
                res_final_mean,
                xerr=beh_final_std,
                yerr=yerr,
                fmt=style.marker,
                markersize=7,
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                alpha=0.9,
                label=name,
            )
        else:
            ax_scatter.scatter(
                beh_final_mean,
                res_final_mean,
                s=90,
                marker=style.marker,
                color=color,
                label=name,
            )
        if show_score_labels:
            ax_scatter.annotate(
                f"({beh_final_mean:.2g}, {res_final_mean:.2g})",
                xy=(beh_final_mean, res_final_mean),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=score_label_fontsize,
                ha="left",
                va="center",
            )

    if plot_pareto:
        frontier = pareto_frontier(np.asarray(mean_final_points, dtype=float), maximize_y=True)
        if frontier.shape[0] > 0:
            ax_scatter.plot(
                frontier[:, 0],
                frontier[:, 1],
                color=pareto_color,
                linestyle=pareto_linestyle,
                linewidth=pareto_linewidth,
                alpha=pareto_alpha,
                zorder=pareto_zorder,
                label=pareto_label,
            )

    ax_scatter.set_xlabel("Behavior Score")
    ax_scatter.set_ylabel(result_ylabel)
    if result_ylog:
        ax_scatter.set_yscale("log")

    ax_beh.set_xlabel("Number of Rounds")
    ax_beh.set_ylabel(behavior_ylabel)

    ax_res.set_xlabel("Number of Rounds")
    ax_res.set_ylabel(result_ylabel)
    if result_ylog:
        ax_res.set_yscale("log")
    if result_xlog:
        ax_res.set_xscale("log")

    if grid:
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.6)

    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    deduped = dict(zip(labels, handles))
    fig.legend(
        handles=deduped.values(),
        labels=deduped.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=legend_ncol or max(1, len(deduped)),
        fontsize="small",
        frameon=False,
    )

    plt.subplots_adjust(top=0.82)
    return fig, axes


def safe_filename(value: str) -> str:
    """Create a filesystem-safe stem for generated figure names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
