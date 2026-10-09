"""Audit, compare, and visualize empirical-versus-authoritative inferer runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIR_KEYS = ["run_id", "case", "method", "seed", "round"]
METRICS: dict[str, tuple[str, str]] = {
    "adjacency_precision": ("Graph", "higher"),
    "adjacency_recall": ("Graph", "higher"),
    "adjacency_f1": ("Graph", "higher"),
    "directed_precision": ("Graph", "higher"),
    "directed_recall": ("Graph", "higher"),
    "directed_f1": ("Graph", "higher"),
    "shd": ("Graph", "lower"),
    "pehe": ("CATE", "lower"),
    "rmse": ("CATE", "lower"),
    "ate_absolute_error": ("CATE", "lower"),
    "runtime_s": ("Performance", "lower"),
    "cumulative_runtime_s": ("Performance", "lower"),
}
PRIMARY_METRICS = ("shd", "pehe")
MAX_FAILURE_DETAIL = 500
RUNTIME_LOG_THRESHOLD = 20
PROVENANCE_FIELDS = (
    "project",
    "algorithm",
    "version",
    "source",
    "citation",
    "license",
    "adapter_version",
    "implementation_kind",
)


def discover_run_directories(
    inputs: Iterable[Path | str], *, latest: bool = False
) -> list[Path]:
    """Find comparison directories containing a combined JSONL artifact."""
    discovered: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_file() and path.suffix == ".jsonl":
            discovered.add(path.parent.resolve())
        elif (path / "combined_comparison.jsonl").is_file():
            discovered.add(path.resolve())
        elif path.is_dir():
            discovered.update(
                candidate.parent.resolve()
                for candidate in path.rglob("combined_comparison.jsonl")
            )
    ordered = sorted(discovered, key=lambda path: (path.name, path.as_posix()))
    return ordered[-1:] if latest and ordered else ordered


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"Malformed JSON at {path}:{line_number}: {exc}"
            raise ValueError(message) from exc
        if not isinstance(value, dict):
            message = f"Expected an object at {path}:{line_number}"
            raise TypeError(message)
        rows.append(value)
    return rows


def load_records(run_directories: Iterable[Path]) -> pd.DataFrame:
    """Load combined per-run records while retaining artifact identity."""
    rows: list[dict[str, Any]] = []
    for directory in run_directories:
        source = directory / "combined_comparison.jsonl"
        rows.extend(
            [
                {
                    "round": 1,
                    **row,
                    "run_id": directory.name,
                    "source_artifact": source.as_posix(),
                }
                for row in _read_jsonl(source)
            ]
        )
    return pd.DataFrame(rows)


def audit_records(records: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912
    """Return machine-readable integrity, pairing, metric, and provenance findings."""
    findings: list[dict[str, Any]] = []

    def add(
        severity: str, code: str, message: str, row: pd.Series | None = None
    ) -> None:
        findings.append(
            {
                "severity": severity,
                "code": code,
                "run_id": "" if row is None else row.get("run_id", ""),
                "case": "" if row is None else row.get("case", ""),
                "method": "" if row is None else row.get("method", ""),
                "message": message,
            }
        )

    required = {*PAIR_KEYS, "implementation", "status", "runtime_s", "backend"}
    missing_columns = sorted(required - set(records.columns))
    if missing_columns:
        add("error", "missing_columns", f"Missing required columns: {missing_columns}")
        return pd.DataFrame(findings)

    allowed_implementations = {"empirical", "authoritative"}
    allowed_statuses = {"ok", "error"}
    for _, row in records.iterrows():
        if row["implementation"] not in allowed_implementations:
            add("error", "invalid_implementation", str(row["implementation"]), row)
        if row["status"] not in allowed_statuses:
            add("error", "invalid_status", str(row["status"]), row)
        runtime = row.get("runtime_s")
        if (
            not isinstance(runtime, int | float)
            or not math.isfinite(float(runtime))
            or runtime < 0
        ):
            add("error", "invalid_runtime", f"Invalid runtime: {runtime!r}", row)
        for metric in METRICS:
            value = row.get(metric)
            if pd.notna(value) and (
                not isinstance(value, int | float) or not math.isfinite(float(value))
            ):
                add("error", "invalid_metric", f"{metric}={value!r}", row)
        if row["status"] == "ok" and not any(
            pd.notna(row.get(metric)) for metric in METRICS
        ):
            add(
                "error",
                "successful_row_without_metrics",
                "Successful row has no metrics",
                row,
            )
        backend = row.get("backend")
        if row["implementation"] == "authoritative":
            if not isinstance(backend, dict):
                add(
                    "error",
                    "missing_provenance",
                    "Authoritative backend is not an object",
                    row,
                )
            else:
                missing = [
                    field for field in PROVENANCE_FIELDS if not backend.get(field)
                ]
                if missing:
                    add(
                        "error",
                        "incomplete_provenance",
                        f"Missing provenance fields: {missing}",
                        row,
                    )
                if backend.get("implementation_kind") != "authoritative":
                    add(
                        "error",
                        "wrong_provenance_kind",
                        "Expected authoritative provenance",
                        row,
                    )

    duplicate_mask = records.duplicated([*PAIR_KEYS, "implementation"], keep=False)
    for _, row in records.loc[duplicate_mask].iterrows():
        add(
            "error",
            "duplicate_implementation",
            "Duplicate result for one pair side",
            row,
        )

    for _keys, group in records.groupby(PAIR_KEYS, dropna=False):
        implementations = set(group["implementation"])
        missing = allowed_implementations - implementations
        representative = group.iloc[0]
        if missing:
            add(
                "error",
                "incomplete_pair",
                f"Missing pair side(s): {sorted(missing)}",
                representative,
            )
        digests = set(
            group.get("input_digest", pd.Series(dtype=str)).dropna().astype(str)
        )
        if not digests:
            add(
                "warning",
                "missing_input_digest",
                "Cannot prove identical paired inputs",
                representative,
            )
        elif len(digests) != 1:
            add(
                "error",
                "input_digest_mismatch",
                f"Pair has digests: {sorted(digests)}",
                representative,
            )
        statuses = set(group["status"])
        if statuses != {"ok"}:
            add(
                "warning",
                "unavailable_pair",
                f"Pair statuses: {sorted(statuses)}",
                representative,
            )

    if not findings:
        findings.append(
            {
                "severity": "info",
                "code": "audit_passed",
                "run_id": "",
                "case": "",
                "method": "",
                "message": "All artifact, pairing, provenance, and metric checks passed.",
            }
        )
    return pd.DataFrame(findings)


def paired_differences(records: pd.DataFrame) -> pd.DataFrame:
    """Create metric-level rows only where both implementations succeeded."""
    empirical = records.loc[records["implementation"] == "empirical"].copy()
    authoritative = records.loc[records["implementation"] == "authoritative"].copy()
    pairs = empirical.merge(
        authoritative, on=PAIR_KEYS, suffixes=("_empirical", "_authoritative")
    )
    pairs = pairs.loc[
        (pairs["status_empirical"] == "ok") & (pairs["status_authoritative"] == "ok")
    ]
    differences: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        for metric, (family, direction) in METRICS.items():
            empirical_value = pair.get(f"{metric}_empirical")
            authoritative_value = pair.get(f"{metric}_authoritative")
            if pd.isna(empirical_value) or pd.isna(authoritative_value):
                continue
            empirical_number = float(empirical_value)
            authoritative_number = float(authoritative_value)
            raw_delta = authoritative_number - empirical_number
            improvement = raw_delta if direction == "higher" else -raw_delta
            percent_change = (
                100.0 * raw_delta / abs(empirical_number)
                if not np.isclose(empirical_number, 0.0)
                else np.nan
            )
            differences.append(
                {
                    **{key: pair[key] for key in PAIR_KEYS},
                    "metric": metric,
                    "family": family,
                    "direction": direction,
                    "empirical": empirical_number,
                    "authoritative": authoritative_number,
                    "authoritative_minus_empirical": raw_delta,
                    "improvement": improvement,
                    "percent_change": percent_change,
                    "winner": (
                        "authoritative"
                        if improvement > 0
                        else "empirical" if improvement < 0 else "tie"
                    ),
                }
            )
    return pd.DataFrame(differences)


def summarize_pairs(
    differences: pd.DataFrame,
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 911,
) -> pd.DataFrame:
    """Summarize paired changes with deterministic percentile bootstrap intervals."""
    if differences.empty:
        return pd.DataFrame()
    final_differences = (
        differences.sort_values("round")
        .groupby(["run_id", "case", "method", "seed", "metric"], as_index=False)
        .tail(1)
    )
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, Any]] = []
    for (method, metric), group in final_differences.groupby(["method", "metric"]):
        values = group["improvement"].to_numpy(dtype=float)
        if len(values) == 1:
            ci_low = ci_high = float(values[0])
        else:
            samples = rng.choice(
                values, size=(bootstrap_resamples, len(values)), replace=True
            )
            bootstrap_means = samples.mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        summaries.append(
            {
                "method": method,
                "metric": metric,
                "family": group["family"].iloc[0],
                "direction": group["direction"].iloc[0],
                "n_pairs": len(group),
                "mean_empirical": group["empirical"].mean(),
                "mean_authoritative": group["authoritative"].mean(),
                "mean_improvement": values.mean(),
                "median_improvement": np.median(values),
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "authoritative_win_rate": float((values > 0).mean()),
                "empirical_win_rate": float((values < 0).mean()),
                "tie_rate": float((values == 0).mean()),
            }
        )
    return pd.DataFrame(summaries).sort_values(["family", "method", "metric"])


def availability_table(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize execution coverage and failure rates by implementation and method."""
    table = (
        records.assign(success=records["status"].eq("ok"))
        .groupby(["method", "implementation"], as_index=False)
        .agg(
            runs=("status", "size"),
            successes=("success", "sum"),
            mean_runtime_s=("runtime_s", "mean"),
        )
    )
    table["failures"] = table["runs"] - table["successes"]
    table["success_rate"] = table["successes"] / table["runs"]
    return table


def failure_table(records: pd.DataFrame) -> pd.DataFrame:
    """Count normalized failure messages without discarding backend identity."""
    failures = records.loc[records["status"] != "ok"].copy()
    if failures.empty:
        return pd.DataFrame(columns=["method", "implementation", "error", "count"])
    if "error" not in failures:
        failures["error"] = "Unspecified failure"
    else:
        failures["error"] = failures["error"].fillna("Unspecified failure")
    return (
        failures.groupby(["method", "implementation", "error"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )


def provenance_table(records: pd.DataFrame) -> pd.DataFrame:
    """Flatten and deduplicate backend provenance records."""
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        backend = row.get("backend")
        if not isinstance(backend, dict):
            continue
        rows.append(
            {
                "method": row["method"],
                "implementation": row["implementation"],
                **{field: backend.get(field, "") for field in PROVENANCE_FIELDS},
            }
        )
    return (
        pd.DataFrame(rows).drop_duplicates().sort_values(["method", "implementation"])
    )


def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _message_figure(title: str, message: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.axis("off")
    axis.set_title(title, fontweight="bold")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return figure


def _plot_availability(availability: pd.DataFrame) -> plt.Figure:
    pivot = availability.pivot(
        index="method", columns="implementation", values="success_rate"
    )
    pivot = pivot.reindex(columns=["empirical", "authoritative"])
    figure, axis = plt.subplots(figsize=(8, max(4, 0.55 * len(pivot))))
    image = axis.imshow(
        pivot.fillna(0).to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"
    )
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            label = "NA" if pd.isna(value) else f"{value:.0%}"
            axis.text(column_index, row_index, label, ha="center", va="center")
    axis.set_title("Backend execution success rate")
    figure.colorbar(image, ax=axis, label="Successful fraction")
    return figure


def _plot_improvement(summary: pd.DataFrame) -> plt.Figure:
    quality = summary.loc[summary.get("family", pd.Series(dtype=str)) != "Performance"]
    if quality.empty:
        return _message_figure(
            "Paired empirical-versus-authoritative improvements",
            "No case has successful results from both implementations.\n"
            "Inspect backend_failures.csv and execution_coverage.png before interpreting quality.",
        )
    pivot = quality.pivot(index="method", columns="metric", values="mean_improvement")
    limit = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    figure, axis = plt.subplots(
        figsize=(max(9, 0.85 * len(pivot.columns)), max(4, 0.6 * len(pivot)))
    )
    image = axis.imshow(
        pivot.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
    )
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            if pd.notna(value):
                axis.text(
                    column_index, row_index, f"{value:.3g}", ha="center", va="center"
                )
    axis.set_title("Mean paired improvement (positive favors authoritative)")
    figure.colorbar(image, ax=axis, label="Direction-normalized improvement")
    return figure


def _plot_primary_scatter(differences: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        subset = differences.loc[
            differences.get("metric", pd.Series(dtype=str)) == metric
        ]
        if subset.empty:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"No paired {metric.upper()} results",
                ha="center",
                va="center",
            )
            continue
        for method, group in subset.groupby("method"):
            axis.scatter(
                group["empirical"], group["authoritative"], label=method, alpha=0.8
            )
        low = min(subset["empirical"].min(), subset["authoritative"].min())
        high = max(subset["empirical"].max(), subset["authoritative"].max())
        if np.isclose(low, high):
            high = low + 1.0
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        axis.set_xlabel("Empirical")
        axis.set_ylabel("Authoritative")
        axis.set_title(f"Paired {metric.upper()} (lower is better)")
        axis.legend(fontsize=8)
    return figure


def _plot_runtime(differences: pd.DataFrame) -> plt.Figure:
    subset = differences.loc[
        differences.get("metric", pd.Series(dtype=str)) == "runtime_s"
    ].copy()
    if subset.empty:
        return _message_figure(
            "Runtime ratio", "No successful paired runtimes are available."
        )
    positive = subset.loc[
        (subset["empirical"] > 0) & (subset["authoritative"] > 0)
    ].copy()
    positive["log2_ratio"] = np.log2(positive["authoritative"] / positive["empirical"])
    grouped = positive.groupby("method")["log2_ratio"].mean().sort_values()
    figure, axis = plt.subplots(figsize=(8, max(4, 0.55 * len(grouped))))
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in grouped]
    axis.barh(grouped.index, grouped.values, color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("log2(authoritative runtime / empirical runtime)")
    axis.set_title("Paired runtime ratio (right means authoritative is slower)")
    return figure


def _plot_case_improvement(differences: pd.DataFrame) -> plt.Figure:
    subset = differences.loc[
        differences.get("metric", pd.Series(dtype=str)).isin(PRIMARY_METRICS)
    ]
    if subset.empty:
        return _message_figure(
            "Case-level quality changes",
            "No paired primary-metric results are available.",
        )
    subset = subset.assign(
        label=subset["method"].astype(str) + " / " + subset["metric"].astype(str)
    )
    pivot = subset.pivot_table(
        index="case", columns="label", values="improvement", aggfunc="mean"
    )
    limit = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    figure, axis = plt.subplots(
        figsize=(max(9, 0.9 * len(pivot.columns)), max(4, 0.55 * len(pivot)))
    )
    image = axis.imshow(
        pivot.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
    )
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title(
        "Case-level primary-metric improvement (positive favors authoritative)"
    )
    figure.colorbar(image, ax=axis)
    return figure


def _plot_failures(failures: pd.DataFrame) -> plt.Figure:
    if failures.empty:
        return _message_figure("Backend failures", "No backend failures were recorded.")
    grouped = (
        failures.groupby(["method", "implementation"])["count"].sum().sort_values()
    )
    labels = [" / ".join(index) for index in grouped.index]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.5 * len(grouped))))
    axis.barh(labels, grouped.values, color="#d95f02")
    axis.set_xlabel("Failed executions")
    axis.set_title("Backend failure counts")
    return figure


def _plot_absolute_overview(records: pd.DataFrame) -> plt.Figure:
    metrics = ["adjacency_f1", "directed_f1", "shd", "pehe", "ate_absolute_error"]
    successful = records.loc[records["status"] == "ok"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, metric in zip(axes.flat, metrics, strict=False):
        values = (
            successful.dropna(subset=[metric])
            if metric in successful
            else pd.DataFrame()
        )
        if values.empty:
            axis.axis("off")
            axis.text(0.5, 0.5, f"No {metric} values", ha="center", va="center")
            continue
        grouped = values.groupby(["method", "implementation"])[metric].mean()
        labels = [
            f"{method}\n{implementation[:4]}."
            for method, implementation in grouped.index
        ]
        axis.bar(
            range(len(grouped)),
            grouped.values,
            color=[
                "#7570b3" if key[1] == "empirical" else "#1b9e77"
                for key in grouped.index
            ],
        )
        axis.set_xticks(
            range(len(grouped)), labels, rotation=40, ha="right", fontsize=8
        )
        axis.set_title(metric.replace("_", " ").title())
    axes.flat[-1].axis("off")
    figure.suptitle(
        "Descriptive metric overview (not necessarily paired)", fontweight="bold"
    )
    return figure


def _plot_inferer_side_by_side(  # noqa: PLR0915
    method: str, records: pd.DataFrame, differences: pd.DataFrame
) -> plt.Figure:
    """Show actual paired values, case-level primary scores, and runtimes for one inferer."""
    paired = differences.loc[
        differences.get("method", pd.Series(dtype=str)) == method
    ].copy()
    if paired.empty:
        method_records = records.loc[records["method"] == method]
        failures = method_records.loc[method_records["status"] != "ok"]
        detail = "; ".join(
            sorted(
                {
                    f"{row['implementation']}: {row.get('error', 'failed')}"
                    for _, row in failures.iterrows()
                }
            )
        )
        if len(detail) > MAX_FAILURE_DETAIL:
            detail = detail[: MAX_FAILURE_DETAIL - 3] + "..."
        return _message_figure(
            f"{method}: empirical vs. authoritative",
            "No successful paired cases are available.\n" + detail,
        )

    paired = (
        paired.sort_values("round")
        .groupby(["run_id", "case", "method", "seed", "metric"], as_index=False)
        .tail(1)
    )

    family = str(paired["family"].iloc[0])
    primary_metric = "shd" if family == "Graph" else "pehe"
    quality_metrics = (
        ["adjacency_f1", "directed_f1", "shd"]
        if family == "Graph"
        else ["pehe", "rmse", "ate_absolute_error"]
    )
    available_metrics = [
        metric for metric in quality_metrics if metric in set(paired["metric"])
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    colors = {"empirical": "#7570b3", "authoritative": "#1b9e77"}

    means = paired.loc[paired["metric"].isin(available_metrics)].groupby("metric")
    positions = np.arange(len(available_metrics), dtype=float)
    width = 0.36
    empirical_means = [
        means.get_group(metric)["empirical"].mean() for metric in available_metrics
    ]
    authoritative_means = [
        means.get_group(metric)["authoritative"].mean() for metric in available_metrics
    ]
    empirical_bars = axes[0].bar(
        positions - width / 2,
        empirical_means,
        width,
        label="Empirical",
        color=colors["empirical"],
    )
    authoritative_bars = axes[0].bar(
        positions + width / 2,
        authoritative_means,
        width,
        label="Authoritative",
        color=colors["authoritative"],
    )
    labels: list[str] = []
    for metric in available_metrics:
        group = means.get_group(metric)
        improvement = group["improvement"].mean()
        winner = (
            "A better" if improvement > 0 else "E better" if improvement < 0 else "tie"
        )
        labels.append(f"{metric}\n{winner}")
    axes[0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Mean paired value (native metric scale)")
    axes[0].set_title("Quality metrics")
    axes[0].legend()
    axes[0].bar_label(empirical_bars, fmt="%.3g", padding=2, fontsize=8)
    axes[0].bar_label(authoritative_bars, fmt="%.3g", padding=2, fontsize=8)

    primary = paired.loc[paired["metric"] == primary_metric].sort_values(
        ["case", "seed"]
    )
    case_labels = [
        f"{case}\nseed {seed}"
        for case, seed in zip(primary["case"], primary["seed"], strict=True)
    ]
    case_positions = np.arange(len(primary), dtype=float)
    empirical_primary_bars = axes[1].bar(
        case_positions - width / 2,
        primary["empirical"],
        width,
        color=colors["empirical"],
    )
    authoritative_primary_bars = axes[1].bar(
        case_positions + width / 2,
        primary["authoritative"],
        width,
        color=colors["authoritative"],
    )
    axes[1].set_xticks(case_positions, case_labels, rotation=30, ha="right")
    axes[1].set_ylabel(primary_metric.upper())
    axes[1].set_title(f"Case-level {primary_metric.upper()} (lower is better)")
    axes[1].bar_label(empirical_primary_bars, fmt="%.3g", padding=2, fontsize=8)
    axes[1].bar_label(authoritative_primary_bars, fmt="%.3g", padding=2, fontsize=8)

    runtime = paired.loc[paired["metric"] == "runtime_s"].sort_values(["case", "seed"])
    runtime_positions = np.arange(len(runtime), dtype=float)
    empirical_runtime_bars = axes[2].bar(
        runtime_positions - width / 2,
        runtime["empirical"],
        width,
        color=colors["empirical"],
    )
    authoritative_runtime_bars = axes[2].bar(
        runtime_positions + width / 2,
        runtime["authoritative"],
        width,
        color=colors["authoritative"],
    )
    axes[2].set_xticks(
        runtime_positions,
        [
            f"{case}\nseed {seed}"
            for case, seed in zip(runtime["case"], runtime["seed"], strict=True)
        ],
        rotation=30,
        ha="right",
    )
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Runtime (lower is better)")
    axes[2].bar_label(empirical_runtime_bars, fmt="%.3g", padding=2, fontsize=8)
    axes[2].bar_label(authoritative_runtime_bars, fmt="%.3g", padding=2, fontsize=8)
    positive_runtimes = runtime.loc[
        (runtime["empirical"] > 0) & (runtime["authoritative"] > 0),
        ["empirical", "authoritative"],
    ].to_numpy()
    if (
        positive_runtimes.size
        and positive_runtimes.max() / positive_runtimes.min() > RUNTIME_LOG_THRESHOLD
    ):
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Seconds (log scale)")

    note = (
        "PC note: directed scores treat unresolved CPDAG edges as unoriented; "
        "CPDAG-aware SHD is the primary structural comparison."
        if method == "pc"
        else (
            "A = authoritative; E = empirical. Authoritative code is a reference "
            "implementation, not a guarantee of better estimates."
        )
    )
    figure.suptitle(f"{method}: empirical vs. authoritative", fontweight="bold")
    figure.text(0.5, 0.01, note, ha="center", fontsize=9)
    return figure


def _plot_round_trajectory(
    method: str,
    case: str,
    records: pd.DataFrame,
    differences: pd.DataFrame,
) -> plt.Figure:
    """Plot empirical and authoritative behavior over buffered observation rounds."""
    paired = differences.loc[
        (differences.get("method", pd.Series(dtype=str)) == method)
        & (differences.get("case", pd.Series(dtype=str)) == case)
    ].copy()
    if paired.empty:
        case_records = records.loc[
            (records["method"] == method) & (records["case"] == case)
        ]
        statuses = (
            case_records.groupby("implementation")["status"]
            .apply(
                lambda values: f"{int(values.eq('ok').sum())}/{len(values)} successful"
            )
            .to_dict()
        )
        return _message_figure(
            f"{method} / {case}: round trajectories",
            "No round has successful results from both implementations.\n"
            + "; ".join(f"{name}: {value}" for name, value in statuses.items()),
        )

    family = str(paired.loc[paired["family"] != "Performance", "family"].iloc[0])
    primary_metric = "shd" if family == "Graph" else "pehe"
    secondary_metrics = (
        ["adjacency_f1", "directed_f1"]
        if family == "Graph"
        else ["rmse", "ate_absolute_error"]
    )
    colors = {"empirical": "#7570b3", "authoritative": "#1b9e77"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    primary = paired.loc[paired["metric"] == primary_metric]
    for implementation in ("empirical", "authoritative"):
        values = primary.groupby("round")[implementation].mean().sort_index()
        axes[0, 0].plot(
            values.index,
            values.values,
            marker="o",
            linewidth=2,
            color=colors[implementation],
            label=implementation.title(),
        )
    axes[0, 0].set_ylabel(primary_metric.upper())
    axes[0, 0].set_title(f"{primary_metric.upper()} (lower is better)")
    axes[0, 0].legend()

    line_styles = {secondary_metrics[0]: "-", secondary_metrics[1]: "--"}
    for metric in secondary_metrics:
        metric_rows = paired.loc[paired["metric"] == metric]
        for implementation in ("empirical", "authoritative"):
            values = metric_rows.groupby("round")[implementation].mean().sort_index()
            axes[0, 1].plot(
                values.index,
                values.values,
                marker="o",
                linestyle=line_styles[metric],
                color=colors[implementation],
                label=f"{implementation.title()} / {metric}",
            )
    axes[0, 1].set_ylabel("Metric value")
    axes[0, 1].set_title("Secondary quality metrics")
    axes[0, 1].legend(fontsize=8)

    improvement = primary.groupby("round")["improvement"].mean().sort_index()
    axes[1, 0].plot(
        improvement.index,
        improvement.values,
        marker="o",
        linewidth=2,
        color="#2166ac",
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].fill_between(
        improvement.index,
        0,
        improvement.values,
        where=improvement.values >= 0,
        color="#1b9e77",
        alpha=0.2,
    )
    axes[1, 0].fill_between(
        improvement.index,
        0,
        improvement.values,
        where=improvement.values < 0,
        color="#7570b3",
        alpha=0.2,
    )
    axes[1, 0].set_ylabel("Direction-normalized improvement")
    axes[1, 0].set_title("Primary-metric delta (+ favors authoritative)")

    runtime = paired.loc[paired["metric"] == "runtime_s"]
    runtime_values: list[float] = []
    for implementation in ("empirical", "authoritative"):
        values = runtime.groupby("round")[implementation].mean().sort_index()
        runtime_values.extend(values.loc[values > 0].tolist())
        axes[1, 1].plot(
            values.index,
            values.values,
            marker="o",
            linewidth=2,
            color=colors[implementation],
            label=implementation.title(),
        )
    if (
        runtime_values
        and max(runtime_values) / min(runtime_values) > RUNTIME_LOG_THRESHOLD
    ):
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_ylabel("Seconds (log scale)")
    else:
        axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].set_title("Per-round update + inference runtime")
    axes[1, 1].legend()

    rounds = sorted(paired["round"].dropna().unique())
    for axis in axes[1, :]:
        axis.set_xlabel("Round")
    for axis in axes.flat:
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"{method} / {case}: behavior across cumulative observation rounds",
        fontweight="bold",
    )
    return figure


def _plot_round_vs_all_metrics(
    method: str,
    case: str,
    records: pd.DataFrame,
    differences: pd.DataFrame,
) -> plt.Figure:
    """Plot round against every paired metric available for one method and case."""
    paired = differences.loc[
        (differences.get("method", pd.Series(dtype=str)) == method)
        & (differences.get("case", pd.Series(dtype=str)) == case)
    ].copy()
    if paired.empty:
        return _plot_round_trajectory(method, case, records, differences)

    metrics = [metric for metric in METRICS if metric in set(paired["metric"])]
    column_count = 3
    row_count = math.ceil(len(metrics) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, 4 * row_count),
        squeeze=False,
        sharex=True,
    )
    colors = {"empirical": "#7570b3", "authoritative": "#1b9e77"}
    rounds = sorted(paired["round"].dropna().unique())
    for axis, metric in zip(axes.flat, metrics, strict=False):
        metric_rows = paired.loc[paired["metric"] == metric]
        plotted_values: list[float] = []
        for implementation in ("empirical", "authoritative"):
            values = metric_rows.groupby("round")[implementation].mean().sort_index()
            plotted_values.extend(values.loc[values > 0].tolist())
            axis.plot(
                values.index,
                values.values,
                marker="o",
                linewidth=2,
                color=colors[implementation],
                label=implementation.title(),
            )
        direction = METRICS[metric][1]
        axis.set_title(f"{metric.replace('_', ' ').title()} ({direction} is better)")
        axis.set_xlabel("Round")
        axis.set_ylabel(metric)
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
        if (
            metric in {"runtime_s", "cumulative_runtime_s"}
            and plotted_values
            and max(plotted_values) / min(plotted_values) > RUNTIME_LOG_THRESHOLD
        ):
            axis.set_yscale("log")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    figure.suptitle(
        f"{method} / {case}: round versus all paired metrics",
        fontweight="bold",
    )
    return figure


def generate_figures(  # noqa: PLR0913, PLR0917
    records: pd.DataFrame,
    differences: pd.DataFrame,
    summary: pd.DataFrame,
    availability: pd.DataFrame,
    failures: pd.DataFrame,
    directory: Path,
) -> list[str]:
    """Generate the complete PNG/PDF comparison figure set."""
    base_figures = {
        "execution_coverage": _plot_availability(availability),
        "paired_improvement_heatmap": _plot_improvement(summary),
        "primary_metric_scatter": _plot_primary_scatter(differences),
        "runtime_ratio": _plot_runtime(differences),
        "case_improvement_heatmap": _plot_case_improvement(differences),
        "backend_failures": _plot_failures(failures),
        "descriptive_metric_overview": _plot_absolute_overview(records),
    }
    figure_names: list[str] = []
    for stem, figure in base_figures.items():
        _save_figure(figure, directory, stem)
        figure_names.append(stem)
    for method in sorted(records["method"].dropna().astype(str).unique()):
        safe_method = re.sub(r"[^a-zA-Z0-9_-]+", "_", method).strip("_")
        side_by_side_stem = f"inferer_{safe_method}_side_by_side"
        _save_figure(
            _plot_inferer_side_by_side(method, records, differences),
            directory,
            side_by_side_stem,
        )
        figure_names.append(side_by_side_stem)
        cases = sorted(
            records.loc[records["method"] == method, "case"]
            .dropna()
            .astype(str)
            .unique()
        )
        for case in cases:
            safe_case = re.sub(r"[^a-zA-Z0-9_-]+", "_", case).strip("_")
            stem = f"inferer_{safe_method}_{safe_case}_round_trajectories"
            _save_figure(
                _plot_round_trajectory(method, case, records, differences),
                directory,
                stem,
            )
            figure_names.append(stem)
            all_metrics_stem = f"inferer_{safe_method}_{safe_case}_round_vs_all_metrics"
            _save_figure(
                _plot_round_vs_all_metrics(method, case, records, differences),
                directory,
                all_metrics_stem,
            )
            figure_names.append(all_metrics_stem)
    return figure_names


def inspect_figure_artifacts(directory: Path, figures: Iterable[str]) -> pd.DataFrame:
    """Verify every rendered PNG/PDF and record reproducibility metadata."""
    rows: list[dict[str, Any]] = []
    for stem in figures:
        for file_format in ("png", "pdf"):
            path = directory / f"{stem}.{file_format}"
            content = path.read_bytes() if path.is_file() else b""
            width = height = np.nan
            valid = bool(content)
            validation = "non-empty file"
            if valid and file_format == "png":
                try:
                    pixels = plt.imread(path)
                    height, width = pixels.shape[:2]
                    validation = "decoded PNG"
                except (OSError, ValueError, SyntaxError):
                    valid = False
                    validation = "PNG decode failed"
            elif valid and file_format == "pdf":
                valid = content.startswith(b"%PDF-") and content.rstrip().endswith(
                    b"%%EOF"
                )
                validation = "PDF header/trailer" if valid else "invalid PDF framing"
            rows.append(
                {
                    "figure": stem,
                    "format": file_format,
                    "path": path.as_posix(),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest() if content else "",
                    "width_px": width,
                    "height_px": height,
                    "valid": valid,
                    "validation": validation,
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(
    frame: pd.DataFrame, columns: list[str], *, limit: int = 30
) -> list[str]:
    if frame.empty:
        return ["_No rows available._"]
    selected = frame.loc[:, [column for column in columns if column in frame]].head(
        limit
    )
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in selected.iterrows():
        values = []
        for column in headers:
            value = row[column]
            values.append(
                f"{value:.4g}"
                if isinstance(value, float)
                else str(value).replace("|", "\\|")
            )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_analysis_report(  # noqa: PLR0913
    directory: Path,
    *,
    run_directories: list[Path],
    records: pd.DataFrame,
    audit: pd.DataFrame,
    differences: pd.DataFrame,
    summary: pd.DataFrame,
    availability: pd.DataFrame,
    failures: pd.DataFrame,
    provenance: pd.DataFrame,
    figure_artifacts: pd.DataFrame,
    figures: list[str],
) -> None:
    """Write CSV/JSON audit products and a self-contained Markdown interpretation."""
    export_records = records.copy()
    if "backend" in export_records:
        export_records["backend"] = export_records["backend"].map(
            lambda value: (
                json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            )
        )
    export_records.to_csv(directory / "normalized_records.csv", index=False)
    audit.to_csv(directory / "artifact_audit.csv", index=False)
    differences.to_csv(directory / "paired_differences.csv", index=False)
    summary.to_csv(directory / "paired_summary.csv", index=False)
    availability.to_csv(directory / "execution_coverage.csv", index=False)
    failures.to_csv(directory / "backend_failures.csv", index=False)
    provenance.to_csv(directory / "backend_provenance.csv", index=False)
    figure_artifacts.to_csv(directory / "figure_artifacts.csv", index=False)

    successful_round_pairs = (
        differences[PAIR_KEYS].drop_duplicates().shape[0]
        if not differences.empty
        else 0
    )
    successful_cases = (
        differences[["run_id", "case", "method", "seed"]].drop_duplicates().shape[0]
        if not differences.empty
        else 0
    )
    error_count = int((audit["severity"] == "error").sum()) if not audit.empty else 0
    warning_count = (
        int((audit["severity"] == "warning").sum()) if not audit.empty else 0
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_directories": [path.as_posix() for path in run_directories],
        "record_count": len(records),
        "successful_case_count": successful_cases,
        "successful_round_pair_count": successful_round_pairs,
        "audit_errors": error_count,
        "audit_warnings": warning_count,
        "figures": figures,
        "figure_artifacts_valid": bool(figure_artifacts["valid"].all()),
    }
    (directory / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Empirical vs. authoritative inferer analysis",
        "",
        "## Interpretation guardrail",
        "",
        (
            f"This report found **{successful_cases} paired cases** containing "
            f"**{successful_round_pairs} successful paired round-results**. "
            "Quality deltas use only successful pairs; descriptive plots may include "
            "unpaired successful runs."
        ),
        (
            "Positive `mean_improvement` always favors the authoritative implementation "
            "after accounting for whether a metric is higher- or lower-is-better."
        ),
        "",
        "## Round trajectory contract",
        "",
        (
            "Round `r` evaluates each persistent inferer after it receives the same new sample "
            "chunk and retains chunks from rounds `1..r`. Trajectory figures therefore show "
            "behavior as the cumulative evidence grows, not independent resamples."
        ),
        (
            "Per-round runtime measures that round's update plus inference call. Summary quality "
            "tables use the final successful round of each case; the trajectory figures retain "
            "all successful paired rounds."
        ),
        "",
        "## Artifact audit",
        "",
        f"- Input run directories: {len(run_directories)}",
        f"- Records: {len(records)}",
        f"- Errors: {error_count}",
        f"- Warnings: {warning_count}",
        "",
        *_markdown_table(
            audit, ["severity", "code", "run_id", "case", "method", "message"]
        ),
        "",
        "## Execution coverage",
        "",
        *_markdown_table(
            availability,
            [
                "method",
                "implementation",
                "runs",
                "successes",
                "failures",
                "success_rate",
                "mean_runtime_s",
            ],
        ),
        "",
        "## Paired quality and runtime summary",
        "",
        *_markdown_table(
            summary,
            [
                "family",
                "method",
                "metric",
                "n_pairs",
                "mean_empirical",
                "mean_authoritative",
                "mean_improvement",
                "improvement_ci_low",
                "improvement_ci_high",
                "authoritative_win_rate",
            ],
        ),
        "",
        "## How to interpret an authoritative loss",
        "",
        (
            "`Authoritative` means the adapter invokes the pinned upstream implementation; "
            "it does **not** mean that implementation must score better on every finite sample."
        ),
        "",
        (
            "Differences can come from sampling variation, different defaults and regularization, "
            "algorithm assumptions, conservative outputs, or tuning embedded in the "
            "empirical code. "
            "The paired confidence intervals and case-level panels show how much evidence the run "
            "actually provides."
        ),
        "",
        (
            "For PC, use CPDAG-aware SHD as the primary result. The authoritative PC "
            "implementation can leave directions unresolved when they are not identifiable "
            "from observational data. "
            "Its directed precision/recall can therefore be lower while adjacency and CPDAG-aware "
            "SHD are equal; that is conservatism, not necessarily an inferior graph estimate."
        ),
        "",
        *_markdown_table(
            summary.loc[
                summary.get("metric", pd.Series(dtype=str)).isin(PRIMARY_METRICS)
            ],
            [
                "method",
                "metric",
                "n_pairs",
                "mean_empirical",
                "mean_authoritative",
                "mean_improvement",
                "improvement_ci_low",
                "improvement_ci_high",
                "authoritative_win_rate",
                "empirical_win_rate",
                "tie_rate",
            ],
        ),
        "",
        "## Backend failures",
        "",
        *_markdown_table(failures, ["method", "implementation", "count", "error"]),
        "",
        "## Backend provenance",
        "",
        *_markdown_table(
            provenance,
            [
                "method",
                "implementation",
                "project",
                "algorithm",
                "version",
                "license",
                "adapter_version",
            ],
        ),
        "",
        "## Figures",
        "",
        f"All rendered files valid: **{bool(figure_artifacts['valid'].all())}**.",
        "See `figure_artifacts.csv` for dimensions, sizes, checksums, and validation results.",
        "",
    ]
    lines.extend(
        f"- [{stem}.png]({stem}.png) / [{stem}.pdf]({stem}.pdf)" for stem in figures
    )
    lines.extend(
        [
            "",
            "## Files for further analysis",
            "",
            "- `normalized_records.csv`: every loaded execution.",
            (
                "- `paired_differences.csv`: case-level paired metric values and "
                "direction-normalized deltas."
            ),
            "- `paired_summary.csv`: bootstrap intervals and empirical/authoritative win rates.",
            "- `execution_coverage.csv` and `backend_failures.csv`: availability diagnostics.",
            "- `artifact_audit.csv`: pairing, digest, provenance, and numerical-integrity checks.",
        ]
    )
    (directory / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_comparisons(
    inputs: Iterable[Path | str],
    output: Path,
    *,
    latest: bool = False,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 911,
) -> dict[str, Any]:
    """Run the complete artifact audit, paired analysis, exports, and figure pipeline."""
    run_directories = discover_run_directories(inputs, latest=latest)
    if not run_directories:
        message = "No comparison run containing combined_comparison.jsonl was found."
        raise FileNotFoundError(message)
    output.mkdir(parents=True, exist_ok=False)
    records = load_records(run_directories)
    if records.empty:
        message = "Comparison artifacts contain no records."
        raise ValueError(message)
    audit = audit_records(records)
    differences = paired_differences(records)
    summary = summarize_pairs(
        differences,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    availability = availability_table(records)
    failures = failure_table(records)
    provenance = provenance_table(records)
    figures = generate_figures(
        records, differences, summary, availability, failures, output
    )
    figure_artifacts = inspect_figure_artifacts(output, figures)
    invalid_figures = figure_artifacts.loc[~figure_artifacts["valid"]]
    if not invalid_figures.empty:
        audit = pd.concat(
            [
                audit,
                pd.DataFrame(
                    {
                        "severity": "error",
                        "code": "invalid_figure_artifact",
                        "run_id": "",
                        "case": "",
                        "method": "",
                        "message": invalid_figures["path"],
                    }
                ),
            ],
            ignore_index=True,
        )
    write_analysis_report(
        output,
        run_directories=run_directories,
        records=records,
        audit=audit,
        differences=differences,
        summary=summary,
        availability=availability,
        failures=failures,
        provenance=provenance,
        figure_artifacts=figure_artifacts,
        figures=figures,
    )
    return {
        "output": output,
        "records": records,
        "audit": audit,
        "differences": differences,
        "summary": summary,
        "availability": availability,
        "failures": failures,
        "provenance": provenance,
        "figure_artifacts": figure_artifacts,
    }


def main() -> None:
    """CLI entrypoint for comprehensive comparison post-processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="*", type=Path, default=[Path("runs/comparisons")]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--latest", action="store_true", help="Analyze only the newest discovered run"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=911)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero after reporting audit errors",
    )
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("post_processing/comparison_reports") / timestamp
    result = analyze_comparisons(
        args.inputs,
        output,
        latest=args.latest,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(output)
    if args.strict and (result["audit"]["severity"] == "error").any():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
