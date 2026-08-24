"""Publication-grade figures and evidence tables for the rebuttal notebook."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from post_processing.rebuttal_utils import (
DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    MISSION_ERROR_LABELS,
    align_run_to_budgets,
    anytime_auc,
    bootstrap_mean_ci,
    display_name,
    group_runs,
    interaction_difference,
    paired_effect,
    pareto_lower,
    trajectory_instability,
)

WHOLE_TRAJECTORY_RUN_COUNT = 20

COLORS = {
    "Cho2016ActiveGBN": "#0072B2",
    "ABCI_Balanced": "#D55E00",
    "ABCI_Fast": "#E69F00",
    "OptimalEffectDesign_Balanced": "#009E73",
    "OptimalEffectDesign_InterventionHeavy": "#CC79A7",
    "OptimalEffectDesign_Fast": "#56B4E9",
}
INFERER_MARKERS = {
    "pc": "o",
    "lingam": "s",
    "notears": "^",
    "dag_discovery": "D",
    "honest_causal_tree": "o",
}


def configure_matplotlib() -> None:
    """Set a compact, colorblind-safe paper style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, output_dir: Path | str, stem: str) -> list[Path]:
    """Export one figure as vector PDF and high-resolution PNG."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{stem}.pdf", output / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=300)
    return paths


def _last_value(run: Sequence[Mapping[str, Any]], key: str) -> float:
    for record in reversed(run):
        value = record.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return math.nan


def _by_seed(
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    mission: str,
    inferer: str,
    agent: str,
    *,
    metric: str = "final_error",
    x_key: str = "total_cumulative",
    max_budget: float | None = None,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for (run_mission, run_inferer, run_agent, seed), run in grouped.items():
        if (run_mission, run_inferer, run_agent) != (mission, inferer, agent):
            continue
        if metric == "final_error":
            value = _last_value(run, "error")
        elif metric == "auc":
            value = anytime_auc(run, x_key=x_key, max_budget=max_budget)
        else:
            value = _last_value(run, metric)
        if math.isfinite(value):
            values[seed] = value
    return values


def _curve_summary(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    budgets: Sequence[float],
    *,
    x_key: str,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> tuple[list[float], list[float], list[float]]:
    import numpy as np

    aligned = np.asarray(
        [
            [
                np.nan if value is None else float(value)
                for value in align_run_to_budgets(run, budgets, x_key=x_key)
            ]
            for run in runs
        ],
        dtype=float,
    )
    means = np.nanmean(aligned, axis=0)
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    bootstrap = np.empty((resamples, aligned.shape[1]), dtype=float)
    for start in range(0, resamples, 500):
        stop = min(resamples, start + 500)
        indices = rng.integers(0, aligned.shape[0], size=(stop - start, aligned.shape[0]))
        bootstrap[start:stop] = np.nanmean(aligned[indices], axis=1)
    lower, upper = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0)
    return means.tolist(), lower.tolist(), upper.tolist()


def _selected_runs(
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    mission: str,
    inferer: str,
    agent: str,
) -> list[Sequence[Mapping[str, Any]]]:
    return [
        run
        for (run_mission, run_inferer, run_agent, _seed), run in sorted(grouped.items())
        if (run_mission, run_inferer, run_agent) == (mission, inferer, agent)
    ]


def _matched_graph_seeds(
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    count: int = WHOLE_TRAJECTORY_RUN_COUNT,
) -> tuple[int, ...]:
    """Select a deterministic common graph-discovery seed subset."""
    seed_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for (mission, inferer, agent, seed) in grouped:
        if mission == "graph_discovery":
            seed_sets[(inferer, agent)].add(seed)
    if not seed_sets:
        return ()
    common = set.intersection(*seed_sets.values())
    if len(common) < count:
        raise RuntimeError(
            f"Whole-trajectory analysis requires {count} common seeds; "
            f"found {len(common)}."
        )
    return tuple(sorted(common)[:count])


def figure_decider_inferer_interaction(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
) -> tuple[list[Path], dict[str, Any]]:
    """Create the primary matched-seed graph interaction figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    grouped = group_runs(records)
    agents = ["Cho2016ActiveGBN", "ABCI_Balanced"]
    inferers = ["pc", "lingam"]
    final = {
        (inferer, agent): _by_seed(
            grouped, "graph_discovery", inferer, agent
        )
        for inferer in inferers
        for agent in agents
    }
    if any(len(values) < 2 for values in final.values()):
        raise RuntimeError("Headline graph interaction requires matched PC and LiNGAM seeds.")

    interaction = interaction_difference(
        final[("lingam", "ABCI_Balanced")],
        final[("lingam", "Cho2016ActiveGBN")],
        final[("pc", "ABCI_Balanced")],
        final[("pc", "Cho2016ActiveGBN")],
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.4),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    x = np.arange(len(inferers))
    for agent in agents:
        estimates = [
            bootstrap_mean_ci(list(final[(inferer, agent)].values()))
            for inferer in inferers
        ]
        y = [estimate.estimate for estimate in estimates]
        yerr = [
            [estimate.estimate - estimate.ci_low for estimate in estimates],
            [estimate.ci_high - estimate.estimate for estimate in estimates],
        ]
        axes[0].errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            color=COLORS[agent],
            label=display_name(agent),
        )
    axes[0].set_xticks(x, ["PC", "LiNGAM"])
    axes[0].set_ylabel(MISSION_ERROR_LABELS["graph_discovery"] + " ↓")
    axes[0].set_title("Acquisition-policy ranking depends on the inferer")
    axes[0].legend(frameon=False)

    effects = []
    for inferer in inferers:
        effects.append(
            paired_effect(
                final[(inferer, "ABCI_Balanced")],
                final[(inferer, "Cho2016ActiveGBN")],
            )
        )
    effects.append(interaction)
    labels = ["PC: ABCI − ActiveGBN", "LiNGAM: ABCI − ActiveGBN", "Interaction ΔΔ"]
    y_positions = np.arange(len(effects))[::-1]
    for position, estimate in zip(y_positions, effects):
        axes[1].errorbar(
            estimate.estimate,
            position,
            xerr=[
                [estimate.estimate - estimate.ci_low],
                [estimate.ci_high - estimate.estimate],
            ],
            fmt="o",
            color="#333333",
            capsize=4,
        )
    axes[1].axvline(0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions, labels)
    axes[1].set_xlabel("Paired SHD difference (positive = ABCI worse)")
    axes[1].set_title("Matched-seed effects (95% bootstrap CI)")
    fig.suptitle("A non-additive decider–inferer interaction", fontweight="bold")
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig01_decider_inferer_interaction")
    plt.close(fig)
    return paths, {
        "estimate": interaction.estimate,
        "ci_low": interaction.ci_low,
        "ci_high": interaction.ci_high,
        "n": interaction.n,
        "win_rate_left": interaction.win_rate_left,
    }


def figure_error_vs_accumulated_evidence(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Plot graph error against actual accumulated evidence and anytime AUC."""
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    grouped = group_runs(records)
    trajectory_seeds = set(_matched_graph_seeds(grouped))
    configurations = [
        ("pc", "Cho2016ActiveGBN"),
        ("pc", "ABCI_Balanced"),
        ("lingam", "Cho2016ActiveGBN"),
        ("lingam", "ABCI_Balanced"),
    ]
    x_specs = [
        ("total_cumulative", "Total collected samples"),
        ("observational_cumulative", "Observational samples"),
        ("interventional_cumulative", "Interventional samples"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.35))
    auc_rows: list[dict[str, Any]] = []
    linestyles = {"pc": "-", "lingam": "--"}

    for axis, (x_key, x_label) in zip(axes[:3], x_specs):
        for inferer, agent in configurations:
            runs = [
                run
                for run in _selected_runs(
                    grouped, "graph_discovery", inferer, agent
                )
                if int(run[0]["seed"]) in trajectory_seeds
            ]
            if not runs:
                continue
            common_min = max(float(run[0][x_key]) for run in runs)
            common_max = min(_last_value(run, x_key) for run in runs)
            budgets = np.linspace(common_min, common_max, 70)
            mean, lower, upper = _curve_summary(runs, budgets, x_key=x_key)
            label = f"{display_name(agent)} + {inferer.upper()}"
            axis.plot(
                budgets,
                mean,
                color=COLORS[agent],
                linestyle=linestyles[inferer],
                linewidth=1.8,
                label=label,
            )
            axis.fill_between(budgets, lower, upper, color=COLORS[agent], alpha=0.11)
        axis.set_xlabel(x_label)
        axis.set_ylabel("SHD ↓" if axis is axes[0] else "")
        axis.set_title(f"Error vs. {x_label.lower()}")

    for inferer, agent in configurations:
        by_seed = _by_seed(
            grouped,
            "graph_discovery",
            inferer,
            agent,
            metric="auc",
            max_budget=600,
        )
        by_seed = {
            seed: value
            for seed, value in by_seed.items()
            if seed in trajectory_seeds
        }
        estimate = bootstrap_mean_ci(list(by_seed.values()))
        auc_rows.append(
            {
                "inferer": inferer,
                "agent": agent,
                "estimate": estimate.estimate,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "n": estimate.n,
            }
        )
    y = np.arange(len(auc_rows))[::-1]
    for position, row in zip(y, auc_rows):
        axes[3].errorbar(
            row["estimate"],
            position,
            xerr=[
                [row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]],
            ],
            fmt=INFERER_MARKERS[row["inferer"]],
            color=COLORS[row["agent"]],
            capsize=3,
        )
    axes[3].set_yticks(
        y,
        [f"{display_name(row['agent'])}\n+ {row['inferer'].upper()}" for row in auc_rows],
    )
    axes[3].set_xlabel(
        "Average SHD throughout acquisition up to 600 collected samples"
    )
    axes[3].set_title(
        "Whole-trajectory error\nacross 20 matched-seed runs"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Round counts conceal different evidence-acquisition paths", fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    paths = save_figure(fig, output_dir, "fig02_error_vs_accumulated_evidence")
    plt.close(fig)
    return paths, auc_rows


def _configuration_summaries(
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    mission: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Sequence[Mapping[str, Any]]]] = defaultdict(list)
    for (run_mission, inferer, agent, _seed), run in grouped.items():
        if run_mission == mission:
            buckets[(inferer, agent)].append(run)
    rows: list[dict[str, Any]] = []
    for (inferer, agent), runs in sorted(buckets.items()):
        errors = [_last_value(run, "error") for run in runs]
        observational = [_last_value(run, "observational_cumulative") for run in runs]
        interventional = [_last_value(run, "interventional_cumulative") for run in runs]
        aucs = [anytime_auc(run, max_budget=600) for run in runs]
        rows.append(
            {
                "inferer": inferer,
                "agent": agent,
                "error": sum(errors) / len(errors),
                "observational": sum(observational) / len(observational),
                "interventional": sum(interventional) / len(interventional),
                "auc": sum(aucs) / len(aucs),
                "n": len(runs),
            }
        )
    return rows


def figure_acquisition_quality_map(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Relate observational/interventional allocation to graph quality."""
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    grouped = group_runs(records)
    rows = _configuration_summaries(grouped, "graph_discovery")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))
    errors = np.asarray([row["error"] for row in rows])
    color_min, color_max = float(errors.min()), float(errors.max())
    inferer_offsets = {
        "dag_discovery": (-8.0, -8.0),
        "lingam": (-3.0, 3.0),
        "notears": (3.0, -3.0),
        "pc": (8.0, 8.0),
    }
    for row in rows:
        offset_x, offset_y = inferer_offsets.get(row["inferer"], (0.0, 0.0))
        axes[0].scatter(
            row["observational"] + offset_x,
            row["interventional"] + offset_y,
            c=[row["error"]],
            cmap="viridis_r",
            vmin=color_min,
            vmax=color_max,
            marker=INFERER_MARKERS.get(row["inferer"], "o"),
            s=70,
            edgecolor="black",
            linewidth=0.5,
        )
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(color_min, color_max), cmap="viridis_r"
    )
    fig.colorbar(scalar, ax=axes[0], label="Final SHD ↓")
    axes[0].set_xlabel("Observational samples")
    axes[0].set_ylabel("Interventional samples")
    axes[0].set_title("Acquisition profiles occupy distinct regions")
    axes[0].text(
        0.02,
        0.02,
        "Small marker offsets reveal overlapping inferers",
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    for inferer in sorted({row["inferer"] for row in rows}):
        subset = [row for row in rows if row["inferer"] == inferer]
        fraction = [
            row["interventional"] / (row["observational"] + row["interventional"])
            for row in subset
        ]
        axes[1].scatter(
            fraction,
            [row["auc"] for row in subset],
            marker=INFERER_MARKERS.get(inferer, "o"),
            s=65,
            label=inferer,
            alpha=0.85,
        )
    axes[1].set_xlabel("Fraction of collected samples that are interventional")
    axes[1].set_ylabel("Anytime SHD AUC ↓")
    axes[1].set_title("The same allocation is not equally useful to every inferer")
    axes[1].legend(title="Inferer", frameon=False, ncol=2)
    fig.suptitle("Acquisition mix and inference quality must be evaluated jointly", fontweight="bold")
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig03_acquisition_quality_map")
    plt.close(fig)
    return paths, rows


def _treatment_contrasts(
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    inferer = "honest_causal_tree"
    definitions = [
        (
            "Allocation contrast",
            "OptimalEffectDesign_InterventionHeavy",
            "OptimalEffectDesign_Balanced",
        ),
        (
            "Decider-family contrast",
            "ABCI_Balanced",
            "OptimalEffectDesign_Balanced",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, left, right in definitions:
        for metric, max_budget in (("final_error", None), ("auc", 400)):
            left_values = _by_seed(
                grouped,
                "treatment_effect",
                inferer,
                left,
                metric=metric,
                max_budget=max_budget,
            )
            right_values = _by_seed(
                grouped,
                "treatment_effect",
                inferer,
                right,
                metric=metric,
                max_budget=max_budget,
            )
            estimate = paired_effect(left_values, right_values)
            rows.append(
                {
                    "contrast": label,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "estimate": estimate.estimate,
                    "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high,
                    "n": estimate.n,
                    "win_rate_left": estimate.win_rate_left,
                }
            )
    return rows


def figure_treatment_effect_repaired(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Create the repaired, valid honest-causal-tree treatment figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    rebuttal_records = [
        record
        for record in records
        if record.get("collection") == "rebuttal"
        and record.get("mission") == "treatment_effect"
        and record.get("inferer") == "honest_causal_tree"
        and record.get("valid") is True
    ]
    grouped = group_runs(rebuttal_records)
    seeds = {
        seed
        for (mission, inferer, _agent, seed) in grouped
        if mission == "treatment_effect" and inferer == "honest_causal_tree"
    }
    if len(seeds) != 21:
        raise RuntimeError(
            "Headline treatment figure requires 21 valid repaired seeds; "
            f"found {len(seeds)}."
        )

    agents = [
        "OptimalEffectDesign_Balanced",
        "OptimalEffectDesign_InterventionHeavy",
        "ABCI_Balanced",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for axis, (x_key, x_label, maximum) in zip(
        axes[:2],
        [
            ("total_cumulative", "Total collected samples", 400),
            ("interventional_cumulative", "Interventional samples", 400),
        ],
    ):
        budgets = np.linspace(4, maximum, 70)
        for agent in agents:
            runs = _selected_runs(
                grouped, "treatment_effect", "honest_causal_tree", agent
            )
            mean, lower, upper = _curve_summary(runs, budgets, x_key=x_key)
            axis.plot(
                budgets,
                mean,
                color=COLORS[agent],
                linewidth=1.8,
                label=display_name(agent),
            )
            axis.fill_between(budgets, lower, upper, color=COLORS[agent], alpha=0.12)
        axis.set_xlabel(x_label)
        axis.set_ylabel("PEHE ↓" if axis is axes[0] else "")
        axis.set_title(f"PEHE vs. {x_label.lower()}")

    contrasts = _treatment_contrasts(grouped)
    y = np.arange(len(contrasts))[::-1]
    for position, row in zip(y, contrasts):
        axes[2].errorbar(
            row["estimate"],
            position,
            xerr=[
                [row["estimate"] - row["ci_low"]],
                [row["ci_high"] - row["estimate"]],
            ],
            fmt="o",
            color="#333333",
            capsize=3,
        )
    axes[2].axvline(0, color="#666666", linestyle="--", linewidth=1)
    axes[2].set_yticks(
        y,
        [
            f"{row['contrast']}\n"
            f"{'final PEHE' if row['metric'] == 'final_error' else 'anytime PEHE AUC'}"
            for row in contrasts
        ],
    )
    axes[2].set_xlabel("Paired difference (left − right; lower is better)")
    axes[2].set_title("Predeclared contrasts (95% bootstrap CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Valid fixed-horizon treatment-effect comparison (21 matched seeds)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    paths = save_figure(fig, output_dir, "fig04_treatment_effect_repaired")
    plt.close(fig)
    return paths, contrasts


def _dominance_probability(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[float]]]:
    agents = sorted({row["agent"] for row in rows})
    by_agent_seed = {
        agent: {
            row["seed"]: row["error"]
            for row in rows
            if row["agent"] == agent
        }
        for agent in agents
    }
    matrix: list[list[float]] = []
    for left in agents:
        line: list[float] = []
        for right in agents:
            seeds = set(by_agent_seed[left]) & set(by_agent_seed[right])
            if not seeds:
                line.append(math.nan)
            else:
                line.append(
                    sum(by_agent_seed[left][seed] < by_agent_seed[right][seed] for seed in seeds)
                    / len(seeds)
                )
        matrix.append(line)
    return agents, matrix


def create_supplementary_gallery(
    records: Iterable[Mapping[str, Any]],
    run_qc: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Create non-redundant diagnostic figures for the supplement."""
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    output_paths: list[Path] = []
    grouped = group_runs(records)
    trajectory_seeds = set(_matched_graph_seeds(grouped))
    graph_rows = _configuration_summaries(grouped, "graph_discovery")

    # Heatmap and rank view.
    agents = sorted({row["agent"] for row in graph_rows})
    inferers = sorted({row["inferer"] for row in graph_rows})
    heat = np.full((len(agents), len(inferers)), np.nan)
    for row in graph_rows:
        heat[agents.index(row["agent"]), inferers.index(row["inferer"])] = row["error"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    image = axes[0].imshow(heat, cmap="viridis_r", aspect="auto")
    axes[0].set_xticks(range(len(inferers)), inferers, rotation=30, ha="right")
    axes[0].set_yticks(range(len(agents)), [display_name(agent) for agent in agents])
    axes[0].set_title("Mean final SHD")
    fig.colorbar(image, ax=axes[0], label="SHD ↓")
    for inferer in inferers:
        subset = sorted(
            [row for row in graph_rows if row["inferer"] == inferer],
            key=lambda row: row["error"],
        )
        axes[1].plot(
            range(1, len(subset) + 1),
            [row["error"] for row in subset],
            marker=INFERER_MARKERS.get(inferer, "o"),
            label=inferer,
        )
    axes[1].set_xlabel("Within-inferer rank")
    axes[1].set_ylabel("Mean final SHD ↓")
    axes[1].set_title("Rank profiles differ across inferers")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp01_full_interaction_heatmap")
    plt.close(fig)

    # Paired-seed slope plot.
    final_abci_pc = _by_seed(grouped, "graph_discovery", "pc", "ABCI_Balanced")
    final_cho_pc = _by_seed(grouped, "graph_discovery", "pc", "Cho2016ActiveGBN")
    final_abci_li = _by_seed(grouped, "graph_discovery", "lingam", "ABCI_Balanced")
    final_cho_li = _by_seed(grouped, "graph_discovery", "lingam", "Cho2016ActiveGBN")
    common = sorted(
        set(final_abci_pc) & set(final_cho_pc) & set(final_abci_li) & set(final_cho_li)
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for seed in common:
        differences = [
            final_abci_pc[seed] - final_cho_pc[seed],
            final_abci_li[seed] - final_cho_li[seed],
        ]
        ax.plot([0, 1], differences, color="#999999", alpha=0.38, linewidth=0.8)
    ax.plot(
        [0, 1],
        [
            sum(final_abci_pc[s] - final_cho_pc[s] for s in common) / len(common),
            sum(final_abci_li[s] - final_cho_li[s] for s in common) / len(common),
        ],
        color="#111111",
        linewidth=3,
        marker="o",
        label="Mean paired difference",
    )
    ax.axhline(0, color="#666666", linestyle="--")
    ax.set_xticks([0, 1], ["PC", "LiNGAM"])
    ax.set_ylabel("ABCI-inspired − ActiveGBN-inspired SHD")
    ax.set_title("Matched seeds reveal the interaction, not independent rounds")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp02_paired_seed_interaction")
    plt.close(fig)

    # Dominance matrix for PC agents.
    pc_rows = []
    for (mission, inferer, agent, seed), run in grouped.items():
        if mission == "graph_discovery" and inferer == "pc":
            pc_rows.append({"agent": agent, "seed": seed, "error": _last_value(run, "error")})
    dominance_agents, dominance = _dominance_probability(pc_rows)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(dominance, cmap="RdBu_r", vmin=0, vmax=1)
    labels = [display_name(agent) for agent in dominance_agents]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Compared agent")
    ax.set_ylabel("Row agent")
    ax.set_title("Probability row agent has lower SHD on a matched seed (PC)")
    fig.colorbar(image, ax=ax, label="Matched-seed win probability")
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp03_dominance_probability")
    plt.close(fig)

    # AUC forest across all graph configurations.
    auc_rows = []
    for inferer in inferers:
        for agent in agents:
            values = _by_seed(
                grouped,
                "graph_discovery",
                inferer,
                agent,
                metric="auc",
                max_budget=600,
            )
            values = {
                seed: value
                for seed, value in values.items()
                if seed in trajectory_seeds
            }
            if values:
                estimate = bootstrap_mean_ci(list(values.values()))
                auc_rows.append((inferer, agent, estimate))
    auc_rows.sort(key=lambda item: item[2].estimate)
    fig, ax = plt.subplots(figsize=(7.2, max(5.5, len(auc_rows) * 0.22)))
    y = np.arange(len(auc_rows))[::-1]
    for position, (inferer, _agent, estimate) in zip(y, auc_rows):
        ax.errorbar(
            estimate.estimate,
            position,
            xerr=[
                [estimate.estimate - estimate.ci_low],
                [estimate.ci_high - estimate.estimate],
            ],
            fmt=INFERER_MARKERS.get(inferer, "o"),
            color="#333333",
            capsize=2,
        )
    ax.set_yticks(
        y, [f"{display_name(agent)} + {inferer}" for inferer, agent, _ in auc_rows]
    )
    ax.set_xlabel(
        "Average SHD throughout acquisition up to 600 collected samples"
    )
    ax.set_title(
        "Whole-trajectory graph-discovery error aggregated across "
        "20 matched-seed runs"
    )
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp04_anytime_auc_forest")
    plt.close(fig)

    # Instability and rebound.
    instability_rows = []
    for (mission, inferer, agent, _seed), run in grouped.items():
        if mission != "graph_discovery":
            continue
        stats = trajectory_instability(run)
        instability_rows.append((inferer, agent, stats))
    buckets: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for inferer, agent, stats in instability_rows:
        buckets[(inferer, agent)].append(stats)
    summary = [
        (
            inferer,
            agent,
            sum(item["worsening_fraction"] for item in values) / len(values),
            sum(item["max_rebound"] for item in values) / len(values),
        )
        for (inferer, agent), values in buckets.items()
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    for inferer in inferers:
        subset = [item for item in summary if item[0] == inferer]
        ax.scatter(
            [item[2] for item in subset],
            [item[3] for item in subset],
            marker=INFERER_MARKERS.get(inferer, "o"),
            s=60,
            label=inferer,
        )
    ax.set_xlabel("Fraction of new-evidence steps that worsen SHD")
    ax.set_ylabel("Mean maximum SHD rebound")
    ax.set_title("Additional evidence can transiently degrade an estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp05_trajectory_instability")
    plt.close(fig)

    # Raw resource Pareto view.
    resource_rows = []
    for (mission, inferer, agent, _seed), run in grouped.items():
        if mission != "graph_discovery":
            continue
        resource_rows.append(
            {
                "inferer": inferer,
                "agent": agent,
                "error": _last_value(run, "error"),
                "time": _last_value(run, "time_used_s"),
                "memory": _last_value(run, "memory_used_mb"),
                "samples": _last_value(run, "total_cumulative"),
            }
        )
    aggregate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resource_rows:
        aggregate[(row["inferer"], row["agent"])].append(row)
    points = []
    for (inferer, agent), values in aggregate.items():
        points.append(
            {
                "inferer": inferer,
                "agent": agent,
                **{
                    key: sum(value[key] for value in values if math.isfinite(value[key]))
                    / sum(math.isfinite(value[key]) for value in values)
                    for key in ("error", "time", "memory", "samples")
                },
            }
        )
    frontier = pareto_lower((row["time"], row["error"]) for row in points)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    for row in points:
        axes[0].scatter(
            row["time"],
            row["error"],
            marker=INFERER_MARKERS.get(row["inferer"], "o"),
            alpha=0.8,
        )
        axes[1].scatter(
            row["memory"],
            row["error"],
            marker=INFERER_MARKERS.get(row["inferer"], "o"),
            alpha=0.8,
        )
    if frontier:
        axes[0].plot(
            [point[0] for point in frontier],
            [point[1] for point in frontier],
            color="black",
            linestyle="--",
            label="Lower-is-better frontier",
        )
    axes[0].set(xlabel="Wall-clock time used (s)", ylabel="Final SHD ↓")
    axes[1].set(xlabel="Measured memory used (MB)", ylabel="Final SHD ↓")
    axes[0].legend(frameon=False)
    fig.suptitle("Raw computational resources avoid composite-score ambiguity")
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp06_raw_resource_pareto")
    plt.close(fig)

    # 3D acquisition trajectories.
    fig = plt.figure(figsize=(8.2, 5.4))
    axis = fig.add_subplot(111, projection="3d")
    for inferer, agent in [
        ("pc", "Cho2016ActiveGBN"),
        ("pc", "ABCI_Balanced"),
        ("lingam", "Cho2016ActiveGBN"),
        ("lingam", "ABCI_Balanced"),
    ]:
        runs = _selected_runs(grouped, "graph_discovery", inferer, agent)
        if not runs:
            continue
        run = runs[0]
        axis.plot(
            [record["observational_cumulative"] for record in run],
            [record["interventional_cumulative"] for record in run],
            [record["error"] for record in run],
            color=COLORS[agent],
            linestyle="-" if inferer == "pc" else "--",
            label=f"{display_name(agent)} + {inferer}",
        )
    axis.set_xlabel("Observational samples")
    axis.set_ylabel("Interventional samples")
    axis.set_zlabel("SHD ↓")
    axis.set_title("Representative seed: acquisition-error trajectories")
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp07_acquisition_trajectories_3d")
    plt.close(fig)

    # SCM log-error diagnostics.
    scm_rows = _configuration_summaries(grouped, "scm_discovery")
    if scm_rows:
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        ordered = sorted(scm_rows, key=lambda row: row["error"])
        ax.bar(
            range(len(ordered)),
            [row["error"] for row in ordered],
            color="#56B4E9",
        )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(
            range(len(ordered)),
            [display_name(row["agent"]) for row in ordered],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("SCM distributional error ↓ (symlog)")
        ax.set_title("Exploratory: mechanism recovery exposes extreme numerical failures")
        fig.tight_layout()
        output_paths += save_figure(fig, output_dir, "supp08_scm_error_diagnostics")
        plt.close(fig)

    # QC counts and invalidation reasons.
    status_counts = Counter(
        (
            str(row["experiment_id"]).split("/")[1],
            "valid" if row.get("valid") else "invalid",
        )
        for row in run_qc
    )
    missions = sorted({mission for mission, _status in status_counts})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    x = np.arange(len(missions))
    valid = [status_counts[(mission, "valid")] for mission in missions]
    invalid = [status_counts[(mission, "invalid")] for mission in missions]
    axes[0].bar(x, valid, label="Valid", color="#009E73")
    axes[0].bar(x, invalid, bottom=valid, label="Invalidated", color="#D55E00")
    axes[0].set_xticks(x, missions, rotation=30, ha="right")
    axes[0].set_ylabel("Latest agent transcripts")
    axes[0].set_title("Run inclusion audit")
    axes[0].legend(frameon=False)
    reasons = Counter(
        (row.get("invalidation_reason") or "unknown").split(":", 1)[0]
        for row in run_qc
        if not row.get("valid")
    )
    reason_items = reasons.most_common()
    axes[1].barh(
        range(len(reason_items)),
        [count for _reason, count in reason_items],
        color="#D55E00",
    )
    axes[1].set_yticks(
        range(len(reason_items)), [reason for reason, _count in reason_items]
    )
    axes[1].set_xlabel("Invalidated transcripts")
    axes[1].set_title("Invalidation reasons (excluded from analyses)")
    fig.tight_layout()
    output_paths += save_figure(fig, output_dir, "supp09_run_quality_audit")
    plt.close(fig)

    return output_paths


def build_evidence_table(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the reviewer-facing numerical claims with matched-seed uncertainty."""
    grouped = group_runs(records)
    graph = interaction_difference(
        _by_seed(grouped, "graph_discovery", "lingam", "ABCI_Balanced"),
        _by_seed(grouped, "graph_discovery", "lingam", "Cho2016ActiveGBN"),
        _by_seed(grouped, "graph_discovery", "pc", "ABCI_Balanced"),
        _by_seed(grouped, "graph_discovery", "pc", "Cho2016ActiveGBN"),
    )
    rows = [
        {
            "reviewer_question": (
                "Does acquisition policy interact non-trivially with the inferer?"
            ),
            "finding": "Decider-inferer interaction (LiNGAM contrast minus PC contrast)",
            "metric": "SHD difference-in-differences",
            "estimate": graph.estimate,
            "ci_low": graph.ci_low,
            "ci_high": graph.ci_high,
            "n_seeds": graph.n,
            "interpretation": (
                "Positive values mean the ABCI-inspired acquisition policy is relatively "
                "less compatible with LiNGAM than with PC."
            ),
        }
    ]
    treatment_records = [
        record
        for record in records
        if record.get("collection") == "rebuttal"
        and record.get("mission") == "treatment_effect"
        and record.get("valid") is True
    ]
    if treatment_records:
        treatment_grouped = group_runs(treatment_records)
        for contrast in _treatment_contrasts(treatment_grouped):
            reviewer_question = (
                "How does interventional allocation affect treatment-effect "
                "cost-quality?"
                if contrast["contrast"] == "Allocation contrast"
                else (
                    "Do different decider families change treatment-effect "
                    "accuracy?"
                )
            )
            rows.append(
                {
                    "reviewer_question": reviewer_question,
                    "finding": contrast["contrast"],
                    "metric": (
                        "Final PEHE difference"
                        if contrast["metric"] == "final_error"
                        else "Anytime PEHE AUC difference"
                    ),
                    "estimate": contrast["estimate"],
                    "ci_low": contrast["ci_low"],
                    "ci_high": contrast["ci_high"],
                    "n_seeds": contrast["n"],
                    "interpretation": (
                        f"{display_name(contrast['left'])} minus "
                        f"{display_name(contrast['right'])}; negative favors the left."
                    ),
                }
            )
    return rows


def write_evidence_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the compact evidence table used for captions and rebuttal prose."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Evidence table is empty.")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
