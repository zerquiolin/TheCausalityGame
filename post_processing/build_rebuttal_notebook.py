"""Build the source notebook for the NeurIPS rebuttal analysis."""

from __future__ import annotations

import json
from pathlib import Path


def code(source: str) -> dict:
    """Return a code cell with stable source formatting."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (source.strip() + "\n").splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    """Return a Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (source.strip() + "\n").splitlines(keepends=True),
    }


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "cells": [],
    }
    notebook["cells"] = [
        markdown(
            """
# NeurIPS rebuttal analysis

This notebook produces reviewer-facing evidence from round-level game transcripts.
It treats SHD, PEHE, mechanism error, and resource use as quantities to minimize.
Seeds—not rounds—are the independent experimental units.

Invalidated runs are retained in the quality-control audit but excluded from every
statistical analysis. Agent names use **method-inspired** language because these
implementations are not claimed to reproduce the original methods.
"""
        ),
        code(
            """
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

NOTEBOOK_DIR = Path.cwd()
REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "post_processing" else NOTEBOOK_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_processing.rebuttal_utils import (
    load_round_cache,
    scan_round_records,
    write_round_cache,
)
from post_processing.rebuttal_figures import (
    build_evidence_table,
    create_supplementary_gallery,
    figure_acquisition_quality_map,
    figure_decider_inferer_interaction,
    figure_error_vs_accumulated_evidence,
    figure_treatment_effect_repaired,
    write_evidence_csv,
)
"""
        ),
        markdown(
            """
## Configuration

The treatment headline is intentionally strict: Figure 4 is generated only when
all 21 repaired seeds have valid transcripts. It never falls back to the
invalidated treatment runs used for quality-control diagnostics.
"""
        ),
        code(
            """
RUN_ROOTS = [
    REPO_ROOT / "runs" / "seeded",
    REPO_ROOT / "runs" / "rebuttal",
]
MANIFEST_ROOTS = [
    REPO_ROOT / "scripts" / "problem_instances",
    REPO_ROOT / "scripts" / "problem_instances_rebuttal",
]
CACHE_PATH = REPO_ROOT / "post_processing" / "rebuttal_results.json"
FIGURES_DIR = REPO_ROOT / "post_processing" / "figures_rebuttal"
EVIDENCE_PATH = REPO_ROOT / "post_processing" / "rebuttal_evidence.csv"

REBUILD_CACHE = True
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        markdown("## Round-level extraction and run-quality audit"),
        code(
            """
if REBUILD_CACHE or not CACHE_PATH.exists():
    records, run_qc = scan_round_records(RUN_ROOTS, MANIFEST_ROOTS)
    write_round_cache(CACHE_PATH, records, run_qc)
else:
    records, run_qc = load_round_cache(CACHE_PATH)

print(f"{len(records):,} round records from {len(run_qc):,} latest agent transcripts")
print(f"Cache schema written to: {CACHE_PATH}")
"""
        ),
        code(
            """
qc = pd.DataFrame(run_qc)
qc["mission"] = qc["experiment_id"].str.split("/").str[1]
qc_summary = (
    qc.groupby(["mission", "valid"], dropna=False)
    .size()
    .rename("agent_transcripts")
    .reset_index()
)
display(qc_summary)

invalid_reasons = (
    qc.loc[~qc["valid"]]
    .groupby(["mission", "invalidation_reason"], dropna=False)
    .size()
    .rename("agent_transcripts")
    .reset_index()
    .sort_values("agent_transcripts", ascending=False)
)
display(invalid_reasons.head(20))
"""
        ),
        markdown(
            """
## Headline 1 — decider–inferer interaction

The plotted estimand is

`(ABCI-inspired − ActiveGBN-inspired) under LiNGAM`
minus
`(ABCI-inspired − ActiveGBN-inspired) under PC`.

A positive value means the ABCI-inspired acquisition policy is relatively less
compatible with LiNGAM than with PC.
"""
        ),
        code(
            """
fig01_paths, graph_interaction = figure_decider_inferer_interaction(
    records, FIGURES_DIR
)
display(pd.DataFrame([graph_interaction]))
fig01_paths
"""
        ),
        markdown("## Headline 2 — result error against accumulated evidence"),
        code(
            """
fig02_paths, graph_auc = figure_error_vs_accumulated_evidence(
    records, FIGURES_DIR
)
display(pd.DataFrame(graph_auc))
fig02_paths
"""
        ),
        markdown("## Headline 3 — acquisition composition and inference quality"),
        code(
            """
fig03_paths, acquisition_rows = figure_acquisition_quality_map(
    records, FIGURES_DIR
)
display(pd.DataFrame(acquisition_rows).sort_values(["inferer", "error"]))
fig03_paths
"""
        ),
        markdown(
            """
## Headline 4 — repaired treatment-effect experiment

These runs have a fixed 100-round/400-sample horizon, 400-second time limit, and
2 MB measured-memory limit. Termination is forced by the horizon; this experiment
does **not** provide evidence of adaptive stopping.

The repaired PEHE query excludes post-treatment descendants from the covariate
set. This preserves the total-effect estimand instead of inadvertently blocking
mediated paths; the regression test uses the benchmark path `Z -> X -> Y`.
"""
        ),
        code(
            """
fig04_paths, treatment_contrasts = figure_treatment_effect_repaired(
    records, FIGURES_DIR
)
display(pd.DataFrame(treatment_contrasts))
fig04_paths
"""
        ),
        markdown(
            """
## Supplementary diagnostic gallery

These plots answer distinct diagnostic questions. They are not additional
headline claims and should be selected for the supplement only when they support
the final rebuttal text.
"""
        ),
        code(
            """
supplementary_paths = create_supplementary_gallery(
    records, run_qc, FIGURES_DIR
)
pd.DataFrame({"figure": supplementary_paths})
"""
        ),
        markdown("## Auditable reviewer evidence table"),
        code(
            """
evidence = build_evidence_table(records)
write_evidence_csv(EVIDENCE_PATH, evidence)
evidence_df = pd.DataFrame(evidence)
display(evidence_df)
print(f"Wrote {EVIDENCE_PATH}")
"""
        ),
        markdown(
            """
## Interpretation guardrails

- Report the implemented agents, not the original cited methods.
- Treat confidence intervals as descriptive matched-seed uncertainty, not formal
  hypothesis-test declarations.
- Never count rounds as independent replicates.
- Never reuse or truncate invalidated treatment traces.
- Use raw observational/interventional samples, wall-clock time, and memory when
  discussing cost. The composite behavior metric depends on budget denominators.
- A fixed-dataset benchmark can compare terminal estimates, but it cannot recover
  the acquisition trajectories, allocation choices, anytime AUC, or interaction
  costs shown here.
"""
        ),
    ]

    output = Path(__file__).with_name("rebuttal_analysis.ipynb")
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
