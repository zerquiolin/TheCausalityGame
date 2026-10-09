# Inferer comparisons

These scripts compare the retained handwritten (`empirical`) inferers with the selected
authoritative upstream implementations on identical data and seeds. They report differences;
they do not assume that an authoritative backend must obtain a better score.

Install the Python backends:

```bash
pip install -e '.[analysis,authoritative]'
```

Install R and then install the pinned R packages into a repository-local library:

```bash
Rscript scripts/comparisons/bootstrap_r.R .r-library
export TCG_R_LIBS="$PWD/.r-library"
```

Run all comparisons:

```bash
python -m scripts.comparisons.run_all --rounds 5
```

Each round receives a new, disjoint sample chunk while each inferer retains its cumulative buffer.
Both implementations receive identical chunks, round numbers, queries, and cumulative-input
digests. Use `--rounds 1` for the former single-batch behavior and `--synthetic-only` for a quick
graph run. Results are written as per-round JSONL, aggregate CSV, and Markdown under
`runs/comparisons/<timestamp>/`; authoritative provenance is retained for both successful and
failed rows. The custom DAG, SCM, and symbolic-SCM composites are excluded because they have no
authoritative upstream implementation.

## Post-process and inspect the comparison

Analyze the newest run and create a timestamped report:

```bash
python -m post_processing.comparison_analysis runs/comparisons --latest
```

Analyze several runs together at a fixed destination:

```bash
python -m post_processing.comparison_analysis \
  runs/comparisons/20260829T161318Z \
  runs/comparisons/20260830T090000Z \
  --output post_processing/comparison_reports/full-study
```

Add `--strict` in CI to return exit code 2 when the input or generated-figure audit finds an error.
Warnings (for example, a backend that was unavailable) remain visible without failing the command.

The report includes:

- a self-contained `analysis.md` and `analysis_summary.json`;
- normalized records, backend availability/failures, and immutable provenance tables;
- paired case-level differences and deterministic bootstrap confidence intervals;
- coverage, quality, runtime, failure, and descriptive figures in PNG and PDF;
- per-inferer, per-case round trajectories for primary error, secondary quality metrics,
  direction-normalized improvement, and update-plus-inference runtime;
- `round_vs_all_metrics` small multiples covering every available quality metric plus per-round
  and cumulative runtime;
- `figure_artifacts.csv`, which validates both render formats and records dimensions, byte sizes,
  and SHA-256 checksums;
- an artifact audit that checks pairing, identical-input digests, duplicate rows, finite metrics,
  successful-result contracts, and authoritative provenance.

Only cases where both implementations succeeded enter paired quality and runtime claims. If an
authoritative dependency is unavailable, the report shows that fact and produces explicit
"no paired results" figures instead of treating the empirical result as a comparison win.
