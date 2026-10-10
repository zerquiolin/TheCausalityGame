# Publication implementation review

Historical record for the invalid identical-seed campaign. The author resolved the precedence choice; see [IMPLEMENTATION_REVIEW_V4.md](IMPLEMENTATION_REVIEW_V4.md) for the corrected fallback policy and rerun.

Prepared 2026-10-09. **All current changes are uncommitted for author review.** Package base HEAD: `9c27fe5b8e532eb2583ff2e81bce11485f48e4bd`. Paper base: `5126ce1158b9708a3c493d4c7107536da93cc01f`. Root guidelines are local files outside either repository.

## Changes to review

1. **Scientific runtime correctness:** canonical independent acquisition/node streams, explicit manifest seed routing, post-inform answers, frozen historical estimates, whole-batch sample/experiment/cost preflight, categorical/numerical domain checks, structural graph checks, custom metric mounting and copied hook payloads.
2. **Author-approved termination contract:** a forced stop permits a final response from acquired evidence. Valid responses receive scores; missing, failed and invalid final responses have `Feedback.result=None`. Transcript headers record `forced_stop`, resource `termination_reason`, `response_status` and diagnostic `response_error`. Initially exhausted budgets permit a round-zero response without sampling. Prior estimates are not reused when that response fails.
3. **Required construction features:** `SyntheticSCMGenerator`, discrete BIF loading/legacy asset migration, dataset-derived SCMs with configurable scikit-learn outcome estimators, Laplace/Dirichlet noise, intervention-price hierarchy, experiment/financial budgets, recursive resource stopping and arbitrary linear behavior metrics.
4. **Traceability and execution:** structured graph exports/full budgets/terminal scores, installed package and normalized manifest fingerprints, installed dependency versions, preserved failed runs, source-aware resume, subprocess deadlines/process-group termination and portable clean-wheel CI configuration.
5. **Paper alignment:** conditional identifiability, explicit DAG/CPDAG SHD semantics and final-response termination protocol. The four root context documents distinguish current implementation from historical findings.

## Verified local evidence

- Python 3.12.13 / R 4.6.1: **215 passed, zero skipped**, 23.22 seconds, including seven actual upstream integration cases. Full log `/tmp/tcg-final-tests.log`.
- Statement coverage **5,454/7,507 = 72.65%**; no branch-coverage claim. Coverage `/tmp/tcg-final-coverage.json`.
- Touched Python files pass Ruff `E4,E7,E9,F,I,B,UP`; formatting applied. This does not establish whole-repository strict typing or lint compliance.
- `scripts/build_catalog.py --output-dir runs/catalog-v3` constructs and samples **52 distinct current-spec tasks**: 46 migrated JSON SCMs, three synthetic variants and three dataset-derived examples; zero failed assets. `munin_sn1.json.disabled` is explicitly excluded. Report `runs/catalog-v3/catalog_report.json`. This is construction validation, not agent performance across all tasks.
- LaTeX compilation succeeds: `/tmp/tcg-paper-v3/neurips_2026.pdf`. Existing broad manuscript/style issues remain tracked.
- Final wheel `/tmp/tcg-v3-review-wheel/thecausalitygame-0.1.0-py3-none-any.whl`, SHA-256 `988a6df634185281d31935c29cff2934330d194134674bbf96ebaba07322a1f3`. Installed into the previously prepared isolated base-only environment outside the checkout; pip check, hybrid generator and graph CLI smoke checked. This reuses a constrained environment, not a fresh dependency resolution. The verified wheel predates the final analysis-only duplicate-seed guard; rebuild the release wheel after resolving that blocker. Scientific package sources and their fingerprint are unchanged by this guard. Remote Linux/Windows/macOS CI is configured but not executed for this uncommitted change.
- Scientific package source fingerprint: `fd421edb06652f08709e553bca1e7389b00ead0c8e94ee78d85d6ee47212c0b8`. It covers package `.py`/`.R` files, not analysis scripts, assets or manuscript; those require separate hashes when archiving.

## New experiment protocol and commands

Historical caches, figure exports, comparison studies and earlier failed pilot timestamps are preserved. New study root is `runs/publication-v3`. Results use independent stream namespaces `acquisition-v3`/`node-v3`, with `post-inform-v2` response timing. Do not pool earlier scores into this campaign.

```sh
TCG_R_LIBS=.r-library MPLCONFIGDIR=/tmp/tcg-mpl-v3 python scripts/problem_instances_rebuttal/run_rebuttal.py --runs-dir runs/publication-v3 --timeout-s 1800
MPLCONFIGDIR=/tmp/tcg-mpl-v3 python scripts/analyze_publication.py --runs-dir runs/publication-v3
```

The runner preserves timestamped artifacts/logs and writes `campaign_progress.json`. It resumes only complete matching source/normalized-manifest runs with all active agents and finite final scores. A deliberately unscored final response is not considered a complete scored study result; inspect its transcript rather than silently dropping it. Deadlines apply per manifest subprocess, while package compute budgets are checked at agent-call boundaries.

The analysis script requires all 21 matching manifests and their complete active-agent inventory before exporting treatment-only plots, numerical contrasts and round-level cache. It does not invoke the general graph evidence builder or mix historical graph data. Early honest-tree estimates remain explicitly labelled `fit_status=insufficient_data`, `provisional_estimate=zero` until both arms have enough observations for disjoint honest fitting. The figure includes these deployed provisional estimates; report their count and assess this convention before publication.

## Known limits and remaining pre-submission work

- No arbitrary-SCM identifiability theorem. Synthetic numerical roots sample uniformly from their domain; numerical descendants are clipped after mechanism/noise evaluation. Clipping changes the structural mechanism and must be described. Four scalar noise choices are exposed by the generator; Dirichlet is a categorical simplex noise class.
- Dataset examples embed complete-case training rows and rebuild fitted models from configuration. They are not fitted checkpoints. Dataset licensing, original provenance, train/evaluation splits and causal assumptions still require review.
- Graph validation checks local structural consistency/chordality/semidirected cycles, not maximal CPDAG orientation or equivalence-class identifiability. Hook payload copying is not a Python execution sandbox.
- Memory budgets measure acquired sample-table volume, not peak RSS. Compute time excludes simulation/evaluator work and includes deferred inference; individual calls are not preempted by the soft package limit.
- The entire nine-family 189-manifest historical campaign is not rerun. New treatment results cannot establish old graph claims or all manuscript empirical conclusions.
- Promised graph classification/regression/functional/noise-divergence metric families, missing BART/causal forest/EA baselines, GIES dependency/history disclosure, Croissant/data metadata and broad fitted-history reconstruction remain open in root task trackers. Do not treat this review as all publication requirements completed.
- Author decision D5 preserves current reproducibility wording; local tests do not prove cross-platform byte equality.

## Campaign result

Execution completed: **21 manifests, 147 valid agent trajectories, 14,700 scored rounds**, 147 forced-stop final responses, zero unscored finals, and 147 explicitly provisional zero-estimate rounds. Elapsed scheduling time approximately 17.5 minutes. **Publication analysis is blocked**: every manifest declares `seeds.scm=911`, overriding its distinct serialized SCM RNG state. The scientific trajectories are identical across purported seed replicates; there is only one effective SCM seed. Degenerate confidence intervals must not be published. Generated plots/cache/contrasts are quarantined under `runs/publication-v3/analysis/invalid_seed_replicates`; `analysis/report.json` marks publication ineligible. The exporter now rejects duplicate effective SCM states before plotting.

Await author choice between normalizing each manifest's explicit SCM seed to its filename seed (recommended) or changing the precedence contract, then rerun into a new preserved study root. No current changes have been committed.
