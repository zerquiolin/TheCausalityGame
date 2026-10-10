# Pre-submission package validation

Historical validation record for the earlier committed pass. Current uncommitted implementation, seed fallback policy and rerun are documented in [IMPLEMENTATION_REVIEW_V4.md](IMPLEMENTATION_REVIEW_V4.md). Historical claims below must not be read as current missing-feature status.

Prepared 2026-10-09 against package base commit
`2c7ac3d7cdac6f58ac575a745ed4afef4270a158` and manuscript revision
`5126ce1158b9708a3c493d4c7107536da93cc01f`. All changes in this audit are uncommitted
for author review. See the repository diff for compatibility fixes from the earlier
R repair as well as the current environment/test changes.

## Five-step scope

1. Declare and test Python 3.12; align Python metadata and static-tool targets.
2. Provide tested dependency constraints, explicit R setup/provenance, CI and clean-wheel checks.
3. Map implemented behavior to manuscript claims and keep unsupported claims open.
4. Add runtime integration and benchmark-input regressions, preserving scientific semantics.
5. Execute bounded comparisons and an actual rebuttal-manifest smoke run; audit reports,
   estimate larger work, and preserve historical evidence.

The larger audit backlog remains in `../../TASKS_PACKAGE.md` and `../../TASKS_PAPER.md` at
the TCG workspace root. Completion of this validation pass does not close missing
generators, baselines, metric families or publication result reproduction.

## Changes and scientific implications

- Python 3.12 is the declared support family. Current backend pins are unchanged.
- The new constraints use installed distribution metadata rather than Conda build URLs.
- Matplotlib is declared in runtime dependencies to make base CLI imports resolvable.
- The CLI accepts `tcg run manifest.json --run-dir output`, `tcg manifest.json --run-dir
  output`, and the global-output-option form. Its earlier optional positional argument
  competed with subcommand parsing. Regression tests cover both forms and malformed JSON.
- Hill comparison data normalizes only its string-coded binary treatment to numeric 0/1
  before either inferer receives it. Both sides still receive identical data/digests.
  The original SCM/manifests and adapter input contracts are unchanged.
- R local-build provenance records verified source HEAD and patch identity rather than
  inferring installed commit identity from a constant or package version.
- README now describes sample-only `Agent.inform`; metric feedback is recorded in
  transcripts. No feedback interface or lifecycle timing was changed.

## Claim-to-implementation matrix

Paths prefixed `P/` refer to `TheCausalityGameNeurips2026/`; package paths are relative
to this repository. Line references refer to the above manuscript revision.

| Claim / scientific contract | Implemented and checked | Publication action still needed |
| --- | --- | --- |
| DAG recovery outputs, `P/section/standard_components.tex:21` says CPDAG | Public PC produces partial graphs; NOTEARS/DirectLiNGAM produce DAGs. SHD counts missing/extra=1, reversal=1, matching unresolved adjacency=0. Mixed-graph tests remain passing. | Update paper to the approved DAG/CPDAG contract. Do not call it CPDAG-to-CPDAG distance. |
| Six published inferers, `P/section/appendix.tex` method descriptions | Integration tests call actual PC/NOTEARS/DirectLiNGAM, S_RF/EstimateCate, TOT tree and honest causal tree. Exact references are in `TheCausalityGame/agent/inferers/REFERENCES.md`. | Label native assumptions, compatibility patches, and historical empirical results separately. S_RF is an RF S-learner, not BART or a causal forest. |
| `P/section/abstract.tex:7`: “over 31 standard strategies” and “over 50 benchmark tasks” | Current method classes, pairings, variants, seeds and raw assets are distinct counting units; existence of a class is not empirical verification. | Complete the versioned method/task inventory. Missing promised BART/causal forest/EA and generators remain required work. |
| `P/section/standard_components.tex:17`: functional similarity plus noise-model divergence | Current SCM score mixes moment/category error and SHD; no explicit noise-divergence metric was added here. | Required metric-family work and figure formula/version alignment stay open. |
| Effect evaluator and selected subgroup | Existing PEHE uses baseline-only input features, fixed subgroup row and shared treatment query frames. Hill regression remains passing. | Describe the approved subgroup scope; add analytic nonempty-baseline/shape regressions. Do not reinterpret this as a population-wide evaluator. |
| Runtime feedback and answer timing | Real runtime tests establish act→answer→sample acquisition→inform→evaluation. Transcript feedback is not passed to Agent.inform. Submission captures the previous update; round exhaustion retains the pre-update result. | Give manuscript lifecycle pseudocode and specify forced termination explicitly. High coverage is not endorsement of all current timing semantics. |
| `P/section/appendix.tex:556`: “Every resulting SCM is identifiable” | No unconditional identifiability theorem or generator guarantee is established by these tests. | Apply the approved qualification by mission/access/interventions/model assumptions. |
| Seeded reproducibility | Same-treatment chunking and independently constructed agent traces agree in sequential/thread/process tests. | Treatment-discovery-order independence and all cross-platform guarantees remain unverified; retain the author's protected wording while tracking the implementation gaps. |
| Rebuttal/experimental figures | Fresh comparison outputs are independently audited; the ten-round rebuttal copy exercises seven actual agents. Cached paper evidence remains historical. | Version-bind raw transcripts, old empirical implementations, metrics, exclusions and figure exports before reproducing paper claims. |

## Final verification

- Full suite: **162 passed, 0 skipped in 16.40 s**; statement coverage **4,670/6,726 = 69.43%**.
- Environment/Game/Runner coverage: **96.55% / 100% / 91.18%**.
- Clean installed wheel outside checkout: **24 integration/CLI cases passed in 7.64 s**, packaged resources and installed NOTEARS revision verified, pip check passed.
- The installed CLI also completed all seven agents of the reduced Hill manifest and the runnable minimal graph example. A second fresh base-only wheel environment passes pip check/imports and executes the graph example without lingam, causal-learn or NOTEARS installed.
- R bootstrap succeeds, records all three exact sources/patches, and repeats without rebuilding.
- Scoped Ruff/Black/R syntax checks pass; adapter mypy passes 11 files. The updated remote CI is not yet executed.
- The named Conda editable install is updated to this checkout.

## Tests

`tests/integration/conftest.py` supplies a fresh tiny real SCM/manifest for each test.
`tests/integration/test_runtime.py` adds 14 cases for lifecycle/latent filtering,
submission and forced-round timing, custom behavior metrics, invalid interventions,
exact/overspent samples, fake-clock exhaustion, end hooks, stream chunking, three
execution modes, inactive agents, setup failures and empty runs. Timing metadata is
excluded from per-agent scientific equivalence comparisons. No sleep is used in tests.
`tests/unit/scripts/test_comparison_reports.py` adds the Hill coding/backend-payload
regression; `tests/unit/infra/test_cli.py` adds four CLI cases.

Remaining runtime checks include duplicate IDs, result-type custom metrics, setup-hook
failures, executor-failure simulation, protected hook state and treatment-order seed
invariance. Low-coverage deciders and promised feature families remain in the tracker.
Do not turn statement coverage into a claim that scientific conclusions are verified.

Final verification results and exact artifact paths are recorded in
[`validation_summary.json`](validation_summary.json).

## Bounded benchmark validation and failure policy

The initial smoke used 120 samples, two rounds and seed 911 across all comparison
cases. It exposed Hill's string/numeric mismatch. Preserve that diagnostic run; it is
not a paired performance result. After the coding fix, the default-size validation ran:

```sh
export TCG_R_LIBS="$PWD/.r-library"
python -m scripts.comparisons.run_all --seed 911 --samples 500 --rounds 5 --output runs/pre_submission/verified
python -m post_processing.comparison_analysis runs/pre_submission/verified --latest --strict --output post_processing/comparison_reports/pre_submission_verified --bootstrap-resamples 500
```

This is 210 round records: 105 upstream, 105 empirical, across three graph cases
(Gaussian/non-Gaussian chains and RC circuit) and four effect cases (linear/smooth/
piecewise synthetic and Hill). All 105 upstream rows succeeded. Overall 205 succeeded,
five failed: empirical transformed-outcome cannot fit a propensity logistic regression
with zero Hill covariates. This historical implementation is retained unchanged; its
failure is not counted as an upstream performance win. There are 100 successful paired
rounds across 20 successful case/seed pairs. Input hashes agree for each pair.

Measured update-plus-inference compute sums to 32.29 seconds. The strict report audit
has zero errors and five warnings, with all 55 figure stems valid in PNG and PDF.
Bootstrap intervals from a single seed cannot support between-seed publication claims;
500 resamples here validate report execution only. Synthetic Gaussian data do not meet
DirectLiNGAM's non-Gaussian identification assumptions; interpret that case diagnostically.
RC numerical scaling also caused warnings in historical empirical NOTEARS.

An actual seven-agent rebuttal manifest for seed 911 was copied under
`runs/pre_submission/manuscript_smoke/manifest.json`. Only the output ID, 10-round/
40-sample/30-second budget, two workers and logging level were changed. All seven
transcripts contain ten rounds and are non-invalidated; execution took about ten
seconds after initialization. This smoke is not the original 100-round result.

## Larger-run cost, preservation and resume boundaries

- Rebuttal dry-run: 21 pending manifests, seven agents each, 100 rounds = up to 14,700
  agent rounds. No full rerun was launched during this audit. The ten-round smoke's
  linear projection is approximately 35 minutes at two workers, but later fitting
  costs and contention can increase it substantially; plan tens of minutes to hours.
- The 400-second budget counts selected act/inform work; evaluation/sampling/hooks are
  paused. It is not a reliable total wall-clock deadline. The R bridge permits each
  subprocess up to 3,600 seconds. Use an external scheduler/process deadline for a
  time-limited reproduction job.
- `scripts/problem_instances_rebuttal/run_rebuttal.py --dry-run` safely inventories
  work. `--seed 911 --limit 1 --runs-dir <new-study-root>` is the first full-manifest
  pilot. Record its wall time before expanding to all 21 seeds.
- Resume is manifest-level, based on any timestamp containing all non-invalidated
  agent transcripts. It does not check the code/metric revision and does not resume
  an interrupted agent mid-round. Use a fresh study root after implementation changes;
  keep its source revision, constraints, original manifests and stdout/stderr together.
- The comparison runner writes reports after a suite finishes and has no incremental
  checkpoint/resume. Keep each seed in its own output root and analyze completed runs
  together. Do not promise interruption-safe round checkpoints that do not exist.
- Legacy `scripts/main-seeded.py` deletes newly failed timestamp directories by default
  and can recursively retry errors. Avoid it for evidence-preserving reproduction until
  those behaviors are repaired. `main.py`/`main-inferers.py` can launch many long runs.
- Cached `post_processing/experiment_results.json`, `rebuttal_results.json` and paper
  PDFs were preserved. Their raw paths refer to another workspace and public inferer
  paths have changed lineage. No new score is substituted into an old manuscript panel.

Full multi-seed reproduction and paper-figure regeneration remain separate scientific
work, requiring the author audit and a versioned study configuration before launch.
