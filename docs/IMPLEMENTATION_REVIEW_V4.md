# Publication implementation review: corrected seed fallbacks

Prepared 2026-10-10. **Changes remain uncommitted for author review.** Package base HEAD: `9c27fe5b8e532eb2583ff2e81bce11485f48e4bd`; manuscript base: `5126ce1158b9708a3c493d4c7107536da93cc01f`.

The implemented runtime, generators/converters, acquisition costs/stopping, graph/domain validation, hook isolation, structured artifacts and paper-alignment changes are described in [IMPLEMENTATION_REVIEW_V3.md](IMPLEMENTATION_REVIEW_V3.md). This revision supersedes its seed precedence and publication-campaign status. Its old identical-seed results remain execution checks only.

## Author-approved fallback policy

`core/lib/utils/random_state_serialization.py::resolve_scm_random_state` implements:

1. Use a present serialized SCM RNG state, preserving the full stream state and position.
2. Without that state, use `ProblemInstanceSpec.seeds["scm"]`, or `seeds["global"]` if no named SCM seed exists.
3. Without either, use deterministic default **911**.

Malformed serialized state raises an error rather than silently selecting another seed. `core/runtime/game.py` and `scripts/analyze_publication.py` share this resolver. Acquisition streams derive from the resolved initial SCM state; only an explicitly supplied `seeds["acquisition"]` selects a separate acquisition stream. Evaluator sampling cannot change that captured identity.

Explicit component `params.seed`/`params.random_state` remain authoritative. Without either, existing component routing uses family seed, agent seed, then global seed; these fallback seeds are namespaced into supported constructor seeds. Without a manifest fallback, the component constructor retains its default. This is configuration/state restoration, not a promise of complete fitted-agent checkpoint restoration.

The 21 Hill manifests were left unchanged: each already contains a different serialized SCM state. They now resolve to **21 distinct effective SCM states**, despite the common lower-priority `seeds.scm=911`. The previous source-first override caused `publication-v3` to repeat one scientific trajectory 21 times. Those raw runs are preserved and their plots/cache/contrasts are quarantined under `runs/publication-v3/analysis/invalid_seed_replicates`; its report explicitly marks publication ineligible.

Forced termination follows the author's earlier decision: a valid final response receives a score, while a missing/failed/invalid response has no score. Transcripts record the resource boundary, forced-stop flag and response status. Answers follow observation updates; a failed final response cannot reuse an earlier estimate.

## Verification

- **224 passed, zero skipped in 25.14 seconds**, including all seven real upstream integration cases. Regression tests protect serialized-state continuation, conflicting named seeds, named/global fallback, default911, malformed-state failure and end-to-end sampling. Log `/tmp/tcg-v4-tests.log`.
- Statement coverage **5,458/7,511 = 72.67%**, not branch coverage. `/tmp/tcg-v4-coverage.json`.
- Scientific package source fingerprint: `c0d1661e0b775c3cd1a75a3343917dd38e5d41fc04d36e72582d6e5b2e0082ff`. Analysis scripts, assets and manuscript are separately hashed in the final evidence inventory; they are not covered by this package fingerprint.
- Current wheel `/tmp/tcg-v4-review-wheel/thecausalitygame-0.1.0-py3-none-any.whl`, SHA-256 `cd6af84f04bfb1f2c4de7a10b971721d58f38e8ce6b10732cbbaf1baa6358cea`. Installed into the existing constrained base-only environment outside the checkout; pip check, full serialized-state continuation, packaged R resources, absent research extras, hybrid generator sampling and graph CLI smoke all pass. This reuses a constrained environment rather than performing a new dependency resolution.
- A fresh construction/sample check under this source revision verifies **52 tasks**, zero failed assets: `runs/catalog-v4/catalog_report.json`. This is not a comparative agent benchmark across 52 tasks or a dataset-license audit.
- Manuscript LaTeX compiled after termination, graph scoring and generator-domain clarification. The reproducibility wording remains unchanged under author decision D5; local tests do not establish cross-platform byte identity. Remote multi-platform CI remains pending review and commit.

## Campaign and figure acceptance

Corrected study root: `runs/publication-v4`. The original manifests and all earlier runs remain preserved.

```sh
TCG_R_LIBS=.r-library MPLCONFIGDIR=/tmp/tcg-mpl-v4 python scripts/problem_instances_rebuttal/run_rebuttal.py --runs-dir runs/publication-v4 --timeout-s 1800
MPLCONFIGDIR=/tmp/tcg-mpl-v4 python scripts/analyze_publication.py --runs-dir runs/publication-v4
```

Resume requires matching package source/normalized manifest hashes and all active agent transcripts with finite final scores. Analysis requires all 21 matching manifests, the full 147-agent inventory, 21 distinct resolved SCM states and 21 distinct acquired-data traces per agent configuration before exporting a treatment-only figure and contrasts. It rejects duplicate effective seeds and mixed-source latest timestamps. Historical graph data are not pooled with new treatment data.

Honest-tree rounds before enough observations exist in both treatment arms retain explicitly labelled provisional zero estimates. Count these separately; a provisional estimate is not evidence of a completed upstream fit. The current anytime figure includes this deployed estimator convention.

## Remaining publication work

The broader campaign of 189 manifests across nine families, promised missing metric/baseline families, dataset provenance/licenses/splits, Croissant metadata, strict whole-project typing, fitted-history reconstruction and remote CI remain open in the root task trackers. This change does not certify all manuscript claims or all submission requirements.

## Final results

Completed successfully: **21 distinct effective SCM states, 147 valid agent trajectories, 14,700 scored rounds**, zero failed manifests and zero missing final scores. All seven agent configurations have **21 distinct acquired-data traces**. All 147 final responses are valid forced-stop responses; the 147 initial provisional zero-estimate rounds are explicitly recorded. Campaign elapsed time: **1,099.67 seconds (18.33 minutes)**.

The PNG export and independently rendered PDF were visually checked. Current outputs:

- `runs/publication-v4/analysis/fig04_treatment_effect_repaired.pdf` and `.png`.
- `runs/publication-v4/analysis/treatment_contrasts.csv` and `round_cache.json`.
- `runs/publication-v4/analysis/report.json`: resolved-state and input-trace checks, counts and contrasts.
- `docs/publication_validation_v4.json`: durable verification summary, source/analysis/manifest/output hashes and all 147 raw transcript hashes.

Matched-seed 95% bootstrap intervals (left minus right; negative favors left):

| Contrast | Estimate | 95% interval | Seeds |
| --- | ---: | --- | ---: |
| Intervention-heavy minus balanced, final PEHE | 0.022330 | [-0.014013, 0.055276] | 21 |
| Intervention-heavy minus balanced, anytime PEHE AUC | 0.009304 | [-0.018996, 0.038518] | 21 |
| ABCI-balanced minus optimal-effect-design-balanced, final PEHE | -0.000941 | [-0.032336, 0.034110] | 21 |
| ABCI-balanced minus optimal-effect-design-balanced, anytime PEHE AUC | -0.013533 | [-0.023967, -0.002597] | 21 |

Three intervals include zero; only the decider-family anytime contrast excludes zero. These are bounded Hill/honest-tree results under the declared acquisition and provisional-estimate convention, not universal superiority claims. The old degenerate intervals and any narrative based on them must not be reused. Historical graph figures and manuscript empirical claims were not silently replaced by this treatment-only evidence.

The local implementation and corrected treatment study are ready for author review. Nothing has been committed or pushed.
