# Inferer implementations and references

The root modules for published methods are thin adapters. They invoke the pinned upstream code
listed below; no source from the GPL R packages is vendored. Historical handwritten behavior is
preserved under `TheCausalityGame.agent.inferers.empirical`.

| Public inferer | Authoritative code | Pin | License | Primary reference |
| --- | --- | --- | --- | --- |
| PC | [py-why/causal-learn](https://github.com/py-why/causal-learn) `causallearn.search.ConstraintBased.PC.pc` | `causal-learn==0.1.4.8` | MIT | Zheng et al. (2024), “Causal-learn: Causal Discovery in Python,” *JMLR* 25(60), 1–8; Spirtes, Glymour, and Scheines (2000), *Causation, Prediction, and Search*, 2nd ed. |
| NOTEARS | [xunzheng/notears](https://github.com/xunzheng/notears) `notears.linear.notears_linear` | `4a9ab19fe502e392503da331773c5223d82d3666` | Apache-2.0 | Zheng et al. (2018), “DAGs with NO TEARS: Continuous Optimization for Structure Learning,” *NeurIPS 31*. |
| DirectLiNGAM | [cdt15/lingam](https://github.com/cdt15/lingam) `lingam.DirectLiNGAM` | `lingam==1.13.0` | MIT | Shimizu et al. (2011), “DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model,” *JMLR* 12, 1225–1248. |
| Outcome regression / S-learner | [forestry-labs/causalToolbox](https://github.com/forestry-labs/causalToolbox) `S_RF` and `EstimateCate` | `75acc74ae689fddebe63a43597ff21f3ec233bd4` | GPL-3.0-or-later | Künzel et al. (2019), “Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning,” *PNAS* 116(10), 4156–4165. |
| Transformed-outcome tree | [susanathey/causalTree](https://github.com/susanathey/causalTree) `causalTree`, `split.Rule="TOT"`, `cv.option="TOT"` | `51a8762bff0ab17b971bce852cb37bece2d8240f` | GPL-2 or GPL-3 | Athey and Imbens (2016), “Recursive Partitioning for Heterogeneous Causal Effects,” *PNAS* 113(27), 7353–7360. |
| Honest causal tree | [susanathey/causalTree](https://github.com/susanathey/causalTree) `honest.causalTree` with distinct splitting and estimation samples | `51a8762bff0ab17b971bce852cb37bece2d8240f` | GPL-2 or GPL-3 | Athey and Imbens (2016), same reference as above. |

The R bootstrap records nested dependency pins and applies build/API compatibility patches. The pinned `causalToolbox::S_RF` source passes `OOBhonest` to Rforestry, but the pinned Rforestry API has no such argument or model slot. For this adapter's `S_RF` fit plus `EstimateCate` prediction path, the local build omits that unsupported argument and preserves feature names when the model has one covariate. It does not enable OOB-honest inference or local-conformal intervals; do not claim those features. Rforestry's core-count guard also treats `detectCores() == NA` as an unknown count while preserving the requested `nthread`. See `scripts/comparisons/r-backends.lock.json` and the bootstrap script for patch scope.

`DAGDiscoveryInferer`, `SCMDiscoveryInferer`, and `SparseSymbolicSCMDiscoveryInferer` are
project-specific composites. No author-maintained upstream implementation exists, so their public
paths are compatibility wrappers around the empirical implementations and report
`implementation_kind="empirical"`.

For publication setup, the local R build records verified checkout HEAD and patch identity
in `TCGSourceCommit`/`TCGCompatibilityPatch`. The R adapter checks source stamps and reports
them alongside package versions; S_RF also records its nested Rforestry source. A package
version or an expected commit constant alone is insufficient provenance. Older unstamped
local libraries require rerunning `scripts/comparisons/bootstrap_r.R`.
