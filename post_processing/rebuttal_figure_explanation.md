# Rebuttal explanation script: graph-discovery figures

This script is written for a roughly four-minute explanation. All reported
intervals are deterministic 95% paired- or seed-bootstrap percentile intervals.
Final-SHD analyses use 21 seeds; the whole-trajectory visualization uses the
lowest 20 numerical seed IDs, excluding seed 989 by a deterministic,
performance-independent rule. SHD and anytime SHD AUC are errors, so lower is
better. Agent labels refer to method-inspired implementations, not exact
reproductions of the named methods.

## Opening

> These figures address a limitation of reporting only terminal performance or
> performance by round. In this framework, a round is not a fixed unit of
> information: different deciders acquire different numbers and mixtures of
> observational and interventional samples. We therefore compare methods using
> the evidence actually collected, and we treat seeds—not rounds—as the
> independent experimental units.

## Figure 1: error versus accumulated evidence

File: `fig02_error_vs_accumulated_evidence`

> The first three panels align every run by cumulative total, observational, and
> interventional samples. At each budget, the curve uses the latest estimate
> already available at that budget; it never uses a future estimate. The bands
> describe uncertainty across the 20 matched seeds used for this visualization.
>
> The main observation is that acquisition and inference cannot be evaluated
> independently. With PC, the ActiveGBN-inspired and balanced ABCI-inspired
> policies have similar whole-trajectory error: their normalized SHD AUCs are
> 3.52 and 3.31. Their paired difference is -0.21, with a 95% interval from
> -0.48 to 0.10, so we do not claim a reliable advantage between them under PC.
>
> Under LiNGAM, the same comparison changes substantially. The
> ActiveGBN-inspired trajectory has an AUC of 5.39, whereas the balanced
> ABCI-inspired trajectory has an AUC of 7.65. The paired difference is 2.26
> SHD-AUC units, with a 95% interval from 1.92 to 2.60. Thus, the balanced
> ABCI-inspired acquisition path is markedly less compatible with LiNGAM,
> although it is competitive with ActiveGBN-inspired acquisition under PC.
>
> Separating observational and interventional evidence is important because the
> total-sample panel alone hides how that evidence was obtained. These panels
> show the framework's value: it exposes the acquisition path, the downstream
> inferer, and their joint effect on error at a common evidence budget.

Transition:

> The next figure asks whether this is an isolated four-configuration result or
> part of a broader decider-by-inferer pattern.

## Figure 2: full decider-by-inferer interaction

File: `supp01_full_interaction_heatmap`

> The heatmap reports mean final SHD for every implemented decider-inferer
> combination. Yellow denotes lower error and purple denotes higher error. The
> right panel sorts configurations within each inferer and shows the shape and
> spread of its performance profile; it does not track the identity of one
> decider across ranks.
>
> The heatmap shows that no single decider has a context-free performance
> ranking. With the generic DAG-discovery inferer, mean final SHD ranges from
> 1.33 to 2.14. With PC, the deciders are much more tightly clustered, from 3.48
> to 3.90. In contrast, LiNGAM ranges from 3.67 to 6.33, and NOTEARS ranges from
> 3.62 to 6.19.
>
> This variation is the substantive result: the inferer changes both the
> absolute error level and how sensitive performance is to the acquisition
> policy. In particular, the balanced ABCI-inspired policy is among the
> lowest-error PC combinations at 3.48 SHD, but reaches 6.33 with LiNGAM. The
> ActiveGBN-inspired policy moves from 3.71 with PC to 3.67 with LiNGAM. This
> reversal motivates testing an interaction rather than declaring one decider
> universally superior.

Transition:

> The heatmap is an aggregate view. We next verify the key interaction within
> the same problem seeds.

## Figure 3: matched-seed interaction

File: `supp02_paired_seed_interaction`

> Each gray line is one seed. Its vertical value is balanced ABCI-inspired SHD
> minus ActiveGBN-inspired SHD, so values below zero favor ABCI-inspired
> acquisition and values above zero favor ActiveGBN-inspired acquisition.
>
> Under PC, the mean paired difference is -0.24 SHD, with a 95% interval from
> -1.19 to 0.71. The seed-level outcomes are mixed: ABCI-inspired acquisition is
> better on 8 seeds, tied on 4, and worse on 9. We therefore interpret the PC
> contrast as uncertain.
>
> Under LiNGAM, the mean paired difference rises to 2.67 SHD, with a 95%
> interval from 1.95 to 3.38. ABCI-inspired acquisition is worse on 18 of the
> 21 seeds, tied on 2, and better on only 1.
>
> The matched-seed difference-in-differences is 2.90 SHD, with a 95% interval
> from 1.71 to 4.29. This is the central non-additive result: changing the
> inferer changes the relative effect of the acquisition policy. The analysis
> uses 21 seeds as the independent units and does not inflate the sample size by
> treating rounds as independent observations.

Transition:

> Finally, we test whether this conclusion depends on looking only at the last
> estimate.

## Figure 4: anytime AUC forest

File: `supp04_anytime_auc_forest`

> This forest plot summarizes the complete error trajectory through 600
> collected samples. The normalized anytime AUC is the average left-continuous
> SHD over that evidence range, so it rewards configurations that obtain useful
> estimates early and maintain them. Lower values are better; each interval
> describes uncertainty over the same 20 matched seeds.
>
> The forest reveals three broad performance regimes. Configurations using the
> generic DAG-discovery inferer are strongest, led by
> ActiveGBN-inspired acquisition at an AUC of approximately 1.57.
> PC configurations form a compact middle group around 3.31 to 3.52. LiNGAM and
> NOTEARS configurations are more acquisition-sensitive and extend from about
> 5.35 to 8.65.
>
> The same interaction seen at the final estimate therefore persists across the
> full trajectory. For ABCI-inspired versus ActiveGBN-inspired acquisition, the
> AUC interaction difference is 2.47, with a paired-bootstrap interval from
> 2.09 to 2.86. This makes the conclusion stronger than a terminal-only result:
> the incompatibility affects the quality of estimates throughout evidence
> acquisition, not merely at the stopping point.

## Closing

> Taken together, these figures show why the game abstraction is useful. It does
> not merely rank estimators on a fixed dataset. It measures how a decider
> gathers observational and interventional evidence, how an inferer converts
> that evidence into a causal graph, and how their compatibility changes both
> terminal accuracy and anytime performance. The defensible claim is not that
> one implemented method is universally best; it is that acquisition and
> inference exhibit a reproducible, practically important interaction.

## Compact written-rebuttal version

> To separate round count from information acquired, we aligned graph-discovery
> runs by actual cumulative observational and interventional samples and used
> normalized anytime SHD AUC through 600 samples. Under PC, balanced
> ABCI-inspired versus ActiveGBN-inspired acquisition produced an uncertain AUC
> difference of -0.21 (95% paired-bootstrap CI [-0.48, 0.10]). Under LiNGAM,
> however, the corresponding difference was 2.26 [1.92, 2.60], favoring
> ActiveGBN-inspired acquisition. The final-SHD matched-seed interaction was
> 2.90 [1.71, 4.29] across 21 seeds, and the anytime-AUC interaction was 2.47
> [2.09, 2.86] across the deterministic 20-seed subset. The full
> decider-by-inferer heatmap shows that this is part of
> a broader pattern: PC configurations are comparatively insensitive to the
> decider, whereas LiNGAM and NOTEARS exhibit substantially larger performance
> spreads. These results support a joint acquisition-inference evaluation rather
> than a context-free ranking of deciders.

## Claims to avoid

- Do not call the method-inspired agents exact reproductions of the cited
  methods.
- Do not claim that overlapping or non-overlapping marginal confidence
  intervals constitute a hypothesis test; use the matched contrasts where
  available.
- Do not call rounds independent replicates.
- Do not say that more interventional data alone causes lower error. The plots
  compare complete acquisition-inference configurations.
- Do not describe the right panel of the heatmap as tracking individual agents;
  it shows sorted within-inferer performance profiles.
- Do not claim universal superiority. The primary evidence is an interaction:
  relative performance depends on the inferer.
