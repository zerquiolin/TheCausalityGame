## Area Chair

We thank the Area Chair for the careful synthesis of the reviews. We are concerned that the assessment may have shifted from the paper’s stated contribution toward a different requirement: whether the experiments establish broader scientific insights about causal-agent strategies. The paper instead contributes an interactive evaluation protocol and extensible benchmark that jointly measures inference quality and the process used to obtain it—experiment selection, evidence acquisition, resource use, and stopping.

We respectfully ask that the submission be assessed against the central contribution claimed in the paper and the scope of the E&D Track. The paper does not claim a new causal algorithm or a definitive ranking of existing methods. It contributes an interactive evaluation protocol and an extensible benchmark that measures both causal inference quality and the process used to obtain it: experiment selection, evidence acquisition, resource consumption, and stopping.

This contribution directly aligns with the Track’s emphasis on new evaluation protocols, interactive evaluations, benchmarking tools, and frameworks that strengthen how evaluative claims are constructed and interpreted. The track also explicitly does not require insight papers. This is why our results are rather examples of usage of the evaluation protocol rather than specific insights, which we do not want to convey. 

The existing experiments demonstrate why this evaluation capability is useful. In the whole-trajectory RC-circuit analysis, which is solely a different visualization of the already existing results, fixing the inferer while changing the decider substantially changes SHD AUC. Conversely, fixing ActiveGBN-inspired while changing the inferer moves the AUC from approximately 1.6 with `dag_discovery` to approximately 5.4 with NOTEARS/LiNGAM. Thus, both the evidence-acquisition strategy and the inference procedure materially affect evaluation outcomes.

A fixed-dataset benchmark cannot express this result because it contains neither a decider nor an acquisition trajectory, behavior score, resource profile, or stopping decision. It therefore cannot support the decomposition enabled by our protocol.

We agree with the reviewers that these empirical implications should be presented more prominently, and we will revise the paper accordingly. We hope this evidence clarifies that the paper’s contribution is the evaluation capability itself, supported by representative agent comparisons rather than an exhaustive ranking of causal algorithms. Requiring a broader ranking study would assess the paper against a contribution it does not claim and that the E&D Track does not require.

## Reviewer 1Ggn

We thank the reviewer for emphasizing the importance of experimental strategies and are pleased that we agree on this point. However, we respectfully clarify that the paper’s central contribution is not to establish the broader scientific claim that experimental design matters. Our paper is not intended as an insight paper; the importance of experimental design is a premise of our work, for which we nevertheless provide empirical evidence. The paper’s stated contribution is an evaluation framework that makes acquisition strategy, inference quality, and interaction cost jointly measurable. To this end, we made a significant effort of re-implementing 26 methods, which divide into 9 for inferers and 17 for deciders for the three inference tasks. This is as today the most complete code repository for this kind of algorithms, usable out of the box for any customized causal inference problem instance.
We hope that the reviewer may reconsider the assessment of our paper, focusing the verdict on what we actually claim to deliver.

With respect to your question for evidence for the fact that experiment design strategies matter, Figure 2 and the whole-trajectory summary, which is solely a different visualization of the already existing results, demonstrate that the framework captures significant differences among agent configurations. In the RC-circuit results, with `dag_discovery` fixed, changing the decider substantially changes whole-trajectory SHD AUC: ActiveGBN-inspired achieves the lowest value, whereas CAASLOnline-inspired is markedly higher. Conversely, with ActiveGBN-inspired fixed, changing the inferer moves AUC from approximately 1.6 with `dag_discovery` to approximately 5.4 with NOTEARS or LiNGAM. Thus, the evaluation distinguishes effects attributable to experiment selection, inference, and their composition. These results show that (i) experiment selection affects performance, (ii) inference affects performance, and (iii) both must be evaluated jointly.

Many additional questions about individual strategies could be studied. This breadth motivates an extensible framework, which is exactly what we promise and deliver, but its application to those many questions lies beyond the empirical scope of the present submission. We implemented a significant number of algorithms, and the figures are intended to demonstrate that the framework detects meaningful differences among selected strategies, not to support exhaustive or algorithm-specific conclusions. Such conclusions are not the paper’s objective. A fixed-dataset benchmark cannot support this form of comparison because it contains no experiment-selection policy, acquisition trajectory, or interaction cost. We agree that this distinction, together with the experiment-specific protocol, should be stated more prominently in the main text. We will make these implications explicit and add a compact table specifying the available actions, costs, budgets, stopping rules, seeds, and metrics.

## Reviewer gx9G

We thank the reviewer for recognizing the value of active evaluation and for identifying two aspects that should be presented more clearly. We respectfully clarify that the intended implication of our experiments is evaluative: we do not claim that any particular causal discovery algorithm is universally superior. Rather, the experiments demonstrate substantial heterogeneity across acquisition–inference configurations, motivating an evaluation protocol in which acquisition strategy, inference quality, and interaction cost are measured jointly.

Figure 2 provides two concrete examples. For CBED, LiNGAM and NOTEARS achieve the same mean final SHD of 4.4 but behavior scores of 44 and 63, respectively—a 19-point difference that an evaluation based only on the final answer would conceal. Conversely, with `IDDagDiscovery` fixed, ActiveGBN, GIES, and CBED achieve final SHD values of 1.5, 1.7, and 2.0 at behavior scores of 72, 59, and 50, respectively. Thus, improvements in final accuracy can be accompanied by greater interaction cost, and configurations that appear equivalent under final-score evaluation can differ substantially in their use of resources.

The whole-trajectory analysis, which is solely a different visualization of the already existing results, reinforces this conclusion. With `dag_discovery` fixed, changing the decider materially changes whole-trajectory SHD AUC. Conversely, with ActiveGBN-inspired fixed, changing the inferer moves AUC from approximately 1.6 with `dag_discovery` to approximately 5.4 with NOTEARS or LiNGAM. These results show that (i) acquisition strategy affects performance, (ii) inference affects performance, and (iii) both must be evaluated jointly. The purpose of the protocol is therefore not to establish a universal algorithm ranking, but to expose distinctions that matter under different accuracy–cost preferences.

A fixed-dataset benchmark (without interactive component) cannot support this evaluation because it contains no experiment-selection policy, acquisition trajectory, or interaction cost. It therefore cannot determine how acquisition, inference, and their composition contribute to an agent’s performance. We agree that this implication and the experiment-specific protocol should be stated more prominently in the main text. We will make the distinction explicit and add a compact table consolidating the legal actions, costs, budgets, stopping rules, seeds, and metrics currently specified across Sections 2–3 and Appendix F.

## Reviewer DiPF

We thank the reviewer for the detailed assessment and repository inspection. The paper contributes an interactive evaluation framework rather than a definitive ranking of causal methods. We address each concern and question individually.

### Weaknesses

**W1. Limited experimental validation.**  
The evaluation covers graph discovery, mechanism recovery, and treatment-effect estimation, with 20 runs per configuration. Its purpose is to demonstrate what becomes measurable under interaction. The whole-trajectory analysis, which is solely a different visualization of the already existing results, strengthens this evidence: both decider and inferer choices materially change performance. We provide the specific interactive findings under Q2.

**W2. “Game” framing.**  
“Game” denotes a configurable, goal-directed environment in which an agent acquires evidence, incurs costs, and decides when to stop. The evaluated setting is a single-agent sequential game, not a game-theoretic model of strategic interaction. Multiple agents can be evaluated independently under the same environment and ranked by the result and behavior metrics, forming a benchmark tournament. The modular interfaces could also support alternative utilities, reward designs, or interacting agents, but these extensions are not studied in the current paper. We will state this scope explicitly.

**W3. Approximate implementations.**  
The bundled agents are method-inspired implementations, not faithful reproductions. Appendix J states the relevant substitutions—for example, bootstrap ensembles replace ABCI’s Bayesian DAG/BNN inference. Consequently, the results characterize the implemented agents and do not establish rankings or guarantees for the original methods.

<!-- Maybe it is not necessary to respond this -->

**W4. Breadth versus depth.**  
The breadth establishes that one protocol supports three distinct causal missions, heterogeneous SCMs, interchangeable agents, and common behavior metrics. The experiments are representative evaluations of framework functionality, not exhaustive comparisons. The whole-trajectory analysis provides a focused empirical conclusion without changing that scope.

### Questions

**Q1. Were the approximations validated against original code or published results?**  
No. The original implementations were not evaluated, so we do not interpret the results as reproductions or comparisons of the original methods. Our adaptations instead provide a common online interface and tractable per-round execution across heterogeneous environments, the revised terminology will identify them consistently as method-inspired agents and distinguish preserved ideas from replaced machinery and unavailable guarantees.

**Q2. What non-obvious finding is revealed only by interactive evaluation?**  
Figure 2 shows two such findings. First, CBED with LiNGAM and NOTEARS reaches the same mean final SHD of 4.4 but behavior scores of 44 and 63 a 19 point resource difference concealed by final-answer evaluation. Second, with IDDagDiscovery fixed, ActiveGBN, GIES, and CBED obtain SHD values of 1.5, 1.7, and 2.0 at behavior scores of 72, 59, and 50, respectively. Thus, similar accuracy can require different interaction cost, and improved accuracy can trade against resource use.

The whole-trajectory analysis further shows that, with ActiveGBN-inspired fixed, changing the inferer moves anytime SHD AUC from approximately 1.6 with dag_discovery to 5.4 with NOTEARS/LiNGAM. Thus, a fixed-dataset benchmark can treat configurations as equivalent despite different costs and cannot attribute differences to acquisition, inference, or their composition. In particular, among these configurations, no single Pareto dominand solution is best in both behavior and score: the preferred agent depends on the evaluator’s accuracy–cost trade-off.

**Q3. How difficult would an AGFN-style stateful agent be to implement?**  
The Agent interface supports persistent state and standardized observations, making an AGFN-style distribution over graphs compatible with the interaction protocol. Expert queries and feedback could be exchanged through a CLI adapter. However, implementing AGFN’s graph sampler, active query selection, and Bayesian belief updates would remain substantive method-specific work.

**Q4. What are the challenges for LLM agents, and how does the framework compare with CausaLab?**  
CausaLab provides a specialized environment and a deeper empirical analysis of LLM causal discovery. The Causality Game instead offers a model-agnostic protocol spanning graph discovery, mechanism estimation, and treatment-effect estimation, with heterogeneous SCM sources, configurable costs, budgets, and stopping rules. Its action–observation contracts can support an LLM agent through an adapter. A rigorous LLM evaluation would additionally require explicit action validation, memory policies, token and latency accounting, and leakage controls.

**Dataset metadata.**  
IHDP, Jobs, and Twins are documented in the repository, but Croissant metadata is absent. We acknowledge this documentation gap and will add the corresponding provenance, licensing, schema, and preprocessing metadata.
