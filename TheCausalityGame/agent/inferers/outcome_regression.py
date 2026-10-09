"""Authoritative S-learner adapter backed by causalToolbox."""

from __future__ import annotations

from typing import override

from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.agent.inferers._r_cate import AuthoritativeRCATEInferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.lib.errors.inferer import ObsoleteInfererParameterError

CAUSAL_TOOLBOX_COMMIT = "75acc74ae689fddebe63a43597ff21f3ec233bd4"


class OutcomeRegressionInferer(AuthoritativeRCATEInferer):
    """Use causalToolbox's author-maintained random-forest S-learner."""

    backend_method = "s_learner"
    capability = "outcome_regression"
    provenance = BackendProvenance(
        project="forestry-labs/causalToolbox",
        algorithm="S_RF",
        version=CAUSAL_TOOLBOX_COMMIT,
        source=f"https://github.com/forestry-labs/causalToolbox/tree/{CAUSAL_TOOLBOX_COMMIT}",
        citation="Künzel et al., PNAS 116(10), 2019",
        license="GPL-3.0",
    )

    def __init__(self, *, nthread: int = 1, seed: int = 911) -> None:
        self.nthread = int(nthread)
        self.seed = int(seed)
        super().__init__(nthread=self.nthread, seed=self.seed)

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> OutcomeRegressionInferer:
        params = spec.params or {}
        if "alpha" in params:
            raise ObsoleteInfererParameterError(
                "alpha configured the empirical ridge S-learner and is not an S_RF parameter."
            )
        return cls(**params)
