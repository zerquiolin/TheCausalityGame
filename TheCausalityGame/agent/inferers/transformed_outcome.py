"""Authoritative transformed-outcome tree adapter backed by causalTree."""

from __future__ import annotations

from typing import override

from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.agent.inferers._r_cate import AuthoritativeRCATEInferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.lib.errors.inferer import ObsoleteInfererParameterError

CAUSAL_TREE_COMMIT = "51a8762bff0ab17b971bce852cb37bece2d8240f"


class TransformedOutcomeInferer(AuthoritativeRCATEInferer):
    """Use causalTree's published ``TOT`` splitting and CV rule."""

    backend_method = "transformed_outcome"
    capability = "transformed_outcome"
    provenance = BackendProvenance(
        project="susanathey/causalTree",
        algorithm="causalTree split.Rule=TOT",
        version=CAUSAL_TREE_COMMIT,
        source=f"https://github.com/susanathey/causalTree/tree/{CAUSAL_TREE_COMMIT}",
        citation="Athey and Imbens, PNAS 113(27), 2016",
        license="GPL-3.0",
    )

    def __init__(
        self,
        *,
        minsize: int = 2,
        xval: int = 5,
        cp: float = 0.0,
        split_bucket: bool = False,
        seed: int = 911,
    ) -> None:
        self.minsize = int(minsize)
        self.xval = int(xval)
        self.cp = float(cp)
        self.split_bucket = bool(split_bucket)
        self.seed = int(seed)
        super().__init__(
            minsize=self.minsize,
            xval=self.xval,
            cp=self.cp,
            split_bucket=self.split_bucket,
            seed=self.seed,
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> TransformedOutcomeInferer:
        params = spec.params or {}
        obsolete = {"alpha", "propensity_alpha", "min_group_size", "propensity_clip"}
        if obsolete & params.keys():
            raise ObsoleteInfererParameterError(
                "The supplied parameters belong to the empirical transformed-outcome regressor."
            )
        return cls(**params)
