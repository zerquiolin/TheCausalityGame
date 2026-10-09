"""Authoritative honest causal-tree adapter backed by causalTree."""

from __future__ import annotations

from typing import override

from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.agent.inferers._r_cate import AuthoritativeRCATEInferer
from TheCausalityGame.agent.inferers.transformed_outcome import CAUSAL_TREE_COMMIT
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.lib.errors.inferer import ObsoleteInfererParameterError


class HonestCausalTreeInferer(AuthoritativeRCATEInferer):
    """Use causalTree's one-step honest causal-tree implementation."""

    backend_method = "honest_causal_tree"
    capability = "honest_causal_tree"
    provenance = BackendProvenance(
        project="susanathey/causalTree",
        algorithm="honest.causalTree split.Rule=CT",
        version=CAUSAL_TREE_COMMIT,
        source=f"https://github.com/susanathey/causalTree/tree/{CAUSAL_TREE_COMMIT}",
        citation="Athey and Imbens, PNAS 113(27), 2016",
        license="GPL-3.0",
    )

    def __init__(
        self,
        *,
        honest_fraction: float = 0.5,
        minsize: int = 2,
        xval: int = 5,
        cp: float = 0.0,
        split_bucket: bool = True,
        seed: int = 911,
    ) -> None:
        self.honest_fraction = float(honest_fraction)
        self.minsize = int(minsize)
        self.xval = int(xval)
        self.cp = float(cp)
        self.split_bucket = bool(split_bucket)
        self.seed = int(seed)
        super().__init__(
            honest_fraction=self.honest_fraction,
            minsize=self.minsize,
            xval=self.xval,
            cp=self.cp,
            split_bucket=self.split_bucket,
            seed=self.seed,
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> HonestCausalTreeInferer:
        params = spec.params or {}
        obsolete = {
            "max_depth",
            "min_leaf_size",
            "min_treatment_count",
            "max_candidates_per_feature",
        }
        if obsolete & params.keys():
            raise ObsoleteInfererParameterError(
                "The supplied parameters belong to the empirical handwritten tree."
            )
        if "random_state" in params:
            params = {**params, "seed": params["random_state"]}
            del params["random_state"]
        return cls(**params)
