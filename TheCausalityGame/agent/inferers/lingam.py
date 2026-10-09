"""Authoritative adapter for the official DirectLiNGAM package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import override

import networkx as nx

from TheCausalityGame.agent.inferers._data import (
    adjacency_to_digraph,
    numeric_matrix,
    observational_frame,
)
from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.inferer import (
    InfererDependencyError,
    ObsoleteInfererParameterError,
)


class LiNGAMInferer(Inferer):
    """Delegate structure learning to ``lingam.DirectLiNGAM``."""

    def __init__(
        self,
        *,
        random_state: int = 911,
        measure: str = "pwling",
        apply_prior_knowledge_softly: bool = False,
    ) -> None:
        self.random_state = int(random_state)
        self.measure = str(measure)
        self.apply_prior_knowledge_softly = bool(apply_prior_knowledge_softly)
        self._samples: list[Samples] = []
        self._dag: nx.DiGraph | None = None

    @staticmethod
    def _provenance() -> BackendProvenance:
        try:
            installed = version("lingam")
        except PackageNotFoundError:
            installed = "not-installed"
        return BackendProvenance(
            project="cdt15/lingam",
            algorithm="DirectLiNGAM",
            version=installed,
            source="https://github.com/cdt15/lingam",
            citation="Shimizu et al., JMLR 12, 2011",
            license="MIT",
        )

    @override
    def update(self, observation: RoundObservation) -> None:
        self._samples.extend(observation.samples)
        self._dag = None

    @override
    def answer(self) -> nx.DiGraph:
        if self._dag is not None:
            return self._dag
        frame = observational_frame(self._samples)
        if frame.empty:
            self._dag = nx.DiGraph()
            return self._dag
        data = numeric_matrix(frame, algorithm="DirectLiNGAM")
        try:
            from lingam import DirectLiNGAM
        except ImportError as exc:
            raise InfererDependencyError(
                "LiNGAMInferer requires lingam==1.13.0; install the authoritative extra."
            ) from exc
        model = DirectLiNGAM(
            random_state=self.random_state,
            measure=self.measure,
            apply_prior_knowledge_softly=self.apply_prior_knowledge_softly,
        )
        model.fit(data)
        # DirectLiNGAM stores child-by-parent coefficients; transpose for source-by-target.
        self._dag = adjacency_to_digraph(
            model.adjacency_matrix_,
            list(frame.columns),
            transpose=True,
        )
        return self._dag

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={
                "observational_rows": len(observational_frame(self._samples)),
                "backend": self._provenance().to_dict(),
            },
            capabilities=("dag_discovery", "lingam"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "random_state": self.random_state,
                "measure": self.measure,
                "apply_prior_knowledge_softly": self.apply_prior_knowledge_softly,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> LiNGAMInferer:
        params = spec.params or {}
        obsolete = {"ridge", "coef_threshold"} & params.keys()
        if obsolete:
            raise ObsoleteInfererParameterError(
                f"Parameters {sorted(obsolete)} belong to the empirical approximation."
            )
        return cls(**params)
