"""Authoritative adapter for the authors' linear NOTEARS implementation."""

from __future__ import annotations

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

NOTEARS_COMMIT = "4a9ab19fe502e392503da331773c5223d82d3666"


class NOTEARSInferer(Inferer):
    """Delegate linear structure learning to ``notears.linear.notears_linear``."""

    def __init__(
        self,
        *,
        lambda1: float = 0.1,
        loss_type: str = "l2",
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ) -> None:
        self.lambda1 = float(lambda1)
        self.loss_type = str(loss_type)
        self.max_iter = int(max_iter)
        self.h_tol = float(h_tol)
        self.rho_max = float(rho_max)
        self.w_threshold = float(w_threshold)
        self._samples: list[Samples] = []
        self._dag: nx.DiGraph | None = None

    @staticmethod
    def _provenance() -> BackendProvenance:
        return BackendProvenance(
            project="xunzheng/notears",
            algorithm="NOTEARS linear",
            version=NOTEARS_COMMIT,
            source=f"https://github.com/xunzheng/notears/tree/{NOTEARS_COMMIT}",
            citation="Zheng et al., NeurIPS 2018",
            license="Apache-2.0",
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
        data = numeric_matrix(frame, algorithm="NOTEARS")
        try:
            from notears.linear import notears_linear
        except ImportError as exc:
            raise InfererDependencyError(
                "NOTEARSInferer requires the pinned xunzheng/notears dependency; "
                "install the authoritative extra."
            ) from exc
        adjacency = notears_linear(
            data,
            lambda1=self.lambda1,
            loss_type=self.loss_type,
            max_iter=self.max_iter,
            h_tol=self.h_tol,
            rho_max=self.rho_max,
            w_threshold=self.w_threshold,
        )
        self._dag = adjacency_to_digraph(adjacency, list(frame.columns))
        return self._dag

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={
                "observational_rows": len(observational_frame(self._samples)),
                "backend": self._provenance().to_dict(),
            },
            capabilities=("dag_discovery", "notears"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "lambda1": self.lambda1,
                "loss_type": self.loss_type,
                "max_iter": self.max_iter,
                "h_tol": self.h_tol,
                "rho_max": self.rho_max,
                "w_threshold": self.w_threshold,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> NOTEARSInferer:
        params = spec.params or {}
        if "lambda_l2" in params:
            raise ObsoleteInfererParameterError(
                "lambda_l2 belonged to the empirical NOTEARS approximation; use lambda1."
            )
        return cls(**params)
