"""Authoritative adapter for the PC algorithm from causal-learn."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, override

from TheCausalityGame.agent.inferers._data import numeric_matrix, observational_frame
from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.inferer import (
    InfererDependencyError,
    ObsoleteInfererParameterError,
)


class PCInferer(Inferer):
    """Run causal-learn's PC implementation and preserve its CPDAG output."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        indep_test: str = "fisherz",
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        show_progress: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.indep_test = str(indep_test)
        self.stable = bool(stable)
        self.uc_rule = int(uc_rule)
        self.uc_priority = int(uc_priority)
        self.show_progress = bool(show_progress)
        self._samples: list[Samples] = []
        self._graph: PartiallyDirectedGraph | None = None

    @staticmethod
    def _provenance() -> BackendProvenance:
        try:
            installed = version("causal-learn")
        except PackageNotFoundError:
            installed = "not-installed"
        return BackendProvenance(
            project="py-why/causal-learn",
            algorithm="PC",
            version=installed,
            source="https://github.com/py-why/causal-learn",
            citation="Zheng et al., JMLR 25(60), 2024; Spirtes et al., 2000",
            license="MIT",
        )

    @override
    def update(self, observation: RoundObservation) -> None:
        self._samples.extend(observation.samples)
        self._graph = None

    @staticmethod
    def _endpoint_name(endpoint: Any) -> str:
        name = getattr(endpoint, "name", None)
        return str(name if name is not None else endpoint).lower()

    @classmethod
    def _convert_graph(
        cls, causal_graph: Any, columns: list[str]
    ) -> PartiallyDirectedGraph:
        directed: set[tuple[str, str]] = set()
        undirected: set[tuple[str, str]] = set()
        for edge in causal_graph.G.get_graph_edges():
            node1 = str(edge.get_node1().get_name())
            node2 = str(edge.get_node2().get_name())
            endpoint1 = cls._endpoint_name(edge.get_endpoint1())
            endpoint2 = cls._endpoint_name(edge.get_endpoint2())
            if "tail" in endpoint1 and "arrow" in endpoint2:
                directed.add((node1, node2))
            elif "arrow" in endpoint1 and "tail" in endpoint2:
                directed.add((node2, node1))
            else:
                undirected.add((node1, node2) if node1 < node2 else (node2, node1))
        return PartiallyDirectedGraph(
            nodes=tuple(columns),
            directed_edges=tuple(sorted(directed)),
            undirected_edges=tuple(sorted(undirected)),
        )

    @override
    def answer(self) -> PartiallyDirectedGraph:
        if self._graph is not None:
            return self._graph
        frame = observational_frame(self._samples)
        if frame.empty:
            self._graph = PartiallyDirectedGraph()
            return self._graph
        data = numeric_matrix(frame, algorithm="PC")
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError as exc:
            raise InfererDependencyError(
                "PCInferer requires causal-learn==0.1.4.8; install the authoritative extra."
            ) from exc
        result = pc(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            stable=self.stable,
            uc_rule=self.uc_rule,
            uc_priority=self.uc_priority,
            show_progress=self.show_progress,
            node_names=list(frame.columns),
        )
        self._graph = self._convert_graph(result, list(frame.columns))
        return self._graph

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={
                "observational_rows": len(observational_frame(self._samples)),
                "backend": self._provenance().to_dict(),
            },
            capabilities=("dag_discovery", "pc"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "alpha": self.alpha,
                "indep_test": self.indep_test,
                "stable": self.stable,
                "uc_rule": self.uc_rule,
                "uc_priority": self.uc_priority,
                "show_progress": self.show_progress,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> PCInferer:
        params = spec.params or {}
        if "is_numerical" in params:
            raise ObsoleteInfererParameterError(
                "is_numerical configured the empirical PC approximation; use causal-learn's "
                "indep_test parameter, for example 'fisherz' or 'chisq'."
            )
        return cls(**params)
