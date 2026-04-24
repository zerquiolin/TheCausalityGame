"""PC-based inferer for DAG discovery."""

from __future__ import annotations

from itertools import combinations
from typing import override

import networkx as nx
import pandas as pd

from TheCausalityGame.agent.inferers._dag_common import empty_graph, observational_frame
from TheCausalityGame.agent.inferers.dag import (
    apply_meek_r1,
    ci_test_continuous_nonlinear,
    ci_test_discrete_chi2_stratified,
    ci_test_gaussian,
    orient_v_structures,
)
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path

GAUSSIAN_CI_MAX_CONDITIONING = 2


class PCInferer(Inferer):
    """Discover a DAG from observational data using PC-style CI tests."""

    def __init__(self, *, is_numerical: bool = True, alpha: float = 0.05) -> None:
        self.is_numerical = is_numerical
        self.alpha = alpha
        self._samples: list[Samples] = []
        self._dag: nx.DiGraph | None = None

    @override
    def update(self, observation: RoundObservation) -> None:
        self._samples.extend(observation.samples)
        self._dag = None

    @override
    def answer(self) -> nx.DiGraph:
        if self._dag is not None:
            return self._dag

        obs_df = observational_frame(self._samples)
        if obs_df.empty:
            self._dag = nx.DiGraph()
            return self._dag

        skeleton, sep_sets = self._pc_skeleton(obs_df)
        directed = orient_v_structures(skeleton, sep_sets)
        directed = apply_meek_r1(skeleton, directed)

        dag = empty_graph(list(obs_df.columns))
        dag.add_edges_from(directed)
        for source, target in sorted(skeleton.edges()):
            if dag.has_edge(source, target) or dag.has_edge(target, source):
                continue
            dag.add_edge(source, target)
            if nx.is_directed_acyclic_graph(dag):
                continue
            dag.remove_edge(source, target)
            dag.add_edge(target, source)
            if nx.is_directed_acyclic_graph(dag):
                continue
            dag.remove_edge(target, source)
        self._dag = dag
        return dag

    @override
    def snapshot(self) -> BeliefSnapshot:
        obs_rows = len(observational_frame(self._samples))
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"observational_rows": obs_rows},
            capabilities=("dag_discovery", "pc"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "is_numerical": self.is_numerical,
                "alpha": self.alpha,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> PCInferer:
        params = spec.params or {}
        return cls(
            is_numerical=bool(params.get("is_numerical", True)),
            alpha=float(params.get("alpha", 0.05)),
        )

    def _pc_skeleton(
        self,
        obs_df: pd.DataFrame,
    ) -> tuple[nx.Graph, dict[frozenset[str], set[str]]]:
        """Run a small PC skeleton search over observational data."""
        columns = list(obs_df.columns)
        graph = nx.Graph()
        graph.add_nodes_from(columns)
        for source, target in combinations(columns, 2):
            graph.add_edge(source, target)

        sep_sets: dict[frozenset[str], set[str]] = {}
        cond_size = 0
        while True:
            removed_any = False
            edges_snapshot = list(graph.edges())
            for source, target in edges_snapshot:
                if not graph.has_edge(source, target):
                    continue

                search_space = (
                    [node for node in graph.neighbors(source) if node != target],
                    [node for node in graph.neighbors(target) if node != source],
                )
                tested = False
                for neighbors in search_space:
                    if len(neighbors) < cond_size:
                        continue
                    tested = True
                    for conditioning in combinations(neighbors, cond_size):
                        if self._independent(obs_df, source, target, conditioning):
                            graph.remove_edge(source, target)
                            sep_sets[frozenset((source, target))] = set(conditioning)
                            removed_any = True
                            break
                    if not graph.has_edge(source, target):
                        break

                if tested:
                    continue

            max_neighbors = max(
                (len(list(graph.neighbors(node))) for node in graph.nodes()),
                default=0,
            )
            if max_neighbors <= cond_size:
                break
            if not removed_any and cond_size >= max_neighbors:
                break
            cond_size += 1

        return graph, sep_sets

    def _independent(
        self,
        obs_df: pd.DataFrame,
        source: str,
        target: str,
        conditioning: tuple[str, ...],
    ) -> bool:
        """Dispatch to the configured CI test."""
        if self.is_numerical:
            if len(conditioning) <= GAUSSIAN_CI_MAX_CONDITIONING:
                return ci_test_gaussian(obs_df, source, target, conditioning, self.alpha)
            return ci_test_continuous_nonlinear(
                obs_df,
                source,
                target,
                conditioning,
                self.alpha,
                n_perm=50,
                seed=911,
            )
        return ci_test_discrete_chi2_stratified(
            obs_df,
            source,
            target,
            conditioning,
            self.alpha,
        )
