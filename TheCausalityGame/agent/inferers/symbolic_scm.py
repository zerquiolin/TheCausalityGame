"""Symbolic SCM discovery inferer."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, override

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.linear_model import Lasso

from TheCausalityGame.agent.inferers.dag import learn_dag_from_samples  # type: ignore
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import SamplesCollection
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


@dataclass(frozen=True)
class SymbolicMechanism:
    """A symbolic mechanism for one SCM variable."""

    node: str
    parents: list[str]
    expression: sp.Basic

    def evaluate(self, parent_data: pd.DataFrame) -> np.ndarray:
        """Evaluate the mechanism over parent values."""
        if not self.parents:
            value = float(self.expression)
            return np.full(len(parent_data), value, dtype=float)

        missing = [parent for parent in self.parents if parent not in parent_data.columns]
        if missing:
            raise KeyError(f"Missing parent columns for {self.node}: {missing}")  # noqa: TRY003

        fn = sp.lambdify(self.parents, self.expression, modules="numpy")
        values = fn(*tuple(parent_data[self.parents].to_numpy(dtype=float).T))
        if np.isscalar(values):
            return np.full(len(parent_data), float(values), dtype=float)
        return np.asarray(values, dtype=float)


class EstimatedSymbolicSCM:
    """Estimated SCM represented by symbolic node mechanisms."""

    def __init__(
        self,
        dag: nx.DiGraph,
        mechanisms: dict[str, SymbolicMechanism],
    ) -> None:
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("EstimatedSymbolicSCM requires a DAG.")  # noqa: TRY003
        self.dag = dag
        self.mechanisms = mechanisms

    def mechanism_for(self, node: str) -> SymbolicMechanism:
        """Return the symbolic mechanism for a node."""
        return self.mechanisms[node]

    def evaluate_mechanism(self, node: str, parent_data: pd.DataFrame) -> np.ndarray:
        """Evaluate one symbolic mechanism over parent data."""
        return self.mechanism_for(node).evaluate(parent_data)

    def expressions(self) -> dict[str, str]:
        """Return JSON-friendly expression strings by node."""
        return {node: str(mechanism.expression) for node, mechanism in self.mechanisms.items()}


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean()) >= 0.95


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _candidate_terms(
    df: pd.DataFrame,
    parents: list[str],
    *,
    reciprocal_eps: float,
    min_reciprocal_valid_fraction: float,
) -> list[tuple[sp.Basic, np.ndarray]]:
    """Build symbolic candidate terms and their numeric columns."""
    terms: list[tuple[sp.Basic, np.ndarray]] = [
        (sp.Integer(1), np.ones(len(df), dtype=float))
    ]
    parent_values: dict[str, np.ndarray] = {
        parent: df[parent].to_numpy(dtype=float) for parent in parents
    }
    symbols = {parent: sp.Symbol(parent) for parent in parents}

    for parent in parents:
        values = parent_values[parent]
        symbol = symbols[parent]
        terms.append((symbol, values))
        terms.append((symbol**2, values**2))

    for left, right in combinations(parents, 2):
        terms.append((symbols[left] * symbols[right], parent_values[left] * parent_values[right]))

    reciprocals: dict[str, np.ndarray] = {}
    for parent in parents:
        values = parent_values[parent]
        safe = np.abs(values) > reciprocal_eps
        if float(np.mean(safe)) < min_reciprocal_valid_fraction:
            continue
        reciprocal = np.full(len(values), np.nan, dtype=float)
        reciprocal[safe] = 1.0 / values[safe]
        reciprocals[parent] = reciprocal
        terms.append((1 / symbols[parent], reciprocal))

    for numerator in parents:
        for denominator, reciprocal in reciprocals.items():
            if numerator == denominator:
                continue
            terms.append(
                (
                    symbols[numerator] / symbols[denominator],
                    parent_values[numerator] * reciprocal,
                )
            )

    return terms


def fit_sparse_symbolic_mechanism_for_node(
    *,
    node: str,
    parents: list[str],
    df: pd.DataFrame,
    alpha: float,
    coefficient_threshold: float,
    reciprocal_eps: float,
    min_reciprocal_valid_fraction: float,
) -> SymbolicMechanism | None:  # noqa: PLR0913
    """Fit a sparse symbolic mechanism for one numeric node."""
    if node not in df.columns:
        return None

    sub_cols = [node] + [parent for parent in parents if parent in df.columns]
    sub = df[sub_cols].copy()
    sub[node] = _coerce_numeric(sub[node])
    for parent in parents:
        if parent in sub.columns:
            sub[parent] = _coerce_numeric(sub[parent])

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return None

    available_parents = [parent for parent in parents if parent in sub.columns]
    y = sub[node].to_numpy(dtype=float)

    if not available_parents:
        return SymbolicMechanism(
            node=node,
            parents=[],
            expression=sp.Float(float(np.mean(y))),
        )

    terms = _candidate_terms(
        sub,
        available_parents,
        reciprocal_eps=reciprocal_eps,
        min_reciprocal_valid_fraction=min_reciprocal_valid_fraction,
    )
    design = np.column_stack([values for _, values in terms])
    finite_mask = np.isfinite(design).all(axis=1) & np.isfinite(y)
    if int(np.sum(finite_mask)) < max(5, len(terms)):
        return SymbolicMechanism(
            node=node,
            parents=[],
            expression=sp.Float(float(np.mean(y[np.isfinite(y)]))),
        )

    design = design[finite_mask]
    y = y[finite_mask]

    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20_000)
    model.fit(design, y)

    selected = [
        index
        for index, coefficient in enumerate(model.coef_)
        if abs(float(coefficient)) >= coefficient_threshold
    ]
    if selected:
        refit_design = design[:, selected]
        refit_coefficients, *_ = np.linalg.lstsq(refit_design, y, rcond=None)
    else:
        refit_coefficients = np.asarray([], dtype=float)

    expression: sp.Basic = sp.Integer(0)
    for coefficient, index in zip(refit_coefficients, selected):
        if abs(float(coefficient)) < coefficient_threshold:
            continue
        term, _ = terms[index]
        expression += sp.Float(float(coefficient)) * term

    if expression == 0:
        expression = sp.Float(float(np.mean(y)))

    return SymbolicMechanism(
        node=node,
        parents=available_parents,
        expression=sp.simplify(expression),
    )


class SparseSymbolicSCMDiscoveryInferer(Inferer):
    """Discover numeric SCM mechanisms as sparse SymPy expressions."""

    def __init__(
        self,
        *,
        dag_alpha: float = 0.05,
        lasso_alpha: float = 0.001,
        coefficient_threshold: float = 1e-6,
        reciprocal_eps: float = 1e-6,
        min_reciprocal_valid_fraction: float = 0.95,
        seed: int = 911,
    ) -> None:  # noqa: PLR0913
        self.dag_alpha = dag_alpha
        self.lasso_alpha = lasso_alpha
        self.coefficient_threshold = coefficient_threshold
        self.reciprocal_eps = reciprocal_eps
        self.min_reciprocal_valid_fraction = min_reciprocal_valid_fraction
        self.seed = seed
        self._obs_df: pd.DataFrame | None = None
        self._schema_columns: set[str] = set()
        self._int_batches: list[tuple[set[str], pd.DataFrame]] = []
        self._cached_scm: EstimatedSymbolicSCM | None = None

    def _training_df_for_node(self, node: str) -> pd.DataFrame:
        """Return rows where this node was not intervened on."""
        parts: list[pd.DataFrame] = []
        if self._obs_df is not None and not self._obs_df.empty:
            parts.append(self._obs_df)

        for keys, df in self._int_batches:
            if node not in keys:
                parts.append(df)

        if parts:
            return pd.concat(parts, ignore_index=True, sort=False)
        return pd.DataFrame(columns=sorted(self._schema_columns))

    @override
    def update(self, observation: RoundObservation) -> None:
        samples: SamplesCollection = observation.samples

        for batch in samples:
            if batch.data is None:
                continue

            self._schema_columns |= set(batch.data.columns)
            if batch.data.empty:
                continue

            if batch.kind == "observational":
                self._obs_df = (
                    batch.data.copy()
                    if self._obs_df is None
                    else pd.concat([self._obs_df, batch.data], ignore_index=True, sort=False)
                )
            elif batch.kind == "interventional":
                if batch.interventions:
                    self._int_batches.append((set(batch.interventions.keys()), batch.data.copy()))
            else:
                raise ValueError(f"Unknown batch.kind: {batch.kind}")  # noqa: TRY003

        self._cached_scm = None

    @override
    def answer(self) -> EstimatedSymbolicSCM:
        if self._cached_scm is not None:
            return self._cached_scm

        dag = nx.DiGraph()
        if self._obs_df is not None and not self._obs_df.empty:
            interventional_batches: dict[str, list[pd.DataFrame]] = {}
            for keys, df in self._int_batches:
                for key in keys:
                    interventional_batches.setdefault(key, []).append(df)

            dag = learn_dag_from_samples(
                obs_df=self._obs_df,
                interventional_batches=interventional_batches,
                is_numerical=True,
                alpha=self.dag_alpha,
                seed=self.seed,
            )
            dag.add_nodes_from(self._obs_df.columns)

        dag.add_nodes_from(sorted(self._schema_columns))

        mechanisms: dict[str, SymbolicMechanism] = {}
        var_is_numeric: dict[str, bool] = {}
        for column in sorted(self._schema_columns):
            df_node = self._training_df_for_node(column)
            if column in df_node.columns:
                var_is_numeric[column] = _is_numeric_series(df_node[column])

        for node in dag.nodes:
            if not var_is_numeric.get(node, False):
                continue
            parents = [
                parent
                for parent in dag.predecessors(node)
                if var_is_numeric.get(parent, False)
            ]
            mechanism = fit_sparse_symbolic_mechanism_for_node(
                node=node,
                parents=parents,
                df=self._training_df_for_node(node),
                alpha=self.lasso_alpha,
                coefficient_threshold=self.coefficient_threshold,
                reciprocal_eps=self.reciprocal_eps,
                min_reciprocal_valid_fraction=self.min_reciprocal_valid_fraction,
            )
            if mechanism is not None:
                mechanisms[node] = mechanism

        self._cached_scm = EstimatedSymbolicSCM(dag=dag, mechanisms=mechanisms)
        return self._cached_scm

    @override
    def snapshot(self) -> BeliefSnapshot:
        estimate = self.answer()
        return BeliefSnapshot(
            estimate=estimate,
            summary={
                "schema_columns": sorted(self._schema_columns),
                "expressions": estimate.expressions(),
            },
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "dag_alpha": self.dag_alpha,
                "lasso_alpha": self.lasso_alpha,
                "coefficient_threshold": self.coefficient_threshold,
                "reciprocal_eps": self.reciprocal_eps,
                "min_reciprocal_valid_fraction": self.min_reciprocal_valid_fraction,
                "seed": self.seed,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> SparseSymbolicSCMDiscoveryInferer:
        params = spec.params or {}
        return cls(
            dag_alpha=float(params.get("dag_alpha", 0.05)),
            lasso_alpha=float(params.get("lasso_alpha", 0.001)),
            coefficient_threshold=float(params.get("coefficient_threshold", 1e-6)),
            reciprocal_eps=float(params.get("reciprocal_eps", 1e-6)),
            min_reciprocal_valid_fraction=float(
                params.get("min_reciprocal_valid_fraction", 0.95)
            ),
            seed=int(params.get("seed", 911)),
        )
