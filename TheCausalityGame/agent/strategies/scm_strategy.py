from __future__ import annotations
from typing import Optional

from TheCausalityGame.agent.strategies.dag_strategy import learn_dag_from_samples  # type: ignore
from TheCausalityGame.core.contracts.dto.environment import SamplesCollection
from TheCausalityGame.core.infrastructure.strategy import Strategy

# ruff: noqa

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------
def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    # consider numeric if most values are coercible
    return float(coerced.notna().mean()) >= 0.95


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# Mechanisms
# ----------------------------
@dataclass
class NumericANMMechanism:
    """X = g(Pa) + eps, with eps sampled from residuals (bootstrap)."""

    node: str
    parents: list[str]
    coef: Optional[np.ndarray]  # includes intercept; None for root
    feature_info: dict[str, Any]  # categorical parent levels for one-hot, etc.
    residuals: np.ndarray  # residual pool (or marginal samples for root)
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def _design_row(self, parent_values: dict[str, Any]) -> np.ndarray:
        # intercept
        feats: list[float] = [1.0]
        # numeric parents
        for p in self.feature_info["num_parents"]:
            feats.append(float(parent_values[p]))
        # categorical parents (one-hot, drop-first = False)
        for p in self.feature_info["cat_parents"]:
            levels: list[Any] = self.feature_info["cat_levels"][p]
            pv = parent_values[p]
            for lvl in levels:
                feats.append(1.0 if pv == lvl else 0.0)
        return np.asarray(feats, dtype=float)

    def sample(self, parent_values: dict[str, Any], rng: np.random.Generator) -> float:
        # Root: residuals store empirical marginal samples
        if not self.parents or self.coef is None:
            x = float(rng.choice(self.residuals))
        else:
            mu = float(self._design_row(parent_values) @ self.coef)
            eps = float(rng.choice(self.residuals))
            x = mu + eps

        if self.clip_min is not None:
            x = max(self.clip_min, x)
        if self.clip_max is not None:
            x = min(self.clip_max, x)
        return x


@dataclass
class DiscreteCPTMechanism:
    """p(X | Pa) estimated via a CPT with optional binning for numeric parents."""

    node: str
    parents: list[str]
    categories: list[Any]
    parent_info: dict[str, Any]  # bins for numeric parents, levels for categorical parents
    cpt: dict[tuple[Any, ...], np.ndarray]  # key: parent signature -> probs over categories
    marginal: np.ndarray  # fallback probs over categories

    def _parent_signature(self, parent_values: dict[str, Any]) -> tuple[Any, ...]:
        sig: list[Any] = []
        for p in self.parents:
            if p in self.parent_info["num_bins"]:
                edges = self.parent_info["num_bins"][p]
                # digitize into bins 0..K-1
                v = float(parent_values[p])
                b = int(np.digitize([v], edges[1:-1], right=False)[0])
                sig.append(b)
            else:
                # categorical parent
                sig.append(parent_values[p])
        return tuple(sig)

    def sample(self, parent_values: dict[str, Any], rng: np.random.Generator) -> Any:
        sig = self._parent_signature(parent_values)
        probs = self.cpt.get(sig, self.marginal)
        idx = int(rng.choice(len(self.categories), p=probs))
        return self.categories[idx]


class EstimatedANMSCM:
    """Estimated SCM supporting sampling under do-interventions."""

    def __init__(self, dag: nx.DiGraph, mechanisms: dict[str, Any], seed: int = 0):
        self.dag = dag
        self.mechanisms = mechanisms
        self._seed = seed

        # Validate topological order (best-effort)
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("EstimatedANMSCM requires a DAG (acyclic).")

    def generate_samples(
        self,
        num_samples: int,
        interventions: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = _safe_rng(self._seed if seed is None else seed)
        interventions = interventions or {}

        order = list(nx.topological_sort(self.dag))
        rows: list[dict[str, Any]] = []

        # If the DAG has no nodes, we can't generate any variables; return n empty rows.
        # If it has nodes but topological_sort somehow returns empty (shouldn't happen),
        # return the right columns filled with NaNs.
        if len(order) == 0:
            cols = list(self.dag.nodes)
            if len(cols) == 0:
                return pd.DataFrame(index=range(num_samples))
            return pd.DataFrame({c: [np.nan] * num_samples for c in cols})

        for _ in range(num_samples):
            row: dict[str, Any] = {}
            for node in order:
                if node in interventions:
                    row[node] = interventions[node]
                    continue

                mech = self.mechanisms[node]
                parent_vals = {p: row[p] for p in self.dag.predecessors(node)}

                if isinstance(mech, NumericANMMechanism):
                    row[node] = mech.sample(parent_vals, rng)
                elif isinstance(mech, DiscreteCPTMechanism):
                    row[node] = mech.sample(parent_vals, rng)
                else:
                    raise TypeError(f"Unknown mechanism type for {node}: {type(mech)}")

            rows.append(row)

        return pd.DataFrame(rows)


def fit_anm_mechanisms(
    dag: nx.DiGraph,
    df: pd.DataFrame,
    *,
    bins_numeric_parents: int = 4,
    clip_numeric_to_data_range: bool = True,
) -> dict[str, Any]:
    """
    Fit node-wise mechanisms:
    - Numeric nodes: linear ANM + residual bootstrap
    - Categorical nodes: CPT with binned numeric parents
    """
    mechs: dict[str, Any] = {}

    # Decide variable types from data
    var_is_numeric: dict[str, bool] = {c: _is_numeric_series(df[c]) for c in df.columns}

    for node in dag.nodes:
        parents = list(dag.predecessors(node))

        if var_is_numeric.get(node, False):
            # ---- numeric ANM ----
            sub_cols = [node] + parents
            sub = df[sub_cols].copy()

            # Coerce numeric target
            sub[node] = _coerce_numeric(sub[node])
            # Keep parent columns as-is for one-hot if categorical
            for p in parents:
                if var_is_numeric.get(p, False):
                    sub[p] = _coerce_numeric(sub[p])

            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

            # Root numeric: empirical marginal
            if len(parents) == 0 or sub.shape[0] < 20:
                vals = (
                    _coerce_numeric(df[node])
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if vals.size == 0:
                    vals = np.array([0.0], dtype=float)
                mechs[node] = NumericANMMechanism(
                    node=node,
                    parents=[],
                    coef=None,
                    feature_info={"num_parents": [], "cat_parents": [], "cat_levels": {}},
                    residuals=vals,
                    clip_min=float(vals.min()) if clip_numeric_to_data_range else None,
                    clip_max=float(vals.max()) if clip_numeric_to_data_range else None,
                )
                continue

            # Build design matrix: numeric parents + one-hot for categorical parents
            num_parents = [p for p in parents if var_is_numeric.get(p, False)]
            cat_parents = [p for p in parents if not var_is_numeric.get(p, False)]

            X_parts = []
            # numeric parent matrix
            if num_parents:
                X_parts.append(sub[num_parents].to_numpy(dtype=float))

            cat_levels: dict[str, list[Any]] = {}
            if cat_parents:
                # deterministic level order from observed data
                for p in cat_parents:
                    levels = list(pd.Series(sub[p].astype(object)).dropna().unique())
                    cat_levels[p] = levels
                # one-hot
                for p in cat_parents:
                    levels = cat_levels[p]
                    oh = np.column_stack([(sub[p] == lvl).to_numpy(dtype=float) for lvl in levels])
                    X_parts.append(oh)

            X = np.column_stack(X_parts) if X_parts else np.zeros((sub.shape[0], 0), dtype=float)
            X = np.column_stack([np.ones(X.shape[0]), X])  # intercept

            y = sub[node].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ coef
            resid = y - y_hat

            # residual pool
            if resid.size == 0:
                resid = np.array([0.0], dtype=float)

            clip_min = float(y.min()) if clip_numeric_to_data_range else None
            clip_max = float(y.max()) if clip_numeric_to_data_range else None

            mechs[node] = NumericANMMechanism(
                node=node,
                parents=parents,
                coef=coef,
                feature_info={
                    "num_parents": num_parents,
                    "cat_parents": cat_parents,
                    "cat_levels": cat_levels,
                },
                residuals=resid,
                clip_min=clip_min,
                clip_max=clip_max,
            )

        else:
            # ---- categorical CPT ----
            # Build a CPT p(node | parents). Numeric parents are binned into quantiles.
            sub_cols = [node] + parents
            sub = df[sub_cols].copy()

            # establish categories
            cats = list(pd.Series(sub[node].astype(object)).dropna().unique())
            if not cats:
                cats = [None]

            # parent binning info
            parent_info = {"num_bins": {}, "cat_levels": {}}
            for p in parents:
                if var_is_numeric.get(p, False):
                    sp = _coerce_numeric(sub[p]).replace([np.inf, -np.inf], np.nan).dropna()
                    if sp.size < 10:
                        # fallback single bin
                        edges = np.array([0.0, 1.0], dtype=float)
                    else:
                        qs = np.linspace(0, 1, bins_numeric_parents + 1)
                        edges = np.quantile(sp.to_numpy(dtype=float), qs)
                        # ensure strictly increasing
                        edges = np.unique(edges)
                        if edges.size < 2:
                            edges = np.array([sp.min(), sp.max() + 1e-6], dtype=float)
                    parent_info["num_bins"][p] = edges
                else:
                    parent_info["cat_levels"][p] = list(
                        pd.Series(sub[p].astype(object)).dropna().unique()
                    )

            # drop rows with missing in node or parents (after numeric coercion for numeric parents)
            for p in parents:
                if var_is_numeric.get(p, False):
                    sub[p] = _coerce_numeric(sub[p])
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

            # marginal
            node_counts = sub[node].astype(object).value_counts()
            marginal = np.array([node_counts.get(c, 0) for c in cats], dtype=float)
            if marginal.sum() == 0:
                marginal = np.ones(len(cats), dtype=float)
            marginal = marginal / marginal.sum()

            # CPT
            cpt: dict[tuple[Any, ...], np.ndarray] = {}

            def parent_sig(row: pd.Series) -> tuple[Any, ...]:
                sig = []
                for p in parents:
                    if p in parent_info["num_bins"]:
                        edges = parent_info["num_bins"][p]
                        v = float(row[p])
                        b = int(np.digitize([v], edges[1:-1], right=False)[0])
                        sig.append(b)
                    else:
                        sig.append(row[p])
                return tuple(sig)

            if parents:
                grouped: dict[tuple[Any, ...], pd.Series] = {}
                for _, r in sub.iterrows():
                    key = parent_sig(r)
                    grouped.setdefault(key, [])
                    grouped[key].append(r[node])

                for key, ys in grouped.items():
                    ys = pd.Series(ys, dtype=object)
                    counts = ys.value_counts()
                    probs = np.array([counts.get(c, 0) for c in cats], dtype=float)
                    if probs.sum() == 0:
                        probs = marginal.copy()
                    else:
                        probs = probs / probs.sum()
                    cpt[key] = probs
            else:
                # no parents: just marginal
                cpt[tuple()] = marginal.copy()

            mechs[node] = DiscreteCPTMechanism(
                node=node,
                parents=parents,
                categories=cats,
                parent_info=parent_info,
                cpt=cpt,
                marginal=marginal,
            )

    return mechs


class SCMDiscoveryStrategy(Strategy):
    """
    Learns an estimated SCM:
      1) learn DAG from observational + interventional data
      2) fit ANM mechanisms from data
      3) answer() returns an EstimatedANMSCM that can sample under do()
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        seed: int = 911,
        bins_numeric_parents: int = 4,
        use_interventional_for_mechanisms: bool = True,
    ):
        self.alpha = alpha
        self.seed = seed
        self.bins_numeric_parents = bins_numeric_parents
        self.use_interventional_for_mechanisms = use_interventional_for_mechanisms

        self._obs_df: Optional[pd.DataFrame] = None
        self._schema_columns: set[str] = set()
        # store (intervention_keys_set, df)
        self._int_batches: list[tuple[set[str], pd.DataFrame]] = []
        self._cached_scm: Optional[EstimatedANMSCM] = None

    def initialize(self) -> None:
        self._obs_df = None
        self._schema_columns = set()
        self._int_batches = []
        self._cached_scm = None
        self._is_initialized = True

    def learn(self, samples: SamplesCollection) -> None:
        if not self.is_initialized:
            raise RuntimeError("Strategy must be initialized before learning.")

        for batch in samples:
            if batch.data is None or batch.data.empty:
                continue
            self._schema_columns |= set(batch.data.columns)

            if batch.kind == "observational":
                self._obs_df = (
                    batch.data.copy()
                    if self._obs_df is None
                    else pd.concat([self._obs_df, batch.data], ignore_index=True, sort=False)
                )

            elif batch.kind == "interventional":
                if not batch.interventions:
                    continue
                int_keys = set(batch.interventions.keys())
                self._int_batches.append((int_keys, batch.data.copy()))
            else:
                raise ValueError(f"Unknown batch.kind: {batch.kind}")

        self._cached_scm = None

    def answer(self) -> EstimatedANMSCM:
        if not self.is_initialized:
            raise RuntimeError("Strategy must be initialized before answering.")

        if self._cached_scm is not None:
            return self._cached_scm

        if self._obs_df is None or self._obs_df.empty:
            # If we never collected observational data, we cannot run PC reliably.
            # Still, we can build an SCM over the known variable schema so sampling
            # returns the requested number of rows/columns.
            dag = nx.DiGraph()
            dag.add_nodes_from(sorted(self._schema_columns))

            # Fit root mechanisms from whatever data we have (interventional batches).
            if self._int_batches:
                mech_df = self._build_mechanism_training_df(dag)
                mechanisms = fit_anm_mechanisms(
                    dag=dag,
                    df=mech_df,
                    bins_numeric_parents=self.bins_numeric_parents,
                    clip_numeric_to_data_range=True,
                )
            else:
                mechanisms = {}

            self._cached_scm = EstimatedANMSCM(dag=dag, mechanisms=mechanisms, seed=self.seed)
            return self._cached_scm

        # Build the interventional_batches dict[str, list[pd.DataFrame]] expected by your DAG code
        interventional_batches: dict[str, list[pd.DataFrame]] = {}
        for keys, df in self._int_batches:
            for k in keys:
                interventional_batches.setdefault(k, []).append(df)

        dag = learn_dag_from_samples(
            obs_df=self._obs_df,
            interventional_batches=interventional_batches,
            is_numerical=True,  # TODO: adapt if needed
            alpha=self.alpha,
            seed=self.seed,
        )

        # Ensure the DAG contains all variables we have seen.
        dag.add_nodes_from(self._obs_df.columns)
        dag.add_nodes_from(sorted(self._schema_columns))

        # ---- 2) fit mechanisms ----
        mech_df = (
            self._build_mechanism_training_df(dag)
            if self.use_interventional_for_mechanisms
            else self._obs_df
        )
        mechanisms = fit_anm_mechanisms(
            dag=dag,
            df=mech_df,
            bins_numeric_parents=self.bins_numeric_parents,
            clip_numeric_to_data_range=True,
        )

        self._cached_scm = EstimatedANMSCM(dag=dag, mechanisms=mechanisms, seed=self.seed)
        return self._cached_scm

    def _build_mechanism_training_df(self, dag: nx.DiGraph) -> pd.DataFrame:
        """
        Optional: include interventional rows when fitting mechanisms for nodes that were NOT intervened on.
        This leverages invariance: mechanisms for non-intervened nodes remain valid under do(other vars).
        """
        parts: list[pd.DataFrame] = []
        if self._obs_df is not None and not self._obs_df.empty:
            parts.append(self._obs_df)

        if not self._int_batches:
            # If we have no interventional batches either, return an empty frame with schema columns.
            if parts:
                return parts[0]
            return pd.DataFrame(columns=sorted(self._schema_columns))

        # We include each interventional batch wholesale; the mechanism fitter is node-wise and
        # can still learn conditionals. If you want to be stricter, you can filter per-node later.
        # A simple improvement: drop rows where a node itself was intervened when fitting that node;
        # implement that later if needed.
        for _, df in self._int_batches:
            parts.append(df)

        return pd.concat(parts, ignore_index=True, sort=False)
