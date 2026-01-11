# ruff: noqa # pylint: skip-file # mypy: ignore-errors
# type: ignore
"""The Causality Game - DAG Discovery Strategy Implementation."""

import itertools
import logging
from collections.abc import Iterable
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, combine_pvalues, ks_2samp, pearsonr  # type: ignore

# Optional: ML regressors for nonlinear conditional-independence testing
from sklearn.ensemble import HistGradientBoostingRegressor

_SKLEARN_AVAILABLE = True

from TheCausalityGame.core.contracts.dto.environment import SamplesCollection
from TheCausalityGame.core.infrastructure.strategy import Strategy  # adjust import if needed


# ---------------------------------------------------------------------
# Conditional independence tests
# ---------------------------------------------------------------------
def ci_test_gaussian(df: pd.DataFrame, x: str, y: str, z: Iterable[str], alpha: float) -> bool:
    """Gaussian CI test via partial correlation using residualization (defensive).

    Returns True if independent at level alpha.
    Handles object dtypes, non-numeric values, infinities, and constant vectors.
    """
    z = list(z)
    cols = [x, y] + z

    sub = df[cols].copy()

    def _to_float(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce").astype(float)

    for c in cols:
        sub[c] = _to_float(sub[c])

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    # Not enough data => treat as independent (conservative for deletions)
    if sub.shape[0] < 5:
        return True

    def _pval_safe(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size < 3 or b.size < 3:
            return 1.0
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return 1.0
        _, p = pearsonr(a, b)
        return 1.0 if np.isnan(p) else float(p)

    # Unconditional Pearson
    if len(z) == 0:
        p = _pval_safe(sub[x].to_numpy(), sub[y].to_numpy())
        return p > alpha

    # Residualize on Z (with intercept)
    Z = sub[z].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(Z.shape[0]), Z])

    def _residuals(target: str) -> np.ndarray:
        yv = sub[target].to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(Z, yv, rcond=None)
        return yv - Z @ coef

    rx = _residuals(x)
    ry = _residuals(y)

    p = _pval_safe(rx, ry)
    return p > alpha


# ---------------------------------------------------------------------
# Nonlinear conditional independence (CI) for continuous variables
# ---------------------------------------------------------------------


def _rbf_kernel(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    # Pairwise squared distances
    d2 = (x - x.T) ** 2

    if sigma is None:
        # Median heuristic
        tri = d2[np.triu_indices(n, k=1)]
        med = float(np.median(tri)) if tri.size else 1.0
        sigma = np.sqrt(max(med, 1e-12))

    gamma = 1.0 / (2.0 * (sigma**2) + 1e-12)
    return np.exp(-gamma * d2)


def _hsic_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased-ish HSIC using RBF kernels and centering."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if n < 5:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0

    K = _rbf_kernel(x)
    L = _rbf_kernel(y)

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


def _hsic_pvalue(x: np.ndarray, y: np.ndarray, *, n_perm: int = 50, seed: int = 0) -> float:
    """Permutation p-value for HSIC."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if n < 10:
        return 1.0

    obs = _hsic_stat(x, y)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if _hsic_stat(x, yp) >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def ci_test_continuous_nonlinear(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: Iterable[str],
    alpha: float,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> bool:
    """Nonlinear CI test: X ⟂ Y | Z.

    Procedure:
    1) Build a design matrix for Z (one-hot for categoricals).
    2) Fit flexible regressors x~Z and y~Z (HGBR if sklearn available; else linear).
    3) Compute residuals r_x, r_y.
    4) Test r_x ⟂ r_y via HSIC permutation test.

    Returns True if independent at level alpha.
    """
    z = list(z)
    cols = [x, y] + z
    sub = df[cols].copy()

    # Coerce x/y to numeric
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    for c in z:
        # leave Z as-is; we will one-hot it
        pass

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub.shape[0] < 30:
        # Too little data => do not delete edges
        return False

    if len(z) == 0:
        p = _hsic_pvalue(sub[x].to_numpy(), sub[y].to_numpy(), n_perm=n_perm, seed=seed)
        return p > alpha

    Zdf = pd.get_dummies(sub[z], drop_first=False)
    Zm = Zdf.to_numpy(dtype=float)

    def _fit_and_resid(target: str) -> np.ndarray:
        yt = sub[target].to_numpy(dtype=float)
        if _SKLEARN_AVAILABLE and HistGradientBoostingRegressor is not None:
            model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, random_state=seed)
            model.fit(Zm, yt)
            yhat = model.predict(Zm)
        else:
            # Fallback linear residualization with intercept
            X = np.column_stack([np.ones(Zm.shape[0]), Zm])
            coef, *_ = np.linalg.lstsq(X, yt, rcond=None)
            yhat = X @ coef
        return yt - yhat

    rx = _fit_and_resid(x)
    ry = _fit_and_resid(y)

    p = _hsic_pvalue(rx, ry, n_perm=n_perm, seed=seed)
    return p > alpha


def ci_test_discrete_chi2_stratified(
    df: pd.DataFrame, x: str, y: str, z: Iterable[str], alpha: float
) -> bool:
    """
    Discrete CI test via stratified chi-square with Fisher p-value combination.

    Returns True if independent at level alpha.

    NOTE: This is a heuristic; CMH / exact methods can be better.
    """
    z = list(z)
    sub = df[[x, y, *z]].dropna()  # type: ignore
    if sub.empty:
        return True

    if len(z) == 0:
        table = pd.crosstab(sub[x], sub[y])  # type: ignore
        if table.size == 0:
            return True
        _, p, _, _ = chi2_contingency(table)
        return p > alpha  # type: ignore

    pvals = []
    for _, block in sub.groupby(z, dropna=True):  # type: ignore
        if block[x].nunique() < 2 or block[y].nunique() < 2:
            continue
        table = pd.crosstab(block[x], block[y])  # type: ignore
        if table.size == 0:
            continue
        _, p, _, _ = chi2_contingency(table)
        pvals.append(p)  # type: ignore

    if not pvals:
        return True

    _, p_comb = combine_pvalues(pvals, method="fisher")
    return p_comb > alpha  # type: ignore


# ---------------------------------------------------------------------
# PC skeleton + v-structures + Meek R1 (minimal CPDAG orientation)
# ---------------------------------------------------------------------
def pc_skeleton(  # type: ignore
    obs_df: pd.DataFrame,
    is_numerical: bool,
    alpha: float,
) -> tuple[nx.Graph, dict[frozenset[str], set[str]]]:  # type: ignore
    """Generate PC skeleton from observational data."""
    if obs_df is None or obs_df.empty:  # type: ignore
        return nx.Graph(), {}  # type: ignore

    variables = list(obs_df.columns)
    G = nx.complete_graph(variables)  # type: ignore # undirected complete graph

    sep_sets: dict[frozenset[str], set[str]] = {}

    def ci_indep(X: str, Y: str, Z: Iterable[str]) -> bool:  # noqa: N803
        if is_numerical:
            # Nonlinear CI is needed for mechanisms like q/C or (V-Vc)/R.
            return ci_test_continuous_nonlinear(obs_df, X, Y, Z, alpha, n_perm=50, seed=0)
        return ci_test_discrete_chi2_stratified(obs_df, X, Y, Z, alpha)

    l = 0  # noqa: E741
    # classic PC: increase conditioning set size until nothing removable
    while True:
        any_removed = False
        edges_snapshot = list(G.edges())
        for X, Y in edges_snapshot:
            adj_X = set(G.neighbors(X)) - {Y}
            if len(adj_X) < l:
                continue

            for Z in itertools.combinations(adj_X, l):
                if ci_indep(X, Y, Z):
                    if G.has_edge(X, Y):
                        G.remove_edge(X, Y)
                    sep_sets[frozenset({X, Y})] = set(Z)
                    any_removed = True
                    break

        if not any_removed:
            # if nothing removed at this l, move to next l
            l += 1

        # stopping condition: if l exceeds max possible adjacency size
        max_adj = max((len(list(G.neighbors(v))) for v in G.nodes()), default=0)
        if l > max_adj:
            break

    return G, sep_sets


def orient_v_structures(
    skeleton: nx.Graph, sep_sets: dict[frozenset, set[str]]
) -> set[tuple[str, str]]:
    directed: set[tuple[str, str]] = set()
    nodes = list(skeleton.nodes())

    for Z in nodes:
        adj = list(skeleton.neighbors(Z))
        for X, Y in itertools.combinations(adj, 2):
            if skeleton.has_edge(X, Y):
                continue  # shielded
            # unshielded triple X - Z - Y: orient if Z not in SepSet(X,Y)
            if Z not in sep_sets.get(frozenset({X, Y}), set()):
                directed.add((X, Z))
                directed.add((Y, Z))

    return directed


def apply_meek_r1(skeleton: nx.Graph, directed: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """
    Meek R1: X -> Y - Z and X not adjacent Z => orient Y -> Z.
    We treat 'undirected edges' as those still present in `skeleton`
    and not yet oriented in either direction.
    """

    def is_adj(a: str, b: str) -> bool:
        if skeleton.has_edge(a, b):
            return True
        if (a, b) in directed or (b, a) in directed:
            return True
        return False

    changed = True
    while changed:
        changed = False

        # undirected edges are those in skeleton not already oriented
        undirected_edges = sorted(
            (u, v)
            for (u, v) in skeleton.edges()
            if (u, v) not in directed and (v, u) not in directed
        )

        for Y, Z in undirected_edges:
            # look for some X such that X -> Y and X not adj Z
            for X, Y2 in sorted(directed):
                if Y2 != Y:
                    continue
                if not is_adj(X, Z):
                    directed.add((Y, Z))
                    changed = True
                    break
            if changed:
                break

    return directed


# ---------------------------------------------------------------------
# Interventional orientation helpers (simple + statistically calibrated)
# ---------------------------------------------------------------------
def intervention_changes_target_discrete(
    baseline: pd.Series,
    interventional: pd.Series,
    alpha: float,
) -> bool:
    base_counts = baseline.value_counts()
    int_counts = interventional.value_counts()
    cats = sorted(set(base_counts.index).union(set(int_counts.index)))

    a = np.array([base_counts.get(c, 0) for c in cats], dtype=float)
    b = np.array([int_counts.get(c, 0) for c in cats], dtype=float)

    table = np.vstack([a, b])
    # if degenerate, no evidence
    if table.sum() == 0 or (table[0].sum() == 0) or (table[1].sum() == 0):
        return False

    _, p, _, _ = chi2_contingency(table)
    return p < alpha


def intervention_changes_target_numeric(
    baseline: pd.Series,
    interventional: pd.Series,
    alpha: float,
) -> bool:
    base = baseline.dropna().to_numpy()
    inter = interventional.dropna().to_numpy()
    if len(base) < 5 or len(inter) < 5:
        return False
    # nonparametric distribution shift
    _, p = ks_2samp(base, inter)
    return p < alpha


# ---------------------------------------------------------------------
# Main learner: PC on observational, then orient remaining edges by interventions
# ---------------------------------------------------------------------
def learn_dag_from_samples(
    obs_df: pd.DataFrame,
    interventional_batches: dict[str, list[pd.DataFrame]],
    is_numerical: bool,
    alpha: float,
    seed: int,
) -> nx.DiGraph:
    if obs_df is None or obs_df.empty:
        return nx.DiGraph()

    skeleton, sep_sets = pc_skeleton(obs_df=obs_df, is_numerical=is_numerical, alpha=alpha)
    directed = orient_v_structures(skeleton, sep_sets)
    directed = apply_meek_r1(skeleton, directed)

    pooled_interventions: dict[str, pd.DataFrame] = {}
    for var, batches in interventional_batches.items():
        if not batches:
            continue
        pooled_interventions[var] = pd.concat(batches, axis=0, ignore_index=True, sort=False)

    # -----------------------------------------------------------------
    # Interventions: (i) orient remaining skeleton edges, and
    # (ii) add missing edges using an intervention-derived partial order.
    #
    # Key causal fact (under causal sufficiency + perfect interventions):
    # If do(X) changes the distribution of Y, then X is an ancestor of Y.
    # We use this to recover edges that PC may miss under nonlinear/deterministic mechanisms.
    # To avoid adding mediated edges (e.g., C -> I when C -> Vc -> I),
    # we perform a simple transitive reduction on the "affects" relation.
    # -----------------------------------------------------------------

    # Build affects[X][Y] from pooled interventions
    vars_ = list(obs_df.columns)
    affects: dict[str, set[str]] = {v: set() for v in vars_}

    for X, x_int in pooled_interventions.items():
        if x_int is None or x_int.empty:
            continue
        for Y in vars_:
            if Y == X:
                continue
            if Y not in x_int.columns or Y not in obs_df.columns:
                continue
            # Does do(X) change Y?
            if is_numerical:
                base = pd.to_numeric(obs_df[Y], errors="coerce").dropna().to_numpy(dtype=float)
                inter = pd.to_numeric(x_int[Y], errors="coerce").dropna().to_numpy(dtype=float)
                if base.size < 20 or inter.size < 20:
                    continue
                _, p = ks_2samp(base, inter)
                if p < alpha:
                    affects[X].add(Y)
            else:
                base = obs_df[Y].dropna()
                inter = x_int[Y].dropna()
                if base.empty or inter.empty:
                    continue
                # 2xK chi-square on counts
                base_counts = base.value_counts()
                inter_counts = inter.value_counts()
                cats = sorted(set(base_counts.index).union(set(inter_counts.index)))
                a = np.array([base_counts.get(c, 0) for c in cats], dtype=float)
                b = np.array([inter_counts.get(c, 0) for c in cats], dtype=float)
                table = np.vstack([a, b])
                if table.sum() > 0 and table[0].sum() > 0 and table[1].sum() > 0:
                    _, p, _, _ = chi2_contingency(table)
                    if p < alpha:
                        affects[X].add(Y)

    # (A) Orient remaining skeleton edges using affects relation
    remaining_undirected = [
        (u, v) for (u, v) in skeleton.edges() if (u, v) not in directed and (v, u) not in directed
    ]

    for X, Y in remaining_undirected:
        x_affects_y = Y in affects.get(X, set())
        y_affects_x = X in affects.get(Y, set())
        if x_affects_y and not y_affects_x:
            directed.add((X, Y))
        elif y_affects_x and not x_affects_y:
            directed.add((Y, X))

    # (B) Add missing edges by transitive reduction of the affects relation
    # Add X->Y if X affects Y and there is no mediator Z such that X affects Z and Z affects Y.
    extra_direct: set[tuple[str, str]] = set()
    for X in vars_:
        for Y in affects.get(X, set()):
            if X == Y:
                continue
            mediated = False
            for Z in vars_:
                if Z == X or Z == Y:
                    continue
                if Z in affects.get(X, set()) and Y in affects.get(Z, set()):
                    mediated = True
                    break
            if not mediated:
                extra_direct.add((X, Y))

    directed |= extra_direct

    # Build final DAG (we omit ambiguous edges)
    G = nx.DiGraph()
    G.add_nodes_from(obs_df.columns)
    G.add_edges_from(directed)

    # Ensure acyclic (PC output should be acyclic in oriented parts, but intervention orientation can conflict)
    # If cycles appear, break them deterministically.
    rs = np.random.RandomState(seed)
    try:
        while True:
            cycle = nx.find_cycle(G, orientation="original")
            i = rs.choice(len(cycle))
            e = cycle[i][:2]
            logging.info(f"Removing edge {e} to break cycle.")
            G.remove_edge(*e)
    except nx.NetworkXNoCycle:
        pass

    return G


# ---------------------------------------------------------------------
# Concrete Strategy implementation
# ---------------------------------------------------------------------
class DAGDiscoveryStrategy(Strategy):
    """
    Concrete strategy:
    - learn(): incrementally stores observational/interventional data
    - answer(): recomputes a DAG estimate from all stored data
    """

    def __init__(self, *, is_numerical: bool = True, alpha: float = 0.05, seed: int = 911):
        self.is_numerical = is_numerical
        self.alpha = alpha
        self.seed = seed

        self._obs_df: Optional[pd.DataFrame] = None
        self._interventional_batches: dict[str, list[pd.DataFrame]] = {}
        self._dag: Optional[nx.DiGraph] = None

    def initialize(self) -> None:
        self._obs_df = None
        self._interventional_batches = {}
        self._dag = None
        self._is_initialized = True

    def learn(self, samples: SamplesCollection) -> None:
        if not self.is_initialized:
            raise RuntimeError("Strategy must be initialized before learning.")

        for batch in samples:
            if batch.data is None or batch.data.empty:
                continue

            if batch.kind == "observational":
                if self._obs_df is None:
                    self._obs_df = batch.data.copy()
                else:
                    # align columns, union schema
                    self._obs_df = pd.concat(
                        [self._obs_df, batch.data], axis=0, ignore_index=True, sort=False
                    )

            elif batch.kind == "interventional":
                # We assume batch.interventions is like {"X": some_value} or {"X": ... , "Y": ...}
                # This code groups the batch under each intervened variable key.
                if not batch.interventions:
                    continue
                for int_var in batch.interventions.keys():
                    self._interventional_batches.setdefault(int_var, []).append(batch.data.copy())
            else:
                raise ValueError(f"Unknown batch.kind: {batch.kind}")

        # Invalidate cached answer
        self._dag = None

    def answer(self) -> nx.DiGraph:
        if not self.is_initialized:
            raise RuntimeError("Strategy must be initialized before answering.")

        if self._dag is not None:
            return self._dag

        if self._obs_df is None or self._obs_df.empty:
            self._dag = nx.DiGraph()
            return self._dag

        self._dag = learn_dag_from_samples(
            obs_df=self._obs_df,
            interventional_batches=self._interventional_batches,
            is_numerical=self.is_numerical,
            alpha=self.alpha,
            seed=self.seed,
        )
        return self._dag
