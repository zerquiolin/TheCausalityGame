from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _binary_entropy(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@dataclass
class BeliefSummary:
    columns: list[str]
    edge_prob: np.ndarray  # [d,d] p[i,j] ~ P(i -> j)
    edge_entropy: np.ndarray  # [d,d] entropy of p[i,j]
    total_entropy: float
    means: np.ndarray  # [d]
    cov: np.ndarray  # [d,d]
    df_numeric: pd.DataFrame  # stored numeric data used to fit


class RidgeBootstrapEdgeBelief:
    """
    Lightweight uncertainty over edges via bootstrap ridge regressions.

    Adds:
      - stores numeric dataframe, mean, covariance for fantasies (Tigas).
    """

    def __init__(
        self,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        seed: int | None = None,
    ) -> None:
        self.n_bootstrap = int(n_bootstrap)
        self.ridge_lambda = float(ridge_lambda)
        self.coef_threshold = float(coef_threshold)
        self.rng = np.random.default_rng(seed)

        self._summary: BeliefSummary | None = None

    def summary(self) -> BeliefSummary | None:
        return self._summary

    @staticmethod
    def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype("category").cat.codes.astype(float)
        df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
        return df

    def fit(self, samples_collection: list[Any]) -> BeliefSummary | None:
        if not samples_collection:
            self._summary = None
            return None

        frames: list[pd.DataFrame] = []
        for s in samples_collection:
            if getattr(s, "data", None) is None:
                continue
            if len(s.data) == 0:
                continue
            frames.append(s.data)

        if not frames:
            self._summary = None
            return None

        df = pd.concat(frames, axis=0, ignore_index=True)
        df = self._to_numeric_df(df)

        if df.shape[0] < 8 or df.shape[1] < 2:
            self._summary = None
            return None

        cols = list(df.columns)
        X_full = df.to_numpy(dtype=float)
        n, d = X_full.shape

        means = X_full.mean(axis=0)
        cov = np.cov(X_full, rowvar=False)
        # numerical stability
        cov = cov + 1e-8 * np.eye(d)

        present_counts = np.zeros((d, d), dtype=float)
        idx_all = np.arange(n)

        for _ in range(self.n_bootstrap):
            boot_idx = self.rng.choice(idx_all, size=n, replace=True)
            Xb = X_full[boot_idx, :]

            for j in range(d):
                y = Xb[:, j]
                X = np.delete(Xb, j, axis=1)

                y_c = y - y.mean()
                X_c = X - X.mean(axis=0, keepdims=True)

                XtX = X_c.T @ X_c
                XtX.flat[:: XtX.shape[0] + 1] += self.ridge_lambda
                Xty = X_c.T @ y_c

                try:
                    beta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(XtX) @ Xty

                k = 0
                for i in range(d):
                    if i == j:
                        continue
                    if abs(beta[k]) > self.coef_threshold:
                        present_counts[i, j] += 1.0
                    k += 1

        edge_prob = present_counts / float(self.n_bootstrap)
        np.fill_diagonal(edge_prob, 0.0)

        edge_entropy = _binary_entropy(edge_prob)
        np.fill_diagonal(edge_entropy, 0.0)

        self._summary = BeliefSummary(
            columns=cols,
            edge_prob=edge_prob,
            edge_entropy=edge_entropy,
            total_entropy=float(edge_entropy.sum()),
            means=means,
            cov=cov,
            df_numeric=df,
        )
        return self._summary

    def incident_uncertainty(self, var: str) -> float:
        s = self._summary
        if s is None or var not in s.columns:
            return 0.0
        k = s.columns.index(var)
        return float(s.edge_entropy[k, :].sum() + s.edge_entropy[:, k].sum())

    def outgoing_uncertainty(self, var: str) -> float:
        s = self._summary
        if s is None or var not in s.columns:
            return 0.0
        k = s.columns.index(var)
        return float(s.edge_entropy[k, :].sum())

    def canonical_value(self, domain: list[Any]) -> Any:
        # Cho-style: do not optimize value; pick canonical / baseline.
        if not domain:
            return 0
        # numeric [low, high] -> choose low (KO / baseline)
        if len(domain) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in domain[:2]
        ):
            return float(domain[0])
        # categorical -> first category
        return domain[0]

    def value_signal(self, var: str, value: Any) -> float:
        s = self._summary
        if s is None or var not in s.columns:
            return 1.0
        k = s.columns.index(var)
        mu = float(s.means[k])
        try:
            v = float(value)
            return abs(v - mu)
        except (TypeError, ValueError):
            return 1.0

    def fantasize_do(
        self, var: str, value: Any, n: int, rng: np.random.Generator
    ) -> pd.DataFrame | None:
        """
        Generate fantasized interventional samples under a Gaussian approximation:
          - set var to value
          - sample all other variables from conditional MVN given var=value
        """
        s = self._summary
        if s is None or var not in s.columns or n <= 0:
            return None

        cols = s.columns
        d = len(cols)
        k = cols.index(var)

        try:
            v = float(value)
        except (TypeError, ValueError):
            # If non-numeric value slipped in after encoding, just use mean
            v = float(s.means[k])

        mu = s.means
        Sigma = s.cov

        rest = [i for i in range(d) if i != k]
        Sigma_kk = Sigma[k, k]
        Sigma_kk = max(1e-12, Sigma_kk)

        Sigma_rk = Sigma[np.ix_(rest, [k])]  # [d-1,1]
        Sigma_kr = Sigma[np.ix_([k], rest)]  # [1,d-1]
        Sigma_rr = Sigma[np.ix_(rest, rest)]  # [d-1,d-1]

        # Conditional mean/cov of X_rest | X_k = v
        mu_r = mu[rest] + (Sigma_rk[:, 0] / Sigma_kk) * (v - mu[k])
        cov_r = Sigma_rr - (Sigma_rk @ Sigma_kr) / Sigma_kk

        # Numerical stabilization: enforce symmetry and project to PSD.
        cov_r = 0.5 * (cov_r + cov_r.T)
        # Add small jitter, then clip negative eigenvalues.
        cov_r = cov_r + 1e-8 * np.eye(d - 1)
        try:
            w, V = np.linalg.eigh(cov_r)
            w = np.maximum(w, 1e-10)
            cov_r = (V * w) @ V.T
            cov_r = 0.5 * (cov_r + cov_r.T)
        except np.linalg.LinAlgError:
            # Fallback: diagonal covariance
            cov_r = np.diag(np.maximum(np.diag(cov_r), 1e-10))

        Xr = rng.multivariate_normal(mean=mu_r, cov=cov_r, size=n)
        X = np.zeros((n, d), dtype=float)
        X[:, k] = v
        X[:, rest] = Xr

        return pd.DataFrame(X, columns=cols)

    def entropy_of_df(self, df_numeric: pd.DataFrame, n_bootstrap: int | None = None) -> float:
        """
        Compute total edge entropy for a hypothetical dataset (used in Tigas lookahead).
        Uses the same bootstrap ridge procedure but can use fewer bootstraps for speed.
        """
        df = self._to_numeric_df(df_numeric)
        if df.shape[0] < 8 or df.shape[1] < 2:
            return float("inf")

        X_full = df.to_numpy(dtype=float)
        n, d = X_full.shape

        B = int(n_bootstrap) if n_bootstrap is not None else self.n_bootstrap
        present_counts = np.zeros((d, d), dtype=float)
        idx_all = np.arange(n)

        for _ in range(B):
            boot_idx = self.rng.choice(idx_all, size=n, replace=True)
            Xb = X_full[boot_idx, :]

            for j in range(d):
                y = Xb[:, j]
                X = np.delete(Xb, j, axis=1)

                y_c = y - y.mean()
                X_c = X - X.mean(axis=0, keepdims=True)

                XtX = X_c.T @ X_c
                XtX.flat[:: XtX.shape[0] + 1] += self.ridge_lambda
                Xty = X_c.T @ y_c

                try:
                    beta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(XtX) @ Xty

                k = 0
                for i in range(d):
                    if i == j:
                        continue
                    if abs(beta[k]) > self.coef_threshold:
                        present_counts[i, j] += 1.0
                    k += 1

        edge_prob = present_counts / float(B)
        np.fill_diagonal(edge_prob, 0.0)

        edge_entropy = _binary_entropy(edge_prob)
        np.fill_diagonal(edge_entropy, 0.0)

        return float(edge_entropy.sum())

    def sample_linear_dag_ensemble(
        self,
        n_graphs: int = 32,
        ridge_lambda: float | None = None,
        coef_threshold: float | None = None,
        seed: int | None = None,
    ) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
        """\
        Sample an ensemble of DAG linear-Gaussian SEMs from the current numeric dataset.

        Returns a list of tuples:
          (order, B, sigma2)
        where:
          - order is a topological ordering (list of variable indices)
          - B is a [d,d] coefficient matrix with B[i,j] = weight i->j
          - sigma2 is a [d] vector of noise variances for each node.

        This provides a lightweight posterior proxy over DAGs for minimax/Thompson/EIG agents.
        """
        s = self._summary
        if s is None:
            return []

        lam = float(ridge_lambda) if ridge_lambda is not None else float(self.ridge_lambda)
        thr = float(coef_threshold) if coef_threshold is not None else float(self.coef_threshold)

        X_full = s.df_numeric.to_numpy(dtype=float)
        n, d = X_full.shape
        if n < 8 or d < 2:
            return []

        rng = np.random.default_rng(seed)

        models: list[tuple[list[int], np.ndarray, np.ndarray]] = []
        for _ in range(int(n_graphs)):
            order = list(rng.permutation(d))
            pos = {node: k for k, node in enumerate(order)}

            B = np.zeros((d, d), dtype=float)
            sigma2 = np.ones((d,), dtype=float)

            # Fit each node as linear regression on its predecessors in this ordering.
            for node in order:
                preds = [p for p in order if pos[p] < pos[node]]
                y = X_full[:, node]

                if len(preds) == 0:
                    resid = y - y.mean()
                    sigma2[node] = float(np.var(resid) + 1e-8)
                    continue

                Xp = X_full[:, preds]

                # Center
                y_c = y - y.mean()
                Xp_c = Xp - Xp.mean(axis=0, keepdims=True)

                XtX = Xp_c.T @ Xp_c
                XtX.flat[:: XtX.shape[0] + 1] += lam
                Xty = Xp_c.T @ y_c

                try:
                    beta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(XtX) @ Xty

                # Threshold to form sparse parents
                for kk, p in enumerate(preds):
                    if abs(beta[kk]) > thr:
                        B[p, node] = float(beta[kk])

                # Noise variance from residuals
                y_hat = Xp_c @ beta
                resid = y_c - y_hat
                sigma2[node] = float(np.var(resid) + 1e-8)

            models.append((order, B, sigma2))

        return models
