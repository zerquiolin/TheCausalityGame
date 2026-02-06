from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


def _df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("category").cat.codes.astype(float)
    return df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")


def _sample_value(rng, domain):
    dom = list(domain)
    if not dom:
        return 0

    # Numeric bounds convention: [low, high]
    if len(dom) >= 2 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]):
        low, high = float(dom[0]), float(dom[1])
        if high < low:
            low, high = high, low
        if np.isclose(low, high):
            return float(low)
        return float(rng.uniform(low, high))

    # categorical/enumerated
    return dom[rng.integers(0, len(dom))]


class GIESAgent(CommonAgent):
    """
    Greedy Interventional Equivalence Search (GIES), Hauser & Bühlmann (2012).

    We call the actual GIES implementation (pip package `gies`) with the Gaussian BIC score.
    It returns a CPDAG adjacency matrix A where:
      - A[i,j] != 0 and A[j,i] == 0 implies i -> j
      - A[i,j] != 0 and A[j,i] != 0 implies i - j (undirected)
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 3,
        num_inter: int = 3,
        debug: int = 0,
        phases: list[str] | None = None,
        iterate: bool = True,
        k_intervene: int = 1,  # number of variables to intervene on at once
        seed: int | None = 911,
    ) -> None:
        self.id = id
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self.k_intervene = int(max(1, k_intervene))
        self.debug = int(debug)
        self.phases = phases or ["forward", "backward", "turning"]
        self.iterate = bool(iterate)
        self.rng = np.random.default_rng(seed)

        self._columns: list[str] | None = None
        self._cpdag: np.ndarray | None = None
        self._score: float | None = None

    @override
    def inform(self, samples_collection) -> None:
        self.strategy.learn(samples_collection)

        try:
            import gies  # type: ignore
        except Exception:
            # If not installed, just skip learning
            self._cpdag = None
            self._score = None
            return

        # Gather and numeric-encode all samples
        frames = []
        interventions = []
        for s in list(samples_collection):
            if getattr(s, "data", None) is None or len(s.data) == 0:
                continue
            df = _df_to_numeric(s.data)
            if len(df) == 0:
                continue
            frames.append(df)
            interventions.append(getattr(s, "interventions", None))

        if not frames:
            self._cpdag = None
            self._score = None
            return

        # Fix a consistent column order
        if self._columns is None:
            self._columns = list(frames[0].columns)
        cols = self._columns

        # Group by intervention target set (environment)
        # GIES expects: data = [X_env0, X_env1, ...], I = [targets_env0, targets_env1, ...]
        env_map: dict[tuple[int, ...], list[np.ndarray]] = {}

        for df, inter in zip(frames, interventions):
            df = df.reindex(columns=cols)
            X = df.to_numpy(dtype=float)

            if inter is None:
                key = tuple()  # observational environment
            else:
                # map intervened variable names to indices
                idxs = []
                for name in inter.keys():
                    if name in cols:
                        idxs.append(cols.index(name))
                key = tuple(sorted(set(idxs)))

            env_map.setdefault(key, []).append(X)

        data_list: list[np.ndarray] = []
        I_list: list[list[int]] = []

        for key, mats in env_map.items():
            data_list.append(np.vstack(mats))
            I_list.append(list(key))

        # Run GIES (Gaussian BIC score)
        A_hat, score = gies.fit_bic(
            data_list,
            I_list,
            A0=None,
            phases=self.phases,
            iterate=self.iterate,
            debug=self.debug,
        )

        self._cpdag = A_hat.astype(float)
        self._score = float(score)

    def _pick_value(self, domain: list[Any]) -> Any:
        dom = list(domain)
        if not dom:
            return 0

        # Numeric bounds
        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
        ):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low
            if np.isclose(low, high):
                return float(low)
            return float(self.rng.uniform(low, high))

        # Categorical/enumerated
        return dom[self.rng.integers(0, len(dom))]

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        # If no learned CPDAG yet: random intervention
        if self._cpdag is None or self._columns is None:
            vars_sorted = sorted(list(available_actions.experiments), key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = self._pick_value(list(var.domain))
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        cols = self._columns
        A = self._cpdag

        def undirected_neighbors(i: int) -> set[int]:
            nbrs: set[int] = set()
            for j in range(A.shape[0]):
                if i == j:
                    continue
                if A[i, j] != 0.0 and A[j, i] != 0.0:
                    nbrs.add(j)
            return nbrs

        col_index = set(cols)
        cands = sorted(
            [v for v in available_actions.experiments if v.name in col_index],
            key=lambda v: str(v.name),
        )
        if not cands:
            vars_sorted = sorted(list(available_actions.experiments), key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = _sample_value(self.rng, var.domain)
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        chosen: list[Any] = []
        covered_edges: set[tuple[int, int]] = set()  # undirected edges as (min(i,j), max(i,j))

        k = min(self.k_intervene, len(cands))
        for _ in range(k):
            best_var = None
            best_gain = -1

            for v in cands:
                if v in chosen:
                    continue
                i = cols.index(v.name)
                gain = 0
                for j in undirected_neighbors(i):
                    e = (min(i, j), max(i, j))
                    if e not in covered_edges:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            if best_var is None:
                break

            chosen.append(best_var)
            i = cols.index(best_var.name)
            for j in undirected_neighbors(i):
                covered_edges.add((min(i, j), max(i, j)))

        treatment = {v.name: _sample_value(self.rng, v.domain) for v in chosen}
        decision.add_experiment(treatment=treatment, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "debug": self.debug,
            "phases": self.phases,
            "iterate": self.iterate,
            "k_intervene": self.k_intervene,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> GIESAgent:
        if not spec.params:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.num_obs or None,  # type: ignore
            num_inter=spec.params.num_inter or None,  # type: ignore
            debug=spec.params.debug or None,  # type: ignore
            phases=spec.params.phases or None,  # type: ignore
            iterate=spec.params.iterate or None,  # type: ignore
            k_intervene=spec.params.k_intervene or None,  # type: ignore
        )
