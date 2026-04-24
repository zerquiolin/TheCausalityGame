"""GIES-style interventional design decider."""

from __future__ import annotations

import importlib
from typing import override

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path

DOMAIN_BOUNDS_COUNT = 2


def _df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("category").cat.codes.astype(float)
    return df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")


def _sample_value(rng: np.random.Generator, domain: list[object]) -> object:
    dom = list(domain)
    if not dom:
        return 0

    if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
        isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:DOMAIN_BOUNDS_COUNT]
    ):
        low, high = float(dom[0]), float(dom[1])  # type: ignore
        if high < low:
            low, high = high, low
        if np.isclose(low, high):
            return float(low)
        return float(rng.uniform(low, high))

    return dom[rng.integers(0, len(dom))]


class GIESDecider(Decider):
    """Greedy interventional design using a learned GIES CPDAG."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 3,
        num_inter: int = 3,
        debug: int = 0,
        phases: list[str] | None = None,
        iterate: bool = True,
        k_intervene: int = 1,
        seed: int | None = 911,
    ) -> None:
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
    def update(self, observation: RoundObservation) -> None:
        try:
            gies = importlib.import_module("gies")
        except ImportError:
            self._cpdag = None
            self._score = None
            return

        frames = []
        interventions = []
        for s in list(observation.samples):
            if getattr(s, "data", None) is None or len(s.data) == 0:
                continue
            df = _df_to_numeric(s.data)
            if len(df) == 0:
                continue
            frames.append(df)  # type: ignore
            interventions.append(getattr(s, "interventions", None))  # type: ignore

        if not frames:
            self._cpdag = None
            self._score = None
            return

        if self._columns is None:
            self._columns = list(frames[0].columns)  # type: ignore
        cols = self._columns

        env_map: dict[tuple[int, ...], list[np.ndarray]] = {}
        for frame, inter in zip(frames, interventions, strict=True):  # type: ignore
            numeric_frame = frame.reindex(columns=cols)  # type: ignore
            values = numeric_frame.to_numpy(dtype=float)  # type: ignore

            if inter is None:
                key = ()
            else:
                idxs = [cols.index(name) for name in inter if name in cols]  # type: ignore
                key = tuple(sorted(set(idxs)))

            env_map.setdefault(key, []).append(values)  # type: ignore

        data_list: list[np.ndarray] = []
        intervention_list: list[list[int]] = []
        for key, mats in env_map.items():
            data_list.append(np.vstack(mats))
            intervention_list.append(list(key))

        adjacency_hat, score = gies.fit_bic(
            data_list,
            intervention_list,
            A0=None,
            phases=self.phases,
            iterate=self.iterate,
            debug=self.debug,
        )

        self._cpdag = adjacency_hat.astype(float)
        self._score = float(score)

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        del round_info, belief
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        if self._cpdag is None or self._columns is None:
            vars_sorted = sorted(available_actions.experiments, key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = _sample_value(self.rng, list(var.domain))
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        cols = self._columns
        adjacency = self._cpdag

        def undirected_neighbors(i: int) -> set[int]:
            nbrs: set[int] = set()
            for j in range(adjacency.shape[0]):
                if i == j:
                    continue
                if adjacency[i, j] != 0.0 and adjacency[j, i] != 0.0:
                    nbrs.add(j)
            return nbrs

        col_index = set(cols)
        cands = sorted(
            [v for v in available_actions.experiments if v.name in col_index],
            key=lambda v: str(v.name),
        )
        if not cands:
            vars_sorted = sorted(available_actions.experiments, key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = _sample_value(self.rng, list(var.domain))
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        chosen: list[object] = []
        covered_edges: set[tuple[int, int]] = set()

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

        treatment = {v.name: _sample_value(self.rng, list(v.domain)) for v in chosen}  # type: ignore
        decision.add_experiment(treatment=treatment, n=self._num_inter)  # type: ignore
        return decision

    @override
    def to_spec(self) -> DeciderSpec:
        return DeciderSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
                "debug": self.debug,
                "phases": self.phases,
                "iterate": self.iterate,
                "k_intervene": self.k_intervene,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> GIESDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 3)),
            num_inter=int(params.get("num_inter", 3)),
            debug=int(params.get("debug", 0)),
            phases=list(params.get("phases", ["forward", "backward", "turning"])),
            iterate=bool(params.get("iterate", True)),
            k_intervene=int(params.get("k_intervene", 1)),
        )
