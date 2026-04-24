"""Annadani-style CAASL online decider."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    ExperimentVariable,
    RoundInfo,
)
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path

DOMAIN_BOUNDS_COUNT = 2


class Annadani2024CAASLOnlineDecider(Decider):
    """CAASL-like online decider with a lightweight learned policy."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 1,
        num_inter: int = 1,
        n_bootstrap: int = 24,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        lr: float = 0.05,
        temperature: float = 1.0,
        num_value_candidates: int = 5,
        seed: int | None = 911,
    ) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self._belief = RidgeBootstrapEdgeBelief(
            n_bootstrap=n_bootstrap,
            ridge_lambda=ridge_lambda,
            coef_threshold=coef_threshold,
            seed=seed,
        )
        self.lr = float(lr)
        self.temperature = float(max(1e-6, temperature))
        self.rng = np.random.default_rng(seed)
        self.num_value_candidates = int(max(2, num_value_candidates))
        self.k = 6
        self.w = np.zeros(self.k, dtype=float)
        self._last_phi: np.ndarray | None = None
        self._entropy_before: float | None = None

    @override
    def update(self, observation: RoundObservation) -> None:
        before = self._entropy_before
        self._belief.fit(list(observation.samples))
        after = self._belief.summary().total_entropy if self._belief.summary() else None  # type: ignore

        if before is None or after is None or self._last_phi is None:
            self._entropy_before = None
            self._last_phi = None
            return

        reward = before - after
        self.w += self.lr * reward * self._last_phi
        self._entropy_before = None
        self._last_phi = None

    def _enumerate_actions(self, available_actions: AvailableActions) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        for var in available_actions.experiments:
            dom = list(var.domain)
            if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
                isinstance(x, (int, float, np.integer, np.floating))
                for x in dom[:DOMAIN_BOUNDS_COUNT]
            ):
                low, high = float(dom[0]), float(dom[1])
                if high < low:
                    low, high = high, low

                if np.isclose(low, high):
                    candidates = [float(low)]
                else:
                    candidates = np.linspace(
                        low,
                        high,
                        num=self.num_value_candidates,
                        dtype=float,
                    ).tolist()

                out.extend((var.name, float(v)) for v in candidates)
                continue

            out.extend((var.name, v) for v in dom)

        return out

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    def _candidate_values(self, var: ExperimentVariable) -> list[int | float | str]:
        dom = list(var.domain)
        if not dom:
            return []

        if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:DOMAIN_BOUNDS_COUNT]
        ):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low
            if np.isclose(low, high):
                return [float(low)]
            return (
                np.linspace(low, high, num=self.num_value_candidates, dtype=float)
                .astype(float)
                .tolist()
            )

        return dom

    def _phi(self, var: str, val: object, rounds_left: int | None) -> np.ndarray:
        s = self._belief.summary()
        u = float(self._belief.outgoing_uncertainty(var))
        total_u = 1.0
        if s is not None:
            try:
                total_u = max(1.0, float(s.edge_entropy.sum()))
            except (AttributeError, TypeError, ValueError):
                total_u = 1.0
        u_n = float(u / total_u)

        try:
            sig = float(self._belief.value_signal(var, val))
        except (AttributeError, TypeError, ValueError):
            sig = 0.0
        sig = float(min(sig, 1e6))
        sig_n = float(sig / (1.0 + sig))

        rl = 0.0
        if rounds_left is not None:
            rl = float(1.0 / (1.0 + max(0, int(rounds_left))))

        try:
            float(val)  # type: ignore
            is_num = 1.0
        except (TypeError, ValueError):
            is_num = 0.0

        return np.array([1.0, u_n, sig_n, rl, is_num, u_n * sig_n], dtype=float)

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        del belief
        decision = Decision.experiment()
        summ = self._belief.summary()
        self._entropy_before = summ.total_entropy if summ else None

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        rounds_left = round_info.budget_snapshot.rounds_left if round_info.budget_snapshot else None

        if summ is None:
            for var in available_actions.experiments:
                dom = list(var.domain)
                if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
                    isinstance(x, (int, float, np.integer, np.floating))
                    for x in dom[:DOMAIN_BOUNDS_COUNT]
                ):
                    low, high = float(dom[0]), float(dom[1])
                    if high < low:
                        low, high = high, low
                    val = float(self.rng.uniform(low, high))
                else:
                    vals = self._candidate_values(var)
                    val = vals[self.rng.integers(0, len(vals))]

                decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        actions = self._enumerate_actions(available_actions)
        if not actions:
            return decision

        phis = np.stack([self._phi(a, v, rounds_left) for (a, v) in actions], axis=0)
        logits = (phis @ self.w) / self.temperature
        probs = self._softmax(logits)

        idx = int(self.rng.choice(len(actions), p=probs))
        var, val = actions[idx]
        self._last_phi = phis[idx]
        decision.add_experiment(treatment={var: val}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> DeciderSpec:
        return DeciderSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
                "n_bootstrap": self._belief.n_bootstrap,
                "ridge_lambda": self._belief.ridge_lambda,
                "coef_threshold": self._belief.coef_threshold,
                "lr": self.lr,
                "temperature": self.temperature,
                "num_value_candidates": self.num_value_candidates,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> Annadani2024CAASLOnlineDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 1)),
            n_bootstrap=int(params.get("n_bootstrap", 24)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            lr=float(params.get("lr", 0.05)),
            temperature=float(params.get("temperature", 1.0)),
            num_value_candidates=int(params.get("num_value_candidates", 5)),
        )
