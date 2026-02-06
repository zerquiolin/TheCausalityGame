from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class Annadani2024CAASLOnlineAgent(CommonAgent):
    """
    CAASL-like amortized policy (online proxy):
      - chooses action via learned policy π(a | belief-summary)
      - updates via REINFORCE-style rule using entropy-reduction reward
      - does NOT do lookahead fantasies

    Later you can swap the linear policy for a trained transformer while keeping:
      - action masking from AvailableActions
      - reward plumbing via inform()
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 1,
        num_inter: int = 1,
        n_bootstrap: int = 24,  # keep belief cheap, policy shouldn't be heavy
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        lr: float = 0.05,
        temperature: float = 1.0,
        num_value_candidates: int = 5,  # sampled values for numeric ranges
        seed: int | None = 911,
    ) -> None:
        self.id = id
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

        # small linear policy over hand-designed features
        self.k = 6
        self.w = np.zeros(self.k, dtype=float)

        # saved for policy-gradient update
        self._last_phi: np.ndarray | None = None
        self._entropy_before: float | None = None

    @override
    def inform(self, samples_collection) -> None:
        self.strategy.learn(samples_collection)
        before = self._entropy_before
        self._belief.fit(list(samples_collection))
        after = self._belief.summary().total_entropy if self._belief.summary() else None

        if before is None or after is None or self._last_phi is None:
            self._entropy_before = None
            self._last_phi = None
            return

        reward = before - after  # positive if uncertainty reduced

        # REINFORCE-ish update (simple, stable baseline-free bandit update)
        self.w += self.lr * reward * self._last_phi

        self._entropy_before = None
        self._last_phi = None

    def _enumerate_actions(self, available_actions: AvailableActions) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for var in available_actions.experiments:
            dom = list(var.domain)

            # Numeric bounds: treat as [low, high] and sample interior candidate values.
            if len(dom) >= 2 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
            ):
                low, high = float(dom[0]), float(dom[1])
                if high < low:
                    low, high = high, low

                if np.isclose(low, high):
                    candidates = [float(low)]
                else:
                    # Stratified grid across the range.
                    candidates = np.linspace(
                        low, high, num=self.num_value_candidates, dtype=float
                    ).tolist()

                for v in candidates:
                    out.append((var.name, float(v)))
                continue

            # Categorical/enumerated domain: use provided values.
            for v in dom:
                out.append((var.name, v))

        return out

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    def _candidate_values(self, var) -> list[Any]:
        """Return candidate intervention values for a variable domain."""
        dom = list(var.domain)
        if not dom:
            return []

        # Numeric bounds: treat as [low, high] and generate an interior grid.
        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
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

        # Categorical/enumerated domain.
        return dom

    def _phi(self, var: str, val: Any, rounds_left: int | None) -> np.ndarray:
        """Feature map for the linear policy over actions."""
        s = self._belief.summary()

        # Structural uncertainty (where)
        u = float(self._belief.outgoing_uncertainty(var))
        total_u = 1.0
        if s is not None:
            try:
                total_u = max(1.0, float(s.edge_entropy.sum()))
            except Exception:
                total_u = 1.0
        u_n = float(u / total_u)

        # Value signal (how): a simple heuristic from the belief.
        try:
            sig = float(self._belief.value_signal(var, val))
        except Exception:
            sig = 0.0
        sig = float(min(sig, 1e6))
        sig_n = float(sig / (1.0 + sig))

        # Rounds-left feature (budget awareness)
        rl = 0.0
        if rounds_left is not None:
            rl = float(1.0 / (1.0 + max(0, int(rounds_left))))

        # Numeric indicator
        try:
            float(val)
            is_num = 1.0
        except (TypeError, ValueError):
            is_num = 0.0

        # 6-dimensional feature vector (must match self.k)
        return np.array([1.0, u_n, sig_n, rl, is_num, u_n * sig_n], dtype=float)

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()

        # store entropy BEFORE action for reward later
        summ = self._belief.summary()
        self._entropy_before = summ.total_entropy if summ else None

        # always take a little observational data (optional)
        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        rounds_left = round_info.budget_snapshot.rounds_left if round_info.budget_snapshot else None

        if summ is None:
            # no belief yet: random interventions, sampling interior for numeric ranges
            for var in available_actions.experiments:
                dom = list(var.domain)
                if len(dom) >= 2 and all(
                    isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
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

        # save for policy gradient update
        self._last_phi = phis[idx]

        decision.add_experiment(treatment={var: val}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "n_bootstrap": self._belief.n_bootstrap,
            "ridge_lambda": self._belief.ridge_lambda,
            "coef_threshold": self._belief.coef_threshold,
            "lr": self.lr,
            "temperature": self.temperature,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> Annadani2024CAASLOnlineAgent:
        if spec.params is None:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.num_obs or None,  # type: ignore
            num_inter=spec.params.num_inter or None,  # type: ignore
            n_bootstrap=spec.params.n_bootstrap or None,  # type: ignore
            ridge_lambda=spec.params.ridge_lambda or None,  # type: ignore
            coef_threshold=spec.params.coef_threshold or None,  # type: ignore
            lr=spec.params.lr or None,  # type: ignore
            temperature=spec.params.temperature or None,  # type: ignore
        )
