"""The Causality Game - Exhaustive Agent with multiple strategies."""

from typing import Any, override

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    Feedback,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.infrastructure.strategy import Strategy


class ExhaustiveAgent(Agent):
    """
    Agent that performs exhaustive experimentation and learns conditional treatment effects.

    Parameters
    ----------
    id : str
        Unique identifier for the agent.
    num_obs : int, optional
        Number of observational samples to collect, by default 1.
    num_inter : int, optional
        Number of interventional samples per treatment value, by default 1.
    """

    def __init__(self, id: str, num_obs: int = 1, num_inter: int = 1) -> None:
        super().__init__()
        self.id = id
        self._num_obs = num_obs
        self._num_inter = num_inter

    @override
    def set_context(self, ctx: AgentContext) -> None:
        self._context = ctx

        strategies: dict[str, Strategy] = {
            "Conditional Average Treatment Effect Mission": CATEStrategy()
        }

        self.strategy = strategies[self._context.mission["name"]]
        self.strategy.initialize()

    @override
    def act(
        self, round_info: RoundInfo, available_actions: AvailableActions
    ) -> Decision:
        decision = Decision.experiment()

        for var in available_actions.experiments:
            decision.add_experiment(treatment=None, n=self._num_obs)

            low, high = var.domain
            if isinstance(low, str):  # Categorical variable
                for val in var.domain:
                    decision.add_experiment({var.name: val}, n=self._num_inter)
            else:  # Numerical variable
                for val in np.linspace(float(low), float(high), num=5):
                    decision.add_experiment({var.name: val}, n=self._num_inter)

        return decision

    @override
    def inform(self, samples_collection: SamplesCollection, feedback: Feedback) -> None:
        self.strategy.learn(samples_collection, feedback)

    @override
    def answer(self) -> Any:
        return self.strategy.answer()

    @override
    def to_spec(self) -> AgentSpec:
        return AgentSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            params={"num_obs": self._num_obs, "num_inter": self._num_inter},
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> "ExhaustiveAgent":
        return cls(
            id=spec.id,
            num_obs=spec.params["num_obs"],
            num_inter=spec.params["num_inter"],
        )


class CATEStrategy(Strategy):
    """Conditional Average Treatment Effect (CATE) strategy."""

    @override
    def initialize(self) -> None:
        self._model: SGDRegressor | None = None
        self._last: SamplesCollection | None = None
        self._features: list[str]
        self._target: str

    @override
    def learn(self, samples: SamplesCollection, feedback: Feedback) -> None:
        """
        Learn from new samples using partial fit.

        Parameters
        ----------
        samples : SamplesCollection
            Collection of observed/intervened data.
        feedback : Feedback
            Performance metrics (not used in this strategy).
        """
        if self._model is None:
            self._last = samples
            return

        for sample in samples:
            X = sample.data[self._features]
            y: pd.Series[int | float | str] = sample.data[self._target]
            self._model.partial_fit(X, y)

    @override
    def answer(self) -> Any:
        """
        Return callable that computes treatment effect from counterfactual pairs.

        Returns
        -------
        Callable
            A function that computes treatment effects on given data.
        """
        if self._last is None:

            def dummy(
                X: list[str],
                treatment: str,
                outcome: str,
                covariate_values: tuple[pd.DataFrame, pd.DataFrame],
            ):
                return pd.DataFrame(
                    np.zeros(len(covariate_values[0])),
                    columns=["treatment_effect"],
                    dtype=float,
                )

            return dummy

        if self._model is None:

            def before(
                X: list[str],
                treatment: str,
                outcome: str,
                covariate_values: tuple[pd.DataFrame, pd.DataFrame],
            ):
                self._features = [treatment, *X]
                self._target = outcome

                x_non, x_treat = covariate_values
                data = pd.concat([s.data for s in self._last] if self._last else [])

                X_train = data[self._features]
                y_train: pd.Series[int | float | str] = data[self._target]

                model = SGDRegressor(
                    warm_start=True,
                    learning_rate="optimal",
                    random_state=911,
                    max_iter=1000,
                    tol=1e-3,
                    penalty="l2",
                    alpha=0.01,
                )
                # Optional alternative:
                # model = RandomForestRegressor(n_estimators=100, random_state=911)

                model.partial_fit(X_train, y_train)
                self._model = model

                y_non = model.predict(x_non[self._features])
                y_treat = model.predict(x_treat[self._features])

                return pd.DataFrame(y_treat - y_non, columns=["treatment_effect"])

            return before

        def after(
            X: list[str],
            treatment: str,
            outcome: str,
            covariate_values: tuple[pd.DataFrame, pd.DataFrame],
        ):
            x_non, x_treat = covariate_values

            y_non = self._model.predict(x_non[self._features])  # type: ignore
            y_treat = self._model.predict(x_treat[self._features])  # type: ignore

            return pd.DataFrame(y_treat - y_non, columns=["treatment_effect"])

        return after
