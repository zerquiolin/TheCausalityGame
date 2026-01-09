"""The Causality Game - Conditional Average Treatment Effect Strategy."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from TheCausalityGame.core.contracts.dto.environment import (
    SamplesCollection,
)
from TheCausalityGame.core.infrastructure.strategy import Strategy


class CATEStrategy(Strategy):
    """Conditional Average Treatment Effect (CATE) strategy."""

    @override
    def initialize(self) -> None:
        self._model: SGDRegressor | None = None
        self._last: SamplesCollection | None = None
        self._features: list[str]
        self._target: str

    @override
    def learn(self, samples: SamplesCollection) -> None:
        """
        Learn from new samples using partial fit.

        Parameters
        ----------
        samples : SamplesCollection
            Collection of observed/intervened data.
        """
        if self._model is None:
            self._last = samples
            return

        for sample in samples:
            X = sample.data[self._features]  # noqa: N806
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
                X: list[str],  # noqa: ARG001, N803
                treatment: str,  # noqa: ARG001
                outcome: str,  # noqa: ARG001
                covariate_values: tuple[pd.DataFrame, pd.DataFrame],
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    np.zeros(len(covariate_values[0])),
                    columns=["treatment_effect"],
                    dtype=float,
                )

            return dummy

        if self._model is None:

            def before(
                X: list[str],  # noqa: N803
                treatment: str,
                outcome: str,
                covariate_values: tuple[pd.DataFrame, pd.DataFrame],
            ) -> pd.DataFrame:
                self._features = [treatment, *X]
                self._target = outcome

                x_non, x_treat = covariate_values
                data = pd.concat([s.data for s in self._last] if self._last else [])

                X_train = data[self._features]  # noqa: N806
                y_train: pd.Series[int | float | str] = data[self._target]

                model = SGDRegressor(
                    warm_start=True,
                    learning_rate="optimal",
                    random_state=911,
                    max_iter=100,
                    tol=1e-5,
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
            X: list[str],  # noqa: ARG001, N803
            treatment: str,  # noqa: ARG001
            outcome: str,  # noqa: ARG001
            covariate_values: tuple[pd.DataFrame, pd.DataFrame],
        ) -> pd.DataFrame:
            x_non, x_treat = covariate_values

            y_non = self._model.predict(x_non[self._features])  # type: ignore
            y_treat = self._model.predict(x_treat[self._features])  # type: ignore

            return pd.DataFrame(y_treat - y_non, columns=["treatment_effect"])

        return after
