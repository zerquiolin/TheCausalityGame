"""Treatment-effect inferer implementation."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class CATEInferer(Inferer):
    """Infer treatment effects from accumulated observational/interventional data."""

    def __init__(self) -> None:
        self._model: SGDRegressor | None = None
        self._last_data: list[pd.DataFrame] = []
        self._features: list[str] = []
        self._target: str | None = None

    @override
    def update(self, observation: RoundObservation) -> None:
        for sample in observation.samples:
            self._last_data.append(sample.data)

        if self._model is None or self._target is None or not self._features:
            return

        for sample in observation.samples:
            X = sample.data[self._features]  # noqa: N806
            y: pd.Series[int | float | str] = sample.data[self._target]
            self._model.partial_fit(X, y)

    @override
    def answer(self) -> Any:
        if not self._last_data:

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
                data = pd.concat(self._last_data, ignore_index=True, sort=False)

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

            y_non = self._model.predict(x_non[self._features])  # type: ignore[arg-type]
            y_treat = self._model.predict(x_treat[self._features])  # type: ignore[arg-type]

            return pd.DataFrame(y_treat - y_non, columns=["treatment_effect"])

        return after

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(estimate=self.answer())

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(class_=get_class_path(self.__class__))

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> CATEInferer:
        del spec
        return cls()
