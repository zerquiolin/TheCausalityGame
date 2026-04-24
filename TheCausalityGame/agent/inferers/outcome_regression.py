"""Outcome-regression inferer for conditional treatment effects."""

from __future__ import annotations

from typing import Any, override

import pandas as pd

from TheCausalityGame.agent.inferers._cate_common import (
    BufferedSample,
    buffer_samples,
    combined_data,
    fit_tabular_regressor,
    usable_rows,
    with_treatment_interactions,
    zero_answer,
    zero_effect,
)
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class OutcomeRegressionInferer(Inferer):
    """Estimate CATE by predicting potential outcomes with an S-learner."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._samples: list[BufferedSample] = []

    @override
    def update(self, observation: RoundObservation) -> None:
        buffer_samples(self._samples, list(observation.samples))

    @override
    def answer(self) -> Any:
        if not self._samples:
            return zero_answer()

        def estimate(
            X: list[str],  # noqa: N803
            treatment: str,
            outcome: str,
            covariate_values: tuple[pd.DataFrame, pd.DataFrame],
        ) -> pd.DataFrame:
            features = [treatment, *X]
            data = usable_rows(combined_data(self._samples), [*features, outcome])
            if data.empty:
                return zero_effect(covariate_values)

            model = fit_tabular_regressor(
                with_treatment_interactions(data, treatment, X),
                data[outcome],
                alpha=self.alpha,
            )

            control, treated = covariate_values
            missing_features = [
                feature
                for feature in features
                if feature not in control.columns or feature not in treated.columns
            ]
            if missing_features:
                return zero_effect(covariate_values)

            y_control = model.predict(with_treatment_interactions(control, treatment, X))
            y_treated = model.predict(with_treatment_interactions(treated, treatment, X))
            return pd.DataFrame(y_treated - y_control, columns=["treatment_effect"])

        return estimate

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"n_samples": sum(len(sample.data) for sample in self._samples)},
            capabilities=("cate", "outcome_regression"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={"alpha": self.alpha},
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> OutcomeRegressionInferer:
        params = spec.params or {}
        return cls(alpha=float(params.get("alpha", 1.0)))
