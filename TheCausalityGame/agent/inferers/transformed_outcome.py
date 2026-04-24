"""Transformed-outcome inferer for conditional treatment effects."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.inferers._cate_common import (
    BufferedSample,
    buffer_samples,
    combined_data,
    controlled_data,
    fit_binary_propensity,
    fit_tabular_regressor,
    treatment_pair,
    usable_rows,
    zero_answer,
    zero_effect,
)
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class TransformedOutcomeInferer(Inferer):
    """Estimate CATE by regressing IPW-style transformed outcomes."""

    def __init__(
        self,
        alpha: float = 1.0,
        propensity_alpha: float = 1.0,
        min_group_size: int = 2,
        propensity_clip: float = 0.05,
    ) -> None:
        self.alpha = float(alpha)
        self.propensity_alpha = float(propensity_alpha)
        self.min_group_size = int(max(1, min_group_size))
        self.propensity_clip = float(min(max(propensity_clip, 1e-6), 0.49))
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
            values = treatment_pair(treatment, covariate_values)
            if values is None:
                return zero_effect(covariate_values)

            control_value, treated_value = values
            train = self._interventional_training_frame(
                X=X,
                treatment=treatment,
                outcome=outcome,
                control_value=control_value,
                treated_value=treated_value,
            )
            if train is None:
                train = self._observational_training_frame(
                    X=X,
                    treatment=treatment,
                    outcome=outcome,
                    control_value=control_value,
                    treated_value=treated_value,
                )
            if train is None or train.empty:
                return zero_effect(covariate_values)

            model = fit_tabular_regressor(
                train[X],
                train["_transformed_outcome"],
                alpha=self.alpha,
            )

            control, _treated = covariate_values
            if any(feature not in control.columns for feature in X):
                return zero_effect(covariate_values)
            return pd.DataFrame(
                model.predict(control[X]),
                columns=["treatment_effect"],
            )

        return estimate

    def _interventional_training_frame(
        self,
        *,
        X: list[str],  # noqa: N803
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> pd.DataFrame | None:
        data = usable_rows(
            controlled_data(self._samples, treatment, (control_value, treated_value)),
            [*X, treatment, outcome],
        )
        if data.empty:
            return None

        treated_mask = data[treatment] == treated_value
        control_mask = data[treatment] == control_value
        n_treated = int(treated_mask.sum())
        n_control = int(control_mask.sum())
        if n_treated < self.min_group_size or n_control < self.min_group_size:
            return None

        total = n_treated + n_control
        p_treated = n_treated / total
        p_control = n_control / total
        transformed = np.zeros(len(data), dtype=float)
        transformed[treated_mask.to_numpy()] = data.loc[treated_mask, outcome].to_numpy(
            dtype=float
        ) / p_treated
        transformed[control_mask.to_numpy()] = -data.loc[control_mask, outcome].to_numpy(
            dtype=float
        ) / p_control

        train = data[X].copy()
        train["_transformed_outcome"] = transformed
        return train

    def _observational_training_frame(
        self,
        *,
        X: list[str],  # noqa: N803
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> pd.DataFrame | None:
        data = usable_rows(combined_data(self._samples), [*X, treatment, outcome])
        if data.empty:
            return None

        in_pair = data[treatment].isin([control_value, treated_value])
        data = data.loc[in_pair].copy()
        if data.empty:
            return None

        treated_mask = data[treatment] == treated_value
        control_mask = data[treatment] == control_value
        too_few_treated = int(treated_mask.sum()) < self.min_group_size
        too_few_control = int(control_mask.sum()) < self.min_group_size
        if too_few_treated or too_few_control:
            return None

        propensity_model, positive_index = fit_binary_propensity(
            data[X],
            treated_mask.astype(int),
            alpha=self.propensity_alpha,
            random_state=911,
        )
        p_treated = np.clip(
            propensity_model.predict_proba(data[X])[:, positive_index],
            self.propensity_clip,
            1.0 - self.propensity_clip,
        )
        p_control = 1.0 - p_treated

        transformed = np.zeros(len(data), dtype=float)
        transformed[treated_mask.to_numpy()] = data.loc[treated_mask, outcome].to_numpy(
            dtype=float
        ) / p_treated[treated_mask.to_numpy()]
        transformed[control_mask.to_numpy()] = -data.loc[control_mask, outcome].to_numpy(
            dtype=float
        ) / p_control[control_mask.to_numpy()]

        train = data[X].copy()
        train["_transformed_outcome"] = transformed
        return train

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"n_samples": sum(len(sample.data) for sample in self._samples)},
            capabilities=("cate", "transformed_outcome"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "alpha": self.alpha,
                "propensity_alpha": self.propensity_alpha,
                "min_group_size": self.min_group_size,
                "propensity_clip": self.propensity_clip,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> TransformedOutcomeInferer:
        params = spec.params or {}
        return cls(
            alpha=float(params.get("alpha", 1.0)),
            propensity_alpha=float(params.get("propensity_alpha", 1.0)),
            min_group_size=int(params.get("min_group_size", 2)),
            propensity_clip=float(params.get("propensity_clip", 0.05)),
        )
