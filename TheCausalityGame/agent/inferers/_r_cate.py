"""Shared adapter contract for authoritative R CATE implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.inferers._data import combined_frame, zero_cate_answer
from TheCausalityGame.agent.inferers._provenance import BackendProvenance
from TheCausalityGame.agent.inferers._r_bridge import run_r_cate_backend
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.inferer import UnsupportedInfererInputError


class AuthoritativeRCATEInferer(Inferer):
    """Buffer samples and delegate a binary-treatment CATE query to R."""

    backend_method: ClassVar[str]
    capability: ClassVar[str]
    provenance: ClassVar[BackendProvenance]

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._samples: list[Samples] = []
        self._last_backend: dict[str, Any] | None = None

    @override
    def update(self, observation: RoundObservation) -> None:
        self._samples.extend(observation.samples)

    def _payload(
        self,
        *,
        X: list[str],  # noqa: N803
        treatment: str,
        outcome: str,
        covariate_values: tuple[pd.DataFrame, pd.DataFrame],
    ) -> dict[str, Any] | None:
        control, treated = covariate_values
        required = [*X, treatment, outcome]
        data = combined_frame(self._samples)
        if data.empty or any(column not in data.columns for column in required):
            return None
        if any(
            column not in control.columns or column not in treated.columns
            for column in X
        ):
            return None
        if treatment not in control.columns or treatment not in treated.columns:
            return None
        control_values = control[treatment].dropna().unique()
        treated_values = treated[treatment].dropna().unique()
        if len(control_values) != 1 or len(treated_values) != 1:
            raise UnsupportedInfererInputError(
                "CATE queries require one control and one treated value."
            )
        control_value = control_values[0]
        treated_value = treated_values[0]
        if control_value == treated_value:
            raise UnsupportedInfererInputError(
                "Control and treated values must differ."
            )
        observed_treatments = set(data[treatment].dropna().unique())
        requested_treatments = {control_value, treated_value}
        if not observed_treatments.issubset(requested_treatments):
            raise UnsupportedInfererInputError(
                f"{self.backend_method} supports binary treatments only; found "
                f"{len(observed_treatments)} treatment levels."
            )
        usable = data.loc[
            data[treatment].isin([control_value, treated_value]), required
        ].dropna()
        if usable.empty:
            return None
        if any(
            not pd.api.types.is_numeric_dtype(usable[column])
            for column in [*X, outcome]
        ):
            raise UnsupportedInfererInputError(
                f"{self.backend_method} currently supports numeric covariates and outcomes only."
            )
        assignment = (usable[treatment] == treated_value).astype(int)
        binary_arm_count = 2
        if assignment.nunique() != binary_arm_count:
            return None
        if (
            self.backend_method == "honest_causal_tree"
            and assignment.value_counts().min() < 2
        ):
            # Both disjoint samples require at least one row from each arm.
            return None
        feature_names = list(X)
        training_features = usable[X].copy()
        query = control[X].reset_index(drop=True).copy()
        if not feature_names:
            feature_names = [".tcg_intercept"]
            training_features[".tcg_intercept"] = 1.0
            query[".tcg_intercept"] = 1.0
        training_frame = training_features.copy()
        training_frame[outcome] = usable[outcome].to_numpy()
        return {
            "method": self.backend_method,
            "features": feature_names,
            "outcome": outcome,
            "train": training_frame.to_dict(orient="list"),
            "treatment": assignment.tolist(),
            "query": query.to_dict(orient="list"),
            "params": self.params,
        }

    @override
    def answer(self) -> Any:
        if not self._samples:
            return zero_cate_answer()

        frozen = deepcopy(self)

        def estimate(
            X: list[str],  # noqa: N803
            treatment: str,
            outcome: str,
            covariate_values: tuple[pd.DataFrame, pd.DataFrame],
        ) -> pd.DataFrame:
            payload = frozen._payload(
                X=X,
                treatment=treatment,
                outcome=outcome,
                covariate_values=covariate_values,
            )
            if payload is None:
                self._last_backend = {
                    **self.provenance.to_dict(),
                    "fit_status": "insufficient_data",
                    "provisional_estimate": "zero",
                }
                return pd.DataFrame(
                    np.zeros(len(covariate_values[0])),
                    columns=["treatment_effect"],
                )
            response = run_r_cate_backend(payload)
            self._last_backend = response.get("backend")
            if self._last_backend is not None:
                self._last_backend["fit_status"] = "fitted"
            predictions = np.asarray(response["predictions"], dtype=float).reshape(-1)
            if len(predictions) != len(covariate_values[0]):
                raise UnsupportedInfererInputError(
                    "The authoritative R backend returned the wrong prediction count."
                )
            return pd.DataFrame(predictions, columns=["treatment_effect"])

        return estimate

    @override
    def snapshot(self) -> BeliefSnapshot:
        backend = self.provenance.to_dict()
        if self._last_backend:
            backend = {**backend, **self._last_backend}
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={
                "n_samples": sum(len(sample.data) for sample in self._samples),
                "backend": backend,
            },
            capabilities=("cate", self.capability),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__), params=dict(self.params)
        )
