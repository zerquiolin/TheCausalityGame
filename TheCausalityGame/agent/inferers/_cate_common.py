"""Shared helpers for conditional treatment-effect inferers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from TheCausalityGame.core.contracts.dto.environment import Samples

CATECallable = Callable[[list[str], str, str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]
TabularModel = Pipeline


@dataclass(frozen=True, slots=True)
class BufferedSample:
    """Observed sample batch with intervention metadata."""

    data: pd.DataFrame
    interventions: dict[str, object] | None


def buffer_samples(samples: list[BufferedSample], observed: list[Samples]) -> None:
    """Append samples without mutating the source data frames."""
    samples.extend(
        BufferedSample(
            data=sample.data.copy(),
            interventions=dict(sample.interventions) if sample.interventions else None,
        )
        for sample in observed
    )


def zero_effect(covariate_values: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Return a valid zero-effect estimate with the expected output shape."""
    return pd.DataFrame(
        np.zeros(len(covariate_values[0]), dtype=float),
        columns=["treatment_effect"],
    )


def zero_answer() -> CATECallable:
    """Return a valid CATE callable that always predicts no effect."""

    def answer(
        X: list[str],  # noqa: ARG001, N803
        treatment: str,  # noqa: ARG001
        outcome: str,  # noqa: ARG001
        covariate_values: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        return zero_effect(covariate_values)

    return answer


def treatment_pair(
    treatment: str,
    covariate_values: tuple[pd.DataFrame, pd.DataFrame],
) -> tuple[object, object] | None:
    """Extract control and treated values from the query frames."""
    control, treated = covariate_values
    if treatment not in control.columns or treatment not in treated.columns:
        return None
    if control.empty or treated.empty:
        return None
    return control[treatment].iloc[0], treated[treatment].iloc[0]


def combined_data(samples: list[BufferedSample]) -> pd.DataFrame:
    """Combine buffered sample data into one frame."""
    if not samples:
        return pd.DataFrame()
    return pd.concat([sample.data for sample in samples], ignore_index=True, sort=False)


def controlled_data(
    samples: list[BufferedSample],
    treatment: str,
    treatment_values: tuple[object, object],
) -> pd.DataFrame:
    """Return rows from batches where the queried treatment was externally fixed."""
    frames: list[pd.DataFrame] = []
    allowed = set(treatment_values)
    for sample in samples:
        if not sample.interventions or treatment not in sample.interventions:
            continue
        value = sample.interventions[treatment]
        if value not in allowed:
            continue
        frame = sample.data.copy()
        frame[treatment] = value
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def usable_rows(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Drop rows missing any required column."""
    if data.empty or any(column not in data.columns for column in columns):
        return pd.DataFrame(columns=columns)
    return data.dropna(subset=columns).copy()


def with_treatment_interactions(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Add numeric treatment-covariate interactions for heterogeneous effects."""
    features = data[[treatment, *covariates]].copy()
    treatment_values = pd.to_numeric(features[treatment], errors="coerce")
    if treatment_values.isna().all():
        return features
    for covariate in covariates:
        covariate_values = pd.to_numeric(features[covariate], errors="coerce")
        if covariate_values.isna().all():
            continue
        features[f"{treatment}__x__{covariate}"] = (
            treatment_values.fillna(0.0) * covariate_values.fillna(0.0)
        )
    return features


def fit_tabular_regressor(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series | np.ndarray,
    *,
    alpha: float,
) -> TabularModel:
    """Fit a ridge regression pipeline for mixed tabular features."""
    model = Pipeline(
        steps=[
            ("preprocess", _preprocessor(X)),
            ("model", Ridge(alpha=float(max(alpha, 0.0)))),
        ]
    )
    model.fit(X, np.asarray(y, dtype=float))
    return model


def fit_binary_propensity(
    X: pd.DataFrame,  # noqa: N803
    treated: pd.Series | np.ndarray,
    *,
    alpha: float,
    random_state: int,
) -> tuple[TabularModel, int]:
    """Fit a binary propensity model and return the positive-class index."""
    labels = np.asarray(treated, dtype=int)
    inverse_regularization = 1.0 / max(float(alpha), 1e-9)
    model = Pipeline(
        steps=[
            ("preprocess", _preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    C=inverse_regularization,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X, labels)
    classifier = model.named_steps["model"]
    positive_index = int(np.where(classifier.classes_ == 1)[0][0])
    return model, positive_index


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:  # noqa: N803
    """Create a stable sklearn preprocessor for the observed feature columns."""
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [column for column in X.columns if column not in numeric_features]
    transformers: list[tuple[str, object, list[str]]] = []
    if not numeric_features and not categorical_features:
        transformers.append(("constant", ConstantFeatureTransformer(), []))
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("categorical", _one_hot_encoder(), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _one_hot_encoder() -> OneHotEncoder:
    """Build an encoder compatible with recent and older sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class ConstantFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create one constant feature when a CATE query has no covariates."""

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: object | None = None,
    ) -> ConstantFeatureTransformer:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return np.ones((len(X), 1), dtype=float)
