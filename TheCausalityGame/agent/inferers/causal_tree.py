"""Honest causal-tree inferer for conditional treatment effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.inferers._cate_common import (
    BufferedSample,
    buffer_samples,
    combined_data,
    controlled_data,
    treatment_pair,
    usable_rows,
    zero_answer,
    zero_effect,
)
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


@dataclass(slots=True)
class _Split:
    feature: str
    value: object
    is_numeric: bool


@dataclass(slots=True)
class _Node:
    effect: float
    split: _Split | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split is None or self.left is None or self.right is None


class HonestCausalTreeInferer(Inferer):
    """Estimate CATE with an honest recursive partitioning tree."""

    def __init__(  # noqa: PLR0913
        self,
        max_depth: int = 3,
        min_leaf_size: int = 8,
        min_treatment_count: int = 2,
        honest_fraction: float = 0.5,
        max_candidates_per_feature: int = 8,
        random_state: int = 911,
    ) -> None:
        self.max_depth = int(max(1, max_depth))
        self.min_leaf_size = int(max(2, min_leaf_size))
        self.min_treatment_count = int(max(1, min_treatment_count))
        self.honest_fraction = float(min(max(honest_fraction, 0.1), 0.9))
        self.max_candidates_per_feature = int(max(1, max_candidates_per_feature))
        self.random_state = int(random_state)
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
            data = self._training_data(
                X=X,
                treatment=treatment,
                outcome=outcome,
                control_value=control_value,
                treated_value=treated_value,
            )
            if data.empty:
                return zero_effect(covariate_values)

            split_data, estimate_data = self._honest_partition(data)
            if split_data.empty or estimate_data.empty:
                return zero_effect(covariate_values)

            root_effect = self._effect(
                estimate_data,
                treatment,
                outcome,
                control_value,
                treated_value,
            )
            root = self._build_tree(
                split_data=split_data,
                estimate_data=estimate_data,
                features=X,
                treatment=treatment,
                outcome=outcome,
                control_value=control_value,
                treated_value=treated_value,
                depth=0,
                fallback_effect=root_effect,
            )

            control, _treated = covariate_values
            if any(feature not in control.columns for feature in X):
                return zero_effect(covariate_values)
            effects = [
                self._predict_row(root, row)
                for _, row in control[X].reset_index(drop=True).iterrows()
            ]
            return pd.DataFrame(effects, columns=["treatment_effect"], dtype=float)

        return estimate

    def _training_data(
        self,
        *,
        X: list[str],  # noqa: N803
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> pd.DataFrame:
        data = controlled_data(self._samples, treatment, (control_value, treated_value))
        if data.empty:
            data = combined_data(self._samples)
        data = usable_rows(data, [*X, treatment, outcome])
        if data.empty:
            return data
        return data.loc[data[treatment].isin([control_value, treated_value])].copy()

    def _honest_partition(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(data))
        rng.shuffle(indices)
        split_at = round(len(indices) * self.honest_fraction)
        split_at = min(max(split_at, 1), len(indices) - 1)
        split_indices = indices[:split_at]
        estimate_indices = indices[split_at:]
        return (
            data.iloc[split_indices].reset_index(drop=True),
            data.iloc[estimate_indices].reset_index(drop=True),
        )

    def _build_tree(  # noqa: PLR0913
        self,
        *,
        split_data: pd.DataFrame,
        estimate_data: pd.DataFrame,
        features: list[str],
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
        depth: int,
        fallback_effect: float,
    ) -> _Node:
        effect = self._effect_or_fallback(
            estimate_data,
            treatment,
            outcome,
            control_value,
            treated_value,
            fallback_effect,
        )
        if depth >= self.max_depth or len(split_data) < 2 * self.min_leaf_size:
            return _Node(effect=effect)

        split = self._best_split(
            split_data=split_data,
            features=features,
            treatment=treatment,
            outcome=outcome,
            control_value=control_value,
            treated_value=treated_value,
        )
        if split is None:
            return _Node(effect=effect)

        split_left, split_right = self._partition(split_data, split)
        estimate_left, estimate_right = self._partition(estimate_data, split)
        if split_left.empty or split_right.empty:
            return _Node(effect=effect)

        return _Node(
            effect=effect,
            split=split,
            left=self._build_tree(
                split_data=split_left,
                estimate_data=estimate_left,
                features=features,
                treatment=treatment,
                outcome=outcome,
                control_value=control_value,
                treated_value=treated_value,
                depth=depth + 1,
                fallback_effect=effect,
            ),
            right=self._build_tree(
                split_data=split_right,
                estimate_data=estimate_right,
                features=features,
                treatment=treatment,
                outcome=outcome,
                control_value=control_value,
                treated_value=treated_value,
                depth=depth + 1,
                fallback_effect=effect,
            ),
        )

    def _best_split(  # noqa: PLR0913
        self,
        *,
        split_data: pd.DataFrame,
        features: list[str],
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> _Split | None:
        parent_score = self._node_score(
            split_data,
            treatment,
            outcome,
            control_value,
            treated_value,
        )
        best_gain = 0.0
        best_split: _Split | None = None
        for candidate in self._candidate_splits(split_data, features):
            left, right = self._partition(split_data, candidate)
            if not self._valid_leaf(left, treatment, control_value, treated_value):
                continue
            if not self._valid_leaf(right, treatment, control_value, treated_value):
                continue
            child_score = self._node_score(
                left,
                treatment,
                outcome,
                control_value,
                treated_value,
            ) + self._node_score(
                right,
                treatment,
                outcome,
                control_value,
                treated_value,
            )
            gain = child_score - parent_score
            if gain > best_gain:
                best_gain = gain
                best_split = candidate
        return best_split

    def _candidate_splits(self, data: pd.DataFrame, features: list[str]) -> list[_Split]:
        splits: list[_Split] = []
        for feature in features:
            series = data[feature].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                quantiles = np.linspace(0.1, 0.9, self.max_candidates_per_feature)
                values = np.unique(np.quantile(series.to_numpy(dtype=float), quantiles))
                splits.extend(
                    _Split(feature=feature, value=float(value), is_numeric=True)
                    for value in values
                )
            else:
                values = series.astype(str).value_counts().index[: self.max_candidates_per_feature]
                splits.extend(
                    _Split(feature=feature, value=value, is_numeric=False)
                    for value in values
                )
        return splits

    def _partition(self, data: pd.DataFrame, split: _Split) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data.empty:
            return data, data
        if split.is_numeric:
            mask = data[split.feature].astype(float) <= float(split.value)
        else:
            mask = data[split.feature].astype(str) == str(split.value)
        return data.loc[mask].copy(), data.loc[~mask].copy()

    def _valid_leaf(
        self,
        data: pd.DataFrame,
        treatment: str,
        control_value: object,
        treated_value: object,
    ) -> bool:
        if len(data) < self.min_leaf_size:
            return False
        counts = data[treatment].value_counts()
        return (
            int(counts.get(control_value, 0)) >= self.min_treatment_count
            and int(counts.get(treated_value, 0)) >= self.min_treatment_count
        )

    def _node_score(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> float:
        if not self._valid_leaf(data, treatment, control_value, treated_value):
            return 0.0
        effect = self._effect(data, treatment, outcome, control_value, treated_value)
        return float(len(data) * effect * effect)

    def _effect_or_fallback(  # noqa: PLR0913
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
        fallback: float,
    ) -> float:
        if not self._valid_leaf(data, treatment, control_value, treated_value):
            return fallback
        return self._effect(data, treatment, outcome, control_value, treated_value)

    @staticmethod
    def _effect(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_value: object,
        treated_value: object,
    ) -> float:
        treated = data.loc[data[treatment] == treated_value, outcome].astype(float)
        control = data.loc[data[treatment] == control_value, outcome].astype(float)
        if treated.empty or control.empty:
            return 0.0
        return float(treated.mean() - control.mean())

    def _predict_row(self, node: _Node, row: pd.Series[object]) -> float:
        current = node
        while not current.is_leaf and current.split is not None:
            split = current.split
            value = row[split.feature]
            if split.is_numeric:
                go_left = float(value) <= float(split.value)
            else:
                go_left = str(value) == str(split.value)
            next_node = current.left if go_left else current.right
            if next_node is None:
                break
            current = next_node
        return current.effect

    @override
    def snapshot(self) -> BeliefSnapshot:
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"n_samples": sum(len(sample.data) for sample in self._samples)},
            capabilities=("cate", "honest_causal_tree"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "max_depth": self.max_depth,
                "min_leaf_size": self.min_leaf_size,
                "min_treatment_count": self.min_treatment_count,
                "honest_fraction": self.honest_fraction,
                "max_candidates_per_feature": self.max_candidates_per_feature,
                "random_state": self.random_state,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> HonestCausalTreeInferer:
        params = spec.params or {}
        return cls(
            max_depth=int(params.get("max_depth", 3)),
            min_leaf_size=int(params.get("min_leaf_size", 8)),
            min_treatment_count=int(params.get("min_treatment_count", 2)),
            honest_fraction=float(params.get("honest_fraction", 0.5)),
            max_candidates_per_feature=int(params.get("max_candidates_per_feature", 8)),
            random_state=int(params.get("random_state", 911)),
        )
