"""Functional metric for symbolic SCM mechanisms."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.inferers.symbolic_scm import EstimatedSymbolicSCM
from TheCausalityGame.core.contracts.mission import ResultMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.scm_node import NumericalSCMNode
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.metric import (
    NotInitializedError,
    UnsupportedMetricTypeError,
)


class SymbolicMechanismFunctionalErrorMetric(ResultMetric):
    """Score symbolic mechanisms by normalized functional prediction error."""

    name = "Symbolic Mechanism Functional Error"
    description = (
        "Compares discovered symbolic mechanisms against true noiseless numeric SCM "
        "mechanism outputs over sampled parent contexts."
    )
    kinds = ["SymbolicSCM"]  # noqa: RUF012

    def __init__(
        self,
        *,
        num_samples: int = 512,
        missing_penalty: float = 10.0,
        invalid_penalty: float = 10.0,
        seed: int = 911,
    ) -> None:
        self.num_samples = num_samples
        self.missing_penalty = missing_penalty
        self.invalid_penalty = invalid_penalty
        self.seed = seed

    @override
    def mount(self, scm: SCM) -> None:
        self.scm = scm
        self._sample_context = scm.generate_samples(
            num_samples=self.num_samples,
            random_state=np.random.RandomState(self.seed),
        )
        self.is_mounted = True

    def _true_numeric_mechanism_nodes(self) -> list[str]:
        nodes: list[str] = []
        for node_name in self.scm.vars:
            node = self.scm.nodes[node_name]
            if not isinstance(node, NumericalSCMNode):
                continue
            if getattr(node, "evaluation", None) is None:
                continue
            nodes.append(node_name)
        return nodes

    def _true_noiseless_values(self, node_name: str, parent_data: pd.DataFrame) -> np.ndarray:
        node = self.scm.nodes[node_name]
        values = node.generate_values(
            parent_values=parent_data,
            random_state=np.random.RandomState(self.seed),
            cancel_noise=True,
        )
        return np.asarray(values, dtype=float)

    @staticmethod
    def _normalized_mse(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
        true_values = np.asarray(true_values, dtype=float)
        predicted_values = np.asarray(predicted_values, dtype=float)
        finite = np.isfinite(true_values) & np.isfinite(predicted_values)
        if not np.any(finite):
            return 1.0

        true_values = true_values[finite]
        predicted_values = predicted_values[finite]
        mse = float(np.mean((true_values - predicted_values) ** 2))
        scale = float(np.var(true_values))
        return mse / (scale + 1e-12)

    @override
    def evaluate(self, kind: str, result: Any) -> float:
        if not self.is_mounted:
            raise NotInitializedError(self.name)

        if kind != "SymbolicSCM":
            raise UnsupportedMetricTypeError(kind)

        if not isinstance(result, EstimatedSymbolicSCM):
            raise TypeError(  # noqa: TRY003
                f"Expected EstimatedSymbolicSCM result, got {type(result)}"
            )

        target_nodes = self._true_numeric_mechanism_nodes()
        if not target_nodes:
            return 0.0

        scores: list[float] = []
        for node_name in target_nodes:
            mechanism = result.mechanisms.get(node_name)
            if mechanism is None:
                scores.append(self.missing_penalty)
                continue

            parent_data = self._sample_context
            try:
                true_values = self._true_noiseless_values(node_name, parent_data)
                predicted_values = mechanism.evaluate(parent_data)
                scores.append(self._normalized_mse(true_values, predicted_values))
            except Exception:  # noqa: BLE001
                scores.append(self.invalid_penalty)

        return float(np.mean(scores))

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_samples": self.num_samples,
                "missing_penalty": self.missing_penalty,
                "invalid_penalty": self.invalid_penalty,
                "seed": self.seed,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> SymbolicMechanismFunctionalErrorMetric:
        params = spec.params or {}
        return cls(
            num_samples=int(params.get("num_samples", 512)),
            missing_penalty=float(params.get("missing_penalty", 10.0)),
            invalid_penalty=float(params.get("invalid_penalty", 10.0)),
            seed=int(params.get("seed", 911)),
        )
