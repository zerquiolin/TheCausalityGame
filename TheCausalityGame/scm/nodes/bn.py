"""Discrete Bayesian-network nodes with validated, ordered conditional tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.scm_node import CategoricalSCMNode, SCMNodeSpec
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_from_json,
)


class BayesianNetworkSCMNode(CategoricalSCMNode):
    """Sample a finite categorical domain from a normalized CPT."""

    def __init__(
        self,
        name: str,
        parents: list[str],
        domain: list[str],
        probability_distribution: dict | list,
        accessibility: NodeAccessibility = NodeAccessibility.CONTROLLABLE,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            parents=parents,
            parent_mappings=None,
            domain=domain,
            random_state=random_state,
            noise_distribution=None,
        )
        if not domain or len(set(domain)) != len(domain):
            raise ValueError(f"{name} needs a nonempty unique categorical domain.")
        if any("," in str(category) for category in domain):
            raise ValueError("CPT category labels cannot contain commas.")

        def normalized(values: list) -> list[float]:
            probabilities = np.asarray(values, dtype=float).reshape(-1)
            if (
                len(probabilities) != len(domain)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError(f"Invalid probability vector for {name}: {values}")
            return (probabilities / probabilities.sum()).tolist()

        if parents:
            if not isinstance(probability_distribution, dict):
                raise ValueError("A parented BN node needs a conditional table.")
            self.probability_distribution = {
                key: normalized(values)
                for key, values in probability_distribution.items()
            }
        else:
            self.probability_distribution = normalized(probability_distribution)

    def generate_values(
        self,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState | None = None,
        cancel_noise: bool = False,
    ) -> np.ndarray:
        """Use one scalar uniform draw per row, preserving batch/chunk equivalence."""
        rs = self.random_state if random_state is None else random_state
        distributions = self.get_distributions(parent_values)
        uniforms = rs.uniform(size=len(parent_values))
        if cancel_noise:
            uniforms[:] = 0.5
        return np.asarray(
            [
                self.domain[
                    min(
                        np.searchsorted(np.cumsum(dist), u, side="right"),
                        len(self.domain) - 1,
                    )
                ]
                for dist, u in zip(distributions, uniforms, strict=True)
            ],
        )

    def get_distributions(self, parent_values: pd.DataFrame) -> list:
        """Look up CPTs using declared parent order."""
        if not self.parents:
            return [
                list(self.probability_distribution) for _ in range(len(parent_values))
            ]
        if not set(self.parents).issubset(parent_values.columns):
            raise ValueError(f"Missing parents for {self.name}: {self.parents}")
        result = []
        for row in parent_values[self.parents].itertuples(index=False, name=None):
            key = ",".join(map(str, row))
            if key not in self.probability_distribution:
                raise ValueError(f"Missing CPT row for {self.name}: {key}")
            result.append(list(self.probability_distribution[key]))
        return result

    def _to_dict(self) -> dict[str, Any]:
        return {"probability_distribution": self.probability_distribution}

    @classmethod
    def from_spec(cls, spec: SCMNodeSpec) -> BayesianNetworkSCMNode:
        """Reconstruct a node without changing parent ordering or probabilities."""
        return cls(
            name=spec.name,
            parents=spec.parents or [],
            domain=list(spec.domain),
            probability_distribution=spec.probability_distribution,
            accessibility=spec.accessibility,
            random_state=(
                random_state_from_json(spec.random_state) if spec.random_state else None
            ),
        )
