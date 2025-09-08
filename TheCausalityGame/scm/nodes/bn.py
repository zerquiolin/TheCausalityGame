# Math
import numpy as np
import pandas as pd

from causalitygame.lib.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)


# Abstract Base Class
from causalitygame.scm.nodes.abstract import (
    BaseCategoricSCMNode,
)
from causalitygame.lib.constants.nodes import ACCESSIBILITY_OBSERVABLE


class BayesianNetworkSCMNode(BaseCategoricSCMNode):
    def __init__(
        self,
        name: str,
        parents: list,
        values: list,
        probability_distribution: dict | list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            parents=parents,
            parent_mappings=None,
            domain=values,
            random_state=random_state,
            noise_distribution=None,
        )
        self.probability_distribution = probability_distribution

        # sanity check of given distribution
        if isinstance(probability_distribution, list):
            assert all(
                [isinstance(v, (float, np.float64)) for v in probability_distribution]
            ), f"invalid entries in distribution for {name}: {probability_distribution}"
            s = np.sum(probability_distribution)
            assert np.isclose(
                s, 1
            ), f"Invalid distribution for leaf node {name}, which sum up to {sum(probability_distribution)}: {probability_distribution}"
            if s != 1:
                self.probability_distribution /= s
        else:
            for parent_combo, distribution in probability_distribution.items():
                assert all(
                    [isinstance(v, (float, np.float64)) for v in distribution]
                ), f"invalid entries in distribution for {name}: {distribution}"
                s = np.sum(distribution)
                assert np.isclose(
                    s, 1
                ), f"Invalid distribution for parent combination {parent_combo} node {name}, which sum up to {sum(distribution)}: {distribution}"
                if s != 1:
                    probability_distribution[parent_combo] /= s

    def generate_values(
        self, parent_values: pd.DataFrame, random_state, cancel_noise=False
    ) -> str:
        """
        Given a dictionary of parent values, returns a sampled value from the node's distribution.

        Args:
            parent_values (dict): A dictionary mapping parent names to their values.

        Returns:
            str: A sampled value from the node's distribution.
        """

        # get distribution for each of the samples
        distributions = self.get_distributions(parent_values)

        # draw samples from each distribution
        self._init_random_state()
        rs = random_state if random_state else self.random_state
        return [rs.choice(self.domain, p=dist) for dist in distributions]

    def get_distributions(self, parent_values: pd.DataFrame) -> list:
        if not self.parents:
            # Flatten the possibly nested list
            return [
                [
                    p
                    for sub in self.probability_distribution
                    for p in (sub if isinstance(sub, list) else [sub])
                ]
                for _ in range(len(parent_values))
            ]

        # get the distribution for each entry, based on the respective values of the parent variables
        dists = []
        for row in parent_values[self.parents].values:
            key = ",".join(row)
            dists.append(
                [
                    p
                    for sub in self.probability_distribution[key]
                    for p in (sub if isinstance(sub, list) else [sub])
                ]
            )
        return dists

    def _to_dict(self) -> dict:
        data = {
            "class": self.__class__.__name__,
            "name": self.name,
            "parents": self.parents,
            "values": self.domain,
            "accessibility": self.accessibility,
            "probability_distribution": self.probability_distribution,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianNetworkSCMNode":
        # Reconstruct the random state.
        random_state = data["random_state"] if "random_state" in data else None
        return cls(
            name=data["name"],
            parents=data["parents"],
            values=data["values"],
            probability_distribution=(
                data["probability_distribution"]
                if data["parents"]
                else (
                    [v[0] for v in data["probability_distribution"]]
                    if isinstance(data["probability_distribution"][0], list)
                    else [v for v in data["probability_distribution"]]
                )
            ),
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            random_state=random_state,
        )
