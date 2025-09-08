# Math
import numpy as np
import pandas as pd

from time import time

# typing
from typing import Dict, Any, List
from causalitygame.lib.utils.random_state_serialization import (
    random_state_to_json,
    random_state_from_json,
)

# Abstract Base Class
from causalitygame.scm.nodes.abstract import (
    BaseCategoricSCMNode,
    BaseNumericSCMNode,
)
from causalitygame.lib.constants.nodes import ACCESSIBILITY_OBSERVABLE


class DatabaseDefinedSCMNode:

    def enable_eval_mode(self):
        self.eval_mode = True  # enables access to the unrevealed instances

    def disable_eval_mode(self):
        self.eval_mode = False

    def _generate_values(
        self,
        possibilities: np.ndarray,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState = None,
    ):

        self.logger.debug(
            f"Generating values for {self.name} from database. parent_values have shape {parent_values.shape}."
        )

        # take own random state if None is given
        if random_state is None:
            random_state = self.random_state

        # for each sample, find a random value from the node column filtered to rows that are still possible
        vals = []
        column_as_slice = self.df[self.name].values
        for parent_vals, possibilities_to_complete_instance in zip(
            parent_values.values, possibilities
        ):
            vals.append(
                random_state.choice(column_as_slice[possibilities_to_complete_instance])
            )
        return vals

    def get_distribution(self, parent_values: dict) -> list:
        if not self.parents:
            # Flatten the possibly nested list
            return [
                p
                for sub in self.probability_distribution
                for p in (sub if isinstance(sub, list) else [sub])
            ]

        key_ordered = [parent_values[parent] for parent in self.parents]
        key = ",".join(key_ordered)
        # Flatten the possibly nested list for conditional distributions
        return [
            p
            for sub in self.probability_distribution[key]
            for p in (sub if isinstance(sub, list) else [sub])
        ]

    def get_value_distribution(self, parent_values: dict) -> list:
        """
        Given a dictionary of parent values, returns the probability distribution
        over this node's values.

        Args:
            parent_values (dict): A dictionary mapping parent names to their values.

        Returns:
            list: The probability distribution over the node's values.
        """
        return self.get_distribution(parent_values)


class DatabaseDefinedNumericSCMNode(BaseNumericSCMNode, DatabaseDefinedSCMNode):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        revealed_to_agent: list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.cols = list(df.columns)
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            noise_distribution=None,
            parents=self.cols[: self.cols.index(name)],
            domain=sorted(pd.unique(df[name])),
            random_state=random_state,
        )
        self.df = df
        self.revealed_to_agent = revealed_to_agent
        self.eval_mode = False

    def generate_values(
        self,
        possibilities: np.ndarray,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState = None,
    ):
        return self._generate_values(
            parent_values=parent_values,
            random_state=random_state,
            possibilities=possibilities,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseDefinedNumericSCMNode":
        return cls(**data)


class DatabaseDefinedCategoricSCMNode(BaseCategoricSCMNode, DatabaseDefinedSCMNode):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        revealed_to_agent: list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.cols = list(df.columns)
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            noise_distribution=None,
            parents=self.cols[: self.cols.index(name)],
            domain=sorted(pd.unique(df[name])),
            random_state=random_state,
        )
        self.df = df
        self.revealed_to_agent = revealed_to_agent
        self.eval_mode = False

    def generate_values(
        self,
        possibilities: np.ndarray,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState = None,
    ):
        return self._generate_values(
            parent_values=parent_values,
            random_state=random_state,
            possibilities=possibilities,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseDefinedCategoricSCMNode":
        return cls(**data)
