from causalitygame.scm.nodes.abstract import BaseNumericSCMNode, BaseCategoricSCMNode
from causalitygame.generators.outcome.base import OutcomeGenerator

import numpy as np
import pandas as pd

import logging


class ComputedNumericSCMNode(BaseNumericSCMNode):

    def __init__(
        self, name, value_computer, accessibility=..., random_state=..., logger=None
    ):
        super().__init__(
            name,
            None,
            domain=[],
            noise_distribution=None,
            accessibility=accessibility,
            random_state=random_state,
            logger=logger,
        )
        self.value_computer = value_computer

    def generate_values(self, parent_values, random_state):
        assert isinstance(
            parent_values, pd.DataFrame
        ), f"parent_values must be given as a DataFrame but are {type(parent_values)}"
        self.logger.info(
            f"Generating values given parent values of shape {parent_values.shape} with generator {self.value_computer}"
        )
        vals = self.value_computer.generate(
            parent_values.values, random_state=random_state
        )
        self.logger.info(f"Generated {len(vals)} values.")
        assert len(parent_values) == len(
            vals
        ), f"{type(self.value_computer)} returned {len(vals)} values but {len(parent_values)} were expected."
        return vals

    def to_dict(self):
        return {"value_computer": self.value_computer.to_dict()}

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data["value_computer"] = OutcomeGenerator.from_dict(data)
        return ComputedNumericSCMNode(**data)


class ComputedCategoricSCMNode(BaseCategoricSCMNode):

    def __init__(self, name, value_computer, accessibility=..., random_state=...):
        super().__init__(
            name,
            None,
            domain=[],
            noise_distribution=None,
            accessibility=accessibility,
            random_state=random_state,
        )
        self.value_computer = value_computer

    def generate_values(self, parent_values, random_state):
        vals = self.value_computer.generate(
            parent_values.values, random_state=random_state
        )
        assert len(parent_values) == len(
            vals
        ), f"{type(self.value_computer)} returned {len(vals)} values but {len(parent_values)} were expected."
        return vals

    def to_dict(self):
        return {"value_computer": self.value_computer.to_dict()}

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data["value_computer"] = OutcomeGenerator.from_dict(data)
        return ComputedNumericSCMNode(**data)
