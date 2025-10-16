"""The Causality Game - SCM Node contract."""

import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.serializable import Serializable

# Spec
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.constants.nodes import (
    ACCESSIBILITY_CONTROLLABLE,
)
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)

T = TypeVar("T", bound="Serializable")


class SCMNode(Serializable):
    def __init__(
        self,
        name: str,
        evaluation: Callable | None,
        domain: list[float | str],
        noise_distribution: NoiseDistribution,
        accessibility: str = ACCESSIBILITY_CONTROLLABLE,
        parents: list[str] | None = None,
        parent_mappings: dict[str, int | float] | None = None,
        random_state: np.random.RandomState = np.random.RandomState(911),
        logger: logging.Logger = None,
    ):
        """
        SCMNode is class representing a node in a Structural Causal Model (SCM).
        It encapsulates the node's name, evaluation function, domain of possible values,
        parent nodes, and a random state for generating random values.

        Args:
            name (str): The name of the node.
            accessibility (str): accessibility of this variable by the agent (latent, observable, or controllable)
            evaluation (Callable): A function to evaluate the node's value based on its parents.
            domain (List[float | str]): The domain of possible values for the node.
            parents (List[str]): A list of parent node names.
            random_state (np.random.RandomState): Random state for generating random values.
        """
        self.name = name
        self.accessibility = accessibility
        self.evaluation = evaluation
        self.domain = domain
        if not isinstance(domain, list):
            self.domain = list(self.domain)
        self.noise_distribution = noise_distribution
        self.parents = parents
        self.parent_mappings = parent_mappings
        self.random_state = random_state
        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

        # this is just to not break the MRO
        super().__init__()

    def _init_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    def prepare_new_random_state_structure(self, random_state):
        """
            Generates a random structure that is required by this node. By default, this is just a simple RandomState.
            However, if need be and keeping in mind reproducibility, it can be useful to generate several such objects
            so that several random things can be determined for multiple sampled instances in parallel, e.g., noise and category or so.

        Args:
            random_state (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        return np.random.RandomState(random_state.randint(0, 10**5))

    @abstractmethod
    def generate_values(
        self, parent_values: pd.DataFrame, random_state: np.random.RandomState
    ) -> float | str:
        """
        Generates a value for the node based on its parents and noise.

        Args:
            parent_values (dict): A dictionary of parent node values.
            random_state (np.random.RandomState): Random state for generating random values.

        Returns
        -------
            float | str: The generated value for the node.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _to_dict(self) -> dict:
        return {}

    def to_spec(self) -> SCMNodeSpec:
        d = {
            "class_": get_class_path(self.__class__),
            "name": self.name,
            "accessibility": self.accessibility,
            "domain": self.domain,
            "parents": self.parents,
            "parent_mappings": self.parent_mappings if self.parent_mappings else None,
            "noise_distribution": self.noise_distribution.to_dict(),
            "random_state": (
                random_state_to_json(self.random_state) if self.random_state else None
            ),
        }
        d.update(self._to_dict())
        assert "class" in d or "class_" in d, f"Serialized node has no class entry: {d}"

        node = SCMNodeSpec(**d)
        return node


class NumericSCMNode(SCMNode):
    pass


class CategoricSCMNode(SCMNode):
    pass
