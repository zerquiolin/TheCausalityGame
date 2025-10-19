"""The Causality Game - SCM Node contract."""

import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)

T = TypeVar("T", bound="Serializable")


class SCMNode(Serializable):
    """
    Base class for a Structural Causal Model (SCM) node.

    Each node encapsulates its name, domain, parent nodes, evaluation logic, noise,
    and configuration options such as accessibility.

    Parameters
    ----------
    name : str
        Node identifier.
    evaluation : Callable | None
        Evaluation function used to compute values (if applicable).
    domain : list[float | str]
        Possible values the node can take.
    noise_distribution : NoiseDistribution
        Source of noise for the node.
    accessibility : str
        Level of agent access (e.g., controllable, observable, latent).
    parents : list[str] | None
        Names of parent nodes in the SCM graph.
    parent_mappings : dict[str, int | float] | None
        Optional transformation or mapping of parent values.
    random_state : np.random.RandomState
        Random state used for reproducibility.
    logger : logging.Logger | None
        Optional logger instance.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        evaluation: Callable[[pd.DataFrame], float | str] | None,
        domain: list[float | str],
        noise_distribution: NoiseDistribution,
        accessibility: NodeAccessibility = NodeAccessibility.CONTROLLABLE,
        parents: list[str] | None = None,
        parent_mappings: dict[str, int | float] | None = None,
        random_state: np.random.RandomState | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the SCM node."""
        self.name = name
        self.accessibility = accessibility
        self.evaluation = evaluation
        self.domain = domain
        self.noise_distribution = noise_distribution
        self.parents = parents
        self.parent_mappings = parent_mappings
        self.random_state = random_state or np.random.RandomState(911)
        self.logger = logger or logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}"
        )
        super().__init__()

    # def _init_random_state(self):
    #     """Ensure the node has a valid random state."""
    #     if self.random_state is None:
    #         self.random_state = np.random.RandomState()

    def prepare_new_random_state_structure(
        self, random_state: np.random.RandomState
    ) -> np.random.RandomState:
        """
        Prepare a new random state structure for multi-sample generation.

        Parameters
        ----------
        random_state : np.random.RandomState
            Source random generator.

        Returns
        -------
        np.random.RandomState
            A new reproducible random state.
        """
        return np.random.RandomState(random_state.randint(0, 10**5))

    @abstractmethod
    def generate_values(
        self,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState,
        cancel_noise: bool = False,
    ) -> list[int | float | str]:
        """
        Generate the node's value using its parents and internal noise.

        Parameters
        ----------
        parent_values : pd.DataFrame
            Values of the parent nodes.
        random_state : np.random.RandomState
            Source of randomness.

        Returns
        -------
        list of int | float | str
            Generated values for the node.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _to_dict(self) -> dict[str, Any]:
        """Subclass hook to add custom fields to the spec."""
        return {}

    def to_spec(self) -> SCMNodeSpec:
        """
        Convert the node into a serializable specification.

        Returns
        -------
        SCMNodeSpec
            The structured specification of the SCM node.
        """
        d = {
            "class_": get_class_path(self.__class__),
            "name": self.name,
            "accessibility": self.accessibility,
            "domain": self.domain,
            "parents": self.parents if self.parents else None,
            "parent_mappings": self.parent_mappings or None,
            "noise_distribution": self.noise_distribution.to_dict(),
            "random_state": (
                random_state_to_json(self.random_state) if self.random_state else None
            ),
        }
        d.update(self._to_dict())
        assert "class" in d or "class_" in d, f"Serialized node has no class entry: {d}"
        return SCMNodeSpec(**d)  # type: ignore


class NumericSCMNode(SCMNode):
    """Marker class for numeric SCM nodes."""

    pass


class CategoricSCMNode(SCMNode):
    """Marker class for categorical SCM nodes."""

    pass
