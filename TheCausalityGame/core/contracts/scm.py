# Math
import logging

# Typing
from typing import Optional

# Graph
import networkx as nx
import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.constants.nodes import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_LATENT,
    ACCESSIBILITY_OBSERVABLE,
)
from TheCausalityGame.core.contracts.dag import DAG

# Nodes
from TheCausalityGame.core.contracts.scm_node import (
    SCMNode,
)
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.utils.imports import get_class
from TheCausalityGame.core.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)


class SCM(Serializable):
    """
    Structural Causal Model (SCM) that represents a system of variables with causal dependencies.

    It generates data samples according to a DAG and a collection of SCMNodes, where each node defines
    a data generation function. The nodes are evaluated in topological order of the DAG, respecting
    causal dependencies.

    Attributes:
        dag (DAG): The underlying directed acyclic graph defining variable dependencies.
        nodes (List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode]): List of nodes in topological order.
        random_state (np.random.RandomState): Random number generator for reproducibility.
    """

    def __init__(
        self,
        dag: DAG,
        nodes: list[SCMNode],
        random_state: Optional[np.random.RandomState],
        logger: logging.Logger = None,
        name=None,
    ):
        """
        Initializes the SCM with a DAG, a list of nodes, and a random number generator.

        Args:
            dag (DAG): The DAG representing the causal structure.
            nodes (List[SCMNode]): List of SCMNode instances in topological order.
            random_state (np.random.RandomState): NumPy random number generator.
            name (str): Name of the SCM
        """
        self.dag = dag
        self.nodes = {node.name: node for node in nodes}
        self._topologically_sorted_var_names = list(nx.topological_sort(self.dag.graph))
        self.random_state = random_state if random_state else np.random.RandomState(911)
        self.name = name
        self.logger = (
            logger
            if logger is not None
            else logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        )

    @property
    def vars(self):
        return self._topologically_sorted_var_names

    @property
    def controllable_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility == ACCESSIBILITY_CONTROLLABLE
        ]

    @property
    def observable_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [ACCESSIBILITY_CONTROLLABLE, ACCESSIBILITY_OBSERVABLE]
        ]

    @property
    def latent_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility in [ACCESSIBILITY_LATENT]
        ]

    @property
    def outcome_vars(self):
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [ACCESSIBILITY_CONTROLLABLE, ACCESSIBILITY_OBSERVABLE]
            and not self.nodes[n].parents
        ]

    @property
    def leaf_vars(self):
        """
        Returns a list of leaf variables in the SCM, which are nodes with no outgoing edges.

        Returns:
            List[str]: A list of leaf variable names.
        """
        return [n for n in self.vars if self.dag.graph.out_degree(n) == 0]

    def get_random_state(self) -> np.random.RandomState:
        """
        Returns the SCM's random number generator.

        Returns:
            np.random.RandomState: The random generator used for sampling.
        """
        return self.random_state

    def prepare_new_random_state_structure(self, random_state=None):

        # root
        random_state = random_state or self.random_state

        # ask each node for a structure of new random states
        random_structure = {}
        for node_name, node in self.nodes.items():
            random_structure[node_name] = node.prepare_new_random_state_structure(
                random_state
            )
        return random_structure

    def generate_samples(
        self,
        interventions: dict[str, float] = {},
        num_samples: int = 1,
        cancel_noise: bool = False,
        random_state: Optional[dict[str, np.random.RandomState]] = None,
    ) -> list[dict[str, float]]:
        """
        Generates multiple samples from the SCM.

        Args:
            interventions (Dict[str, float], optional): Interventions to apply to nodes.
            num_samples (int): Number of samples to generate.
            random_state (np.random.RandomState, optional): Optional random generator for reproducibility.

        Returns:
            List[Dict[str, float]]: A list of sample dictionaries.
        """
        random_states = random_state or {v: self.random_state for v in self.vars}
        if not isinstance(random_states, dict):
            random_states = {v: random_states for v in self.vars}
        sample = pd.DataFrame(index=range(num_samples))

        for node_name, node in [
            (node_name, self.nodes[node_name]) for node_name in self.vars
        ]:
            if node_name in interventions:
                sample_for_col = [interventions[node_name]] * num_samples
            else:
                sample_for_col = node.generate_values(
                    parent_values=sample,
                    random_state=random_states[node_name],
                    cancel_noise=cancel_noise,
                )

            sample = pd.concat(
                [sample, pd.DataFrame({node_name: sample_for_col})], axis=1
            )
        assert (
            type(sample) is pd.DataFrame
        ), f"sample should be a dataframe but is {type(sample)}"
        return sample

    # TODO: Fix serializaton to a spec class
    def to_dict(self) -> dict:
        """
        Serializes the SCM to a dictionary format.

        Returns:
            Dict: A dictionary representing the SCM's structure and state.
        """
        # Serialize nodes and their parameters
        nodes_data = [node.to_dict() for node in self.nodes.values()]

        # Serialize the random state
        return {
            "vars": nodes_data,
            "edges": self.dag.to_dict()["edges"],
            "random_state": random_state_to_json(self.random_state),
        }

    # TODO: Fix deserialization to the correct class
    @classmethod
    def from_dict(cls, data: dict) -> "SCM":
        """
        Deserializes an SCM instance from a dictionary.

        Args:
            data (Dict): Dictionary containing SCM structure and state.

        Returns:
            SCM: A new SCM instance.
        """
        if "class" in data and data["class"] not in [__class__.__name__]:
            class_name = data.pop("class")
            return get_class(class_name).from_dict(data)

        # Reconstruct the DAG from the dictionary
        nodes = [v["name"] for v in data["vars"]]
        edges = data["edges"]
        dag = DAG.from_dict(
            {"nodes": nodes, "edges": edges}
        )  # TODO: This is now from_spec method

        # Ensure nodes are sorted in topological order
        topological_order = list(nx.topological_sort(dag.graph))
        nodes = []

        # Create nodes in topological order
        for node_as_dict in sorted(
            data["vars"], key=lambda n: topological_order.index(n["name"])
        ):
            # extract parents from edges if they are not explicitly given
            if not "parents" in node_as_dict:
                node_as_dict["parents"] = [
                    e[0] for e in edges if e[1] == node_as_dict["name"]
                ]

            # Generate parent mappings if not provided
            if (
                not "parent_mappings" in node_as_dict
                or not node_as_dict["parent_mappings"]
            ):
                node_as_dict["parent_mappings"] = {
                    node.name: {cat: idx for idx, cat in enumerate(node.domain)}
                    for node in nodes
                    if node.name in (node_as_dict["parents"] or [])
                    and isinstance(node.domain[0], str)
                }

            # get node object
            nodes.append(SCMNode.from_dict(node_as_dict))

        # Reconstruct the random state
        random_state = (
            random_state_from_json(data["random_state"])
            if "random_state" in data
            else None
        )
        return cls(dag, nodes, random_state)
