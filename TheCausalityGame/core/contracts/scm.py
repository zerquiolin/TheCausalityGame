# Math
import logging

# Typing
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
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infra.registry import build_from_spec, get_class_path
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

    Attributes
    ----------
        dag (DAG): The underlying directed acyclic graph defining variable dependencies.
        nodes (List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode]): List of nodes in topological order.
        random_state (np.random.RandomState): Random number generator for reproducibility.
    """

    def __init__(
        self,
        dag: DAG,
        nodes: list[SCMNode],
        random_state: np.random.RandomState | None,
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

        Returns
        -------
            List[str]: A list of leaf variable names.
        """
        return [n for n in self.vars if self.dag.graph.out_degree(n) == 0]

    def get_random_state(self) -> np.random.RandomState:
        """
        Returns the SCM's random number generator.

        Returns
        -------
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
        random_state: dict[str, np.random.RandomState] | None = None,
    ) -> list[dict[str, float]]:
        """
        Generates multiple samples from the SCM.

        Args:
            interventions (Dict[str, float], optional): Interventions to apply to nodes.
            num_samples (int): Number of samples to generate.
            random_state (np.random.RandomState, optional): Optional random generator for reproducibility.

        Returns
        -------
            List[Dict[str, float]]: A list of sample dictionaries.
        """
        random_states = random_state or dict.fromkeys(self.vars, self.random_state)
        if not isinstance(random_states, dict):
            random_states = dict.fromkeys(self.vars, random_states)
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

    def to_spec(self) -> SCMSpec:
        """
        Serializes the SCM to a dictionary format.

        Returns
        -------
            Dict: A dictionary representing the SCM's structure and state.
        """
        # Serialize nodes and their parameters
        nodes_data = [node.to_dict() for node in self.nodes.values()]

        # Serialize the random state
        return SCMSpec(
            class_=get_class_path(self.__class__),
            vars=nodes_data,
            dag=self.dag.to_spec(),
            random_state=random_state_to_json(self.random_state),
            # TODO: Create spec for the DAG.
        )

    @classmethod
    def from_spec(cls, spec: SCMSpec) -> "SCM":
        """
        Deserializes an SCM instance from a dictionary.

        Args:
            data (Dict): Dictionary containing SCM structure and state.

        Returns
        -------
            SCM: A new SCM instance.
        """
        dag = build_from_spec(spec.dag)

        # Ensure nodes are sorted in topological order
        topological_order = list(nx.topological_sort(dag.graph))
        nodes = []

        # Create nodes in topological order
        for node_spec in sorted(
            spec.vars, key=lambda n: topological_order.index(n.name)
        ):
            # TODO: This strategy might be faster.
            # 1) Parents: prefer explicit, else read from the DAG
            parents = (
                node_spec.parents
                or list(dag.graph.predecessors(node_spec.name))
                or None
            )

            # 2) Parent mappings: prefer explicit, else build from already-created nodes
            if node_spec.parent_mappings:
                parent_mappings = node_spec.parent_mappings
            else:
                # index maps only for categorical parents (domain is non-empty and all strings)
                cat_index = {
                    n.name: {cat: i for i, cat in enumerate(n.domain)}
                    for n in nodes
                    if getattr(n, "domain", None)
                    and len(n.domain) > 0
                    and all(isinstance(cat, str) for cat in n.domain)
                }
                parent_mappings = {
                    p: cat_index[p] for p in (parents or []) if p in cat_index
                } or None

            # get node object
            nodes.append(
                build_from_spec(
                    spec=node_spec.model_copy(
                        update={"parents": parents, "parent_mappings": parent_mappings}
                    )
                )
            )

        # Reconstruct the random state
        random_state = (
            random_state_from_json(spec.random_state) if spec.random_state else None
        )

        return cls(dag, nodes, random_state)
