"""The Causality Game - Structural Causal Model (SCM) definition."""

from __future__ import annotations

import logging
from typing import override

import networkx as nx
import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.dag import DAG
from TheCausalityGame.core.contracts.scm_node import SCMNode
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
)
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)


class SCM(Serializable):
    """Structural Causal Model for simulating data from a causal system.

    This class defines a system of variables connected by a directed acyclic graph (DAG).
    Each variable is modeled by a node that generates values based on parent nodes and noise.

    Parameters
    ----------
    dag : DAG
        Causal structure represented as a directed acyclic graph.
    nodes : list of SCMNode
        List of nodes (variables) with associated evaluation logic.
    random_state : np.random.RandomState or None
        Random number generator for reproducibility.
    logger : logging.Logger, optional
        Logger for this SCM.
    name : str, optional
        Optional name for the SCM.
    """

    _spec: str = "TheCausalityGame.core.contracts.specs.scm:SCMSpec"

    def __init__(
        self,
        dag: DAG,
        nodes: list[SCMNode],
        random_state: np.random.RandomState | None = None,
        logger: logging.Logger | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the SCM instance."""
        self.dag = dag
        self.nodes = {node.name: node for node in nodes}
        self._topologically_sorted_var_names: list[str] = list(nx.topological_sort(self.dag.graph))  # type: ignore
        self.random_state = random_state or np.random.RandomState(911)
        self.name = name
        self.logger = logger or logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    @property
    def vars(self) -> list[str]:
        """Return all variable names in topological order."""
        return self._topologically_sorted_var_names

    @property
    def controllable_vars(self) -> list[str]:
        """Return variables that are controllable by the agent."""
        return [
            n for n in self.vars if self.nodes[n].accessibility == NodeAccessibility.CONTROLLABLE
        ]

    @property
    def observable_vars(self) -> list[str]:
        """Return variables that are observable (controllable or directly observable)."""
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [NodeAccessibility.CONTROLLABLE, NodeAccessibility.MEASURABLE]
        ]

    @property
    def latent_vars(self) -> list[str]:
        """Return variables that are latent (not observable or controllable)."""
        return [n for n in self.vars if self.nodes[n].accessibility == NodeAccessibility.LATENT]

    @property
    def outcome_vars(self) -> list[str]:
        """Return observable or controllable variables with no parents (i.e., outcomes)."""
        return [
            n
            for n in self.vars
            if self.nodes[n].accessibility
            in [NodeAccessibility.CONTROLLABLE, NodeAccessibility.MEASURABLE]
            and not self.nodes[n].parents
        ]

    @property
    def leaf_vars(self) -> list[str]:
        """Return variables with no outgoing edges in the DAG."""
        return [n for n in self.vars if self.dag.graph.out_degree(n) == 0]  # type: ignore

    def get_random_state(self) -> np.random.RandomState:
        """Return the current random number generator."""
        return self.random_state

    def prepare_new_random_state_structure(
        self, random_state: np.random.RandomState | None = None
    ) -> dict[str, np.random.RandomState]:
        """Generate a new random state structure for each node.

        Parameters
        ----------
        random_state : np.random.RandomState or None
            Optional base random state. Defaults to internal RNG.

        Returns
        -------
        dict of str to np.random.RandomState
            Random state per node.
        """
        random_state = random_state or self.random_state
        return {
            node_name: node.prepare_new_random_state_structure(random_state)
            for node_name, node in self.nodes.items()
        }

    def generate_samples(
        self,
        interventions: dict[str, float | str] | None = None,
        num_samples: int = 1,
        cancel_noise: bool = False,
        random_state: (dict[str, np.random.RandomState] | np.random.RandomState | None) = None,
    ) -> pd.DataFrame:
        """Generate samples from the SCM given optional interventions.

        Parameters
        ----------
        interventions : dict of str to float, optional
            Intervened values for variables.
        num_samples : int
            Number of samples to generate.
        cancel_noise : bool, default=False
            Whether to cancel noise during generation.
        random_state : dict or np.random.RandomState, optional
            Custom random state(s) for nodes.

        Returns
        -------
        pd.DataFrame
            Generated samples as a DataFrame.
        """
        # Setup random states
        random_states = random_state or dict.fromkeys(self.vars, self.random_state)

        if not isinstance(random_states, dict):
            random_states = dict.fromkeys(self.vars, random_states)

        sample = pd.DataFrame(index=range(num_samples))
        interventions = interventions if interventions is not None else {}

        for node_name in self.vars:
            node = self.nodes[node_name]
            if node_name in interventions:
                sample_for_col: list[int | float | str] = [interventions[node_name]] * num_samples
            else:
                sample_for_col = node.generate_values(
                    parent_values=sample,
                    random_state=random_states[node_name],
                    cancel_noise=cancel_noise,
                )

            sample = pd.concat([sample, pd.DataFrame({node_name: sample_for_col})], axis=1)

        return sample

    @override
    def to_spec(self) -> SCMSpec:
        """Serialize the SCM into a spec object.

        Returns
        -------
        SCMSpec
            Serialized specification of the SCM.
        """
        return SCMSpec(
            class_=get_class_path(self.__class__),
            vars=[node.to_spec() for node in self.nodes.values()],
            dag=self.dag.to_spec(),
            random_state=random_state_to_json(self.random_state),
        )

    @classmethod
    @override
    def from_spec(cls, spec: SCMSpec) -> SCM:
        """Deserialize an SCM instance from a specification.

        Parameters
        ----------
        spec : SCMSpec
            Specification object for SCM.

        Returns
        -------
        SCM
            Deserialized SCM instance.
        """
        dag: DAG = build_from_spec(spec.dag)
        topological_order: list[str] = list(nx.topological_sort(dag.graph))  # type: ignore
        nodes: list[SCMNode] = []

        for node_spec in sorted(spec.vars, key=lambda n: topological_order.index(n.name)):
            parents = (
                node_spec.parents
                or list(dag.graph.predecessors(node_spec.name))  # type: ignore
                or None
            )

            if node_spec.parent_mappings:
                parent_mappings = node_spec.parent_mappings
            else:
                cat_index: dict[str, dict[float | str, int]] = {
                    n.name: {cat: i for i, cat in enumerate(n.domain)}
                    for n in nodes
                    if getattr(n, "domain", None) and all(isinstance(cat, str) for cat in n.domain)
                }
                parent_mappings = {
                    p: cat_index[p] for p in (parents or []) if p in cat_index
                } or None

            updated_spec = node_spec.model_copy(
                update={"parents": parents, "parent_mappings": parent_mappings}
            )
            nodes.append(build_from_spec(updated_spec))

        # random_state = (
        #     random_state_from_json(spec.random_state) if spec.random_state else None
        # )
        random_state = np.random.RandomState(2345)
        return cls(dag, nodes, random_state)
