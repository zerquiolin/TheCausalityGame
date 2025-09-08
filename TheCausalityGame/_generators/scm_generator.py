# Abstract
# Utils
import logging

# Types
from typing import Any, Callable, Dict, List

# Netorks
import networkx as nx

# Science
import numpy as np
import pandas as pd

# Equations
import sympy as sp

from TheCausalityGame.core.contracts.constants.nodes import ACCESSIBILITY_CONTROLLABLE
from TheCausalityGame.core.contracts.dag import DAG
from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.scm import SCM

# Classes
from TheCausalityGame.scm.nodes.sympy import (
    EquationBasedCategoricalSCMNode,
    EquationBasedNumericalSCMNode,
    SerializableCDF,
)

from .abstract import AbstractSCMGenerator


class EquationBasedSCMGenerator(AbstractSCMGenerator):
    """
    Symbolic SCM Generator using user-specified DAG and constraints.

    Builds symbolic expressions for each node in a DAG using sympy and wraps them into SCM-compatible nodes.
    """

    def __init__(
        self,
        dag: DAG,
        variable_types: Dict[str, str],
        variable_domains: Dict[str, Any],
        user_constraints: Dict[str, Any],
        allowed_operations: List[Any],  # TODO: Currently unused.
        allowed_functions: List[Callable[[sp.Expr], sp.Expr]],
        noise_distributions: List[NoiseDistribution],
        random_state: np.random.RandomState = np.random.RandomState(911),
        num_samples_for_cdf_generation: int = 1000,
        logger: logging.Logger = None,
    ):
        """
        Initialize the symbolic SCM generator.

        Args:
            dag (DAG): Directed acyclic graph representing variable dependencies.
            variable_types (Dict[str, str]): Mapping from variable names to "numerical" or "categorical".
            variable_domains (Dict[str, Any]): Domains for each variable (tuple or list).
            user_constraints (Dict[str, Any]): Configs like non-linearity, exponents, etc.
            allowed_operations (List[Any]): Reserved for future symbolic transformations (currently unused).
            allowed_functions (List[Callable]): Functions used in symbolic term generation.
            noise_distributions (List[BaseNoiseDistribution]): Set of noise models to sample from.
            random_state (np.random.RandomState): RNG for reproducibility.
            num_samples_for_cdf_generation (int): Sample size for empirical CDF estimation (categorical only).
            logger (logging.Logger): Optional logger instance.
        """
        self.dag = dag
        self.variable_types = variable_types
        self.variable_domains = variable_domains
        self.user_constraints = user_constraints
        self.allowed_operations = allowed_operations
        self.allowed_functions = allowed_functions
        self.noise_distributions = noise_distributions
        self.num_samples_for_cdf_generation = num_samples_for_cdf_generation
        self.random_state = random_state
        self.logger = logger or logging

    def generate(self) -> SCM:
        """
        Construct a full SCM from the DAG using symbolic equation generation.

        Returns:
            SCM: A complete SCM instance with symbolic equations and empirical CDFs.
        """
        self.logger.info("Generating new SCM.")
        topological_order = list(nx.topological_sort(self.dag.graph))
        self.logger.debug(f"Topological order: {topological_order}")

        # Sample holder to simulate parent values for equation-based nodes
        samples = pd.DataFrame(index=range(self.num_samples_for_cdf_generation))
        nodes = []

        for node_name in topological_order:
            node_type = self.variable_types[node_name]
            parents = list(self.dag.graph.predecessors(node_name))

            # Encode categorical parents into numeric domain values
            parent_mappings = {
                p: {cat: idx for idx, cat in enumerate(self.variable_domains[p])}
                for p in parents
                if self.variable_types[p] == "categorical"
            }

            # Dispatch to correct generator based on variable type
            generator = {
                "numerical": self._generate_numerical_node,
                "categorical": self._generate_categorical_node,
            }.get(node_type)

            if not generator:
                raise ValueError(f"Unsupported node type: {node_type}")

            self.logger.debug(f"Generating node: {node_name}, type: {node_type}")
            node = generator(node_name, parents, parent_mappings, samples)
            nodes.append(node)

            # Simulate values for this node based on current partial SCM
            samples[node_name] = node.generate_values(
                parent_values=samples[parents],
                random_state=self.random_state,
            )

        return SCM(self.dag, nodes, self.random_state)

    def _generate_equation(self, input_vars: List[sp.Symbol]) -> sp.Expr:
        """
        Build a symbolic expression from input variables using allowed functions and coefficients.

        Args:
            input_vars (List[sp.Symbol]): List of symbolic parent variables.

        Returns:
            sp.Expr: A symbolic expression combining all input variables.
        """
        expr = sp.S.Zero
        nonlinear = self.user_constraints.get("allow_non_linearity", True)
        variable_exp = self.user_constraints.get("allow_variable_exponents", False)

        for var in input_vars:
            coeff = round(self.random_state.uniform(-5, 5), 8)

            if nonlinear:
                func = self.random_state.choice(self.allowed_functions)

                # Handle log/sqrt via absolute value to avoid domain errors
                if func in [sp.log, sp.sqrt]:
                    term = coeff * func(1 + sp.Abs(var))
                elif func == sp.exp:
                    exponent = (
                        self.random_state.uniform(0.5, 5)
                        if not variable_exp or self.random_state.random() < 0.7
                        else 1 + sp.Abs(var)
                    )
                    term = coeff * func(exponent)
                else:
                    term = coeff * func(var)
            else:
                term = coeff * var

            expr += term

        return expr

    def _generate_numerical_node(
        self,
        node_name: str,
        parents: List[str],
        parent_mappings: Dict[str, Any],
        samples: pd.DataFrame,
    ) -> EquationBasedNumericalSCMNode:
        """
        Construct a numerical SCM node using a symbolic expression and noise.

        Args:
            node_name (str): Name of the variable.
            parents (List[str]): Names of parent nodes.
            parent_mappings (Dict[str, Any]): Mapping of categorical parent values.
            samples (pd.DataFrame): Sample table for parent value simulation.

        Returns:
            EquationBasedNumericalSCMNode: Generated SCM node for numeric variable.
        """
        noise = self.random_state.choice(self.noise_distributions)

        if not parents:  # Root node
            return EquationBasedNumericalSCMNode(
                name=node_name,
                accessibility=ACCESSIBILITY_CONTROLLABLE,
                evaluation=None,
                domain=self.variable_domains[node_name],
                noise_distribution=noise,
                parents=None,
                parent_mappings=None,
                random_state=self.random_state,
            )

        # Construct symbolic expression from parent variables
        input_symbols = [sp.Symbol(p) for p in parents]
        equation = self._generate_equation(input_symbols)

        return EquationBasedNumericalSCMNode(
            name=node_name,
            accessibility=ACCESSIBILITY_CONTROLLABLE,
            evaluation=equation,
            domain=self.variable_domains[node_name],
            noise_distribution=noise,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=self.random_state,
        )

    def _generate_categorical_node(
        self,
        node_name: str,
        parents: List[str],
        parent_mappings: Dict[str, Any],
        samples: pd.DataFrame,
    ) -> EquationBasedCategoricalSCMNode:
        """
        Construct a categorical SCM node using symbolic expressions per category and empirical CDFs.

        Args:
            node_name (str): Name of the variable.
            parents (List[str]): Names of parent nodes.
            parent_mappings (Dict[str, Any]): Mapping of categorical parent values.
            samples (pd.DataFrame): Sample table for parent value simulation.

        Returns:
            EquationBasedCategoricalSCMNode: Generated SCM node for categorical variable.
        """
        noise = self.random_state.choice(self.noise_distributions)

        if not parents:  # Root node
            return EquationBasedCategoricalSCMNode(
                name=node_name,
                evaluation=None,
                domain=self.variable_domains[node_name],
                noise_distribution=noise,
                cdfs=None,
                parents=None,
                parent_mappings=None,
                random_state=self.random_state,
            )

        equations = {}
        cdfs = {}
        input_symbols = [sp.Symbol(p) for p in parents]
        parent_data = samples[parents]

        for category in self.variable_domains[node_name]:
            # Create a symbolic expression for this class
            equation = self._generate_equation(input_symbols)
            equations[category] = equation

            # Evaluate this symbolic function over sample data to build a CDF
            func = sp.lambdify(parents, equation, modules="numpy")
            values = func(*parent_data.values.T)
            cdfs[category] = SerializableCDF(np.sort(values))

        return EquationBasedCategoricalSCMNode(
            name=node_name,
            accessibility=ACCESSIBILITY_CONTROLLABLE,
            evaluation=equations,
            domain=self.variable_domains[node_name],
            noise_distribution=noise,
            cdfs=cdfs,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=self.random_state,
        )
