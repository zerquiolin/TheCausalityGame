from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import sympy as sp

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.scm_node import (
    CategoricalSCMNode,
    NumericalSCMNode,
)
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_from_json,
)
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


class EquationBasedSCMNode:
    parents: Optional[List[str]]
    evaluation: Any
    random_state: np.random.RandomState
    noise_distribution: NoiseDistribution
    parent_mappings: Optional[Dict[str, Any]]
    symbols_needed_for_evaluation: Union[Dict[str, set[str]], set[str], None]

    def __init__(self) -> None:
        # check that equations and parents coincide and memorize required symbols per node
        def get_symbols_for_formula_while_checking_that_those_are_declared(
            formula: sp.Basic,
        ) -> set[str]:
            symbols = {str(s) for s in formula.free_symbols}
            assert (
                not symbols or self.parents is not None
            ), f"No parents are given (None) even though the formula has symbols: {symbols}"
            undeclared_symbols = symbols.difference(
                set(self.parents) if self.parents else set()
            )
            assert (
                not undeclared_symbols
            ), f"Formula {formula} has undeclared variables {undeclared_symbols} that occur in the formula but not in the parents, which are specified as {self.parents}."
            return symbols

        if isinstance(self.evaluation, dict):
            self.symbols_needed_for_evaluation = {
                eq_name: get_symbols_for_formula_while_checking_that_those_are_declared(
                    eq
                )
                for eq_name, eq in self.evaluation.items()
                if eq is not None
            }
        else:
            self.symbols_needed_for_evaluation = (
                get_symbols_for_formula_while_checking_that_those_are_declared(
                    self.evaluation
                )
                if self.evaluation is not None
                else None
            )


class EquationBasedNumericalSCMNode(NumericalSCMNode, EquationBasedSCMNode):
    def generate_values(
        self,
        parent_values: pd.DataFrame,
        cancel_noise: bool = False,
        random_state: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        # Define random state
        rs = random_state if random_state else self.random_state

        # Check if the node has parents
        if not self.parents:
            # Draw random values uniformly from the domain
            values = rs.uniform(self.domain[0], self.domain[1], size=len(parent_values))
            return values

        # Check if the parent values are provided
        assert set(self.parents).issubset(
            set(parent_values.columns)
        ), "Parent values do not match the expected symbols"

        # Map all the parent values to the parent mappings
        if self.parent_mappings:
            parent_values = parent_values.copy()
            for parent, mapping in self.parent_mappings.items():
                if parent in parent_values.columns:
                    parent_values[parent] = parent_values[parent].map(mapping)

        # Evaluate the expression
        f = sp.lambdify(self.parents, self.evaluation, modules="numpy")
        evaluated = f(*tuple(parent_values[self.parents].values.T))

        assert not np.any(
            np.iscomplex(evaluated)
        ), f"Evaluation of {self.evaluation} lead to complex numbers {evaluated}"

        if cancel_noise:
            return evaluated

        # Add noise to the evaluated value
        noise = self.noise_distribution.generate(size=len(evaluated), random_state=rs)
        return evaluated + noise

    def _to_dict(self) -> Dict[str, Any]:
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        return {
            "equation": str(self.evaluation) if self.evaluation else None,
        }

    @classmethod
    def from_spec(cls, spec: SCMNodeSpec) -> EquationBasedNumericalSCMNode:
        """
        Deserializes the node from a dictionary representation.

        Args:
            spec (SCMNodeSpec): Specification containing node data.

        Returns:
            EquationBasedNumericalSCMNode: An instance of the node.
        """

        # Validations
        evaluation = sp.sympify(spec.equation) if spec.equation else None
        if evaluation is not None:
            assert (
                str(evaluation) == spec.equation
            ), f"Evaluation structure {spec.equation} could not parsed properly. Recovered {str(evaluation)}"

        noise_distribution = (
            build_from_spec(spec.noise_distribution)
            if spec.noise_distribution
            else UniformNoiseDistribution()
        )

        # Return the new class
        cla = cls(
            **{
                "name": spec.name,
                "evaluation": evaluation,
                "domain": spec.domain,
                "noise_distribution": noise_distribution,
                "accessibility": spec.accessibility,
                "parents": spec.parents,
                "parent_mappings": spec.parent_mappings,
                "random_state": (
                    random_state_from_json(spec.random_state)
                    if spec.random_state
                    else np.random.RandomState(911)
                ),
            }
        )

        return cla


class EquationBasedCategoricalSCMNode(CategoricalSCMNode, EquationBasedSCMNode):
    def __init__(
        self,
        name: str,
        evaluation: Optional[Dict[str, sp.Basic]],
        domain: List[Union[float, str]],
        noise_distribution: NoiseDistribution,
        cdfs: Optional[Dict[str, "SerializableCDF"]] = None,
        accessibility: NodeAccessibility = NodeAccessibility.CONTROLLABLE,
        parents: Optional[List[str]] = None,
        parent_mappings: Optional[Dict[str, Union[int, float]]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        # Superclass constructor
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=cast(
                Optional[Callable[[pd.DataFrame], Union[float, str]]], evaluation
            ),
            domain=domain,
            noise_distribution=noise_distribution,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=random_state,
        )
        # Initialize the CDFs
        self.cdfs = cdfs
        # Initialize the noise distribution
        self.domain_noise_distribution = (
            self._noise_to_category_distribution()
            if not domain_distribution
            else domain_distribution
        )

    def prepare_new_random_state_structure(
        self, random_state: np.random.RandomState
    ) -> Dict[str, np.random.RandomState]:
        return {
            "noise": np.random.RandomState(random_state.randint(0, 10**5)),
            "choice": np.random.RandomState(random_state.randint(0, 10**5)),
        }

    def _noise_to_category_distribution(
        self, n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Converts a continuous noise distribution into a discrete probability distribution over given categories.

        Args:
            n_samples (int): Number of samples to draw from the noise distribution.

        Returns:
            Dict[str, float]: A dictionary mapping each category to a probability.
        """
        # Sample from the noise distribution
        samples = self.noise_distribution.generate(size=n_samples)

        # Use quantiles to bin the samples into categories
        quantiles = np.percentile(samples, np.linspace(0, 100, len(self.domain) + 1))

        # Assign samples to bins
        bin_indices = np.digitize(samples, quantiles[1:-1], right=True)

        # Map bin indices to categories
        mapped = [self.domain[i] for i in bin_indices]

        # Count and normalize
        counts = Counter(mapped)
        total = sum(counts.values())
        return {cat: counts.get(cat, 0) / total for cat in self.domain}

    def generate_values(
        self,
        parent_values: pd.DataFrame,
        random_state: Optional[
            Union[np.random.RandomState, Dict[str, np.random.RandomState]]
        ] = None,
        cancel_noise: bool = False,
    ) -> List[Union[str, float]]:

        # Define random state
        if random_state is None:
            rs_noise, rs_choice = (self.random_state, self.random_state)
        elif isinstance(random_state, dict):
            rs_noise, rs_choice = (random_state["noise"], random_state["choice"])
        else:
            rs_noise, rs_choice = (random_state, random_state)

        # now start sampling
        self.logger.info(
            "Drawing %s values for categorical node %s with parents %s",
            len(parent_values),
            self.name,
            self.parents,
        )
        self.logger.debug(f"Parent mapping of {self.name} is %s.", self.parent_mappings)

        # Check if the node has parents
        if not self.parents:
            return rs_noise.choice(
                list(self.domain_noise_distribution.keys()),
                p=list(self.domain_noise_distribution.values()),
                size=len(parent_values),
            ).tolist()

        missing_parents = set(self.parents).difference(set(parent_values.keys()))
        assert (
            not missing_parents
        ), f"Cannot generate value for {self.name} as no values provided for some parents: {missing_parents}"

        if self.evaluation is None:
            msg = f"Cannot generate values for {self.name} because no evaluation was provided."
            raise ValueError(msg)

        if self.cdfs is None:
            msg = (
                f"Cannot generate values for {self.name} because no CDFs were provided."
            )
            raise ValueError(msg)

        if self.symbols_needed_for_evaluation is None:
            msg = (
                f"Cannot generate values for {self.name} because symbol requirements "
                "were not initialized."
            )
            raise ValueError(msg)

        symbol_requirements = cast(
            Dict[str, set[str]], self.symbols_needed_for_evaluation
        )

        # Check that all parent values are provided
        symbols = set()
        for eq_name, eq in self.evaluation.items():
            missing_values = symbol_requirements[eq_name].difference(
                parent_values.keys()
            )
            assert (
                not missing_values
            ), f"Cannot evaluate formula {eq} of variable {self.name} because no values are provided for parent {missing_values}"
            symbols.update(symbol_requirements[eq_name])
        symbols = list(symbols)

        # Evaluate the expression
        possible_categories = list(self.evaluation.keys())
        evaluations = []

        # determine the noise terms once (important to do this here simultaneously for all instances to not confuse the random state)
        noises = self.noise_distribution.generate(
            size=(len(parent_values), len(possible_categories)), random_state=rs_noise
        )

        for i, possible_category in enumerate(possible_categories):

            eq = self.evaluation[possible_category]

            # Evaluate the expression
            f = sp.lambdify(symbols, eq, modules="numpy")
            evaluated = f(*tuple(parent_values[symbols].values.T))

            # Calculate the CDF for the evaluated value and category to obtain values normalized between 0 and 1
            if cancel_noise:
                evaluations.append(self.cdfs[possible_category](evaluated))
            else:
                evaluations.append(
                    self.cdfs[possible_category](evaluated + noises[:, i])
                )

        evaluations = np.array(evaluations).T
        expected_shape = (len(parent_values), len(self.domain))
        assert (
            expected_shape == evaluations.shape
        ), f"Shape of evaluations should be {expected_shape} but was {evaluations.shape}"

        # Normalize the evaluations
        evaluations = np.maximum(
            evaluations, 10**-20
        )  # to avoid that all entries in a row are 0
        evaluations /= evaluations.sum(axis=1)[:, np.newaxis]

        # Check if the evaluations are valid
        assert (
            np.all(0 <= evaluations)
            and np.all(evaluations <= 1)
            and np.all(np.isclose(np.sum(evaluations, axis=1), 1.0))
        ), f"Evaluations are not valid probabilities: {evaluations}"

        # Sample from the categorical distribution
        return [rs_choice.choice(self.domain, p=dist) for dist in evaluations]

    def _to_dict(self) -> Dict[str, Any]:
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        representation = {
            "equation": (
                {cat: str(eq) for cat, eq in self.evaluation.items()}
                if self.evaluation
                else None
            ),
            "cdfs": (
                {cat: self.cdfs[cat].to_list() for cat in self.cdfs}
                if self.cdfs
                else None
            ),
            "domain_distribution": self.domain_noise_distribution,
        }
        return representation

    @classmethod
    def from_spec(cls, spec: SCMNodeSpec) -> EquationBasedCategoricalSCMNode:
        """
        Deserializes the node from a dictionary representation.

        Args:
            spec (SCMNodeSpec): Specification containing node data.

        Returns:
            EquationBasedCategoricalSCMNode: An instance of the node.
        """
        # For categorical nodes, reconstruct the equation dictionary.
        evaluation = (
            {k: sp.sympify(v) for k, v in spec.equation.items()}
            if spec.equation
            else None
        )

        # Reconstruct the CDF mappings from step points.
        cdfs = (
            {
                cat: SerializableCDF.from_list(points)
                for cat, points in spec.cdfs.items()
            }
            if spec.cdfs
            else None
        )

        # Deserialize the noise distribution
        noise_distribution = (
            build_from_spec(spec.noise_distribution)
            if spec.noise_distribution
            else UniformNoiseDistribution()
        )

        # Return the new class
        return cls(
            **{
                "name": spec.name,
                "evaluation": evaluation,
                "domain": spec.domain,
                "noise_distribution": noise_distribution,
                "accessibility": spec.accessibility,
                "parents": spec.parents,
                "parent_mappings": spec.parent_mappings,
                "cdfs": cdfs,
                "random_state": (
                    random_state_from_json(spec.random_state)
                    if spec.random_state
                    else np.random.RandomState(911)
                ),
            }
        )


class SerializableCDF:
    def __init__(self, sorted_samples: np.ndarray) -> None:
        self.sorted_samples = np.array(sorted_samples)
        assert (
            len(self.sorted_samples.shape) == 1
        ), "SerializableCDF needs a one-dimensional vector of values."

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.sorted_samples, x, side="right") / len(
            self.sorted_samples
        )

    def to_list(self) -> List[float]:
        return self.sorted_samples.tolist()

    @classmethod
    def from_list(cls, data: List[float]) -> "SerializableCDF":
        return cls(np.array(data))
