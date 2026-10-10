"""Generate hybrid acyclic SCMs; identifiability requires mission-specific assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode


class BoundedEquationSCMNode(EquationBasedNumericalSCMNode):
    """Clip a generated structural assignment after noise to its declared interval."""

    def generate_values(self, parent_values, cancel_noise=False, random_state=None):
        values = super().generate_values(parent_values, cancel_noise, random_state)
        return np.clip(values, self.domain[0], self.domain[1])


@dataclass(frozen=True)
class SyntheticSCMGenerator:
    """Bound in-degree/depth and use a domain-safe symbolic operator grammar.

    Domains bound interventions and generated outputs (numeric values are clipped).
    Acyclicity does not establish observational or interventional identifiability.
    """

    n_nodes: int = 8
    edge_probability: float = 0.3
    max_indegree: int = 3
    max_depth: int = 4
    categorical_fraction: float = 0.25
    n_categories: int = 3
    latent_fraction: float = 0.1
    controllable_fraction: float = 0.5
    domain: tuple[float, float] = (-2.0, 2.0)
    operators: tuple[str, ...] = ("linear", "sin", "cos", "exp", "log", "polynomial")
    polynomial_degree: int = 2
    noise: str = "uniform"
    seed: int = 911

    def generate(self) -> SCMSpec:
        """Return a complete current SCMSpec, including isolated nodes and RNG state."""
        fractions = (
            self.edge_probability,
            self.categorical_fraction,
            self.latent_fraction,
            self.controllable_fraction,
        )
        if (
            any(not 0 <= p <= 1 for p in fractions)
            or self.latent_fraction + self.controllable_fraction > 1
        ):
            raise ValueError("Invalid topology/type/access proportions.")
        if (
            self.n_nodes < 1
            or self.max_indegree < 0
            or self.max_depth < 0
            or self.n_categories < 1
        ):
            raise ValueError("Invalid node, degree, depth or category count.")
        if (
            not 1 <= self.polynomial_degree <= 4
            or not self.operators
            or set(self.operators)
            - {"linear", "sin", "cos", "exp", "log", "polynomial"}
        ):
            raise ValueError("Unsupported operator grammar/degree.")
        low, high = self.domain
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(
                "Synthetic numerical domain must be a finite nonempty interval."
            )
        noises = {
            "uniform": ("uniform:UniformNoiseDistribution", {"low": -0.1, "high": 0.1}),
            "gaussian": ("gaussian:GaussianNoiseDistribution", {"std": 0.1}),
            "laplace": ("laplace:LaplaceNoiseDistribution", {"scale": 0.1}),
            "dirac": ("dirac:DiracNoiseDistribution", {}),
        }
        if self.noise not in noises:
            raise ValueError("Unsupported numerical noise family.")
        rng = np.random.RandomState(self.seed)
        names = [f"X{i}" for i in rng.permutation(self.n_nodes)]
        depths, edges, specs = {}, [], []
        categorical = {
            name for name in names if rng.uniform() < self.categorical_fraction
        }
        domains = {
            name: (
                [f"c{j}" for j in range(self.n_categories)]
                if name in categorical
                else [low, high]
            )
            for name in names
        }
        for index, name in enumerate(names):
            candidates = [
                p
                for p in names[:index]
                if depths[p] < self.max_depth and rng.uniform() < self.edge_probability
            ]
            rng.shuffle(candidates)
            parents = candidates[: self.max_indegree]
            depths[name] = 0 if not parents else max(depths[p] for p in parents) + 1
            edges.extend((p, name) for p in parents)
            access_draw = rng.uniform()
            access = (
                "latent"
                if access_draw < self.latent_fraction
                else (
                    "controllable"
                    if access_draw < self.latent_fraction + self.controllable_fraction
                    else "measurable"
                )
            )
            mappings = {
                p: {category: j for j, category in enumerate(domains[p])}
                for p in parents
                if p in categorical
            }
            raw = sum(
                (sp.Float(rng.uniform(-1, 1)) * sp.Symbol(p) for p in parents),
                sp.Float(rng.uniform(-0.5, 0.5)),
            )
            operator = rng.choice(self.operators)
            transformed = {
                "linear": raw,
                "sin": sp.sin(raw),
                "cos": sp.cos(raw),
                "exp": sp.exp(sp.tanh(raw)),
                "log": sp.log(1 + raw**2),
                "polynomial": sp.tanh(raw) ** self.polynomial_degree,
            }[operator]
            suffix, params = noises[self.noise]
            spec = {
                "name": name,
                "domain": domains[name],
                "accessibility": access,
                "parents": parents or None,
                "parent_mappings": mappings or None,
                "noise_distribution": {
                    "class_": "TheCausalityGame.scm.noise." + suffix,
                    "params": params,
                },
            }
            if name in categorical:
                spec["class_"] = (
                    "TheCausalityGame.scm.nodes.sympy:EquationBasedCategoricalSCMNode"
                )
                spec["domain_distribution"] = {
                    c: 1 / self.n_categories for c in domains[name]
                }
                if parents:
                    spec["equation"] = {
                        c: str(
                            sp.Float(rng.uniform(-1, 1)) * transformed
                            + j / self.n_categories
                        )
                        for j, c in enumerate(domains[name])
                    }
                    spec["cdfs"] = {
                        c: np.linspace(-10, 10, 201).tolist() for c in domains[name]
                    }
            else:
                spec["class_"] = (
                    "TheCausalityGame.generators.synthetic:BoundedEquationSCMNode"
                )
                if parents:
                    spec["equation"] = str(
                        sp.Float((low + high) / 2)
                        + sp.Float((high - low) / 2) * sp.tanh(transformed)
                    )
            specs.append(spec)
        result = SCMSpec.model_validate(
            {
                "class_": "TheCausalityGame.scm.core:CoreSCM",
                "vars": specs,
                "dag": {
                    "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
                    "nodes": names,
                    "edges": edges,
                },
                "random_state": random_state_to_json(np.random.RandomState(self.seed)),
            }
        )
        build_from_spec(result)
        return result
