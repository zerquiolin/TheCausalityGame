"""Dataset-derived SCMs using compatible rows and configurable outcome regressors."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from TheCausalityGame.core.contracts.scm_node import (
    CategoricalSCMNode,
    NumericalSCMNode,
)
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.contracts.specs.scm_node import SCMNodeSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
)
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
)


class _DatasetNode:
    """Sampling/training configuration is serializable; fitted model is reconstructed."""

    def __init__(
        self,
        *,
        data: list[dict],
        outcome: bool = False,
        factual_lookup: bool = False,
        estimator: dict | None = None,
        seed: int = 911,
        **kwargs,
    ) -> None:
        super().__init__(noise_distribution=None, evaluation=None, **kwargs)
        self.data = pd.DataFrame(data)
        self.outcome, self.factual_lookup, self.estimator, self.seed = (
            outcome,
            factual_lookup,
            estimator,
            seed,
        )
        if self.data.empty or self.data.isna().any().any():
            raise ValueError(
                "Dataset SCM data must be nonempty and contain no missing values."
            )
        self.model = None
        if outcome:
            from TheCausalityGame.core.infrastructure.registry import load_class

            default = (
                RandomForestClassifier
                if isinstance(self, CategoricalSCMNode)
                else RandomForestRegressor
            )
            if estimator is None:
                model = default(n_estimators=50, random_state=seed)
            else:
                module_name, class_name = estimator["class_"].split(":")
                model_class = (
                    getattr(importlib.import_module(module_name), class_name)
                    if module_name.startswith("sklearn.")
                    else load_class(estimator["class_"])
                )
                model = model_class(**estimator.get("params", {}))
                if not callable(getattr(model, "fit", None)) or not callable(
                    getattr(model, "predict", None)
                ):
                    raise TypeError(
                        "An outcome estimator must provide fit and predict."
                    )
                if (
                    hasattr(model, "get_params")
                    and "random_state" in model.get_params()
                    and model.get_params()["random_state"] is None
                ):
                    model.set_params(random_state=seed)
            features = self.parents or []
            if not features:
                raise ValueError(
                    "An outcome node needs at least one non-outcome feature."
                )
            categorical = [
                p for p in features if not pd.api.types.is_numeric_dtype(self.data[p])
            ]
            transformer = ColumnTransformer(
                [
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical,
                    )
                ],
                remainder="passthrough",
            )
            self.model = make_pipeline(transformer, model).fit(
                self.data[features], self.data[self.name]
            )

    def generate_values(
        self, parent_values: pd.DataFrame, random_state=None, cancel_noise=False
    ):
        rs = self.random_state if random_state is None else random_state
        features = self.parents or []
        if self.outcome:
            values = self.model.predict(parent_values[features])
            if not self.factual_lookup:
                return values
        else:
            values = np.empty(len(parent_values), dtype=object)
        for index in range(len(parent_values)):
            row = parent_values.iloc[index][features].to_dict()
            compatible = np.ones(len(self.data), dtype=bool)
            for name, value in row.items():
                compatible &= self.data[name].to_numpy() == value
            candidates = np.flatnonzero(compatible)
            if not len(candidates):
                if self.outcome:
                    continue
                raise ValueError(f"No compatible rows for {self.name}: {row}")
            chosen = (
                candidates[0]
                if cancel_noise
                else candidates[rs.randint(len(candidates))]
            )
            values[index] = self.data.iloc[chosen][self.name]
        return np.asarray(
            values, dtype=object if isinstance(self, CategoricalSCMNode) else float
        )

    def _to_dict(self) -> dict:
        return {
            "params": {
                "data": self.data.to_dict("records"),
                "outcome": self.outcome,
                "factual_lookup": self.factual_lookup,
                "estimator": self.estimator,
                "seed": self.seed,
            }
        }

    @classmethod
    def from_spec(cls, spec: SCMNodeSpec):
        return cls(
            name=spec.name,
            domain=list(spec.domain),
            parents=spec.parents,
            accessibility=spec.accessibility,
            **(spec.params or {}),
        )


class DatasetNumericalNode(_DatasetNode, NumericalSCMNode):
    """Numerical empirical covariate or learned outcome node."""


class DatasetCategoricalNode(_DatasetNode, CategoricalSCMNode):
    """Categorical empirical covariate or learned outcome node."""


def dataset_scm(
    data: pd.DataFrame,
    *,
    outcomes: list[str],
    controllable: list[str],
    latent: list[str] | None = None,
    categorical: list[str] | None = None,
    estimator: dict | None = None,
    factual_lookup: bool = False,
    seed: int = 911,
) -> SCMSpec:
    """Order non-controls first; sample compatibility excludes controllable predecessors.

    Controls have no causal effects on other non-outcome variables. Outcomes are
    trained on all non-outcomes, including latent columns. Exact compatibility is
    intentional; the factory embeds complete-case training data in its node specs.
    """
    if data.empty or not data.columns.is_unique or data.isna().any().any():
        raise ValueError(
            "Dataset must have unique columns, complete rows and observations."
        )
    if not all(isinstance(name, str) for name in data.columns):
        raise ValueError("Dataset column names must be strings.")
    latent = latent or []
    categorical = categorical or [
        c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])
    ]
    if not outcomes or not set(outcomes + controllable + latent + categorical).issubset(
        data.columns
    ):
        raise ValueError(
            "Dataset roles must name existing columns and include an outcome."
        )
    if set(outcomes) & set(controllable + latent) or set(controllable) & set(latent):
        raise ValueError("Outcome, control and latent roles must be disjoint.")
    data = data.copy()
    for name in categorical:
        source_count = data[name].nunique(dropna=False)
        labels = data[name].map(
            lambda value: value if isinstance(value, str) else str(value)
        )
        if labels.nunique(dropna=False) != source_count:
            raise ValueError(
                f"Categorical column {name} has distinct values that collide as string labels."
            )
        data[name] = labels
    rng = np.random.RandomState(seed)
    noncontrol = [c for c in data.columns if c not in outcomes + controllable]
    controls = list(controllable)
    rng.shuffle(noncontrol)
    rng.shuffle(controls)
    order = noncontrol + controls + list(outcomes)
    nodes, edges = [], []
    for index, name in enumerate(order):
        is_outcome = name in outcomes
        parents = (
            [p for p in order[:index] if p not in outcomes]
            if is_outcome
            else [p for p in order[:index] if p in noncontrol]
        )
        edges.extend((p, name) for p in parents)
        domain = (
            sorted(data[name].unique().tolist(), key=str)
            if name in categorical
            else [float(data[name].min()), float(data[name].max())]
        )
        nodes.append(
            {
                "class_": get_class_path(
                    DatasetCategoricalNode
                    if name in categorical
                    else DatasetNumericalNode
                ),
                "name": name,
                "domain": domain,
                "parents": parents or None,
                "accessibility": (
                    "controllable"
                    if name in controllable
                    else "latent" if name in latent else "measurable"
                ),
                "params": {
                    "data": data[parents + [name]].to_dict("records"),
                    "outcome": is_outcome,
                    "factual_lookup": factual_lookup,
                    "estimator": estimator,
                    "seed": seed,
                },
            }
        )
    spec = SCMSpec.model_validate(
        {
            "class_": "TheCausalityGame.scm.core:CoreSCM",
            "vars": nodes,
            "dag": {
                "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
                "nodes": order,
                "edges": edges,
            },
            "random_state": random_state_to_json(np.random.RandomState(seed)),
        }
    )
    build_from_spec(spec)
    return spec
