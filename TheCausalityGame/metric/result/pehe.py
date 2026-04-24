"""The Causality Game - PEHE Result Metric."""

from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.mission import ResultMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.errors.metric import (
    NotInitializedError,
    UnsupportedMetricTypeError,
)
from TheCausalityGame.core.lib.errors.scm import (
    NoControllableNodeError,
    NoNumericLeafNodeError,
)


class PEHEResultMetric(ResultMetric):
    """
    Computes the Precision in Estimation of Heterogeneous Effect (PEHE).

    This metric compares the true Conditional Average Treatment Effect (CATE)
    to the one estimated by the agent.

    Attributes
    ----------
    name : str
        Human-readable name for the metric.
    description : str
        Description of the metric's purpose.
    kinds : list of str
        Types of expected answers this metric is compatible with.
    """

    name = "PEHE"
    description = (
        "Computes the Precision in Estimation of Heterogeneous Effect (PEHE) "
        "between the true and estimated Conditional Average Treatment Effect (CATE)."
    )
    kinds = ["Treatment Effect Function"]  # noqa: RUF012

    @override
    def mount(self, scm: SCM) -> None:
        rs = np.random.RandomState(911)

        # Pick an outcome node with numeric domain
        possible_outcomes = [
            var
            for var in scm.leaf_vars
            if isinstance(scm.nodes[var].domain[0], int | float)
        ]
        if not possible_outcomes:
            raise NoNumericLeafNodeError()

        self.te_node = rs.choice(possible_outcomes)
        self.scm = scm

        # Pick a treatment node with controllable access
        possible_treatments = [
            node.name
            for node in scm.nodes.values()
            if node.accessibility == NodeAccessibility.CONTROLLABLE
        ]
        if not possible_treatments:
            raise NoControllableNodeError()

        self.treatment_node = rs.choice(possible_treatments)
        self.input_features = [
            var
            for var in scm.vars
            if var not in (self.te_node, self.treatment_node)
            and scm.nodes[var].accessibility
            in {NodeAccessibility.CONTROLLABLE, NodeAccessibility.MEASURABLE}
        ]

        # Generate a random covariate configuration (excluding treatment/outcome)
        conditional_samples = scm.generate_samples(
            num_samples=100,
            cancel_noise=True,
            random_state=rs,
        ).drop(columns=[self.te_node, self.treatment_node])

        covariates = conditional_samples.to_dict(orient="records")[rs.choice(100)]  # type: ignore

        # Generate interventional data for both treatment states
        domain_values = scm.nodes[self.treatment_node].domain
        non_treated_samples, treated_samples = [
            scm.generate_samples(
                interventions={
                    self.treatment_node: val,
                    **{str(k): v for k, v in covariates.items()},
                },
                num_samples=100,
                cancel_noise=True,
                random_state=rs,
            )
            for val in domain_values
        ]

        # Ground truth CATE: average effect on outcome node
        self.true_cate: float = (
            treated_samples[self.te_node] - non_treated_samples[self.te_node]
        )[0]
        self.treatment_values = tuple(domain_values)
        self.treatment_samples = (non_treated_samples, treated_samples)
        self.is_mounted = True

    @override
    def evaluate(self, kind: str, result: Any) -> float:
        """
        Evaluate agent's estimated treatment effect against true CATE using PEHE.

        Parameters
        ----------
        kind : str
            The kind of result provided (must match supported kind).
        result : Any
            The callable object returned by the agent that estimates treatment effects.

        Returns
        -------
        float
            The PEHE score between true and estimated effects.

        Raises
        ------
        RuntimeError
            If `mount()` was not called before evaluating.
        ValueError
            If the provided kind is unsupported or if result is not callable.
        """
        if not self.is_mounted:
            raise NotInitializedError(self.name)

        if kind != "Treatment Effect Function":
            raise UnsupportedMetricTypeError(kind)

        # Prepare the input features (exclude treatment/outcome)
        input_features = list(self.input_features)

        # Prepare covariate test samples
        non_treated = self.treatment_samples[0].drop(columns=[self.te_node])
        treated = self.treatment_samples[1].drop(columns=[self.te_node])

        # Get agent's estimated treatment effect
        estimated_df: pd.DataFrame = result(
            X=input_features,
            treatment=self.treatment_node,
            outcome=self.te_node,
            covariate_values=(non_treated, treated),
        )

        estimated = estimated_df["treatment_effect"].to_numpy()

        # Compute PEHE: sqrt(mean squared error)
        difference = self.true_cate - estimated
        pehe = np.sqrt(np.mean(difference**2))

        return pehe

    @override
    def context_metadata(self) -> dict[str, Any]:
        if not self.is_mounted:
            return {}
        return {
            "query_family": "treatment_effect",
            "estimand_kind": "cate",
            "treatment": self.treatment_node,
            "outcome": self.te_node,
            "covariates": list(self.input_features),
            "treatment_values": list(self.treatment_values),
        }

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> "PEHEResultMetric":
        return cls()
