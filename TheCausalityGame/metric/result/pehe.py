# Science
import numpy as np

from TheCausalityGame.core.contracts.mission import ResultMetric

# Types
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path

# Constants
from TheCausalityGame.core.lib.constants.nodes import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_OBSERVABLE,
)


class PEHEResultMetric(ResultMetric):
    """Computes the PEHE metric.

    PEHE = sqrt(mean((true_effects - predicted_effects)^2)).

    Attributes:
        name (str): Human-readable name for the metric.
        output_processor (BaseOutputProcessor): Output processor class for this metric.

    """

    # Attributes
    name = "PEHE"
    description = (
        "Computes the Precision in Estimation of Heterogeneous Effect (PEHE) "
        "between the true and estimated Conditional Average Treatment Effect (CATE)."
    )
    kinds = ["Treatment Effect Function"]

    def mount(self, scm: SCM) -> None:
        # Define a random state for reproducibility
        rs = np.random.RandomState(911)
        # Select the predictive node
        possible_outcomes = [
            var for var in scm.leaf_vars if type(scm.nodes[var].domain[0]) is not str
        ]
        assert len(possible_outcomes) > 0, "No measurable nodes found for TE"
        te_node = rs.choice(possible_outcomes)
        # Select the treatment node
        possible_treatments = [
            node.name
            for node in scm.nodes.values()
            if node.accessibility == ACCESSIBILITY_CONTROLLABLE
        ]
        assert len(possible_treatments) > 0, "No controllable nodes found for TE"
        treatment_node = rs.choice(possible_treatments)
        # Generate values for the covariant nodes
        conditional_samples = scm.generate_samples(
            num_samples=100,
            cancel_noise=True,
            random_state=rs,
        )
        # Drop columns that are either the treatment node or the outcome node
        conditional_samples = conditional_samples.drop(
            columns=[te_node, treatment_node]
        )
        # Convert the result to a Dict
        conditional_samples = conditional_samples.to_dict(orient="records")[
            rs.choice(range(100))
        ]
        # Generate samples for each treatment value
        non_treated_samples, treated_samples = [
            scm.generate_samples(
                interventions={treatment_node: value, **conditional_samples},
                num_samples=100,
                cancel_noise=True,
                random_state=rs,
            )
            for value in scm.nodes[treatment_node].domain
        ]

        # Extract the outcome variable Y
        true_cate = treated_samples[te_node] - non_treated_samples[te_node]
        # Save the treatment node
        self.treatment_node = treatment_node
        # Save the treatment effect node
        self.te_node = te_node
        # Save the treatment samples
        self.treatment_samples = (non_treated_samples, treated_samples)
        # Save the true CATE
        self.true_cate = true_cate[0]
        # Update the is_mounted flag
        self.is_mounted = True
        # Save the scm
        self.scm = scm

    def evaluate(self, kind: str, result: any) -> float:
        # Check if the mission is mounted
        if not self.is_mounted:
            raise RuntimeError("Mission is not mounted. Call mount() first.")

        # Check kind
        if kind != "Treatment Effect Function":
            raise ValueError(f"Unsupported kind: {kind}")

        # Extract the agent's answer
        answer = result

        # Compute the estimated agent's CATE
        estimated = answer(
            X=[
                var
                for var in self.scm.vars
                if var != self.te_node
                and var != self.treatment_node
                and (
                    self.scm.nodes[var].accessibility == ACCESSIBILITY_CONTROLLABLE
                    or self.scm.nodes[var].accessibility == ACCESSIBILITY_OBSERVABLE
                )
            ],
            outcome=self.te_node,  # TODO: Change to outcome
            treatment=self.treatment_node,  # TODO: Change to treatment
            covariate_values=(
                self.treatment_samples[0].drop(columns=[self.te_node]),
                self.treatment_samples[1].drop(columns=[self.te_node]),
            ),
        )

        # Compute the difference
        difference = self.true_cate - estimated["treatment_effect"]

        # Compute the PEHE
        pehe = np.sqrt(np.mean(difference**2))
        return pehe

    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={},
        )

    @classmethod
    def from_spec(cls, spec: MetricSpec) -> "PEHEResultMetric":
        return PEHEResultMetric(**spec.params)
