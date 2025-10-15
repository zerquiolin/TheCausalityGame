from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.infraestructure.registry import (
    build_from_spec,
    get_class_path,
)

# Identify specific metric classes


class ConditionalAverageTreatmentEffectMission(Mission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

    name = "Conditional Average Treatment Effect Mission"
    description = "This mission evaluates the ability to infer the treatment effects in a causal graph given a intervention Z, covariates X, and outcome Y."

    def mount(self, scm: SCM):
        """
        Mount the mission to the given SCM.

        Args:
            scm (SCM): Structural Causal Model to mount the mission to.
        """
        # Mount Behavior Metric
        self.behavior_metric.mount(scm)
        # Mount Deliverable Metric
        self.result_metric.mount(scm)
        # Update the is_mounted flag
        self.is_mounted = True

    def evaluate(self, transcript: Transcript):
        # Check if the mission is mounted
        if not self.is_mounted:
            raise ValueError("Mission is not mounted")

        # Output validations
        user_output = transcript.entries[-1].result
        # Validate user output
        validated_output = self.result_validator.validate(user_output)
        # Evaluate Behavior & Result scores
        behavior_score = self.behavior_metric.evaluate(transcript=transcript)
        deliverable_score = self.result_metric.evaluate(
            kind=self.result_validator.kind(),
            user_output=validated_output,
        )

        return behavior_score, deliverable_score

    def to_spec(self) -> MissionSpec:
        return MissionSpec(
            class_=get_class_path(self.__class__),
            params={},
            behavior_metric=self.behavior_metric.to_spec(),
            result_metric=self.result_metric.to_spec(),
            result_validator=self.result_validator.to_spec(),
        )

    @classmethod
    def from_spec(cls, spec: MissionSpec) -> "ConditionalAverageTreatmentEffectMission":
        return ConditionalAverageTreatmentEffectMission(
            behavior_metric=build_from_spec(spec.behavior_metric),
            result_metric=build_from_spec(spec.result_metric),
            result_validator=build_from_spec(spec.result_validator),
        )
