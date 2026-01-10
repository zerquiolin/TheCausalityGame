"""The Causality Game - Conditional Average Treatment Effect (CATE) Mission."""

from __future__ import annotations

from typing import override

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric, ResultMetric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
)
from TheCausalityGame.core.lib.errors.mission import NotMountedError


class DAGDiscoveryMission(Mission):
    """
    Mission for discovering the underlying Directed Acyclic Graph (DAG).

    This mission evaluates how well an agent estimates the correct Directed Acyclic Graph of the
    true underlying structured derived from the Structural Causal Model.
    The deliverable is expected to be a nx.DiGraph object that represents the discovered DAG.

    Attributes
    ----------
    id : str
        Unique identifier of the mission.
    name : str
        Human-readable name of the mission.
    description : str
        Description of the mission's goal and setup.
    """

    id: str
    name: str
    description: str

    def __init__(  # noqa: PLR0913
        self,
        behavior_metric: BehaviorMetric,
        result_metric: ResultMetric,
        result_validator: ResultValidator,
        id: str = "dag_discovery",
        name: str = "DAG Discovery Mission",
        description: str = (
            "This mission evaluates the ability to discover the DAG in a structural causal model."
        ),
    ) -> None:
        super().__init__(
            behavior_metric=behavior_metric,
            result_metric=result_metric,
            result_validator=result_validator,
        )
        self.id = id
        self.name = name
        self.description = description

    @override
    def mount(self, scm: SCM) -> None:
        self.behavior_metric.mount(scm)
        self.result_metric.mount(scm)
        self.is_mounted = True

    @override
    def evaluate(self, transcript: Transcript) -> tuple[float, float]:
        """
        Evaluate the agent's performance using both behavior and result metrics.

        Parameters
        ----------
        transcript : Transcript
            The full run history and submitted answer from the agent.

        Returns
        -------
        tuple[float, float]
            A tuple of (behavior_score, deliverable_score).

        Raises
        ------
        NotMountedError
            If the mission hasn't been mounted with an SCM.
        """
        if not self.is_mounted:
            raise NotMountedError()

        # Validate output
        raw_result = transcript.entries[-1].result
        validated_result = self.result_validator.validate(raw_result)

        # Evaluate metrics
        behavior_score = self.behavior_metric.evaluate(transcript)
        result_score = self.result_metric.evaluate(
            kind=self.result_validator.kind,
            result=validated_result,
        )

        return behavior_score, result_score

    @override
    def to_spec(self) -> MissionSpec:
        return MissionSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            behavior_metric=self.behavior_metric.to_spec(),
            result_metric=self.result_metric.to_spec(),
            result_validator=self.result_validator.to_spec(),
        )

    @classmethod
    @override
    def from_spec(cls, spec: MissionSpec) -> DAGDiscoveryMission:
        return cls(
            id=spec.id,
            behavior_metric=build_from_spec(spec.behavior_metric),
            result_metric=build_from_spec(spec.result_metric),
            result_validator=build_from_spec(spec.result_validator),
        )
