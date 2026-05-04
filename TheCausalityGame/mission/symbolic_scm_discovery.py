"""Symbolic SCM discovery mission."""

from __future__ import annotations

from typing import Any, override

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric, ResultMetric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec, get_class_path
from TheCausalityGame.core.lib.errors.mission import NotMountedError


class SymbolicSCMDiscoveryMission(Mission):
    """Mission for discovering symbolic SCM mechanisms."""

    id: str
    name: str
    description: str

    def __init__(
        self,
        behavior_metric: BehaviorMetric,
        result_metric: ResultMetric,
        result_validator: ResultValidator,
        id: str = "symbolic_scm_discovery",
        name: str = "Symbolic SCM Discovery Mission",
        description: str = (
            "This mission evaluates the ability to discover symbolic mechanisms "
            "for the variables in a structural causal model."
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
        if not self.is_mounted:
            raise NotMountedError()

        raw_result = transcript.entries[-1].result
        validated_result = self.result_validator.validate(raw_result)

        behavior_score = self.behavior_metric.evaluate(transcript)
        result_score = self.result_metric.evaluate(
            kind=self.result_validator.kind,
            result=validated_result,
        )

        return behavior_score, result_score

    @override
    def context_metadata(self) -> dict[str, Any]:
        return {
            "query_family": "symbolic_scm",
            "target": "symbolic_mechanisms",
            "scope": "numeric",
        }

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
    def from_spec(cls, spec: MissionSpec) -> SymbolicSCMDiscoveryMission:
        return cls(
            id=spec.id,
            behavior_metric=build_from_spec(spec.behavior_metric),
            result_metric=build_from_spec(spec.result_metric),
            result_validator=build_from_spec(spec.result_validator),
        )
