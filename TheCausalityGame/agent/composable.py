"""Composable agent wrapper."""

from __future__ import annotations

from typing import Any, override

from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.infrastructure.registry import build_from_spec, get_class_path
from TheCausalityGame.core.lib.errors.agent import (
    AgentPendingObservationMissingError,
    IncompatibleAgentCompositionError,
)


class ComposableAgent(Agent):
    """Agent wrapper that composes an inferer with a decider."""

    def __init__(self, id: str, inferer: Inferer, decider: Decider) -> None:
        self.id = id
        self.inferer = inferer
        self.decider = decider
        self._pending_round_info: RoundInfo | None = None
        self._pending_decision: Decision | None = None

    @override
    def set_context(self, ctx: AgentContext) -> None:
        super().set_context(ctx)
        self.inferer.set_context(ctx)
        self.decider.set_context(ctx)

    @override
    def set_logger(self, logger: Logger) -> None:
        super().set_logger(logger)
        self.inferer.set_logger(logger)
        self.decider.set_logger(logger)

    @override
    def act(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
    ) -> Decision:
        belief = self.inferer.snapshot()
        required = self.decider.required_capabilities()
        provided = set(belief.capabilities)
        missing = set(required) - provided
        if missing:
            raise IncompatibleAgentCompositionError(missing)

        decision = self.decider.decide(
            round_info=round_info,
            available_actions=available_actions,
            belief=belief,
        )
        self._pending_round_info = round_info
        self._pending_decision = decision
        return decision

    @override
    def inform(self, samples_collection: SamplesCollection) -> None:
        if self._pending_round_info is None or self._pending_decision is None:
            raise AgentPendingObservationMissingError()

        observation = RoundObservation(
            round_info=self._pending_round_info,
            decision=self._pending_decision,
            samples=samples_collection,
        )
        self.inferer.update(observation)
        self.decider.update(observation)
        self._pending_round_info = None
        self._pending_decision = None

    @override
    def answer(self) -> Any:
        return self.inferer.answer()

    @override
    def to_spec(self) -> AgentSpec:
        return AgentSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            inferer=self.inferer.to_spec(),
            decider=self.decider.to_spec(),
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> ComposableAgent:
        if spec.inferer is None or spec.decider is None:
            raise ValueError("ComposableAgent requires both inferer and decider specs.")  # noqa: TRY003

        inferer = build_from_spec(spec.inferer)
        decider = build_from_spec(spec.decider)
        return cls(id=spec.id, inferer=inferer, decider=decider)
