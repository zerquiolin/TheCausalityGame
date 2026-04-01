"""Combined agent wrapper."""

from __future__ import annotations

from typing import Any, override

from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.agent_policy import AgentPolicy
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.infrastructure.registry import build_from_spec, get_class_path
from TheCausalityGame.core.lib.errors.agent import AgentPendingObservationMissingError


class CombinedAgent(Agent):
    """Agent wrapper backed by a unified policy."""

    def __init__(self, id: str, policy: AgentPolicy) -> None:
        self.id = id
        self.policy = policy
        self._pending_round_info: RoundInfo | None = None
        self._pending_decision: Decision | None = None

    @override
    def set_context(self, ctx: AgentContext) -> None:
        super().set_context(ctx)
        self.policy.set_context(ctx)

    @override
    def set_logger(self, logger: Logger) -> None:
        super().set_logger(logger)
        self.policy.set_logger(logger)

    @override
    def act(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
    ) -> Decision:
        decision = self.policy.decide(
            round_info=round_info,
            available_actions=available_actions,
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
        self.policy.update(observation)
        self._pending_round_info = None
        self._pending_decision = None

    @override
    def answer(self) -> Any:
        return self.policy.answer()

    @override
    def to_spec(self) -> AgentSpec:
        return AgentSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            policy=self.policy.to_spec(),
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> CombinedAgent:
        if spec.policy is None:
            raise ValueError("CombinedAgent requires a policy spec.")

        policy = build_from_spec(spec.policy)
        return cls(id=spec.id, policy=policy)
