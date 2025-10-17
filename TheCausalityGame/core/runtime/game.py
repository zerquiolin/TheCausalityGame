from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.runtime.environment import Environment


class Game:
    def __init__(
        self,
        manifest_id: str,
        agent_spec: AgentSpec,
        scm_spec: SCMSpec,
        mission_spec: MissionSpec,
        custom_metrics_specs: list[MetricSpec],
        budget_spec: BudgetSpec,
        hook_manager: HookManager,
        agent_logger: Logger,
        game_logger: Logger,
        environment_logger: Logger,
    ) -> None:
        # Manifest ID
        self.manifest_id = manifest_id
        # Build Agent
        self.agent: Agent = build_from_spec(agent_spec)  # type: ignore
        # Build SCM
        self.scm: SCM = build_from_spec(scm_spec)  # type: ignore
        # Build Mission
        self.mission: Mission = build_from_spec(mission_spec)  # type: ignore
        # Build Custom Metrics
        self.custom_metrics: list[Metric] = [  # type: ignore
            build_from_spec(m) for m in custom_metrics_specs
        ]
        # Agent Context
        agent_ctx = AgentContext(
            mission={
                "name": self.mission.name,
                "description": self.mission.description,
            },
            behavior_metric={
                "name": self.mission.behavior_metric.name,
                "description": self.mission.behavior_metric.description,
            },
            result_metric={
                "name": self.mission.result_metric.name,
                "description": self.mission.result_metric.description,
            },
            custom_metrics=[
                {"name": m.name, "description": m.description}
                for m in self.custom_metrics
            ],
            seed=911,
        )
        self.agent.set_context(agent_ctx)
        # Agent Logger
        self.agent.set_logger(agent_logger)
        # Build Transcript
        self.transcript = Transcript(
            agent_id=self.agent.id,
            mission_id=self.mission.id,
            manifest_id=self.manifest_id,
            entries=[],
        )
        # Build Environment
        self.environment = Environment(
            self.agent,
            self.scm,
            self.mission,
            self.custom_metrics,
            self.transcript,
            budget_spec,
            hook_manager,
            environment_logger,
        )
        # Hook Manager
        self.hook_manager = hook_manager
        # Logger
        self.logger: Logger = game_logger

    def run(self) -> Transcript:
        # Log start
        self.logger.info(f"Starting game run for agent {self.agent.id}.")
        # Flag start
        self.hook_manager.trigger(HookEvent.GAME_START)
        # Run Environment
        self.environment.run()
        # Flag end
        self.hook_manager.trigger(HookEvent.GAME_END)
        # Log end
        self.logger.info(f"Game run ended for agent {self.agent.id}.")
        return self.transcript
