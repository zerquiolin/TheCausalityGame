from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infraestructure.logger import Logger
from TheCausalityGame.core.infraestructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.hooks import HookEvent
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.runtime.environment import Environment


class Game:
    def __init__(
        self,
        manifest_id: str,
        agent_spec: AgentSpec,
        scm_spec: SCMSpec,
        mission_spec: MissionSpec,
        custom_metrics_specs: list[MissionSpec],
        budget_spec: BudgetSpec,
        hook_manager: HookManager,
        logger: Logger,
    ) -> None:
        # Manifest ID
        self.manifest_id = manifest_id
        # Build Agent
        self.agent: Agent = build_from_spec(agent_spec)
        # Build SCM
        self.scm: SCM = build_from_spec(scm_spec)
        # Build Mission
        self.mission: Mission = build_from_spec(mission_spec)
        # Build Custom Metrics
        self.custom_metrics: list[Metric] = [
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
        # Build Environment
        self.environment = Environment(
            self.agent,
            self.scm,
            self.mission,
            self.custom_metrics,
            budget_spec,
            hook_manager,
            logger,
        )
        # Hook Manager
        self.hook_manager = hook_manager
        # Logger
        self.logger: Logger = logger

    def run(self) -> Transcript:
        # Log start
        self.logger.info(f"Starting game run. {self.agent.id}")
        # Flag start
        self.hook_manager.trigger(HookEvent.RUN_START)
        # Run Environment
        transcript_entries = self.environment.run()
        # Build Transcript
        self.transcript = Transcript(
            manifest_id=self.manifest_id,
            agent_id=self.agent.id,
            mission_id=1,  # TODO: Add mission id to MissionSpec
            mission_name=self.mission.name,
            entries=transcript_entries,
        )  # TODO: Pass in the transcript in the hook start
        # Flag end
        self.hook_manager.trigger(HookEvent.RUN_END)
        # Log end
        self.logger.info(f"Game run ended. {self.agent.id}")
        return self.transcript
