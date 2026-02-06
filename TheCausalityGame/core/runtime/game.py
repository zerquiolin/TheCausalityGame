"""The Causality Game - Game Orchestration Module."""

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
    """
    Represents a single run of the Causality Game for one agent.

    Responsible for:
    - Constructing environment components (SCM, Mission, Agent, Metrics)
    - Initializing the agent context and logger
    - Executing the game via the Environment runner

    Parameters
    ----------
    manifest_id : str
        ID of the overarching run (benchmark/experiment).
    agent_spec : AgentSpec
        Specification for the agent to run.
    scm_spec : SCMSpec
        Specification of the structural causal model (SCM).
    mission_spec : MissionSpec
        Specification for the task/goal to solve.
    custom_metrics_specs : list[MetricSpec]
        Optional custom evaluation metrics.
    budget_spec : BudgetSpec
        Resource constraints for the run.
    hook_manager : HookManager
        Manager responsible for runtime hooks.
    agent_logger : Logger
        Logger scoped to the agent.
    game_logger : Logger
        Logger scoped to the game.
    environment_logger : Logger
        Logger scoped to the environment.
    """

    def __init__(  # noqa: PLR0913
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
        # Store manifest
        self.manifest_id = manifest_id

        # Build components from specs
        self.agent: Agent = build_from_spec(agent_spec)
        self.scm: SCM = build_from_spec(scm_spec)
        self.mission: Mission = build_from_spec(mission_spec)
        self.custom_metrics: list[Metric] = [build_from_spec(m) for m in custom_metrics_specs]

        # Set agent context
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
                {"name": m.name, "description": m.description} for m in self.custom_metrics
            ],
        )
        self.agent.set_context(agent_ctx)
        self.agent.set_logger(agent_logger)

        # Create transcript to record game
        self.transcript = Transcript(
            agent_id=self.agent.id,
            mission_id=self.mission.id,
            manifest_id=self.manifest_id,
            entries=[],
            budget=budget_spec,
        )

        # Environment
        self.environment = Environment(
            agent=self.agent,
            scm=self.scm,
            mission=self.mission,
            custom_metrics=self.custom_metrics,
            transcript=self.transcript,
            budget_spec=budget_spec,
            hook_manager=hook_manager,
            logger=environment_logger,
        )

        self.hook_manager = hook_manager
        self.logger = game_logger

    def run(self) -> Transcript:
        """
        Execute the full game run for this agent.

        Returns
        -------
        Transcript
            The full recorded transcript of the run.
        """
        self.logger.info(f"Starting game run for agent {self.agent.id}.")
        self.hook_manager.trigger(HookEvent.GAME_START)

        self.environment.run()

        self.hook_manager.trigger(HookEvent.GAME_END)
        self.logger.info(f"Game run ended for agent {self.agent.id}.")

        return self.transcript
