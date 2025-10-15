import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.infraestructure.artifacts import ArtifactWriter

# logger
from TheCausalityGame.core.infraestructure.logger import Logger
from TheCausalityGame.core.infraestructure.registry import build_from_spec
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.managers.plot import PlotManager
from TheCausalityGame.core.runtime.game import Game


class Runner:
    def __init__(
        self,
        *,
        run_dir: Path = Path("runs"),
        problem_instance: ProblemInstance | str | dict,
    ) -> None:
        # Build problem instance
        if isinstance(problem_instance, str):
            with open(problem_instance, "r") as f:
                problem_instance = json.load(f)

        if isinstance(problem_instance, ProblemInstance):
            problem_instance = problem_instance.to_spec()

        self.problem_instance = problem_instance

        # Validate problem instance
        # TODO: Add a proper validation step

        # Compute max workers for parallel execution
        assert (
            self.problem_instance.run_plan.max_workers or 1
        ) > 0, "max_workers must be non-negative"
        self.workers = self.problem_instance.run_plan.max_workers or max(
            1, min(4, cpu_count() - 1)
        )

        # Run directory
        self.run_dir = Path(run_dir / self.problem_instance.id)

        # Build artifact writer
        self.artifact_writer = ArtifactWriter(run_dir=self.run_dir)

        # Logger (General)
        self.logger = Logger(name="Runner", log_dir=self.artifact_writer.logs_dir)

        self.logger.info(
            f"Starting run for problem instance '{self.problem_instance.id}'"
        )

    def run(self) -> None:
        # Hooks Manager
        self.hook_manager = HookManager(
            hooks=[
                build_from_spec(hook_spec)
                for hook_spec in self.problem_instance.run_plan.hook_plan
            ]
        )
        self.logger.info(
            f"Initialized Hook Manager with hooks: {[hook.id for hook in self.hook_manager.hooks]}"
        )
        # Plot Manager
        self.plot_manager = PlotManager(
            plots=[
                build_from_spec(plot_spec)
                for plot_spec in self.problem_instance.run_plan.plot_plan
            ]
        )
        self.logger.info(
            f"Initialized Plot Manager with plots: {[plot.id for plot in [*self.plot_manager.round_plots, *self.plot_manager.end_plots, *self.plot_manager.benchmark_plots]]}"
        )
        # Check Runtime Plan
        if self.problem_instance.run_plan.execution == "sequential":
            self.logger.info("Running agents sequentially")
            transcripts = self._sequential_run()
        else:
            self.logger.info(
                f"Running agents in parallel using {self.problem_instance.run_plan.parallel_backend} with max workers: {self.workers}"
            )
            transcripts = self._parallel_run()

        # Handle plots after all agents have run
        self.plot_manager.trigger_benchmark_end(transcripts)

    def _run_agent(self, agent: Agent) -> Transcript:
        if agent.id in self.agents_cached:
            self.logger.warning(
                f"Agent with id '{agent.id}' has already been run. Skipping duplicate."
            )
            return

        self.logger.info(f"Running agent '{agent.id}'")
        # Create agent directory
        self.artifact_writer.create_agent_dirs(agent.id)
        # Create logger for the agent
        agent_logger = Logger(
            name=f"Runner.{agent.id}",
            log_dir=self.artifact_writer.runs_dir / agent.id / "logs",
        )
        # Create game
        game = Game(
            manifest_id=self.problem_instance.id,
            agent_spec=agent,
            scm_spec=self.problem_instance.scm,
            mission_spec=self.problem_instance.mission,
            custom_metrics_specs=self.problem_instance.custom_metrics,
            budget_spec=self.problem_instance.run_plan.budget,
            hook_manager=self.hook_manager,
            logger=agent_logger,
        )
        # Run game
        transcript = game.run()
        return transcript

    def _sequential_run(self) -> dict[str, Transcript]:
        transcripts: dict[str, Transcript] = {}
        for agent in self.problem_instance.agents:
            transcript = self._run_agent(agent)
            if transcript:
                transcripts[agent.id] = transcript
        return transcripts

    def _parallel_run(self) -> dict[str, Transcript]:
        Executor = (
            ThreadPoolExecutor
            if self.problem_instance.run_plan.parallel_backend == "thread"
            else ProcessPoolExecutor
        )

        transcripts: dict[str, Transcript] = {}

        with Executor(max_workers=self.workers) as ex:
            futures = {
                ex.submit(self._run_agent, agent): agent.id
                for agent in self.problem_instance.agents
            }
            for future in as_completed(futures):
                transcripts[futures[future]] = future.result()

        return transcripts
