from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.contracts.specs import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.artifacts import ArtifactWriter

# logger
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.enum.runplan import RunPlanParallelBackEnd
from TheCausalityGame.core.lib.enum.runtime import RuntimeMode
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.managers.plot import PlotManager
from TheCausalityGame.core.runtime.game import Game


class Runner:
    def __init__(
        self,
        *,
        run_dir: Path = Path("runs"),
        problem_instance: ProblemInstance | ProblemInstanceSpec,
    ) -> None:
        if isinstance(problem_instance, ProblemInstance):
            problem_instance = problem_instance.to_spec()

        self.problem_instance: ProblemInstanceSpec = problem_instance

        # Problem Instance attributes (simplified)
        self.is_dev = self.problem_instance.runtime.mode == RuntimeMode.DEV

        # Validate problem instance
        # TODO: Add a proper validation step

        # Compute max workers for parallel execution
        assert (
            self.problem_instance.run_plan.max_workers or 1
        ) > 0, "max_workers must be non-negative"
        self.workers = self.problem_instance.run_plan.max_workers or max(
            1, cpu_count() - 3
        )

        # Run directory
        self.run_dir = Path(run_dir / self.problem_instance.id)

        # Build artifact writer
        self.artifact_writer = ArtifactWriter(run_dir=self.run_dir, is_dev=self.is_dev)
        self.artifact_writer.write_provenance()

        # Loggers
        self.logger = self._generate_logger("Runner", self.artifact_writer.logs_dir)
        self.game_logger = self._generate_logger("Game", self.artifact_writer.logs_dir)
        self.environment_logger = self._generate_logger(
            "Environment", self.artifact_writer.logs_dir
        )

        self.logger.info(
            f"Starting run for problem instance '{self.problem_instance.id}'."
        )
        self.logger.warning(
            f"Runtime mode: {'Development' if self.is_dev else 'Production'}."
        )
        self.logger.error(
            f"Number of agents to run: {len(self.problem_instance.agents)}."
        )
        self.logger.critical(
            f"Run plan execution: {self.problem_instance.run_plan.execution}."
        )

        # Cache to avoid running the same agent multiple times
        self.agents_cached: set[str] = set()

    def _generate_logger(self, name: str, log_dir: Path) -> Logger:
        return Logger(
            name=name,
            log_dir=log_dir if self.is_dev else None,
            log_to_console=self.is_dev,
            level=self.problem_instance.runtime.debug_level,
        )

    def run(self) -> None:
        # Plot Manager
        self.plot_manager = PlotManager(plots=self.problem_instance.run_plan.plot_plan)
        self.logger.info(
            f"Initialized Plot Manager with plots: {[plot.id for plot in [*self.plot_manager.round_plots, *self.plot_manager.end_plots, *self.plot_manager.benchmark_plots]]}."
        )
        # Check Runtime Plan
        if self.problem_instance.run_plan.execution == "sequential":
            self.logger.info("Running agents sequentially.")
            transcripts = self._sequential_run()
        else:
            self.logger.info(
                f"Running agents in parallel using {self.problem_instance.run_plan.parallel_backend} with max workers: {self.workers}."
            )
            transcripts = self._parallel_run()

        # TODO: Create a separate funciton to handle plots. (Only run game/round plots for DEV mode)

        # Handle plots after all agents have run
        benchmark_figures = self.plot_manager.trigger_benchmark_end(transcripts)
        for i, fig in enumerate(benchmark_figures):
            fig_path = self.artifact_writer.plots_dir / f"benchmark_plot_{i}.png"
            fig.set_constrained_layout(True)  # type: ignore
            fig.tight_layout()
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)  # type: ignore
            self.logger.info(f"Saved benchmark plot to {fig_path}.")

    def _run_agent(self, agent: AgentSpec) -> Transcript | None:
        if agent.id in self.agents_cached:
            self.logger.warning(
                f"Agent with id '{agent.id}' has already been run. Skipping duplicate."
            )
            return

        # Hooks Manager, here to ensure fresh hooks for each agent
        self.hook_manager = HookManager(hooks=self.problem_instance.run_plan.hook_plan)
        self.logger.info(
            f"Initialized Hook Manager with hooks: {[hook.id for hook in self.hook_manager.hooks]}."
        )

        # (Log) Start running agent
        self.logger.info(f"Running agent '{agent.id}'.")

        # Create agent logger
        agent_logger = self._generate_logger(
            f"Agent {agent.id}", self.artifact_writer.logs_dir
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
            agent_logger=agent_logger,
            game_logger=self.game_logger,
            environment_logger=self.environment_logger,
        )
        # Run game
        transcript = game.run()

        self.logger.info(
            f"Agent '{agent.id}' completed with feedback: {transcript.entries[-1].feedback}"
        )

        # (Log) Save transcript
        self.logger.info(f"Agent '{agent.id}' finished running.")

        return transcript

    def _sequential_run(self) -> dict[str, Transcript]:
        transcripts: dict[str, Transcript] = {}
        agents = self.problem_instance.agents
        for agent in tqdm(
            agents, desc="Running agents (sequential)", unit="agent", leave=False
        ):
            transcript = self._run_agent(agent)
            if transcript:
                transcripts[agent.id] = transcript
        return transcripts

    def _parallel_run(self) -> dict[str, Transcript]:
        executor = (
            ThreadPoolExecutor
            if self.problem_instance.run_plan.parallel_backend
            == RunPlanParallelBackEnd.THREAD
            else ProcessPoolExecutor
        )

        transcripts: dict[str, Transcript] = {}

        with executor(max_workers=self.workers) as ex:
            futures = {
                ex.submit(self._run_agent, agent): agent.id
                for agent in self.problem_instance.agents
            }
            with tqdm(
                total=len(futures),
                desc=f"Running agents ({self.problem_instance.run_plan.parallel_backend.value})",
                unit="agent",
                leave=False,
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        transcripts[futures[future]] = result
                    pbar.update(1)

        return transcripts
