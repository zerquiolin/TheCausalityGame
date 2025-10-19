"""The Causality Game - Core Runtime Runner Module."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib
import matplotlib.figure
from tqdm import tqdm

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.contracts.specs import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.artifacts import ArtifactWriter
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.enum.runplan import RunPlanParallelBackEnd
from TheCausalityGame.core.lib.enum.runtime import RuntimeMode
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.managers.plot import PlotManager
from TheCausalityGame.core.runtime.game import Game


class Runner:
    """
    Main runtime orchestrator for executing agents in the Causality Game.

    Handles agent execution (sequential or parallel), logging, artifacts, and plotting.

    Parameters
    ----------
    run_dir : Path, optional
        The base directory to save artifacts, by default "runs".
    problem_instance : ProblemInstance | ProblemInstanceSpec
        The problem instance definition or its specification.
    """

    def __init__(
        self,
        *,
        run_dir: Path = Path("runs"),
        problem_instance: ProblemInstance | ProblemInstanceSpec,
    ) -> None:
        # Convert to spec if needed
        if isinstance(problem_instance, ProblemInstance):
            problem_instance = problem_instance.to_spec()

        self.problem_instance: ProblemInstanceSpec = problem_instance
        self.is_dev = self.problem_instance.runtime.mode == RuntimeMode.DEV

        # Compute worker pool size
        assert (
            self.problem_instance.run_plan.max_workers or 1
        ) > 0, "max_workers must be non-negative"
        self.workers = self.problem_instance.run_plan.max_workers or max(
            1, cpu_count() - 3
        )

        # Prepare output directories
        self.run_dir = run_dir / self.problem_instance.id
        self.artifact_writer = ArtifactWriter(run_dir=self.run_dir, is_dev=self.is_dev)

        # Initialize loggers
        self.logger = self._generate_logger("Runner", self.artifact_writer.logs_dir)
        self.game_logger = self._generate_logger("Game", self.artifact_writer.logs_dir)
        self.environment_logger = self._generate_logger(
            "Environment", self.artifact_writer.logs_dir
        )
        self.artifact_logger = self._generate_logger(
            "Artifact", self.artifact_writer.logs_dir
        )
        self.artifact_writer.set_logger(self.artifact_logger)

        # Cache executed agents
        self.agents_cached: set[str] = set()

    def _generate_logger(self, name: str, log_dir: Path) -> Logger:
        """Create a scoped logger for the runtime."""
        return Logger(
            name=name,
            log_dir=log_dir if self.is_dev else None,
            log_to_console=self.is_dev,
            level=self.problem_instance.runtime.debug_level,
        )

    def run(self) -> None:
        """Run the main execution entrypoint."""
        execution = self.problem_instance.run_plan.execution
        backend = self.problem_instance.run_plan.parallel_backend

        # Run agents (sequential or parallel)
        if execution == "sequential":
            self.logger.info("Running agents sequentially.")
            transcripts = self._sequential_run()
        else:
            self.logger.info(
                f"Running agents in parallel ({backend}) with {self.workers} workers."
            )
            transcripts = self._parallel_run()

        self.logger.info("All agents have completed execution.")

        # Save transcripts and plots
        self.artifact_writer.write_transcripts(transcripts)
        self._run_plot_manager(transcripts)

    def _run_agent(self, agent: AgentSpec) -> Transcript | None:
        """Execute a single agent."""
        if agent.id in self.agents_cached:
            self.logger.warning(f"Agent '{agent.id}' already executed. Skipping.")
            return

        # Hooks (fresh per agent)
        self.hook_manager = HookManager(
            self.problem_instance.run_plan.hook_plan,
            self.artifact_writer.agents_dir / agent.id / "hooks",
        )
        self.logger.info(
            f"Hook Manager initialized with {[h.id for h in self.hook_manager.hooks]}."
        )

        self.logger.info(f"Running agent '{agent.id}'.")

        agent_logger = self._generate_logger(
            f"Agent {agent.id}", self.artifact_writer.agents_dir / agent.id
        )

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

        transcript = game.run()

        self.logger.info(
            f"Agent '{agent.id}' completed with feedback: {transcript.entries[-1].feedback}"
        )
        return transcript

    def _sequential_run(self) -> dict[str, Transcript]:
        """Run all agents sequentially."""
        transcripts: dict[str, Transcript] = {}
        for agent in tqdm(
            self.problem_instance.agents,
            desc="Running agents (sequential)",
            unit="agent",
            leave=False,
        ):
            transcript = self._run_agent(agent)
            if transcript:
                transcripts[agent.id] = transcript
        return transcripts

    def _parallel_run(self) -> dict[str, Transcript]:
        """Run all agents in parallel using threads or processes."""
        executor_cls = (
            ThreadPoolExecutor
            if self.problem_instance.run_plan.parallel_backend
            == RunPlanParallelBackEnd.THREAD
            else ProcessPoolExecutor
        )

        transcripts: dict[str, Transcript] = {}

        with executor_cls(max_workers=self.workers) as ex:
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

    def _run_plot_manager(self, transcripts: dict[str, Transcript]) -> None:
        """
        Run configured plots and export them via the artifact writer.

        Parameters
        ----------
        transcripts : dict[str, Transcript]
            Transcripts of each agent's execution.
        """
        self.plot_manager = PlotManager(self.problem_instance.run_plan.plot_plan)
        all_plots = [
            *self.plot_manager.round_plots,
            *self.plot_manager.end_plots,
            *self.plot_manager.benchmark_plots,
        ]
        self.logger.info(
            f"Initialized Plot Manager with plots: {[p.id for p in all_plots]}."
        )

        plots: dict[
            str,
            dict[
                str,
                list[list[tuple[str, matplotlib.figure.Figure]]]
                | list[tuple[str, matplotlib.figure.Figure]],
            ]
            | list[tuple[str, matplotlib.figure.Figure]],
        ] = {}

        for agent_id, transcript in transcripts.items():
            round = self.plot_manager.trigger_rounds(transcript)
            end = self.plot_manager.trigger_end(transcript)

            if round or end:
                plots[agent_id] = {
                    k: v for k, v in [("round", round), ("end", end)] if v
                }

            plots[agent_id] = {
                "round": self.plot_manager.trigger_rounds(transcript),
                "end": self.plot_manager.trigger_end(transcript),
            }

        # Benchmark-level plots
        benchmark = self.plot_manager.trigger_benchmark_end(transcripts)
        if benchmark:
            plots["benchmark"] = benchmark

        self.logger.info("Generated all plots, exporting via Artifact Writer.")

        self.artifact_writer.write_plots(plots)
