import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.engine.game import Game
from TheCausalityGame.core.infra.artifacts import ArtifactWriter

# logger
from TheCausalityGame.core.infra.logger import Logger
from TheCausalityGame.core.infra.logging_ import get_logger
from TheCausalityGame.core.infra.registry import build_from_spec
from TheCausalityGame.core.managers.hook import HookManager
from TheCausalityGame.core.managers.plot import PlotManager


class Runner:
    def __init__(
        self, *, run_dir: Path = Path("runs"), problem_instance: ProblemInstance | str
    ) -> None:
        # Build problem instance
        if isinstance(problem_instance, str):
            with open(problem_instance, "r") as f:
                problem_instance_spec = json.load(f)
            problem_instance = build_from_spec(problem_instance_spec)

        self.problem_instance = problem_instance

        # Run directory
        self.run_dir = Path(run_dir / self.problem_instance.id)

        # Build artifact writer
        self.artifact_writer = ArtifactWriter(run_dir=self.run_dir)
        self.artifact_writer.create_run_dir()

        # Logger (General)
        self.logger = Logger(name="Runner", log_dir=self.run_dir / "logs")

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
        # Plot Manager
        self.plot_manager = PlotManager(
            plots=[
                build_from_spec(plot_spec)
                for plot_spec in self.problem_instance.run_plan.plot_plan
            ]
        )
        # Check Runtime Plan
        if self.problem_instance.run_plan.execution == "sequential":
            transcripts = self._sequential_run()
        else:
            transcripts = self._parallel_run()

        # Handle plots after all agents have run
        self.plot_manager.trigger_benchmark_end(transcripts)

    def _run_agent(self, agent: Agent) -> Transcript:
        # Create game
        game = Game(
            manifest_id=self.problem_instance.id,
            agent_spec=agent,
            scm_spec=self.problem_instance.scm,
            mission_spec=self.problem_instance.mission,
            custom_metrics_specs=self.problem_instance.custom_metrics,
            budget_spec=self.problem_instance.run_plan.budget,
            hook_manager=self.hook_manager,
            plot_manager=self.plot_manager,  # TODO: This might not be needed in the future.
        )
        # Run game
        transcript = game.run()
        return transcript

    def _sequential_run(self) -> dict[str, Transcript]:
        transcripts: dict[str, Transcript] = {}
        for agent in self.problem_instance.agents:
            transcript = self._run_agent(agent)
            transcripts[agent.id] = transcript
        return transcripts

    def _parallel_run(self) -> dict[str, Transcript]:
        Executor = (
            ThreadPoolExecutor
            if self.problem_instance.run_plan.parallel_backend == "thread"
            else ProcessPoolExecutor
        )

        workers = self.problem_instance.run_plan.max_workers or max(
            1, min(4, cpu_count() - 1)
        )

        transcripts: dict[str, Transcript] = {}

        with Executor(max_workers=workers) as ex:
            futures = {
                ex.submit(self._run_agent, agent): agent.id
                for agent in self.problem_instance.agents
            }
            for future in as_completed(futures):
                transcripts[futures[future]] = future.result()

        return transcripts
