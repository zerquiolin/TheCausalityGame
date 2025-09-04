from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.enum.run import RunMode

from TheCausalityGame.core.contracts.agent import AgentContext, BaseAgent
from TheCausalityGame.core.contracts.dto import TranscriptEntry
from TheCausalityGame.core.contracts.enums import HookEvent
from TheCausalityGame.core.engine.environment import Environment
from TheCausalityGame.core.engine.evaluator import MetricsEvaluator
from TheCausalityGame.core.infra.logging_ import get_logger
from TheCausalityGame.core.infra.serialization import jsonl_write

HookEmit = Callable[[HookEvent, dict[str, Any]], None]
WriteStep = Callable[[TranscriptEntry], None]

BuildSCM = Callable[[dict[str, Any]], Any]
BuildMission = Callable[[dict[str, Any]], Any]
BuildMetrics = Callable[[dict[str, Any]], MetricsEvaluator | None]
BuildAgent = Callable[[dict[str, Any]], BaseAgent]


@dataclass(slots=True)
class Orchestrator:
    """Runs agents against environments built from a manifest/problem-instance.

    - Provide builder callables for SCM, Mission, and optional MetricsEvaluator.
    - Provide hook emitter and transcript writer, or use sensible defaults.
    - Choose sequential vs parallel either at call time or from a manifest.
    """

    run_dir: Path
    build_scm: BuildSCM
    build_mission: BuildMission
    build_metrics: BuildMetrics | None = None

    # Optional dependencies
    hook_emit: HookEmit = field(default=lambda *_: None)
    write_step: WriteStep | None = None

    # Parallel clamp
    max_parallel_workers: int = 0  # 0/None => choose sensibly at runtime

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # default transcript writer -> JSON Lines file in run_dir
        if self.write_step is None:
            steps_path = self.run_dir / "steps.jsonl"

            def _writer(entry: TranscriptEntry) -> None:
                jsonl_write(steps_path, entry.model_dump())

            self.write_step = _writer

    # ----------------- Build -----------------

    def build_environment(self, manifest: dict[str, Any]) -> Environment:
        """Construct a fresh Environment instance from a manifest/problem instance."""
        logger = get_logger("tcg.orch")
        # TODO: The SCM and Metrics builder must come from the registry.py since they are handlers for paths and specs.
        scm = self.build_scm(manifest)
        mission = self.build_mission(manifest)
        metrics = self.build_metrics(manifest) if self.build_metrics else None
        logger.debug(
            "Environment built",
            extra={"scm": type(scm).__name__, "mission": type(mission).__name__},
        )
        return Environment(scm=scm, mission=mission, metrics=metrics)

    # ----------------- Single run -----------------

    def run_agent(
        self,
        *,
        manifest: dict[str, Any],
        agent_id: str,
        agent_factory: BuildAgent,
        rounds: int,
        trusted: bool = True,
        time_limit_s: float | None = None,
        sample_limit: int | None = None,
        memory_mb_limit: float | None = None,
        game_scenario: dict[str, Any] | None = None,
        base_seed: int = 0,
    ) -> dict[str, Any]:
        """Run ONE agent to completion against a fresh Environment."""
        logger = get_logger("tcg.orch")
        env = self.build_environment(manifest)

        # Build agent and set its context
        agent = agent_factory(
            {"id": agent_id}
        )  # TODO: Again, this should come from the registry. Since the ProblemInstance already has the agents.
        ctx = AgentContext(
            config={},
            manifest_id=str(manifest.get("id", "manifest")),
            agent_id=agent_id,
            base_seed=base_seed,
            game_scenario=game_scenario or {"max_rounds": rounds},
        )
        agent.set_context(ctx)

        # Per-agent transcript file (helps parallel readability)
        run_steps_path = self.run_dir / f"steps.{agent_id}.jsonl"

        def _write_step(entry: TranscriptEntry) -> None:
            jsonl_write(run_steps_path, entry.model_dump())

        logger.info("Starting run", extra={"agent_id": agent_id, "rounds": rounds})
        out = env.run(
            agent_id=agent_id,
            agent=agent,
            rounds=rounds,
            trusted=trusted,
            hook_emit=self.hook_emit,
            write_step=_write_step,
            time_limit_s=time_limit_s,
            sample_limit=sample_limit,
            memory_mb_limit=memory_mb_limit,
        )
        logger.info(
            "Finished run", extra={"agent_id": agent_id, "done": out.get("done")}
        )
        return out

    # ----------------- Many runs (explicit APIs) -----------------

    def run_many_sequential(
        self,
        *,
        manifest: dict[str, Any],
        agents: Iterable[tuple[str, BuildAgent]],
        rounds: int,
        trusted: bool = True,
        time_limit_s: float | None = None,
        sample_limit: int | None = None,
        memory_mb_limit: float | None = None,
        base_seed: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Run multiple agents sequentially; returns a mapping agent_id -> summary."""
        results: dict[str, dict[str, Any]] = {}
        for agent_id, factory in agents:
            summary = self.run_agent(
                manifest=manifest,
                agent_id=agent_id,
                agent_factory=factory,
                rounds=rounds,
                trusted=trusted,
                time_limit_s=time_limit_s,
                sample_limit=sample_limit,
                memory_mb_limit=memory_mb_limit,
                base_seed=base_seed,
                game_scenario={"max_rounds": rounds},
            )
            results[agent_id] = summary
        return results

    def run_many_parallel(
        self,
        *,
        manifest: dict[str, Any],
        agents: Iterable[tuple[str, BuildAgent]],
        rounds: int,
        trusted: bool = True,
        time_limit_s: float | None = None,
        sample_limit: int | None = None,
        memory_mb_limit: float | None = None,
        base_seed: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Run multiple agents in parallel threads; returns mapping agent_id -> summary."""
        from multiprocessing import cpu_count

        workers = self.max_parallel_workers or max(1, min(4, cpu_count() - 1))
        results: dict[str, dict[str, Any]] = {}
        futures = []

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tcg") as ex:
            for agent_id, factory in agents:
                futures.append(
                    ex.submit(
                        self.run_agent,
                        manifest=manifest,
                        agent_id=agent_id,
                        agent_factory=factory,
                        rounds=rounds,
                        trusted=trusted,
                        time_limit_s=time_limit_s,
                        sample_limit=sample_limit,
                        memory_mb_limit=memory_mb_limit,
                        base_seed=base_seed,
                        game_scenario={"max_rounds": rounds},
                    )
                )

            # IMPORTANT: recover by agent_id order if desired; here we map by the same order
            for fut, (agent_id, _) in zip(as_completed(futures), agents, strict=False):
                try:
                    results[agent_id] = fut.result()
                except Exception as e:  # pragma: no cover (rare paths)
                    results[agent_id] = {
                        "done": False,
                        "error": f"{type(e).__name__}: {e}",
                    }

        return results

    # ----------------- Unified entry -----------------

    def run_many(
        self,
        *,
        run_mode: RunMode,
        manifest: dict[str, Any],
        agents: Iterable[tuple[str, BuildAgent]],
        rounds: int,
        trusted: bool = True,
        time_limit_s: float | None = None,
        sample_limit: int | None = None,
        memory_mb_limit: float | None = None,
        base_seed: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Unified entry to run multiple agents either sequentially or in parallel."""
        if run_mode == RunMode.PARALLEL:
            return self.run_many_parallel(
                manifest=manifest,
                agents=agents,
                rounds=rounds,
                trusted=trusted,
                time_limit_s=time_limit_s,
                sample_limit=sample_limit,
                memory_mb_limit=memory_mb_limit,
                base_seed=base_seed,
            )
        return self.run_many_sequential(
            manifest=manifest,
            agents=agents,
            rounds=rounds,
            trusted=trusted,
            time_limit_s=time_limit_s,
            sample_limit=sample_limit,
            memory_mb_limit=memory_mb_limit,
            base_seed=base_seed,
        )

    # ----------------- Manifest-driven entry -----------------

    def run_from_manifest(
        self,
        *,
        manifest: dict[str, Any],
        agents: Iterable[tuple[str, BuildAgent]],
    ) -> dict[str, dict[str, Any]]:
        """High-level helper: consume full manifest (incl. run_plan) and run all agents."""
        plan = manifest.get("run_plan", {}) or {}
        mode_str = (plan.get("mode") or "sequential").lower()
        run_mode = (
            RunMode(mode_str)
            if mode_str in RunMode._value2member_map_
            else RunMode.SEQUENTIAL
        )

        rounds = int(plan.get("rounds", 1))
        budgets = plan.get("budgets", {}) or {}
        time_limit_s = budgets.get("time_s")
        sample_limit = budgets.get("samples")
        memory_mb_limit = budgets.get("memory_mb")

        # clamp max_parallel_workers if specified in run_plan
        if plan.get("max_parallel_workers"):
            self.max_parallel_workers = int(plan["max_parallel_workers"])

        return self.run_many(
            run_mode=run_mode,
            manifest=manifest,
            agents=agents,
            rounds=rounds,
            trusted=True,  # allow by default; your CLI can expose this if desired
            time_limit_s=time_limit_s,
            sample_limit=sample_limit,
            memory_mb_limit=memory_mb_limit,
            base_seed=manifest.get("base_seed", 0),
        )
