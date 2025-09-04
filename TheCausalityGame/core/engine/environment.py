from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple, List, Dict, Optional

from TheCausalityGame.core.contracts.agent import BaseAgent
from TheCausalityGame.core.contracts.decisions import Decision
from TheCausalityGame.core.contracts.dto import (
    Action,
    ActionOutcome,
    AvailableActions,
    ExperimentSpace,
    Observation,
    RoundInfo,
    Samples,
    SamplesBatch,
    StepRecord,
)
from TheCausalityGame.core.contracts.enums import HookEvent, StepKind
from TheCausalityGame.core.engine.evaluator import MetricsEvaluator
from TheCausalityGame.core.infra.budgets import (
    BudgetEnforcer,
    BudgetExceededError,
    BudgetState,
)
from TheCausalityGame.core.infra.determinism import (
    hash_intervention_key,
    make_intervention_seed,
)
from TheCausalityGame.core.infra.logging_ import get_logger

HookEmit = Callable[[HookEvent, dict[str, Any]], None]
WriteStep = Callable[[StepRecord], None]


@dataclass(slots=True)
class Environment:
    """Environment: binds SCM + Mission and runs the round loop for one agent.

    Features:
      - Two decisions: 'experiment' (one or more (interventions, n)), 'answer'
      - Deterministic seeds derived in env
      - Budget enforcement (rounds, time, samples, memory)
      - Hook emission via HookEvent enum
      - Step transcripts via StepKind enum
      - MetricsEvaluator applied outside the hot loop (optional)
    """

    scm: Any
    mission: Any
    metrics: Optional[MetricsEvaluator] = None

    def __post_init__(self) -> None:
        if hasattr(self.mission, "mount"):
            self.mission.mount(self.scm)

    # -------------------- Thin API --------------------

    def generate_samples(
        self, *, n: int, interventions: Mapping[str, Any] | None, seed: int
    ) -> Mapping[str, list[Any]]:
        """Delegate sampling to SCM."""
        return self.scm.generate_samples(n=n, interventions=interventions, seed=seed)

    def mission_validate_submit(self, payload: dict[str, Any], *, trusted: bool) -> Any:
        """Delegate deliverable validation to Mission."""
        return self.mission.validate_submit(payload, trusted=trusted)

    # TODO: The truth is actually calculated within the Metric, not in the Environment nor the Mission.
    def truth_handle_for_metrics(self) -> Any:
        """Let the Mission compute/own ground-truth handle."""
        return self.mission.truth_handle_for_metrics(self.scm)

    # -------------------- Run loop --------------------

    def run(
        self,
        *,
        agent_id: str,
        agent: BaseAgent,
        rounds: int,
        trusted: bool,
        hook_emit: HookEmit,
        write_step: WriteStep,
        # budgets
        time_limit_s: float | None = None,
        sample_limit: int | None = None,
        memory_mb_limit: float | None = None,
    ) -> dict[str, Any]:
        """Execute the full round loop and return a run summary."""
        logger = get_logger("tcg.env")

        budget = BudgetEnforcer(
            BudgetState(
                hard_round_limit=rounds,
                time_limit_s=time_limit_s,
                sample_limit=sample_limit,
                memory_mb_limit=memory_mb_limit,
            )
        )

        done = False
        last_deliverable: dict[str, Any] | None = None
        transcript: list[dict[str, Any]] = []

        hook_emit(HookEvent.RUN_START, {"agent_id": agent_id})

        try:
            for r in range(rounds):
                if done:
                    break

                budget.check_time()
                budget.tick_round()

                hook_emit(HookEvent.ROUND_START, {"agent_id": agent_id, "round": r})

                round_info = RoundInfo(
                    round_index=r,
                    remaining_rounds=rounds - r,
                    budgets_snapshot=budget.snapshot(),
                )
                available = self._available_actions()

                # status step (useful for consumers)
                observation = Observation(
                    kind=StepKind.STATUS.value, payload={"round": r}
                )
                step0 = StepRecord(
                    round_index=r,
                    step_index=0,
                    agent_id=agent_id,
                    mission_id=type(self.mission).__name__,
                    observation=observation,
                )
                write_step(step0)
                transcript.append(step0.model_dump())

                hook_emit(
                    HookEvent.BEFORE_ACT,
                    {
                        "agent_id": agent_id,
                        "round": r,
                        "experiment_vars": list(available.experiment.variables.keys()),
                        "budgets": round_info.budgets_snapshot,
                    },
                )

                decision: Decision = agent.act(round_info, available)

                (
                    action,
                    outcome,
                    observation,
                    rows_generated,
                    bytes_generated,
                ) = self._apply_decision(
                    decision,
                    agent=agent,
                    trusted=trusted,
                    round_index=r,
                    available=available,
                )

                # charge budgets
                if rows_generated > 0:
                    budget.charge_samples(rows_generated)
                if bytes_generated > 0:
                    budget.charge_memory(bytes_generated)

                # TODO: outcome.feedback is currently always None
                agent.inform(outcome)

                # TODO: Both the mission spec and
                step1 = StepRecord(
                    round_index=r,
                    step_index=1,
                    agent_id=agent_id,
                    mission_id=type(self.mission).__name__,
                    action=action,
                    observation=observation,
                    done=action.kind in (StepKind.ACTION_SUBMIT_FINAL.value,),
                )
                write_step(step1)
                transcript.append(step1.model_dump())

                # TODO: I think this part must be simplified and used within the _apply_decision() since the actionoutcome feedback is always None. Here we can use the feedback for the observation or other processes.
                if action.kind == StepKind.ACTION_EXPERIMENT.value:
                    hook_emit(
                        HookEvent.AFTER_ACT,
                        {
                            "agent_id": agent_id,
                            "round": r,
                            "action": action.kind,
                            "rows": (observation.payload or {}).get("rows"),
                            "intervention_keys": (observation.payload or {}).get(
                                "intervention_keys"
                            ),
                            "bytes_generated": bytes_generated,
                            "budgets": budget.snapshot(),
                        },
                    )
                elif action.kind == StepKind.ACTION_SUBMIT_FINAL.value:
                    last_deliverable = None
                    if outcome.feedback and hasattr(outcome.feedback, "deliverable"):
                        last_deliverable = outcome.feedback.deliverable  # type: ignore[assignment]
                    done = True
                    hook_emit(
                        HookEvent.SUBMIT_FINAL, {"agent_id": agent_id, "round": r}
                    )
                    hook_emit(
                        HookEvent.AFTER_ACT,
                        {"agent_id": agent_id, "round": r, "action": action.kind},
                    )

                hook_emit(HookEvent.ROUND_END, {"agent_id": agent_id, "round": r})

            # Final metrics (optional)
            result: dict[str, Any] = {
                "done": done,
                "last_deliverable": last_deliverable,
            }

            # TODO: The metrics should evaluate both behavior and results, but the truth is calculated within the Metric itself.
            # TODO: This section should only call the metric.evaluate(...) or similar. The metric itself is in charge of evaluating both behavior and result metrics.
            if self.metrics is not None:
                behavior_scores = self.metrics.evaluate_behavior(transcript=transcript)
                result["behavior_scores"] = [m.model_dump() for m in behavior_scores]
                if last_deliverable is not None:
                    truth = self.truth_handle_for_metrics()
                    result_scores = self.metrics.evaluate_results(
                        deliverable=last_deliverable, truth=truth
                    )
                    result["result_scores"] = [m.model_dump() for m in result_scores]

            hook_emit(HookEvent.RUN_END, {"agent_id": agent_id})
            logger.info(
                "run finished",
                extra={"agent_id": agent_id, "rounds": rounds, "done": done},
            )
            return result

        except BudgetExceededError as e:
            hook_emit(HookEvent.RUN_END, {"agent_id": agent_id, "error": str(e)})
            logger.warning(
                "run aborted by budget", extra={"agent_id": agent_id, "reason": str(e)}
            )
            return {"done": False, "error": str(e)}

    # -------------------- helpers --------------------

    # TODO: it would be great to have the accesibility and other info as constants.
    def _available_actions(self) -> AvailableActions:
        """Derive experiment space from SCM nodes (controllable domains)."""
        controllable: dict[str, list] = {}
        nodes = getattr(self.scm, "nodes", {}) or {}
        values_iter = getattr(nodes, "values", lambda: [])()
        for node in values_iter:
            if getattr(node, "accessibility", None) == "controllable":
                domain = getattr(node, "domain", None)
                if domain is not None:
                    controllable[getattr(node, "name", "unknown")] = list(domain)
        exp_space = ExperimentSpace(variables=controllable, max_n=None)
        return AvailableActions(experiment=exp_space, answer={})

    def _validate_interventions(
        self, interventions: Mapping[str, Any], exp: ExperimentSpace
    ) -> None:
        """Ensure interventions are within the advertised experiment space."""
        for var, val in interventions.items():
            if var not in exp.variables:
                raise ValueError(f"Unknown intervention variable: {var}")
            if val not in exp.variables[var]:
                raise ValueError(
                    f"Value {val!r} not in domain of {var}: {exp.variables[var]!r}"
                )

    def _apply_decision(
        self,
        decision: Decision,
        *,
        agent: BaseAgent,
        trusted: bool,
        round_index: int,
        available: AvailableActions,
    ) -> Tuple[Action, ActionOutcome, Observation, int, int]:
        """Apply a Decision and return (Action, Outcome, Observation, rows_generated, bytes_generated)."""

        if decision.is_experiment:
            samples_list: List[Samples] = []
            intervention_keys: List[str] = []
            total_rows = 0
            total_bytes = 0

            for spec in decision.experiments:
                iv: Mapping[str, Any] | None = spec.interventions or {}
                self._validate_interventions(iv, available.experiment) if iv else None

                ctx = agent.context
                ikey = hash_intervention_key(
                    base_seed=ctx.base_seed,
                    manifest_id=ctx.manifest_id,
                    agent_id=ctx.agent_id,
                    round_index=round_index,
                    interventions=iv,
                    n=spec.n,
                )
                seed = make_intervention_seed(
                    base_seed=ctx.base_seed,
                    manifest_id=ctx.manifest_id,
                    agent_id=ctx.agent_id,
                    round_index=round_index,
                    interventions=iv,
                    n=spec.n,
                )

                ds = self.generate_samples(n=spec.n, interventions=iv, seed=seed)

                rows = 0
                if isinstance(ds, Mapping) and ds:
                    rows = len(next(iter(ds.values())))
                total_rows += rows

                approx_bytes = self._estimate_dataset_bytes(ds)
                total_bytes += approx_bytes

                samples_list.append(
                    Samples(
                        kind="observational" if not iv else "interventional",
                        data=ds,  # type: ignore[arg-type]
                        n=rows,
                        interventions=iv,
                        seed=seed,
                        key=ikey,
                    )
                )
                intervention_keys.append(ikey)

            action = Action(
                kind=StepKind.ACTION_EXPERIMENT.value,
                payload={"count": len(samples_list), "keys": intervention_keys},
            )
            outcome = ActionOutcome(
                action_kind="experiment",
                action_payload=action.payload,
                samples=SamplesBatch(items=samples_list),
                feedback=None,
            )
            observation = Observation(
                kind=StepKind.DATASET_BATCH.value,
                payload={
                    "rows": [s.n for s in samples_list],
                    "cols": list(samples_list[0].data.keys()) if samples_list else [],
                    "intervention_keys": intervention_keys,
                    "approx_bytes": total_bytes,
                },
            )
            return action, outcome, observation, total_rows, total_bytes

        if decision.is_answer:
            payload = agent.answer()
            validated = self.mission_validate_submit(payload, trusted=trusted)
            action = Action(kind=StepKind.ACTION_SUBMIT_FINAL.value, payload={})
            outcome = ActionOutcome(
                action_kind="answer",
                action_payload={},
                samples=None,
                feedback=None,  # metrics/feedback handled post-loop
            )
            observation = Observation(
                kind=StepKind.FEEDBACK.value,
                payload={"accepted": True, "deliverable": bool(validated)},
            )
            return action, outcome, observation, 0, 0

        action = Action(kind=StepKind.ACTION_UNKNOWN.value, payload={})
        outcome = ActionOutcome(
            action_kind="unknown", action_payload={}, samples=None, feedback=None
        )
        observation = Observation(
            kind=StepKind.STATUS.value, payload={"warning": "unknown-decision"}
        )
        return action, outcome, observation, 0, 0

    # -------------------- memory estimation --------------------

    def _estimate_dataset_bytes(self, ds: Mapping[str, List[Any]] | Any) -> int:
        """Best-effort, fast approximation of a columnar dataset's memory footprint.

        For numpy/pandas outputs, override this method and use `.nbytes` or
        `df.memory_usage(deep=True).sum()`.
        """
        try:
            if not isinstance(ds, Mapping):
                return 0
            seen_ids: set[int] = set()
            total = 0

            def _sizeof(obj: Any) -> int:
                oid = id(obj)
                if oid in seen_ids:
                    return 0
                seen_ids.add(oid)
                try:
                    return sys.getsizeof(obj)
                except Exception:
                    return 0

            total += _sizeof(ds)
            for col, values in ds.items():
                total += _sizeof(col)
                total += _sizeof(values)
                if isinstance(values, list):
                    for v in values:
                        if isinstance(v, (int, float, bool, str)):
                            total += _sizeof(v)
                        else:
                            # conservative small constant for uncommon types
                            total += 16
            return total
        except Exception:
            return 0
