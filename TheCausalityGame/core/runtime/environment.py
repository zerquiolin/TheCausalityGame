"""The Causality Game - Core Simulation Environment."""

import math
from copy import deepcopy

import numpy as np

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    Experiment,
    ExperimentVariable,
    Feedback,
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.mission import (
    BehaviorMetric,
    Mission,
    ResultMetric,
)
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.scm_node import CategoricalSCMNode
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.enum.environment import ActionKind, SamplesKind
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.errors.budget import (
    MemoryBudgetExceededError,
    SamplesBudgetExceededError,
    TimeBudgetExceededError,
)
from TheCausalityGame.core.lib.errors.environment import (
    ExperimentOutOfDomainError,
    NonControllableVariableError,
    UnknownVariableError,
    UnsupportedMetricTypeError,
)
from TheCausalityGame.core.lib.utils.seeding import (
    derive_random_state,
    intervention_identity,
)
from TheCausalityGame.core.managers.budget import (
    AcquisitionBudgetExceededError,
    BudgetManager,
)
from TheCausalityGame.core.managers.hook import HookManager


class Environment:
    """
    Simulation engine for a single game round.

    Coordinates agent actions, SCM sampling, budget management, metrics, and hooks.

    Parameters
    ----------
    agent : Agent
        The agent being evaluated.
    scm : SCM
        Structural Causal Model representing the environment.
    mission : Mission
        The task/goal the agent is trying to solve.
    custom_metrics : list[Metric]
        Additional user-defined metrics.
    transcript : Transcript
        The object used to store all decisions and results.
    budget_spec : BudgetSpec
        Budget limitations (rounds, time, memory, etc).
    hook_manager : HookManager
        Runtime hook manager.
    logger : Logger
        Logger for environment events.
    """

    def __init__(  # noqa: PLR0913
        self,
        agent: Agent,
        scm: SCM,
        mission: Mission,
        custom_metrics: list[Metric],
        transcript: Transcript,
        budget_spec: BudgetSpec,
        hook_manager: HookManager,
        logger: Logger,
        acquisition_seed: int | None = None,
    ) -> None:
        self.agent = agent
        self.scm = scm
        self.mission = mission
        self.custom_metrics = custom_metrics
        self.transcript = transcript
        self.budget = BudgetManager(budget_spec)
        self.hook_manager = hook_manager
        self.logger = logger
        self.stream_identity = (
            acquisition_seed
            if acquisition_seed is not None
            else scm.acquisition_identity
        )

        # Available Actions
        self.available_actions = AvailableActions(
            experiments=[
                ExperimentVariable(name=node.name, domain=node.domain)
                for node in self.scm.nodes.values()
                if node.accessibility == NodeAccessibility.CONTROLLABLE and node.domain
            ],
            answer="submit",
        )

        # Measurable/Controllable Nodes
        self.measurable_nodes = [
            node.name
            for node in self.scm.nodes.values()
            if node.accessibility
            in {NodeAccessibility.MEASURABLE, NodeAccessibility.CONTROLLABLE}
        ]

        # Seeded random states per treatment
        self.random_states: dict[
            str | tuple[tuple[str, int | float | str], ...],
            np.random.RandomState | dict[str, np.random.RandomState],
        ] = {}

        # Mount SCM to mission if the game has not mounted it already.
        if not self.mission.is_mounted:
            self.mission.mount(self.scm)

    def run(self) -> None:
        """Score estimates after learning; stop normally at resource boundaries.

        Compute time covers act/inform/answer, excluding simulation and evaluator
        work. Limits are checked at call boundaries, not by hard preemption.
        """
        self.budget.start_time()
        reason = self.budget.stop_reason()
        for r in range(1, self.budget.rounds_limit + 1):
            if reason:
                break
            entry = TranscriptEntry(round=r)
            self.transcript.entries.append(entry)
            self.budget.pause_time()
            self.hook_manager.trigger(HookEvent.TRANSCRIPTION_START, entry)
            self.hook_manager.trigger(HookEvent.ROUND_START, entry)
            self.hook_manager.trigger(HookEvent.BEFORE_ACT, entry)
            self.budget.resume_time()
            decision = self.agent.act(
                round_info=RoundInfo(round=r, budget_snapshot=self.budget.snapshot()),
                available_actions=deepcopy(self.available_actions),
            )
            entry.decision = deepcopy(decision)
            self.budget.pause_time()
            self.hook_manager.trigger(HookEvent.AFTER_ACT, entry)
            try:
                self.budget.check_time()
                collection = self._apply_decision(decision)
            except SamplesBudgetExceededError:
                reason = "samples"
                collection = None
            except AcquisitionBudgetExceededError as error:
                reason = error.resource
                collection = None
            except TimeBudgetExceededError:
                reason = "time"
                collection = None
            entry.samples_collection = collection
            memory_exceeded = False
            if collection:
                try:
                    self.budget.charge_memory(collection.total_bytes())
                except MemoryBudgetExceededError:
                    memory_exceeded = True
            if reason is None:
                self.hook_manager.trigger(HookEvent.BEFORE_INFORM, entry)
                filtered = SamplesCollection(
                    [
                        Samples(
                            kind=batch.kind,
                            n=batch.n,
                            data=batch.data[self.measurable_nodes].copy(deep=True),
                            interventions=deepcopy(batch.interventions),
                        )
                        for batch in (collection or [])
                    ]
                )
                self.budget.resume_time()
                self.agent.inform(filtered)
                self.budget.pause_time()
                self.hook_manager.trigger(HookEvent.AFTER_INFORM, entry)
            self.budget.resume_time()
            forced_reason = reason or ("memory" if memory_exceeded else None)
            if decision.kind != ActionKind.ANSWER:
                # This round consumes the final available round before responding.
                forced_reason = forced_reason or (
                    "rounds"
                    if r >= self.budget.rounds_limit
                    else self.budget.stop_reason()
                )
            response_error = None
            try:
                entry.result = deepcopy(self.agent.answer())
            except Exception as error:
                if forced_reason is None:
                    raise
                response_error = f"{type(error).__name__}: {error}"
                entry.result = None
            self.budget.pause_time()
            self.budget.tick_round()
            entry.budget_snapshot = self.budget.snapshot()
            self.hook_manager.trigger(HookEvent.BUDGET_SNAPSHOT, entry)
            self.hook_manager.trigger(HookEvent.BEFORE_EVAL, entry)
            if entry.result is None:
                entry.feedback = Feedback(
                    behavior=self.mission.behavior_metric.evaluate(self.transcript),
                    result=None,
                    custom_metrics={},
                )
            else:
                try:
                    entry.feedback = self._get_feedback(self.transcript)
                except Exception as error:
                    if forced_reason is None:
                        raise
                    response_error = f"{type(error).__name__}: {error}"
                    entry.result = None
                    entry.feedback = Feedback(
                        result=None,
                        behavior=self.mission.behavior_metric.evaluate(self.transcript),
                        custom_metrics={},
                    )
                entry.budget_snapshot = self.budget.snapshot()
                entry.feedback.behavior = self.mission.behavior_metric.evaluate(
                    self.transcript
                )
                for metric in self.custom_metrics:
                    if isinstance(metric, BehaviorMetric):
                        entry.feedback.custom_metrics[metric.name] = metric.evaluate(
                            self.transcript
                        )
            if (
                reason is None
                and self.budget.time_limit_s is not None
                and self.budget.seconds_elapsed() >= self.budget.time_limit_s
            ):
                reason = "time"
            entry.custom_attributes["response_status"] = (
                "failed"
                if response_error
                else "missing" if entry.result is None else "provided"
            )
            entry.custom_attributes["response_error"] = response_error
            self.hook_manager.trigger(HookEvent.AFTER_EVAL, entry)
            inferer = getattr(self.agent, "inferer", None)
            if inferer is not None:
                entry.custom_attributes["estimate_metadata"] = deepcopy(
                    inferer.snapshot().summary
                )
            self.hook_manager.trigger(HookEvent.ROUND_END, entry)
            reason = reason or forced_reason
            if reason is None:
                reason = (
                    "submitted"
                    if decision.kind == ActionKind.ANSWER
                    else self.budget.stop_reason()
                )
        if self.transcript.entries:
            self.transcript.finish(
                reason or "rounds",
                deepcopy(self.transcript.entries[-1].result),
                deepcopy(self.transcript.entries[-1].feedback),
            )
        else:
            # Even an initially exhausted budget permits a final response.
            try:
                result = deepcopy(self.agent.answer())
                entry = TranscriptEntry(round=0, result=result)
                self.transcript.entries.append(entry)
                feedback = (
                    self._get_feedback(self.transcript)
                    if result is not None
                    else Feedback(result=None, behavior=0.0)
                )
                entry.feedback = feedback
                entry.custom_attributes["response_status"] = (
                    "provided" if result is not None else "missing"
                )
            except Exception as error:
                result = None
                feedback = Feedback(result=None, behavior=0.0)
                if not self.transcript.entries:
                    self.transcript.entries.append(TranscriptEntry(round=0))
                entry = self.transcript.entries[-1]
                entry.result = None
                entry.feedback = feedback
                entry.custom_attributes.update(
                    response_status="failed",
                    response_error=f"{type(error).__name__}: {error}",
                )
            self.transcript.finish(reason or "rounds", result, feedback)

    def _apply_decision(self, decision: Decision) -> SamplesCollection | None:
        """Apply the agent's decision and return the resulting samples."""
        if decision.kind == ActionKind.ANSWER:
            return None

        for experiment in decision.experiments:
            self._validate_experiment(experiment)
        cost = self.budget.preflight(decision.experiments)
        collection: list[Samples] = []
        staged_streams = {}

        for experiment in decision.experiments:
            treatment, n = experiment.treatment, experiment.n
            self._validate_experiment(experiment)

            hashed = intervention_identity(treatment)
            if hashed in self.random_states and hashed not in staged_streams:
                staged_streams[hashed] = deepcopy(self.random_states[hashed])
            if hashed not in staged_streams:
                base_rs = derive_random_state(
                    "acquisition-v3", self.stream_identity, hashed
                )
                staged_streams[hashed] = self.scm.prepare_new_random_state_structure(
                    base_rs
                )

            samples = self.scm.generate_samples(
                interventions=treatment,
                num_samples=n,
                random_state=staged_streams[hashed],
            )

            if len(samples) != n:
                raise ValueError(
                    f"SCM returned {len(samples)} rows for {n} requested samples."
                )
            collection.append(
                Samples(
                    kind=(
                        SamplesKind.OBSERVATIONAL.value
                        if not treatment
                        else SamplesKind.INTERVENTIONAL.value
                    ),
                    n=n,
                    data=samples,
                    interventions=treatment,
                )
            )

        self.budget.charge_acquisition(decision.experiments, cost)
        self.random_states.update(staged_streams)
        return SamplesCollection(collection)

    def _validate_experiment(self, experiment: Experiment) -> bool:
        """Ensure that experiment is valid and within allowed bounds."""
        if experiment.treatment is None:
            return True  # Observational study

        for name, value in experiment.treatment.items():
            node = self.scm.nodes.get(name)
            if node is None:
                raise UnknownVariableError(name)
            if node.accessibility != NodeAccessibility.CONTROLLABLE:
                raise NonControllableVariableError(name)
            if isinstance(node, CategoricalSCMNode):
                valid = value in node.domain
            else:
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and len(node.domain) == 2
                    and float(node.domain[0]) <= float(value) <= float(node.domain[1])
                )
            if not valid:
                raise ExperimentOutOfDomainError(name, value, node.domain)

        return True

    def _get_feedback(self, transcript: Transcript) -> Feedback:
        """Evaluate the current transcript and return all feedback metrics."""
        feedback = Feedback()

        entry = transcript.entries[-1]
        answer = entry.result
        if callable(answer):
            # Callable inference (including lazy backend fits) is agent compute;
            # metric bookkeeping and pre-mounted truth remain excluded.
            def timed_answer(X, treatment, outcome, covariate_values):
                self.budget.resume_time()
                try:
                    return answer(X, treatment, outcome, covariate_values)
                finally:
                    self.budget.pause_time()

            entry.result = timed_answer

        # Mission metrics
        try:
            behavior_score, result_score = self.mission.evaluate(transcript)
        finally:
            entry.result = answer
        feedback.behavior = behavior_score
        feedback.result = result_score

        # Final answer from transcript
        result = timed_answer if callable(answer) else answer
        custom_scores: dict[str, float] = {}

        for metric in self.custom_metrics:
            if isinstance(metric, BehaviorMetric):
                score = metric.evaluate(transcript)
            elif isinstance(metric, ResultMetric):
                score = metric.evaluate(
                    kind=self.mission.result_validator.kind, result=result
                )
            else:
                raise UnsupportedMetricTypeError(metric_type=type(metric).__name__)
            custom_scores[metric.name] = score

        feedback.custom_metrics = custom_scores
        return feedback
