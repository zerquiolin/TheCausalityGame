"""The Causality Game - Core Simulation Environment."""

import zlib

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
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.enum.environment import ActionKind, SamplesKind
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.errors.environment import (
    ExperimentOutOfDomainError,
    NonControllableVariableError,
    UnknownVariableError,
    UnsupportedMetricTypeError,
)
from TheCausalityGame.core.managers.budget import BudgetManager
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
    ) -> None:
        self.agent = agent
        self.scm = scm
        self.mission = mission
        self.custom_metrics = custom_metrics
        self.transcript = transcript
        self.budget = BudgetManager(budget_spec)
        self.hook_manager = hook_manager
        self.logger = logger

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
            if node.accessibility in {NodeAccessibility.MEASURABLE, NodeAccessibility.CONTROLLABLE}
        ]

        # Seeded random states per treatment
        self.random_states: dict[
            str | tuple[tuple[str, int | float | str], ...],
            np.random.RandomState | dict[str, np.random.RandomState],
        ] = {}

        # Mount SCM to mission
        self.mission.mount(self.scm)

    def run(self) -> None:
        """
        Run the game loop.

        The agent experiments across rounds until they choose to submit an answer
        or reach a budget limit.
        """
        self.budget.start_time()

        for r in range(1, self.budget.rounds_limit + 1):
            self.budget.check_time()
            transcript_entry = TranscriptEntry(round=r)
            self.transcript.entries.append(transcript_entry)

            # Hooks: Start round
            self.hook_manager.trigger(HookEvent.TRANSCRIPTION_START, transcript_entry)
            self.hook_manager.trigger(HookEvent.ROUND_START, transcript_entry)

            self.budget.pause_time()
            self.hook_manager.trigger(HookEvent.BEFORE_ACT, transcript_entry)
            self.budget.resume_time()

            # Agent acts
            decision = self.agent.act(
                round_info=RoundInfo(round=r, budget_snapshot=self.budget.snapshot()),
                available_actions=self.available_actions,
            )
            transcript_entry.decision = decision
            transcript_entry.result = self.agent.answer()

            self.budget.pause_time()
            self.hook_manager.trigger(HookEvent.AFTER_ACT, transcript_entry)
            self.hook_manager.trigger(HookEvent.BEFORE_EVAL, transcript_entry)

            # Execute decision
            samples_collection = self._apply_decision(decision)
            feedback = self._get_feedback(self.transcript)

            transcript_entry.samples_collection = samples_collection
            transcript_entry.feedback = feedback

            self.hook_manager.trigger(HookEvent.AFTER_EVAL, transcript_entry)
            self.hook_manager.trigger(HookEvent.BEFORE_INFORM, transcript_entry)
            self.budget.resume_time()

            # Agent receives feedback and measurable samples only
            filtered_collection = SamplesCollection()
            if samples_collection:
                for s in samples_collection:
                    filtered_collection.append(
                        Samples(
                            kind=s.kind,
                            n=s.n,
                            data=s.data[self.measurable_nodes],
                            interventions=s.interventions,
                        )
                    )

            self.agent.inform(filtered_collection)
            self.budget.pause_time()

            self.hook_manager.trigger(HookEvent.AFTER_INFORM, transcript_entry)
            self.budget.tick_round()

            if samples_collection:
                self.budget.charge_samples(samples_collection.total_n())
                self.budget.charge_memory(samples_collection.total_bytes())

            transcript_entry.budget_snapshot = self.budget.snapshot()
            self.hook_manager.trigger(HookEvent.BUDGET_SNAPSHOT, transcript_entry)
            self.hook_manager.trigger(HookEvent.ROUND_END, transcript_entry)

            if decision.kind == ActionKind.ANSWER:
                break

    def _apply_decision(self, decision: Decision) -> SamplesCollection | None:
        """Apply the agent's decision and return the resulting samples."""
        if decision.kind == ActionKind.ANSWER:
            return None

        collection: list[Samples] = []

        for experiment in decision.experiments:
            treatment, n = experiment.treatment, experiment.n
            self._validate_experiment(experiment)

            hashed = tuple(sorted(treatment.items())) if treatment else SamplesKind.OBSERVATIONAL
            if hashed not in self.random_states:
                seed = zlib.crc32(str(hashed).encode())
                base_rs = np.random.RandomState(seed)
                self.random_states[hashed] = self.scm.prepare_new_random_state_structure(base_rs)

            samples = self.scm.generate_samples(
                interventions=treatment,
                num_samples=n,
                random_state=self.random_states[hashed],
            )

            collection.append(
                Samples(
                    kind=(
                        SamplesKind.OBSERVATIONAL.value
                        if hashed == SamplesKind.OBSERVATIONAL
                        else SamplesKind.INTERVENTIONAL.value
                    ),
                    n=n,
                    data=samples,
                    interventions=treatment,
                )
            )

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
            low, high = node.domain
            if isinstance(value, int | float) and not (float(low) <= float(value) <= float(high)):
                raise ExperimentOutOfDomainError(name, value, node.domain)

        return True

    def _get_feedback(self, transcript: Transcript) -> Feedback:
        """Evaluate the current transcript and return all feedback metrics."""
        feedback = Feedback()

        # Mission metrics
        behavior_score, result_score = self.mission.evaluate(transcript)
        feedback.behavior = behavior_score
        feedback.result = result_score

        # Final answer from transcript
        result = transcript.entries[-1].result
        custom_scores: dict[str, float] = {}

        for metric in self.custom_metrics:
            if isinstance(metric, BehaviorMetric):
                score = metric.evaluate(transcript)
            elif isinstance(metric, ResultMetric):
                score = metric.evaluate(kind=self.mission.result_validator.kind, result=result)
            else:
                raise UnsupportedMetricTypeError(metric_type=type(metric).__name__)
            custom_scores[metric.name] = score

        feedback.custom_metrics = custom_scores
        return feedback
