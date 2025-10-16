import zlib

import numpy as np

from TheCausalityGame.core.contracts.agent import Agent

# DTO
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
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
from TheCausalityGame.core.infrastructure.budgets import (
    BudgetEnforcer,
    BudgetExceededError,
)
from TheCausalityGame.core.infrastructure.decisions import Decision, ExperimentSpec
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.enum.environment import ActionKind
from TheCausalityGame.core.lib.enum.hooks import HookEvent
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.managers.hook import HookManager


class Environment:
    def __init__(
        self,
        agent: Agent,
        scm: SCM,
        mission: Mission,
        custom_metrics: list[Metric],
        transcript: Transcript,
        budget_spec: BudgetSpec,
        hook_manager: HookManager,
        logger: Logger,
    ):
        # Agent
        self.agent = agent
        # SCM
        self.scm = scm
        # Mission
        self.mission = mission
        # Metrics
        self.custom_metrics = custom_metrics
        # Transcript
        self.transcript = transcript
        # Budget Enforcer
        self.budget = BudgetEnforcer(budget_spec)
        # Hook Manager
        self.hook_manager = hook_manager
        # Logger
        self.logger = logger
        # Available Actions
        self.available_actions = AvailableActions(
            experiments=[
                ExperimentVariable(name=node.name, domain=node.domain)
                for node in self.scm.nodes.values() or []
                if node.accessibility == NodeAccessibility.CONTROLLABLE
                and node.domain is not None
            ],
            answer="submit",
        )
        # Measurable Nodes
        self.measurable_nodes = [
            node.name
            for node in self.scm.nodes.values()
            if node.accessibility
            in (NodeAccessibility.MEASURABLE, NodeAccessibility.CONTROLLABLE)
        ]
        # Random States for experiments
        self.random_states: dict[str | tuple, np.random.RandomState] = {}

        # Mount mission
        self.mission.mount(self.scm)

    def run(self) -> None:
        self.budget.start_time()
        for r in range(1, self.budget.rounds_limit + 1):
            # (Budget) Check time budget
            self.budget.check_time()

            # (Transcript) New entry
            transcript_entry = TranscriptEntry(round=r)
            self.transcript.entries.append(transcript_entry)

            # (Hook) Tranacription start
            self.hook_manager.trigger(
                HookEvent.TRANSCRIPTION_START, context=transcript_entry
            )
            # (Hook) Round start
            self.hook_manager.trigger(HookEvent.ROUND_START)

            # (Budget) Pause timer while triggering hooks
            self.budget.pause_time()

            # (Hook) Before act
            self.hook_manager.trigger(HookEvent.BEFORE_ACT)

            # (Budget) Resume timer before asking agent for action
            self.budget.resume_time()

            # Ask agent for action
            decision: Decision = self.agent.act(
                round_info=RoundInfo(round=r, budget_state=self.budget.snapshot()),
                available_actions=self.available_actions,
            )

            # (Transcript) Add decision
            transcript_entry.decision = decision

            # Get partial submission from agent
            answer = self.agent.answer()

            # (Transcript) Add answer
            transcript_entry.result = answer

            # (Budget) Pause timer while triggering hooks
            self.budget.pause_time()

            # (Hook) After act
            self.hook_manager.trigger(HookEvent.AFTER_ACT)
            # (Hook) Before eval
            self.hook_manager.trigger(HookEvent.BEFORE_EVAL)

            # Apply decision
            samples_collection = self._apply_decision(decision)
            # Evaluate run
            feedback = self._get_feedback(self.transcript)

            # (Transcript) Add samples collection
            transcript_entry.samples_collection = samples_collection
            # (Transcript) Add feedback
            transcript_entry.feedback = feedback

            # (Hook) After eval
            self.hook_manager.trigger(HookEvent.AFTER_EVAL)

            # (Hook) Before inform
            self.hook_manager.trigger(HookEvent.BEFORE_INFORM)

            # (Budget) Resume timer before informing agent
            self.budget.resume_time()

            # Filter samples collection to only include measurable nodes
            filtered_samples_collection = SamplesCollection()
            if samples_collection is not None:
                for samples in samples_collection:
                    new_samples = Samples(
                        kind=samples.kind,
                        n=samples.n,
                        data=samples.data[self.measurable_nodes],
                        interventions=samples.interventions,
                    )
                    filtered_samples_collection.append(new_samples)

            # Inform agent
            self.agent.inform(filtered_samples_collection, feedback)

            # (Budget) Pause timer while triggering hooks
            self.budget.pause_time()

            # (Hook) After inform
            self.hook_manager.trigger(HookEvent.AFTER_INFORM)

            # (Budget) Charge round
            self.budget.tick_round()
            # (Budget) Charge samples used
            self.budget.charge_samples(samples_collection.total_n())
            # (Budget) Charge memory used
            self.budget.charge_memory(samples_collection.total_bytes())

            # (Transcript) Add budget snapshot
            transcript_entry.budget_snapshot = self.budget.snapshot()

            # self.logger.warning(f"Budget Snapshot: {transcript_entry.budget_snapshot}")

            # (Hook) New budget snapshot
            self.hook_manager.trigger(HookEvent.BUDGET_SNAPSHOT)

            # (Hook) Round end
            self.hook_manager.trigger(HookEvent.ROUND_END)

            # Check if done
            if decision.kind == ActionKind.ANSWER:
                break

        self.hook_manager.trigger(HookEvent.ROUND_END)

    def _apply_decision(self, decision: Decision) -> SamplesCollection | None:
        if decision.kind == ActionKind.ANSWER:
            return None

        collection: list[Samples] = []

        for experiment in decision.experiments:
            treatment, n = experiment.treatment, experiment.n
            self._validate_experiment(experiment)
            # Hash the treatment to use as a key for random state
            hashed = tuple(sorted(treatment.items())) if treatment else "observational"
            if hashed not in self.random_states:
                # Create a new random state for this treatment
                seed = zlib.crc32(str(hashed).encode())
                rs_base = np.random.RandomState(seed)
                self.random_states[hashed] = (
                    self.scm.prepare_new_random_state_structure(rs_base)
                )

            # Generate samples using the hashed random state
            samples = self.scm.generate_samples(
                interventions=treatment,
                num_samples=n,
                random_state=self.random_states[hashed],
            )

            collection.append(
                Samples(
                    kind=(
                        "observational"
                        if hashed == "observational"
                        else "interventional"
                    ),
                    n=n,
                    data=samples,
                    interventions=treatment,
                )
            )

        return SamplesCollection(collection)

    def _validate_experiment(self, experiment: ExperimentSpec) -> bool:
        if experiment.treatment is None:
            return True  # Observational study

        for treatment in experiment.treatment.items():
            # Vairable name and value
            name, value = treatment
            # Check controllability
            assert self.scm.nodes[name].accessibility == NodeAccessibility.CONTROLLABLE
            # Check domain
            low, high = self.scm.nodes[name].domain
            if value < low or value > high:
                raise ValueError
        return True

    def _get_feedback(self, transcript: Transcript) -> Feedback:
        feedback = Feedback()
        # Mission metrics
        behavior_score, result_score = self.mission.evaluate(transcript)
        feedback.behavior = behavior_score
        feedback.result = result_score
        # Get current answer
        result = transcript.entries[-1].result
        # Custom metrics
        custom_metrics_scores: dict[str, float] = {}
        for metric in self.custom_metrics:
            # Check if metric is behavioral or result
            if isinstance(metric, BehaviorMetric):
                score = metric.evaluate(transcript)
            elif isinstance(metric, ResultMetric):
                score = metric.evaluate(result)
            else:
                raise ValueError(f"Unknown metric type: {type(metric)}")
            # Add score to results
            custom_metrics_scores[metric.name] = score
        feedback.custom_metrics = custom_metrics_scores
        return feedback
