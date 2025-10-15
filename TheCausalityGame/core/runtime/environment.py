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
from TheCausalityGame.core.infraestructure.budgets import (
    BudgetEnforcer,
    BudgetExceededError,
)
from TheCausalityGame.core.infraestructure.decisions import Decision, ExperimentSpec
from TheCausalityGame.core.lib.enum.hooks import HookEvent
from TheCausalityGame.core.managers.hook import HookManager


class Environment:
    def __init__(
        self,
        agent: Agent,
        scm: SCM,
        mission: Mission,
        custom_metrics: list[Metric],
        budget_spec: BudgetSpec,
        hook_manager: HookManager,
        logger,
    ):
        # Agent
        self.agent = agent
        # SCM
        self.scm = scm
        # Mission
        self.mission = mission
        # Metrics
        self.custom_metrics = custom_metrics
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
                if node.accessibility == "controllable" and node.domain is not None
            ],
            answer="submit",
        )
        # Measurable Nodes
        self.measurable_nodes = [
            node.name
            for node in self.scm.nodes.values()
            if node.accessibility in ("measurable", "controllable")
        ]
        # Random States for experiments
        self.random_states: dict[str | tuple, np.random.RandomState] = {}

        # Mount mission
        self.mission.mount(self.scm)

    def run(self) -> list[TranscriptEntry]:
        transcript: list[TranscriptEntry] = []

        try:
            self.budget.start_time()
            for r in range(1, self.budget.rounds_limit + 1):
                # (Budget) Check time budget
                self.budget.check_time()

                # (Transcript) New entry
                transcript_entry = TranscriptEntry(round=r)
                transcript.append(transcript_entry)

                # (Budget) Pause timer while triggering hooks
                self.budget.pause_time()

                # (Hook) Round start
                self.hook_manager.trigger(HookEvent.ROUND_START, transcript[-1])
                # (Hook) Before act
                self.hook_manager.trigger(HookEvent.BEFORE_ACT)

                # (Budget) Resume timer before asking agent for action
                self.budget.resume_time()

                # Ask agent for action
                decision: Decision = self.agent.act(
                    round_info=RoundInfo(
                        round_number=r, budget_state=self.budget.snapshot()
                    ),
                    available_actions=self.available_actions,
                )

                # (Transcript) Add decision
                transcript_entry.decision = decision

                # Get partial submission from agent
                answer = self.agent.answer()

                # (Transcript) Add answer
                transcript_entry.answer = answer

                # (Budget) Pause timer while triggering hooks
                self.budget.pause_time()

                # (Hook) After act
                self.hook_manager.trigger(HookEvent.AFTER_ACT)
                # (Hook) Before eval
                self.hook_manager.trigger(HookEvent.BEFORE_EVAL)

                # Apply decision
                samples_collection = self._apply_decision(decision)
                # Evaluate run
                feedback = self._get_feedback(transcript)

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

                # (Hook) New snapshot
                self.hook_manager.trigger(HookEvent.NEW_SNAPSHOT)

                # (Hook) Round end
                self.hook_manager.trigger(HookEvent.ROUND_END)

                # Check if done
                if decision.kind == "answer":
                    break

        except BudgetExceededError as e:
            self.logger.warning(f"Budget exceeded: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.hook_manager.trigger(HookEvent.ROUND_END)

        return transcript

    def _apply_decision(self, decision: Decision) -> SamplesCollection | None:
        if decision.kind == "answer":
            return None

        collection: list[Samples] = []

        for experiment in decision.experiments:
            treatment, n = experiment.treatment, experiment.n
            # Check experiment validity
            self._validate_experiment(treatment)
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
        for treatment in experiment.treatment.items():
            # Vairable name and value
            name, value = treatment
            # Check controllability
            if self.scm.nodes[name].accessibility != "controllable":
                raise ValueError
            # Check domain
            low, high = self.scm.nodes[name].domain
            if value < low or value > high:
                raise ValueError
        return True

    def _get_feedback(self, transcript: Transcript) -> Feedback:
        feedback = Feedback()
        # Mission metrics
        behavior_score, result_score = self.mission.evaluate(Transcript)
        feedback.behavior = behavior_score
        feedback.result = result_score
        # Get current answer
        answer = transcript.entries[-1].answer
        # Custom metrics
        custom_metrics_scores: dict[str, float] = {}
        for metric in self.custom_metrics:
            # Check if metric is behavioral or result
            if isinstance(metric, BehaviorMetric):
                score = metric.evaluate(transcript)
            elif isinstance(metric, ResultMetric):
                score = metric.evaluate(answer)
            else:
                raise ValueError(f"Unknown metric type: {type(metric)}")
            # Add score to results
            custom_metrics_scores[metric.name] = score
        feedback.custom_metrics = custom_metrics_scores
        return feedback
