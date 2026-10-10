"""Exercise runtime lifecycle, isolation, failures and real artifact execution."""

from __future__ import annotations

import json
from unittest.mock import Mock

import networkx as nx
import pandas as pd
import pytest

from TheCausalityGame.core.contracts.dto.environment import SamplesCollection
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.core.runtime import game as game_module
from TheCausalityGame.core.runtime.game import Game
from TheCausalityGame.core.runtime.runner import Runner


class RecordingAgent:
    """Record boundary calls while returning a graph before and after learning."""

    id = "recording"

    def __init__(
        self, decision: Decision, *, submit_after_learning: bool = True
    ) -> None:
        self.decision = decision
        self.submit_after_learning = submit_after_learning
        self.calls: list[str] = []
        self.informed: list[SamplesCollection] = []

    def set_context(self, context: object) -> None:
        self.context = context

    def set_logger(self, logger: object) -> None:
        self.logger = logger

    def act(self, **kwargs: object) -> Decision:
        self.calls.append("act")
        return (
            Decision.answer()
            if self.informed and self.submit_after_learning
            else self.decision
        )

    def answer(self) -> nx.DiGraph:
        self.calls.append("answer")
        graph = nx.DiGraph()
        graph.add_nodes_from(["X", "L", "Y"])
        if self.informed:
            graph.add_edges_from([("X", "Y"), ("L", "Y")])
        return graph

    def inform(self, samples: SamplesCollection) -> None:
        self.calls.append("inform")
        self.informed.append(samples)


@pytest.fixture
def game_factory(monkeypatch: pytest.MonkeyPatch, tiny_problem: ProblemInstanceSpec):
    """Inject only the recording agent; keep SCM, mission and metrics real."""

    def create(agent: RecordingAgent, budget: BudgetSpec | None = None) -> Game:
        def build(spec: object):
            return agent if isinstance(spec, AgentSpec) else build_from_spec(spec)

        monkeypatch.setattr(game_module, "build_from_spec", build)
        logger = Mock()
        return Game(
            manifest_id=tiny_problem.id,
            agent_spec=tiny_problem.agents[0],
            scm_spec=tiny_problem.scm,
            mission_spec=tiny_problem.mission,
            custom_metrics_specs=[tiny_problem.mission.behavior_metric],
            budget_spec=budget or tiny_problem.run_plan.budget,
            hook_manager=Mock(),
            agent_logger=logger,
            game_logger=logger,
            environment_logger=logger,
        )

    return create


def test_lifecycle_filters_latents_and_scores_pre_update_answer(game_factory) -> None:
    """A submission captures the learned graph and no acquisition on its own round."""
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=4))
    game = game_factory(agent)
    transcript = game.run()

    assert not transcript.invalidated
    assert agent.calls == ["act", "answer", "inform"] * 2
    first, final = transcript.entries
    assert not first.result.edges
    assert set(final.result.edges) == {("X", "Y"), ("L", "Y")}
    assert first.feedback.result > final.feedback.result == 0.0
    assert first.feedback.custom_metrics
    assert set(agent.informed[0][0].data.columns) == {"X", "Y"}
    assert set(first.samples_collection[0].data.columns) == {"X", "L", "Y"}
    assert final.samples_collection is None
    assert not agent.informed[1]
    assert final.budget_snapshot.samples_left == 36
    events = [call.args[0] for call in game.hook_manager.trigger.call_args_list]
    assert events[0] == HookEvent.GAME_START
    assert events[-1] == HookEvent.GAME_END
    assert events.count(HookEvent.ROUND_START) == 2
    assert events.index(HookEvent.AFTER_INFORM) < events.index(HookEvent.BEFORE_EVAL)


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (
            Decision.experiment().add_experiment({"missing": 0.0}, n=1),
            "UnknownVariableError",
        ),
        (
            Decision.experiment().add_experiment({"L": 0.0}, n=1),
            "NonControllableVariableError",
        ),
        (
            Decision.experiment().add_experiment({"X": 2.0}, n=1),
            "ExperimentOutOfDomainError",
        ),
    ],
)
def test_invalid_intervention_is_reported_and_end_hook_runs(
    game_factory, decision, reason
) -> None:
    agent = RecordingAgent(decision)
    game = game_factory(agent)
    transcript = game.run()
    assert transcript.invalidated
    assert reason in transcript.invalidation_reason
    assert not agent.informed
    assert game.hook_manager.trigger.call_args.args[0] == HookEvent.GAME_END


def test_exact_sample_limit_succeeds_and_overspend_invalidates(game_factory) -> None:
    for n, invalidated in ((4, False), (5, True)):
        agent = RecordingAgent(Decision.experiment().add_experiment(None, n=n))
        game = game_factory(agent, BudgetSpec(rounds=2, samples=4))
        transcript = game.run()
        assert transcript.invalidated is invalidated
        if invalidated:
            assert "SamplesBudgetExceededError" in transcript.invalidation_reason
            assert not agent.informed
        else:
            assert transcript.entries[-1].budget_snapshot.samples_left == 0


def test_round_exhaustion_records_current_pre_update_answer(game_factory) -> None:
    """Protect the current timing contract without claiming a final re-estimation."""
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=2))
    transcript = game_factory(agent, BudgetSpec(rounds=1)).run()
    assert not transcript.invalidated
    assert len(transcript.entries) == 1
    assert not transcript.entries[-1].result.edges
    assert agent.informed[0].total_n() == 2


def test_end_hook_failure_invalidates_a_completed_game(game_factory) -> None:
    game = game_factory(RecordingAgent(Decision.answer()))

    def trigger(event, *args):
        if event == HookEvent.GAME_END:
            raise RuntimeError("end hook failed")

    game.hook_manager.trigger.side_effect = trigger
    transcript = game.run()
    assert transcript.invalidated
    assert transcript.invalidation_reason == "RuntimeError: end hook failed"


def test_time_exhaustion_uses_fake_clock(game_factory, monkeypatch) -> None:
    game = game_factory(
        RecordingAgent(Decision.answer()), BudgetSpec(rounds=1, time_s=1)
    )
    monkeypatch.setattr(game.environment.budget, "seconds_elapsed", lambda: 2.0)
    transcript = game.run()
    assert transcript.invalidated
    assert "TimeBudgetExceededError" in transcript.invalidation_reason
    assert not transcript.entries


def test_chunked_observations_preserve_the_same_treatment_stream(game_factory) -> None:
    """Separate fresh games must agree for one stream split into multiple batches."""
    chunked = game_factory(RecordingAgent(Decision.answer()))
    whole = game_factory(RecordingAgent(Decision.answer()))
    apply = chunked.environment._apply_decision
    pieces = [
        apply(Decision.experiment().add_experiment(None, n=n))[0].data for n in (3, 2)
    ]
    expected = whole.environment._apply_decision(
        Decision.experiment().add_experiment(None, n=5)
    )[0].data
    pd.testing.assert_frame_equal(pd.concat(pieces, ignore_index=True), expected)


@pytest.mark.parametrize(
    "execution,backend",
    [("sequential", "thread"), ("parallel", "thread"), ("parallel", "process")],
)
def test_real_runner_backends_write_isolated_equivalent_scientific_artifacts(
    tiny_problem, tmp_path, execution, backend
) -> None:
    data = tiny_problem.model_dump()
    data["run_plan"].update(execution=execution, parallel_backend=backend)
    problem = ProblemInstanceSpec.model_validate(data)
    runner = Runner(problem_instance=problem, run_dir=tmp_path)
    runner.run()
    transcripts = [
        json.loads(path.read_text())
        for path in sorted(runner.artifact_writer.agents_dir.glob("*/transcript.json"))
    ]
    assert len(transcripts) == 2
    assert all(not item["invalidated"] for item in transcripts)
    assert {item["agent_id"] for item in transcripts} == {"first", "second"}
    for a, b in zip(transcripts[0]["entries"], transcripts[1]["entries"], strict=True):
        assert a["samples_collection"] == b["samples_collection"]
        assert a["decision"] == b["decision"]
        assert a["feedback"] == b["feedback"]
    assert (runner.artifact_writer.run_dir / "provenance.json").is_file()


def test_runner_skips_inactive_agent_and_preserves_setup_failure(
    tiny_problem, tmp_path
) -> None:
    inactive = tiny_problem.agents[0].model_copy(update={"active": False})
    broken = tiny_problem.agents[1].model_copy(
        update={"class_": "missing.module:Agent"}
    )
    runner = Runner(
        problem_instance=tiny_problem.model_copy(update={"agents": [inactive, broken]}),
        run_dir=tmp_path,
    )
    runner.run()
    paths = list(runner.artifact_writer.agents_dir.glob("*/transcript.json"))
    assert len(paths) == 1
    transcript = json.loads(paths[0].read_text())
    assert transcript["agent_id"] == "second"
    assert transcript["invalidated"]
    assert transcript["invalidation_reason"]
    assert transcript["entries"] == []


def test_empty_runner_writes_provenance_without_agent_artifacts(
    tiny_problem, tmp_path
) -> None:
    runner = Runner(
        problem_instance=tiny_problem.model_copy(update={"agents": []}),
        run_dir=tmp_path,
    )
    runner.run()
    assert not list(runner.artifact_writer.agents_dir.iterdir())
    assert (runner.artifact_writer.run_dir / "provenance.json").is_file()
