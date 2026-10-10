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


def test_lifecycle_filters_latents_and_scores_post_update_answer(game_factory) -> None:
    """A submission captures the learned graph and no acquisition on its own round."""
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=4))
    game = game_factory(agent)
    transcript = game.run()

    assert not transcript.invalidated
    assert agent.calls == ["act", "inform", "answer"] * 2
    first, final = transcript.entries
    assert set(first.result.edges) == {("X", "Y"), ("L", "Y")}
    assert set(final.result.edges) == {("X", "Y"), ("L", "Y")}
    assert first.feedback.result == final.feedback.result == 0.0
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


def test_exact_sample_limit_and_overspend_stop_before_extra_sampling(
    game_factory,
) -> None:
    for n in (4, 5):
        agent = RecordingAgent(Decision.experiment().add_experiment(None, n=n))
        game = game_factory(agent, BudgetSpec(rounds=2, samples=4))
        transcript = game.run()
        assert not transcript.invalidated
        assert transcript.termination_reason == "samples"
        if n == 5:
            assert not agent.informed
            assert not game.environment.random_states
            assert game.environment.budget.samples_used == 0
        else:
            assert transcript.entries[-1].budget_snapshot.samples_left == 0


def test_round_exhaustion_scores_the_last_acquired_batch(game_factory) -> None:
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=2))
    transcript = game_factory(agent, BudgetSpec(rounds=1)).run()
    assert not transcript.invalidated
    assert transcript.termination_reason == "rounds"
    assert len(transcript.entries) == 1
    assert set(transcript.final_result.edges) == {("X", "Y"), ("L", "Y")}
    assert transcript.final_feedback.result == 0
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
    assert not transcript.invalidated
    assert transcript.termination_reason == "time"
    assert transcript.final_feedback.result == 2.0
    assert transcript.forced_stop
    assert transcript.response_status == "provided"
    assert len(transcript.entries) == 1
    assert transcript.entries[0].round == 0


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
    from TheCausalityGame.core.infrastructure.artifact_contract import (
        validate_publication_transcript,
    )

    provenance = json.loads(
        (runner.artifact_writer.run_dir / "provenance.json").read_text()
    )
    for transcript in transcripts:
        validate_publication_transcript(transcript, provenance)


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


def test_intervention_streams_ignore_first_use_order_and_interleaving(
    game_factory,
) -> None:
    def draw(game, arms, n):
        return game.environment._apply_decision(
            Decision.experiment().add_experiment(arms, n=n)
        )[0].data

    first = game_factory(RecordingAgent(Decision.answer()))
    second = game_factory(RecordingAgent(Decision.answer()))
    a1 = draw(first, {"X": 0}, 3)
    b1 = draw(first, {"X": 1}, 5)
    a2 = draw(first, {"X": 0.0}, 2)
    b2 = draw(second, {"X": 1.0}, 5)
    whole_a = draw(second, {"X": 0}, 5)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(pd.concat([a1, a2], ignore_index=True), whole_a)


def test_rejected_multibatch_acquisition_consumes_no_rng_or_budget(
    game_factory,
) -> None:
    game = game_factory(
        RecordingAgent(Decision.answer()), BudgetSpec(rounds=2, samples=4)
    )
    decision = (
        Decision.experiment().add_experiment(None, n=3).add_experiment({"X": 0}, n=3)
    )
    from TheCausalityGame.core.lib.errors.budget import SamplesBudgetExceededError

    with pytest.raises(SamplesBudgetExceededError):
        game.environment._apply_decision(decision)
    assert not game.environment.random_states
    assert game.environment.budget.samples_used == 0


def test_invalid_second_experiment_fails_before_any_draw(game_factory) -> None:
    game = game_factory(RecordingAgent(Decision.answer()))
    decision = (
        Decision.experiment()
        .add_experiment(None, n=3)
        .add_experiment({"missing": 0}, n=1)
    )
    with pytest.raises(Exception, match="missing"):
        game.environment._apply_decision(decision)
    assert not game.environment.random_states


def test_acquisition_identity_survives_evaluator_sampling(game_factory) -> None:
    first = game_factory(RecordingAgent(Decision.answer()))
    second = game_factory(RecordingAgent(Decision.answer()))
    first.scm.generate_samples(num_samples=20)
    a = first.environment._apply_decision(
        Decision.experiment().add_experiment(None, n=5)
    )[0].data
    b = second.environment._apply_decision(
        Decision.experiment().add_experiment(None, n=5)
    )[0].data
    pd.testing.assert_frame_equal(a, b)


def test_time_limit_after_act_prevents_acquisition(game_factory, monkeypatch) -> None:
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=3))
    game = game_factory(agent, BudgetSpec(rounds=3, time_s=1))
    elapsed = {"seconds": 0.0}
    original = agent.act

    def act(**kwargs):
        elapsed["seconds"] = 2.0
        return original(**kwargs)

    agent.act = act
    monkeypatch.setattr(
        game.environment.budget, "seconds_elapsed", lambda: elapsed["seconds"]
    )
    transcript = game.run()
    assert not transcript.invalidated
    assert transcript.termination_reason == "time"
    assert not game.environment.random_states
    assert not agent.informed


def test_runner_rejects_duplicate_agent_ids(tiny_problem, tmp_path) -> None:
    problem = tiny_problem.model_copy(
        update={"agents": [tiny_problem.agents[0], tiny_problem.agents[0]]}
    )
    with pytest.raises(ValueError, match="unique"):
        Runner(problem_instance=problem, run_dir=tmp_path)


def test_financial_preflight_stops_without_sampling(game_factory) -> None:
    from TheCausalityGame.core.contracts.specs.acquisition import AcquisitionPricing

    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=3))
    budget = BudgetSpec(rounds=3, cost=5, pricing=AcquisitionPricing(per_sample=2))
    game = game_factory(agent, budget)
    transcript = game.run()
    assert not transcript.invalidated
    assert transcript.termination_reason == "cost"
    assert not game.environment.random_states
    assert game.environment.budget.cost_used == 0


def test_copied_answers_preserve_earlier_graphs(game_factory) -> None:
    agent = RecordingAgent(
        Decision.experiment().add_experiment(None, n=2), submit_after_learning=False
    )
    shared = nx.DiGraph()
    shared.add_nodes_from(["X", "L", "Y"])

    def answer():
        if len(agent.informed) == 2:
            shared.add_edge("X", "Y")
        return shared

    agent.answer = answer
    transcript = game_factory(agent, BudgetSpec(rounds=2)).run()
    assert not transcript.invalidated
    assert not transcript.entries[0].result.edges
    assert set(transcript.entries[1].result.edges) == {("X", "Y")}


def test_named_scm_seeds_change_samples_and_replay_identically(
    tiny_problem, tmp_path
) -> None:
    records = []
    for index, seed in enumerate([37, 37, 49]):
        runner = Runner(
            problem_instance=tiny_problem.model_copy(
                update={
                    "scm": tiny_problem.scm.model_copy(update={"random_state": None}),
                    "seeds": {"scm": seed},
                }
            ),
            run_dir=tmp_path / str(index),
        )
        transcript = runner._run_agent(tiny_problem.agents[0])
        assert not transcript.invalidated
        records.append(transcript.entries[0].samples_collection[0].data)
    pd.testing.assert_frame_equal(records[0], records[1])
    assert not records[0].equals(records[2])


def test_hook_payload_mutation_cannot_change_live_entry(tmp_path) -> None:
    from TheCausalityGame.core.contracts.dto.environment import Feedback
    from TheCausalityGame.core.contracts.dto.transcript import TranscriptEntry
    from TheCausalityGame.core.managers.hook import HookManager

    manager = HookManager([], tmp_path)

    class Mutator:
        def run(self, path, entry):
            entry.result.add_edge("X", "Y")
            entry.feedback.result = -100
            entry.custom_attributes["mutated"] = True

    manager.hooks_by_step[HookEvent.AFTER_EVAL].append(Mutator())
    graph = nx.DiGraph()
    graph.add_nodes_from(["X", "Y"])
    entry = TranscriptEntry(round=1, result=graph, feedback=Feedback(result=2))
    manager.trigger(HookEvent.AFTER_EVAL, entry)
    assert not entry.result.edges
    assert entry.feedback.result == 2
    assert not entry.custom_attributes


def test_real_cate_runtime_supports_timed_callable_validation(tmp_path) -> None:
    from pathlib import Path

    path = Path(
        "scripts/problem_instances_rebuttal/treatment_effect/honest_causal_tree/hill_rs-911.json"
    )
    manifest = json.loads(path.read_text())
    manifest["id"] = "cate_runtime_regression"
    manifest["agents"] = [manifest["agents"][0]]
    manifest["agents"][0]["inferer"] = {
        "class_": "TheCausalityGame.agent.inferers.empirical.outcome_regression:OutcomeRegressionInferer"
    }
    manifest["agents"][0]["decider"] = {
        "class_": "TheCausalityGame.agent.deciders.exhaustive:ExhaustiveDecider",
        "params": {"num_obs": 2, "num_inter": 2},
    }
    manifest["run_plan"]["budget"] = {"rounds": 2, "samples": 8}
    runner = Runner(
        problem_instance=ProblemInstanceSpec.model_validate(manifest), run_dir=tmp_path
    )
    transcript = runner._run_agent(runner.problem_instance.agents[0])
    assert not transcript.invalidated
    assert transcript.final_feedback.result is not None
    assert len(transcript.entries) == 2


@pytest.mark.parametrize("response", ["missing", "failed", "invalid"])
@pytest.mark.parametrize("initially_exhausted", [False, True])
def test_forced_stop_without_valid_response_has_no_score(
    game_factory, response, initially_exhausted
):
    class NonrespondingAgent(RecordingAgent):
        def answer(self):
            if response == "failed":
                raise RuntimeError("response unavailable")
            return None if response == "missing" else "invalid graph"

    agent = NonrespondingAgent(Decision.experiment().add_experiment(None, n=2))
    budget = BudgetSpec(rounds=1, samples=0 if initially_exhausted else 2)
    transcript = game_factory(agent, budget).run()
    assert not transcript.invalidated
    assert transcript.forced_stop
    assert transcript.final_feedback.result is None
    assert transcript.final_result is None
    assert transcript.response_status == (
        "missing" if response == "missing" else "failed"
    )
    if response != "missing":
        assert transcript.response_error


def test_serialized_scm_state_wins_over_conflicting_manifest_seed(
    tiny_problem, tmp_path
):
    import numpy as np

    from TheCausalityGame.core.lib.utils.random_state_serialization import (
        random_state_to_json,
    )

    tiny_problem = tiny_problem.model_copy(
        update={
            "scm": tiny_problem.scm.model_copy(
                update={
                    "random_state": random_state_to_json(np.random.RandomState(150))
                }
            )
        }
    )
    states = []
    for seed in [37, 49]:
        runner = Runner(
            problem_instance=tiny_problem.model_copy(update={"seeds": {"scm": seed}}),
            run_dir=tmp_path / str(seed),
        )
        transcript = runner._run_agent(tiny_problem.agents[0])
        assert not transcript.invalidated
        states.append(transcript.entries[0].samples_collection[0].data)
    pd.testing.assert_frame_equal(states[0], states[1])


@pytest.mark.parametrize("manifest_seed", [42, 49])
def test_configured_component_seed_wins_over_manifest_fallback(
    tiny_problem, manifest_seed
):
    from TheCausalityGame.core.contracts.specs.decider import DeciderSpec

    agent = tiny_problem.agents[0].model_copy(
        update={
            "decider": DeciderSpec(
                class_="TheCausalityGame.agent.deciders.cho:Cho2016ActiveGBNDecider",
                params={"seed": 37},
            )
        }
    )
    logger = Mock()
    game = Game(
        manifest_id=tiny_problem.id,
        agent_spec=agent,
        scm_spec=tiny_problem.scm,
        mission_spec=tiny_problem.mission,
        custom_metrics_specs=[],
        budget_spec=tiny_problem.run_plan.budget,
        hook_manager=Mock(),
        agent_logger=logger,
        game_logger=logger,
        environment_logger=logger,
        seeds={"decider": manifest_seed, "agent": 911},
    )
    assert game.agent.decider.seed == 37


def test_game_cannot_reuse_terminal_episode_state(game_factory):
    """Starting another episode requires constructing fresh agent/SCM state."""
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=2))
    game = game_factory(agent)
    transcript = game.run()
    rounds = len(transcript.entries)
    calls = list(agent.calls)
    with pytest.raises(RuntimeError, match="already"):
        game.run()
    assert len(transcript.entries) == rounds
    assert agent.calls == calls
    assert not transcript.invalidated


def test_fresh_games_do_not_share_scm_domain_configuration(game_factory):
    first = game_factory(RecordingAgent(Decision.answer()))
    second = game_factory(RecordingAgent(Decision.answer()))
    first.scm.nodes["X"].domain[0] = -999
    assert second.scm.nodes["X"].domain == [-1.0, 1.0]


def test_wrong_backend_sample_count_is_not_delivered_to_agent(
    game_factory, monkeypatch
):
    agent = RecordingAgent(Decision.experiment().add_experiment(None, n=4))
    game = game_factory(agent)
    original = game.scm.generate_samples
    monkeypatch.setattr(
        game.scm,
        "generate_samples",
        lambda *args, **kwargs: original(*args, **kwargs).iloc[:1],
    )
    transcript = game.run()
    assert (
        transcript.invalidated
    ), "A backend returning fewer rows silently corrupts accounting"
    assert not agent.informed


@pytest.mark.parametrize("warm_stream", [False, True])
@pytest.mark.parametrize("same_arm", [False, True])
def test_failed_acquisition_rolls_back_rng_and_all_budget_counters(
    game_factory, monkeypatch, same_arm, warm_stream
):
    decision = (
        Decision.experiment()
        .add_experiment(None, n=3)
        .add_experiment(None if same_arm else {"X": 0.5}, n=4)
    )
    game = game_factory(RecordingAgent(decision))
    fresh = game_factory(RecordingAgent(decision))
    import pickle

    if warm_stream:
        warmup = Decision.experiment().add_experiment(None, n=1)
        game.environment._apply_decision(warmup)
        fresh.environment._apply_decision(warmup)
    before = pickle.dumps(game.environment.random_states)
    original = game.scm.generate_samples
    calls = 0

    def fail_after_drawing_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("injected sampling failure")
        return result

    monkeypatch.setattr(game.scm, "generate_samples", fail_after_drawing_second_batch)
    with pytest.raises(RuntimeError, match="injected"):
        game.environment._apply_decision(decision)
    budget = game.environment.budget
    assert (budget.samples_used, budget.experiments_used, budget.cost_used) == (
        int(warm_stream),
        int(warm_stream),
        0,
    )
    assert (
        pickle.dumps(game.environment.random_states) == before
    ), "Failed acquisition installed or advanced live streams"
    monkeypatch.setattr(game.scm, "generate_samples", original)
    retried = game.environment._apply_decision(decision)
    expected = fresh.environment._apply_decision(decision)
    for actual_batch, expected_batch in zip(retried, expected, strict=True):
        pd.testing.assert_frame_equal(actual_batch.data, expected_batch.data)
    assert budget.samples_used == 7 + int(warm_stream)
    assert budget.experiments_used == 2 + int(warm_stream)
