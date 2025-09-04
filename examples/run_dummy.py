# examples/run_dummy.py
from TheCausalityGame.agents.dummy_agent import DummyAgent
from TheCausalityGame.core.contracts.dto import RunPlan
from TheCausalityGame.core.runtime.game_runner import GameRunner
from TheCausalityGame.environments.dummy_env import DummyEnvironment
from TheCausalityGame.missions.dummy_mission import DummyMission
from TheCausalityGame.outputs.dummy_output import DummyOutput

if __name__ == "__main__":
    run_plan = RunPlan(
        mission_spec={"mission": "DummyMission"},
        agent_specs=[{"agent": "DummyAgent"}],
        environment_spec={"env": "DummyEnvironment"},
        metric_specs={
            "behavior": [{"metric": "DummyBehavior"}],
            "result": [{"metric": "DummyResult"}],
            "custom_metric_specs": [],
        },
    )

    runner = GameRunner(
        agents=[DummyAgent()],
        mission=DummyMission(),
        environment=DummyEnvironment(),
        outputs=[DummyOutput()],
        run_plan=run_plan,
    )
    runner.run()
