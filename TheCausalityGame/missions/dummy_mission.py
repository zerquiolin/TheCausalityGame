# missions/dummy_mission.py
from TheCausalityGame.core.contracts.dto import Action, MetricResult
from TheCausalityGame.core.contracts.mission import MissionContract


class DummyMission(MissionContract):
    """A minimal mission that considers success if guess == 42."""

    def evaluate(self, action: Action) -> MetricResult:
        result = action.result.get("guess") == 42
        return MetricResult(success=result, score=1.0 if result else 0.0)
