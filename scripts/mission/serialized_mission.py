"""The Causality Game - Test Serialized DAGs."""

# Registry
from TheCausalityGame.core.infrastructure.registry import build_from_spec

# Mission
from TheCausalityGame.mission.conditional_average_treatment_effect import (
    ConditionalAverageTreatmentEffectMission,
)

# Metrics
from TheCausalityGame.mission.metric.behavior.rounds import RoundsBehaviorMetric
from TheCausalityGame.mission.metric.result.pehe import PEHEResultMetric

# Result Validator
from TheCausalityGame.mission.result_validator.tef_validator import (
    TreatmentEffectFunctionValidator,
)

# Create Behavior and Result Metrics
behavior = RoundsBehaviorMetric()
result = PEHEResultMetric()

# Create Result Validator
validator = TreatmentEffectFunctionValidator()

# Create Mission
mission = ConditionalAverageTreatmentEffectMission(
    behavior_metric=behavior,
    result_metric=result,
    result_validator=validator,
)

# Serialize mission to JSON
mission_json = mission.to_json()

# Deserialize mission from JSON
mission_deserialized = build_from_spec(mission_json)


# Check serialization
def deep_dict_equal(d1, d2, path=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            print(f"Key mismatch at {path}: {d1.keys()} vs {d2.keys()}")
            return False
        for key in d1:
            new_path = f"{path}.{key}" if path else key
            if not deep_dict_equal(d1[key], d2[key], new_path):
                return False
        return True

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"List length mismatch at {path}: {len(d1)} vs {len(d2)}")
            return False
        for index, (item1, item2) in enumerate(zip(d1, d2)):
            new_path = f"{path}[{index}]"
            if not deep_dict_equal(item1, item2, new_path):
                return False
        return True

    else:
        if d1 != d2:
            print(f"Value mismatch at {path}: {d1} vs {d2}")
            return False
        return True


deep_dict_equal(mission.to_dict(), mission_deserialized.to_dict())

print("All checks passed.")
