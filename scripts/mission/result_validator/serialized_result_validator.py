"""The Causality Game - Test Serialized DAGs."""

# Registry
from TheCausalityGame.core.infrastructure.registry import build_from_spec

# Result Validator
from TheCausalityGame.mission.result_validator.tef_validator import (
    TreatmentEffectFunctionValidator,
)

# Create Result Validator
validator = TreatmentEffectFunctionValidator()

# Serialize result validator to JSON
validator_json = validator.to_json()

# Deserialize result validator from JSON
validator_deserialized = build_from_spec(spec=validator_json)


# Test values
def fail_fn(X):
    return X**2


def succeed_fn(X, treatment, outcome, covariate_values):
    return 1


try:
    # Expect failure
    validator.validate(fail_fn)
    assert True, "Validation should have failed!"
except ValueError as err:
    # Expect success
    validator.validate(succeed_fn)
    assert True, "Validation should have succeeded!"


# Check if both methods are identical
assert validator.validate(succeed_fn) == validator_deserialized.validate(
    succeed_fn
), "Validators differ!"

# Check serialization
assert (
    validator_json == validator_deserialized.to_json()
), "Serialized Validators differ!"

print("Original and deserialized result validators's are identical.")
