"""The Causality Game - Test Serialized Noises."""

# Registry
from TheCausalityGame.core.infrastructure.registry import build_from_spec

# Noises
from TheCausalityGame.scm.noise.dirac import DiracNoiseDistribution

# Create noise distribution
noise = DiracNoiseDistribution(1.5)

# Serialize noise distribution to JSON
noise_json = noise.to_json()

# Deserialize noise distribution from JSON
noise_deserialized = build_from_spec(spec=noise_json)

# Generate noise values
fist = noise.generate(size=5)
second = noise_deserialized.generate(size=5)

# Check if both generated noises are the same
assert (fist == second).all()

print("Original and deserialized noise distributions are identical.")
