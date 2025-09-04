from __future__ import annotations

from TheCausalityGame.core.infra.determinism import (
    hash_intervention_key,
    make_intervention_seed,
)


def test_intervention_hash_and_seed_stability() -> None:
    key1 = hash_intervention_key(
        base_seed=123,
        manifest_id="m1",
        agent_id="a1",
        round_index=0,
        interventions={"X": 1, "Z": 0},
        n=100,
    )
    key2 = hash_intervention_key(
        base_seed=123,
        manifest_id="m1",
        agent_id="a1",
        round_index=0,
        interventions={"Z": 0, "X": 1},  # different order, same mapping
        n=100,
    )
    assert key1 == key2

    seed1 = make_intervention_seed(
        base_seed=123,
        manifest_id="m1",
        agent_id="a1",
        round_index=0,
        interventions={"X": 1, "Z": 0},
        n=100,
    )
    seed2 = make_intervention_seed(
        base_seed=123,
        manifest_id="m1",
        agent_id="a1",
        round_index=0,
        interventions={"Z": 0, "X": 1},
        n=100,
    )
    assert seed1 == seed2
