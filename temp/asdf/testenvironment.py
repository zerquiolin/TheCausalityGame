from itertools import permutations
import json
import pytest
from causalitygame.evaluators.behavior import ExperimentsBehaviorMetric
from causalitygame.evaluators.deliverable import SHDDeliverableMetric
from causalitygame.game_engine.Environment import Environment
from causalitygame.game_engine.GameInstance import GameInstance
from causalitygame.generators.scm_generator import EquationBasedSCMGenerator
from causalitygame.generators.dag_generator import DAGGenerator
from causalitygame.missions.DAGInferenceMission import DAGInferenceMission
from causalitygame.scm.noises import (
    GaussianNoiseDistribution,
    UniformNoiseDistribution,
)
from scipy.stats import norm, uniform
import sympy as sp
import numpy as np

import logging

# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


# Test DAGs
dag_random_state = np.random.RandomState(911)
dag = DAGGenerator(
    num_nodes=10,
    num_roots=2,
    num_leaves=2,
    edge_density=0.5,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=1,
    max_path_length=4,
    random_state=dag_random_state,
).generate()

# Test SCM
num_nodes = 10
variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes)}
variable_types[f"X{num_nodes}"] = "categorical"
variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes)}
variable_domains[f"X{num_nodes}"] = ["A", "B", "C"]

scm_generator = EquationBasedSCMGenerator(
    dag=dag,
    variable_types=variable_types,
    variable_domains=variable_domains,
    user_constraints={"max_terms": 2},
    allowed_operations=["+", "*"],
    allowed_functions=[lambda x: x**2],
    noise_distributions=[
        GaussianNoiseDistribution(mean=0, std=1),
        UniformNoiseDistribution(low=-1, high=1),
    ],
    random_state=np.random.RandomState(911),
    num_samples_for_cdf_generation=10,
)
scm = scm_generator.generate()


# Test GameInstance
# Generate a mission
behavior_metric = ExperimentsBehaviorMetric()
deliverable_metric = SHDDeliverableMetric()
mission = DAGInferenceMission(
    behavior_metric=behavior_metric, deliverable_metric=deliverable_metric
)
game_instance_random_state = np.random.RandomState(911)
game_instance = GameInstance(100, scm, mission, game_instance_random_state)


@pytest.mark.parametrize(
    "game_instance, agent, random_state_seed, interventions",
    [
        (
            game_instance,
            None,
            911,
            [
                ("X10", "A"),
                ("X4", -1),
                ("X8", 0),
                ("X9", 1),
            ],  # Interventions X4 is 0 with a log
        )
    ],  # The agent is not used in the test
)
def test_that_sample_sequences_are_invariant_to_treatment_organization(
    game_instance, agent, random_state_seed, interventions
):
    """
    Test that the Environment is invariant to the agent.
    """
    # show nodes and available actions
    print("Nodes in the game instance:")
    for node_name, node in game_instance.scm.nodes.items():
        print(f" - {node_name} (type: {node.domain})")
    print("Available actions:")
    # Generate all permutations
    all_permutations = list(permutations(interventions))

    logger.info(f"Executing {sum([len(p) for p in all_permutations])} experiments")

    prev_final_state = None
    for perm in all_permutations:

        # Create the Environment using the GameInstance and Agent
        all_in_env = Environment(
            game_instance=game_instance,
            agent_name="dummy_agent",  # Use a dummy agent name
            agent=agent,
            random_state=np.random.RandomState(random_state_seed),
        )
        partitioned_env = Environment(
            game_instance=game_instance,
            agent_name="dummy_agent",  # Use a dummy agent name
            agent=agent,
            random_state=np.random.RandomState(random_state_seed),
        )

        def experiment(t):
            return [({var: val}, t)]

        for var, val in perm:

            # Perform the intervention 1x2 in the first and 2x1 in the second environment
            all_in_env.apply_action("experiment", experiment(2))
            partitioned_env.apply_action("experiment", experiment(1))
            partitioned_env.apply_action("experiment", experiment(1))

            # check that both environments have the same datasets now
            h1 = all_in_env.get_state()["datasets"]
            h2 = partitioned_env.get_state()["datasets"]

            logger.debug(f"Compare histories after {len(h1)} interventions.")
            assert (
                (var, val),
            ) in h1, (
                f"no intervention data for {var}={val} in state of first environment"
            )
            assert (
                (var, val),
            ) in h2, (
                f"no intervention data for {var}={val} in state of second environment"
            )
            assert sorted(h1.keys()) == sorted(
                h2.keys()
            ), "Histories have different keys"
            for treatment in h1.keys():
                for col in h1[treatment].columns:
                    assert h1[treatment][col].equals(
                        h2[treatment][col]
                    ), f"Different values for col {col} in experiment {treatment}:\n\th1: {list(h1[treatment][col])}\n\th2: {list(h2[treatment][col])}"
                assert h1[treatment].equals(h2[treatment])

        # check that the history is the same as for the previous permutation
        if prev_final_state is not None:
            cur_final_state_all_in = all_in_env.get_state()["datasets"]
            for exp, df_exp in prev_final_state.items():
                assert df_exp.equals(cur_final_state_all_in[exp])
        prev_final_state = all_in_env.get_state()["datasets"]

    logger.info("Done.")
