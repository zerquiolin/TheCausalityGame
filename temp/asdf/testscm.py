import json
import pytest
from causalitygame.generators.scm_generator import EquationBasedSCMGenerator
from causalitygame.scm.abstract import SCM
from causalitygame.scm.dags.DAG import DAG
from causalitygame.generators.dag_generator import DAGGenerator
from causalitygame.lib.utils.random_state_serialization import (
    random_state_from_json,
    random_state_to_json,
)
from scipy.stats import norm, uniform
import numpy as np
from causalitygame.scm.noises import (
    GaussianNoiseDistribution,
    UniformNoiseDistribution,
)

import logging

DATA_FOLDER_PATH = "causalitygame/data"


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
test_dag_a = DAGGenerator(
    num_nodes=10,
    num_roots=2,
    num_leaves=2,
    edge_density=0.5,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=1,
    max_path_length=4,
    random_state=np.random.RandomState(42),
).generate()
test_dag_b = DAGGenerator(
    num_nodes=3,
    num_roots=1,
    num_leaves=1,
    edge_density=0.2,
    max_in_degree=1,
    max_out_degree=1,
    min_path_length=1,
    max_path_length=3,
    random_state=np.random.RandomState(42),
).generate()


def assert_dicts_equal(d1, d2, path="", msg="", atol=None):
    for key in d1:
        assert key in d2, f"Key '{path + key}' missing in second dict"
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            assert_dicts_equal(d1[key], d2[key], path=path + key + ".", msg=msg)
        else:
            if type(d1[key]) in [float, np.float64] and atol is not None:
                assert np.isclose(
                    d1[key], d2[key], atol=atol
                ), f"{msg}\nValue mismatch at '{path + key}': {d1[key]} != {d2[key]}"
            else:
                assert (
                    d1[key] == d2[key]
                ), f"{msg}\nValue mismatch at '{path + key}': {d1[key]} != {d2[key]}"
    for key in d2:
        assert key in d1, f"{msg}\nKey '{path + key}' missing in first dict"


def check_sample_equivalence(samples_a, samples_b, msg=""):
    assert samples_a.shape == samples_b.shape, msg
    assert all(samples_a.columns == samples_b.columns), msg

    for (
        sample_a,
        sample_b,
    ) in zip(samples_a.values, samples_b.values):
        assert np.array_equal(
            sample_a, sample_b
        ), f"{msg}\n\tSample mismatch between {sample_a} and {sample_b}"


@pytest.mark.parametrize(
    "dag, num_nodes",
    [(test_dag_a, len(test_dag_a.nodes)), (test_dag_b, len(test_dag_b.nodes))],
)
def test_scm_generator(dag, num_nodes):
    """Test if SCMGenerator correctly maps DAG to SCM."""

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

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
        random_state=np.random.RandomState(42),
    )

    scm = scm_generator.generate()

    assert isinstance(scm, SCM), "Output should be an instance of SCM."
    assert (
        len(scm.nodes) == num_nodes
    ), "SCM should have the same number of nodes as DAG."


@pytest.mark.parametrize(
    "dag, num_nodes, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 123),
        (test_dag_b, len(test_dag_b.nodes), 911),
    ],
)
def test_scm_reproducibility(dag, num_nodes, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """
    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator_a = EquationBasedSCMGenerator(
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
        random_state=np.random.RandomState(seed),
    )
    scm_a = scm_generator_a.generate()

    scm_generator_b = EquationBasedSCMGenerator(
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
        random_state=np.random.RandomState(seed),
    )
    scm_b = scm_generator_b.generate()

    # Check if the SCMs are equal
    nodes_a = [node.to_dict() for node in scm_a.nodes.values()]
    nodes_b = [node.to_dict() for node in scm_b.nodes.values()]

    assert nodes_a == nodes_b, "SCMs should be equal with the same seed."


@pytest.mark.parametrize(
    "dag, num_nodes, seed1, seed2",
    [
        (test_dag_a, len(test_dag_a.nodes), 123, 321),
        (test_dag_b, len(test_dag_b.nodes), 911, 119),
    ],
)
def test_scm_variability(dag, num_nodes, seed1, seed2):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """
    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator_a = EquationBasedSCMGenerator(
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
        random_state=np.random.RandomState(seed1),
    )
    scm_a = scm_generator_a.generate()

    scm_generator_b = EquationBasedSCMGenerator(
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
        random_state=np.random.RandomState(seed2),
    )
    scm_b = scm_generator_b.generate()

    # Check if the SCMs are not equal
    nodes_a = [node.to_dict() for node in scm_a.nodes.values()]
    nodes_b = [node.to_dict() for node in scm_b.nodes.values()]

    assert nodes_a != nodes_b, "SCMs should not be equal with the different seeds."


@pytest.mark.parametrize(
    "dag, num_nodes, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 123),
        (test_dag_b, len(test_dag_b.nodes), 911),
    ],
)
def test_scm_serialization(dag, num_nodes, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

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
        random_state=np.random.RandomState(seed),
        num_samples_for_cdf_generation=10,
        logger=logger,
    )
    scm = scm_generator.generate()

    # Serialize the SCM
    scm_data = scm.to_dict()

    serializable = False

    try:
        json.dumps(scm_data)
        serializable = True
    except:
        pass

    assert serializable, "SCM should be serializable."


@pytest.mark.parametrize(
    "dag, num_nodes, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 123),
        (test_dag_b, len(test_dag_b.nodes), 911),
    ],
)
def test_scm_deserialization(dag, num_nodes, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

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
        random_state=np.random.RandomState(seed),
        num_samples_for_cdf_generation=10,
        logger=logger,
    )

    logger.debug("Generating SCM")
    scm = scm_generator.generate()

    # Serialize the SCM
    logger.debug("Converting SCM to dictionary")
    scm_data = scm.to_dict()

    # Deserialize the SCM
    logger.debug("Recovering SCM from dictionary")
    scm_deserialized = SCM.from_dict(scm_data)

    # simulate JSON serialization
    logger.debug("Marshalling and unmarshalling dictionary to json")
    scm_deserialized_json = SCM.from_dict(json.loads(json.dumps(scm_data)))

    # Check if the SCMs are not equal
    for node_a, node_b, node_c in zip(
        sorted(scm.nodes.values(), key=lambda n: n.name),
        sorted(scm_deserialized.nodes.values(), key=lambda n: n.name),
        sorted(scm_deserialized_json.nodes.values(), key=lambda n: n.name),
    ):
        assert node_a.name == node_b.name
        assert node_a.name == node_c.name
        assert node_a.accessibility == node_b.accessibility
        assert node_a.accessibility == node_c.accessibility
        assert_dicts_equal(
            node_a.to_dict(),
            node_b.to_dict(),
            msg=f"SCM node {node_a.name} ({type(node_a)}) is not the same after recovering via to_dict and from_dict.",
        )
        assert_dicts_equal(
            node_b.to_dict(),
            node_c.to_dict(),
            msg=f"SCM node {node_a.name} ({type(node_a)}) is not the same after recovering via json.dumps and json.loads.",
        )

    # Check if the random states are equal
    state_a = scm.get_random_state().get_state()
    state_b = scm_deserialized.get_random_state().get_state()
    state_c = scm_deserialized_json.get_random_state().get_state()

    for a, b, c in zip(state_a, state_b, state_c):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b) and np.array_equal(
                b, c
            ), "Random state arrays should be equal after serialization."
        else:
            assert a == b == c, "Random states should be equal after serialization."

    # Generate samples
    samples_a = scm.generate_samples(num_samples=10)
    samples_b = scm_deserialized.generate_samples(num_samples=10)
    samples_c = scm_deserialized_json.generate_samples(num_samples=10)
    check_sample_equivalence(samples_a, samples_b)
    check_sample_equivalence(samples_a, samples_c)


@pytest.mark.parametrize(
    "dag, num_nodes, num_samples, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 10, 123),
        (test_dag_b, len(test_dag_b.nodes), 21, 911),
    ],
)
def test_scm_samples_reproducibility(dag, num_nodes, num_samples, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

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
        random_state=np.random.RandomState(seed),
        num_samples_for_cdf_generation=10,
    )
    scm = scm_generator.generate()
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
        random_state=np.random.RandomState(seed),
        num_samples_for_cdf_generation=10,
    )
    scm_b = scm_generator.generate()

    # Generate samples
    samples_a = scm.generate_samples(num_samples=num_samples)
    samples_b = scm_b.generate_samples(num_samples=num_samples)

    check_sample_equivalence(samples_a, samples_b)


@pytest.mark.parametrize(
    "network_path",
    [
        (f"{DATA_FOLDER_PATH}/scms/literature_cases/small/cancer.json"),
        (f"{DATA_FOLDER_PATH}/scms/literature_cases/small/earthquake.json"),
    ],
)
def test_bayesian_network_scm_deserialization(network_path):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """
    logger.debug("Recoverin SCM from JSON file")
    # Load the JSON file
    with open(network_path, "r") as f:
        scm_data = json.load(f)
        scm_data["random_state"] = random_state_to_json(np.random.RandomState(0))

    logger.debug("Recovering SCM (A) from JSON file")
    scm_deserialized_a = SCM.from_dict(scm_data)

    # simulate JSON serialization
    logger.debug("Recovering SCM (B) from JSON file")
    scm_deserialized_b = SCM.from_dict(scm_data)

    # Check if the SCMs are not equal
    for node_a, node_b in zip(
        sorted(scm_deserialized_a.nodes.values(), key=lambda n: n.name),
        sorted(scm_deserialized_b.nodes.values(), key=lambda n: n.name),
    ):
        assert node_a.name == node_b.name
        assert node_a.accessibility == node_b.accessibility
        assert_dicts_equal(
            node_a.to_dict(),
            node_b.to_dict(),
            msg=f"SCM node {node_a.name} ({type(node_a)}) is not the same after recovering via to_dict and from_dict.",
        )

    logger.debug("Checking if the random states are equal")
    # Check if the random states are equal
    state_a = scm_deserialized_a.get_random_state().get_state()
    state_b = scm_deserialized_b.get_random_state().get_state()

    for a, b in zip(state_a, state_b):
        if isinstance(a, np.ndarray):
            assert np.array_equal(
                a, b
            ), "Random state arrays should be equal after serialization."
        else:
            assert a == b, "Random states should be equal after serialization."

    logger.debug("Checking if the samples are equal")
    # Generate samples
    samples_a = scm_deserialized_a.generate_samples(num_samples=10)
    samples_b = scm_deserialized_b.generate_samples(num_samples=10)
    check_sample_equivalence(
        samples_a,
        samples_b,
        msg=f"Sample of original SCM and SCM via to_dict and from_dict is not the same.",
    )
