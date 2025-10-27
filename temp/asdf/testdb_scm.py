import pandas as pd
import numpy as np
import json
import itertools as it

from causalitygame.generators.outcome.base import DummyOutcomeGenerator
from causalitygame.scm.abstract import SCM
from causalitygame.scm.db import DatabaseSCM
import pytest

import networkx as nx
import logging

from time import time

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

DATA_FOLDER_PATH = "causalitygame/data"


@pytest.mark.parametrize(
    "path_to_csv, intervention_variables, outcome_variables, has_counterfactuals, overwrite_factual_outcomes",
    [
        [a[0], a[1], a[2], a[3], b]
        for a, b in it.product(
            [
                [
                    f"{DATA_FOLDER_PATH}/datasets/ihdp/ihdp.csv",
                    ["treat"],
                    ["YC"],
                    False,
                ],
                [
                    f"{DATA_FOLDER_PATH}/datasets/jobs/nsw.csv",
                    ["treatment"],
                    ["RE78"],
                    False,
                ],
                [
                    f"{DATA_FOLDER_PATH}/datasets/twins/twins.csv",
                    ["selection"],
                    ["mortality"],
                    True,
                ],
            ],
            [False, True],
        )
    ],
)
def test_functionality_of_db_scm(
    path_to_csv,
    intervention_variables,
    outcome_variables,
    has_counterfactuals,
    overwrite_factual_outcomes,
):

    factual_df = pd.read_csv(path_to_csv).head(1000)

    scm = DatabaseSCM(
        factual_df=factual_df,
        intervention_variables=intervention_variables,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=overwrite_factual_outcomes,
        outcome_generators={
            k: DummyOutcomeGenerator(constant=-1) for k in outcome_variables
        },
    )

    # check that topological ordering is correct
    assert factual_df.shape[1] == len(scm.dag.nodes)
    seen_treatments = []
    seen_outputs = []
    for n in nx.topological_sort(scm.dag.graph):
        if n in intervention_variables:
            seen_treatments.append(n)
        elif n in outcome_variables:
            seen_outputs.append(n)
        else:
            assert (
                not seen_treatments
            ), f"incorrect topological ordering. The covariate {n} comes after treatments {intervention_variables}"
            assert (
                not seen_outputs
            ), f"incorrect topological ordering. The covariate {n} comes after outputs {seen_outputs}"

    # check that the controllable variable is indeed controllable
    assert intervention_variables == scm.controllable_vars

    # check whether all data points are contained in the original database if we do not intervene
    for row in scm.generate_samples(num_samples=10**1).values:
        entry_contained_in_original_data = np.any(
            [np.all(row == orig_row) for orig_row in factual_df[scm.var_names].values]
        )
        assert (
            overwrite_factual_outcomes and not entry_contained_in_original_data
        ) or (not overwrite_factual_outcomes and entry_contained_in_original_data)

    # check that we obtain counter-factuals when we intervene
    num_samples = 10**1  # this is a configuration
    num_matches = 0
    num_mismatches = 0
    possible_interventions = list(
        it.product(*[sorted(pd.unique(factual_df[v])) for v in intervention_variables])
    )
    logger.info(
        f"Testing on {len(possible_interventions)} possible interventions whether generated samples have the desired properties."
    )
    for intervention in possible_interventions:

        # generate samples for intervention
        logger.debug(
            f"Generating {num_samples} samples for intervention {intervention}."
        )
        samples = scm.generate_samples(
            interventions={
                var_name: var_val
                for var_name, var_val in zip(intervention_variables, intervention)
            },
            num_samples=num_samples,
        )

        # check samples
        logger.debug(f"Checking {len(samples)} samples for desirable properties.")
        for _, s in samples.iterrows():
            row = np.array([s[v] for v in scm.var_names])
            is_factual = np.any(
                [
                    np.all(row == orig_row)
                    for orig_row in factual_df[scm.var_names].values
                ]
            )
            if is_factual:
                num_matches += 1
                assert (
                    not overwrite_factual_outcomes
                ), f"There should be exclusively counter-factual entries in the database since outcomes are configured to be overwritten, but {row} is a factual contained in the database."
            else:
                num_mismatches += 1
                assert (
                    overwrite_factual_outcomes or not has_counterfactuals
                ), f"The database is marked as complete, so there should be no mismatches, but {row} does not occur in {path_to_csv}."

        # check that we have both factual entries (from the original data) and counter-factual entries
        if not has_counterfactuals:
            assert (
                num_mismatches >= 1
            ), "There should be at least 1 counter-factual entry not contained in the database even though factuals are not overwritten."


@pytest.mark.parametrize(
    "path_to_csv, intervention_variables, outcome_variables, has_counterfactuals, overwrite_factual_outcomes",
    [
        [a[0], a[1], a[2], a[3], b]
        for a, b in it.product(
            [
                [
                    f"{DATA_FOLDER_PATH}/datasets/ihdp/ihdp.csv",
                    ["treat"],
                    ["YC"],
                    False,
                ],
                [
                    f"{DATA_FOLDER_PATH}/datasets/jobs/nsw.csv",
                    ["treatment"],
                    ["RE78"],
                    False,
                ],
                [
                    f"{DATA_FOLDER_PATH}/datasets/twins/twins.csv",
                    ["selection"],
                    ["mortality"],
                    True,
                ],
            ],
            [False, True],
        )
    ],
)
def test_db_sample_generation_speed(
    path_to_csv,
    intervention_variables,
    outcome_variables,
    has_counterfactuals,
    overwrite_factual_outcomes,
):

    factual_df = pd.read_csv(path_to_csv).head(1000)

    scm = DatabaseSCM(
        factual_df=factual_df,
        intervention_variables=intervention_variables,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=overwrite_factual_outcomes,
        outcome_generators={
            k: DummyOutcomeGenerator(constant=-1) for k in outcome_variables
        },
    )

    for num_samples, allowed_time in [
        (1, 10**-1),
        (10, 10**-1),
        (100, 1),
        (1000, 2),
        (10**4, 20),
    ]:  # we must be able to generate 1000 instances in less than a second
        t_start = time()
        scm.generate_samples(num_samples=num_samples)
        runtime = time() - t_start
        assert (
            runtime < allowed_time
        ), f"Runtime to generate {num_samples} samples was {runtime} but must be under {allowed_time}s."
        logger.info(
            f"Generated {num_samples} samples in overall runtime of {runtime}s."
        )


@pytest.mark.parametrize(
    "path_to_csv, intervention_variables, outcome_variables, overwrite_factual_outcomes",
    [
        [a[0], a[1], a[2], b]
        for a, b in it.product(
            [
                [f"{DATA_FOLDER_PATH}/datasets/ihdp/ihdp.csv", ["treat"], ["YC"]],
                [f"{DATA_FOLDER_PATH}/datasets/jobs/nsw.csv", ["treatment"], ["RE78"]],
                [
                    f"{DATA_FOLDER_PATH}/datasets/twins/twins.csv",
                    ["selection"],
                    ["mortality"],
                ],
            ],
            [False, True],
        )
    ],
)
def test_serializability_of_scm(
    path_to_csv, intervention_variables, outcome_variables, overwrite_factual_outcomes
):

    factual_df = pd.read_csv(path_to_csv).head(1000)

    scm = DatabaseSCM(
        factual_df=factual_df,
        intervention_variables=intervention_variables,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=overwrite_factual_outcomes,
        outcome_generators={
            k: DummyOutcomeGenerator(constant=-1) for k in outcome_variables
        },
    )

    def test_df_equalness(df1, df2, msg=""):
        if df1.equals(df2):
            return
        if msg != "":
            msg += "\n"
        assert (
            df1.shape == df2.shape
        ), f"{msg}DataFrames have different shapes. First has {df1.shape}, second has {df2.shape}"
        assert list(df1.columns) == list(df2.columns), "Column names don't match"
        assert list(df1.index) == list(df2.index), "Indices don't match"
        for i in range(len(df1)):
            row1 = df1.iloc[i]
            row2 = df2.iloc[i]
            for field, v1 in row1.items():
                v2 = row2[field]
                assert (
                    v1 == v2
                ), f"Mismatch in field {field} in row {i} of datasets.\n\tFirst is {v1}\n\tSecond is {v2}"

        assert np.array_equal(df1.values, df2.values)

    for _ in range(
        2
    ):  # the 2nd loop iteration is to check whether the sample generation is also identical after there have been samples drawn before already

        if _ == 0:
            logger.info(
                "Checking whether SCM is correctly serialized and deserialized if no samples have been generated yet."
            )
        if _ == 1:
            logger.info(
                "Checking whether SCM is correctly serialized and deserialized if samples have been generated previously."
            )

        # recover from dict
        scm_recovered_inside_python = SCM.from_dict(scm.to_dict())
        scm_recovered_after_json = SCM.from_dict(json.loads(json.dumps(scm.to_dict())))

        # test that properties are equal
        logger.debug("Checking that properties are equal.")
        for other in [scm_recovered_inside_python, scm_recovered_after_json]:
            assert scm.controllable_vars == other.controllable_vars
            assert scm._outcome_vars == other._outcome_vars
            assert list(nx.topological_sort(scm.dag.graph)) == list(
                nx.topological_sort(other.dag.graph)
            )

        # test that dataframes are identical
        logger.debug("Checking that dataframes are equal.")
        test_df_equalness(
            scm.df,
            scm_recovered_inside_python.df,
            msg="DataFrame has changed after internal Python deserialization",
        )
        test_df_equalness(
            scm.df,
            scm_recovered_after_json.df,
            msg="Dataframe has changed after json.loads and json.dumps",
        )

        # test that all nodes of the same type
        for node_a, node_b, node_c in zip(
            scm.nodes, scm_recovered_inside_python.nodes, scm_recovered_after_json.nodes
        ):
            assert type(node_a) == type(
                node_b
            ), f"Node type has changed after Python internal serialization from {type(node_a)} to {type(node_b)}"
            assert type(node_a) == type(
                node_c
            ), f"Node type has changed after JSON serialization from {type(node_a)} to {type(node_c)}"

        # test that generated instances are the same
        logger.debug(f"Checking that sampled instances are equal.")
        num_samples = 10**1
        samples_a = scm.generate_samples(num_samples=num_samples)
        samples_b = scm_recovered_inside_python.generate_samples(
            num_samples=num_samples
        )
        samples_c = scm_recovered_after_json.generate_samples(num_samples=num_samples)
        for i, (sample_a, sample_b, sample_c) in enumerate(
            zip(samples_a, samples_b, samples_c)
        ):
            assert (
                sample_a == sample_b
            ), f"Mismatch between {i}-th sample of original SCM and SCM after Python internal deserialization"
            assert (
                sample_a == sample_c
            ), f"Mismatch between {i}-th sample of original SCM and SCM after JSON deserialization"
