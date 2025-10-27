from causalitygame.repository._base import get_scm_overview
from causalitygame.scm import SCM
from causalitygame.scm.nodes.abstract import BaseNumericSCMNode, BaseCategoricSCMNode
import logging
import json
import pytest


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

DATA_FOLDER_PATH = "causalitygame/data/scms"


@pytest.mark.parametrize(
    "name, filename",
    get_scm_overview(folder=DATA_FOLDER_PATH)[["name", "filename"]].values,
)
def test_usability_of_shipped_scms(name, filename):

    logger.info(f"Trying to load SCM {name} at file {filename}")
    with open(filename) as f:
        scm_as_dict = json.load(f)
        scm = SCM.from_dict(scm_as_dict)

        # test that we can draw samples from this SCM and that all values are in the domains
        sample = scm.generate_samples(num_samples=10)
        assert len(sample) == 10
        for _, obs in sample.iterrows():
            for var, val in obs.items():
                if isinstance(scm.nodes[var], BaseNumericSCMNode):
                    assert (
                        val >= scm.nodes[var].domain[0]
                    ), f"Invalid value for numerical var {var} with value {val} but must be at least {scm.nodes[var].domain[0]}"
                    assert (
                        val <= scm.nodes[var].domain[1]
                    ), f"Invalid value for numerical var {var} with value {val} but must be at most {scm.nodes[var].domain[1]}"
                elif isinstance(scm.nodes[var], BaseCategoricSCMNode):
                    assert (
                        val in scm.nodes[var].domain
                    ), f"Invalid value for categorical var {var} with value {val} but must be in {scm.nodes[var].domain}"
                else:
                    raise Exception(
                        f"{var} is of unknown type ({type(scm.nodes[var])})"
                    )

        # test that we can intervene at least one variable
        if len(scm.controllable_vars) == 0:
            logger.warning(
                f"The SCM with name {name} in file {filename} has no controllable variables defined by default."
            )


@pytest.mark.parametrize(
    "name, filename",
    get_scm_overview(folder=DATA_FOLDER_PATH)[["name", "filename"]].values,
)
def test_playability_of_shipped_scms(name, filename):
    pass
