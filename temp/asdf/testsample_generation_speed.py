import pytest
import json
from causalitygame.scm.abstract import SCM
from pathlib import Path
from time import time
import pandas as pd

DATA_FOLDER_PATH = "causalitygame/data"

base_path = Path(f"{Path(__file__).parent.parent}/{DATA_FOLDER_PATH}/scms/")


@pytest.mark.parametrize(
    "filename",
    [
        f.stem
        for f in Path(f"{base_path}/physics").rglob("*")
        if f.is_file() and str(f).endswith(".json")
    ],
)
def test_physical_problem_sample_generation_speed(filename):

    # load SCM
    with open(f"{base_path}/physics/{filename}.json") as f:
        scm = SCM.from_dict(json.load(f))

    # generate 1000 data points
    n_samples = 10**6
    t_start = time()
    df_data = scm.generate_samples(num_samples=n_samples)
    t_end = time()
    assert len(df_data) == n_samples
    assert (
        t_end - t_start < 1
    ), f"Runtime for {n_samples} must be less than a second but was {t_end - t_start}s"
