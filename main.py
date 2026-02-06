"""Generates problem instance files with different random states for The Causality Game."""

import copy
import json
import os

import numpy as np

from TheCausalityGame.core.lib.utils.random_state_serialization import random_state_to_json


class InvalidScmError(KeyError):
    """Raised when a JSON file does not contain a valid 'scm' object."""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"{file_path} does not contain a top-level 'scm' object")


input_path = os.path.join(os.path.dirname(__file__), "scripts/problem_instances")
output_directory = "rss"

te, te_files = "treatment_effect", ["hill.json", "hill_extended.json"]
gd, gd_files = "graph_discovery", ["rc_circuit.json"]
sd, sd_files = "scm_discovery", ["rc_circuit.json", "newton_second_law.json"]

rss = [
    (seed, np.random.RandomState(seed))
    for seed in (1, 42, 59, 34, 57, 91, 17, 83, 27, 99, 123, 473, 501, 701, 875, 101, 911)
]

for folder, files in [(te, te_files), (gd, gd_files), (sd, sd_files)]:
    output_folder = os.path.join(input_path, output_directory, folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in files:
        full_path = os.path.join(input_path, folder, filename)

        with open(full_path, encoding="utf-8") as f:
            json_content = json.load(f)

        if "scm" not in json_content or not isinstance(json_content["scm"], dict):
            raise InvalidScmError(full_path)

        for seed, rs in rss:
            modified_json = copy.deepcopy(json_content)
            modified_json["scm"]["random_state"] = random_state_to_json(rs)
            modified_json["id"] = f"rss/{modified_json['id']}"

            output_file = f"{os.path.splitext(filename)[0]}_rs-{seed}.json"
            output_path = os.path.join(output_folder, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(modified_json, out_f, indent=4)
