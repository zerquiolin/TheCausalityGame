"""Generate seeded problem-instance files for The Causality Game."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from TheCausalityGame.core.lib.utils.random_state_serialization import random_state_to_json


ROOT = Path(__file__).resolve().parent
PROBLEM_INSTANCES_ROOT = ROOT / "scripts" / "problem_instances"
PROBLEM_INSTANCES_BY_INFERER_ROOT = ROOT / "scripts" / "problem_instances_by_inferer"
SEEDED_ROOT = PROBLEM_INSTANCES_ROOT / "seeded"

# ---------------------------------------------------------------------------
# Edit this configuration to choose which source manifests and seeds generate.
# Mission/inferer values use the seeded folder names, e.g. graph_discovery/pc.
# File values are relative to the source root they come from:
#   - problem_instances_by_inferer/<mission>/<inferer>/<file>.json
#   - problem_instances/<mission>/<file>.json
# ---------------------------------------------------------------------------

# Generate only these mission folders. Leave empty to allow all missions.
INCLUDE_MISSIONS: set[str] = set()

# Skip these mission folders.
EXCLUDE_MISSIONS: set[str] = set()

# Generate only these mission/inferer folders. Leave empty to allow all inferers.
INCLUDE_INFERER_FOLDERS: set[str] = set()

# Skip these mission/inferer folders.
EXCLUDE_INFERER_FOLDERS: set[str] = set()

# Generate only these exact source files. Leave empty to allow all files.
INCLUDE_SOURCE_FILES: set[str] = set()

# Skip these exact source files.
EXCLUDE_SOURCE_FILES: set[str] = set()

# Generate only these seeds. Leave empty to allow all seeds.
INCLUDE_SEEDS: set[int] = set()

# Skip these seeds.
EXCLUDE_SEEDS: set[int] = set()

MISSION_IDS_BY_FOLDER = {
    "graph_discovery": {"dag_discovery_mission"},
    "treatment_effect": {"conditional_average_treatment_effect"},
    "scm_discovery": {"scm_estimation_mission"},
    "symbolic_scm_discovery": {"symbolic_scm_discovery"},
}

INFERER_BY_FOLDER = {
    "dag_discovery": "TheCausalityGame.agent.inferers.dag:DAGDiscoveryInferer",
    "lingam": "TheCausalityGame.agent.inferers.lingam:LiNGAMInferer",
    "notears": "TheCausalityGame.agent.inferers.notears:NOTEARSInferer",
    "pc": "TheCausalityGame.agent.inferers.pc:PCInferer",
    "honest_causal_tree": (
        "TheCausalityGame.agent.inferers.causal_tree:HonestCausalTreeInferer"
    ),
    "outcome_regression": (
        "TheCausalityGame.agent.inferers.outcome_regression:OutcomeRegressionInferer"
    ),
    "transformed_outcome": (
        "TheCausalityGame.agent.inferers.transformed_outcome:TransformedOutcomeInferer"
    ),
    "scm_discovery": "TheCausalityGame.agent.inferers.scm:SCMDiscoveryInferer",
    "sparse_symbolic_scm": (
        "TheCausalityGame.agent.inferers.symbolic_scm:SparseSymbolicSCMDiscoveryInferer"
    ),
}

DECIDERS_BY_MISSION_FOLDER = {
    "graph_discovery": {
        "TheCausalityGame.agent.deciders.cho:Cho2016ActiveGBNDecider",
        "TheCausalityGame.agent.deciders.tigas:Tigas2022CBEDDecider",
        "TheCausalityGame.agent.deciders.annadani:Annadani2024CAASLOnlineDecider",
        "TheCausalityGame.agent.deciders.geng:HeGeng2008MinimaxDecider",
        "TheCausalityGame.agent.deciders.gies:GIESDecider",
        "TheCausalityGame.agent.deciders.abci:ABCIDecider",
    },
    "treatment_effect": {
        "TheCausalityGame.agent.deciders.optimal_effect_design:OptimalEffectDesignDecider",
        "TheCausalityGame.agent.deciders.abci:ABCIDecider",
    },
    "scm_discovery": {
        "TheCausalityGame.agent.deciders.trust_gradient:TrustYourGradientDecider",
        "TheCausalityGame.agent.deciders.abci:ABCIDecider",
    },
    "symbolic_scm_discovery": {
        "TheCausalityGame.agent.deciders.trust_gradient:TrustYourGradientDecider",
    },
}


@dataclass(frozen=True)
class SourceManifest:
    """A source manifest and its target seeded folder context."""

    path: Path
    source_file: str
    mission_folder: str
    inferer_folder: str


def _normalize_path(value: str) -> str:
    return value.strip().strip("/")


def _normalized_set(values: set[str]) -> set[str]:
    return {_normalize_path(value) for value in values if _normalize_path(value)}


def _seeded_random_states() -> list[tuple[int, np.random.RandomState]]:
    seeds = list(np.random.default_rng(911).integers(0, 999, size=20))
    seeds.append(911)  # Just Because
    if INCLUDE_SEEDS:
        seeds = [seed for seed in seeds if seed in INCLUDE_SEEDS]
    if EXCLUDE_SEEDS:
        seeds = [seed for seed in seeds if seed not in EXCLUDE_SEEDS]
    return [(seed, np.random.RandomState(seed)) for seed in seeds]


def _source_manifests() -> list[SourceManifest]:
    manifests: list[SourceManifest] = []

    for path in sorted(PROBLEM_INSTANCES_BY_INFERER_ROOT.rglob("*.json")):
        relative = path.relative_to(PROBLEM_INSTANCES_BY_INFERER_ROOT)
        if len(relative.parts) < 3:
            manifests.append(SourceManifest(path, relative.as_posix(), "", ""))
            continue
        mission_folder, inferer_folder = relative.parts[:2]
        manifests.append(
            SourceManifest(
                path=path,
                source_file=relative.as_posix(),
                mission_folder=mission_folder,
                inferer_folder=inferer_folder,
            )
        )

    for path in sorted((PROBLEM_INSTANCES_ROOT / "scm_discovery").glob("*.json")):
        relative = path.relative_to(PROBLEM_INSTANCES_ROOT)
        manifests.append(
            SourceManifest(
                path=path,
                source_file=relative.as_posix(),
                mission_folder="scm_discovery",
                inferer_folder="scm_discovery",
            )
        )

    for path in sorted((PROBLEM_INSTANCES_ROOT / "symbolic_scm_discovery").glob("*.json")):
        relative = path.relative_to(PROBLEM_INSTANCES_ROOT)
        manifests.append(
            SourceManifest(
                path=path,
                source_file=relative.as_posix(),
                mission_folder="symbolic_scm_discovery",
                inferer_folder="sparse_symbolic_scm",
            )
        )

    return manifests


def _source_allowed(source: SourceManifest) -> bool:
    include_missions = _normalized_set(INCLUDE_MISSIONS)
    exclude_missions = _normalized_set(EXCLUDE_MISSIONS)
    include_inferer_folders = _normalized_set(INCLUDE_INFERER_FOLDERS)
    exclude_inferer_folders = _normalized_set(EXCLUDE_INFERER_FOLDERS)
    include_source_files = _normalized_set(INCLUDE_SOURCE_FILES)
    exclude_source_files = _normalized_set(EXCLUDE_SOURCE_FILES)

    inferer_folder = f"{source.mission_folder}/{source.inferer_folder}"

    if include_missions and source.mission_folder not in include_missions:
        return False
    if source.mission_folder in exclude_missions:
        return False

    if include_inferer_folders and inferer_folder not in include_inferer_folders:
        return False
    if inferer_folder in exclude_inferer_folders:
        return False

    if include_source_files and source.source_file not in include_source_files:
        return False
    return source.source_file not in exclude_source_files


def _active_agents(json_content: dict) -> list[dict]:
    return [
        agent
        for agent in json_content.get("agents", [])
        if agent.get("active", True)
    ]


def _validation_error(source: SourceManifest, json_content: dict) -> str | None:
    if "scm" not in json_content or not isinstance(json_content["scm"], dict):
        return "missing top-level dict 'scm'"

    if source.mission_folder not in MISSION_IDS_BY_FOLDER:
        return f"unknown mission folder '{source.mission_folder}'"

    if source.inferer_folder not in INFERER_BY_FOLDER:
        return f"unknown inferer folder '{source.inferer_folder}'"

    mission_id = json_content.get("mission", {}).get("id")
    valid_mission_ids = MISSION_IDS_BY_FOLDER[source.mission_folder]
    if mission_id not in valid_mission_ids:
        return (
            f"mission id '{mission_id}' does not match folder "
            f"'{source.mission_folder}'"
        )

    expected_inferer = INFERER_BY_FOLDER[source.inferer_folder]
    valid_deciders = DECIDERS_BY_MISSION_FOLDER[source.mission_folder]

    for agent in _active_agents(json_content):
        agent_id = agent.get("id", "<unknown>")
        inferer_class = agent.get("inferer", {}).get("class_")
        if inferer_class != expected_inferer:
            return (
                f"agent '{agent_id}' inferer '{inferer_class}' does not match "
                f"folder '{source.inferer_folder}'"
            )

        decider_class = agent.get("decider", {}).get("class_")
        if decider_class not in valid_deciders:
            return (
                f"agent '{agent_id}' decider '{decider_class}' is not valid for "
                f"mission folder '{source.mission_folder}'"
            )

    return None


def _load_valid_source(source: SourceManifest) -> dict | None:
    with source.path.open(encoding="utf-8") as f:
        json_content = json.load(f)

    validation_error = _validation_error(source, json_content)
    if validation_error:
        print(f"Skipping {source.path}: {validation_error}")
        return None

    return json_content


def _seed_manifest(
    json_content: dict,
    *,
    mission_folder: str,
    inferer_folder: str,
    seed: int,
    random_state: np.random.RandomState,
) -> dict:
    seeded_json = copy.deepcopy(json_content)
    base_name = Path(json_content["id"]).name
    seeded_json["scm"]["random_state"] = random_state_to_json(random_state)
    seeded_json["id"] = f"seeded/{mission_folder}/{inferer_folder}/{base_name}_rs-{seed}"
    seeded_json["run_plan"]["hook_plan"] = []
    return seeded_json


def main() -> None:
    random_states = _seeded_random_states()
    generated = 0
    skipped = 0
    excluded = 0

    for source in _source_manifests():
        if not _source_allowed(source):
            excluded += 1
            continue

        json_content = _load_valid_source(source)
        if json_content is None:
            skipped += 1
            continue

        base_name = source.path.stem
        output_folder = SEEDED_ROOT / source.mission_folder / source.inferer_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        for seed, random_state in random_states:
            seeded_json = _seed_manifest(
                json_content,
                mission_folder=source.mission_folder,
                inferer_folder=source.inferer_folder,
                seed=seed,
                random_state=random_state,
            )

            output_path = output_folder / f"{base_name}_rs-{seed}.json"
            with output_path.open("w", encoding="utf-8") as out_f:
                json.dump(seeded_json, out_f, indent=4)
                out_f.write("\n")
            generated += 1

    print(f"Generated {generated} seeded problem instances.")
    if excluded:
        print(f"Excluded {excluded} source manifests by config.")
    if skipped:
        print(f"Skipped {skipped} invalid source manifests.")


if __name__ == "__main__":
    main()
