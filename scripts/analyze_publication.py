"""Validate the source-isolated treatment campaign before exporting its figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_processing.rebuttal_figures import (
    figure_treatment_effect_repaired,
    write_evidence_csv,
)
from post_processing.rebuttal_utils import scan_round_records, write_round_cache
from scripts.problem_instances_rebuttal.run_rebuttal import DEFAULT_MANIFEST_ROOT, scan
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.artifact_contract import (
    validate_publication_transcript,
)
from TheCausalityGame.core.infrastructure.provenance import (
    manifest_fingerprint,
    source_fingerprint,
)
from TheCausalityGame.core.lib.utils.random_state_serialization import (
    resolve_scm_random_state,
)
from TheCausalityGame.core.lib.utils.seeding import state_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = scan(DEFAULT_MANIFEST_ROOT, args.runs_dir, set())
    if len(targets) != 21 or any(target.state != "complete" for target in targets):
        raise RuntimeError("All 21 source/manifest-matching runs must be complete.")
    effective_states = set()
    effective_state_by_manifest = {}
    for target in targets:
        state = resolve_scm_random_state(
            target.manifest["scm"].get("random_state"), target.manifest.get("seeds")
        )
        identity = state_identity(state)
        effective_states.add(identity)
        effective_state_by_manifest[target.manifest["id"]] = identity
    if len(effective_states) != len(targets):
        raise RuntimeError(
            f"Campaign has {len(effective_states)} distinct effective SCM states "
            f"for {len(targets)} claimed seed replicates; resolve seed precedence before plotting."
        )
    records, qc = scan_round_records(
        [args.runs_dir / "rebuttal"], [DEFAULT_MANIFEST_ROOT]
    )
    expected = {
        (target.manifest["id"], agent["id"])
        for target in targets
        for agent in target.manifest["agents"]
        if agent.get("active", True)
    }
    actual = {(row["experiment_id"], row["agent"]) for row in qc if row["valid"]}
    if actual != expected or len(qc) != len(expected):
        raise RuntimeError("Missing, duplicate, or invalid agent trajectories.")
    manifests = {target.manifest["id"]: target.manifest for target in targets}
    current_source = source_fingerprint()
    acquisition_traces = {}
    for row in qc:
        timestamp = Path(row["transcript_path"]).parents[2]
        provenance = json.loads((timestamp / "provenance.json").read_text())
        expected_hash = manifest_fingerprint(
            ProblemInstanceSpec.model_validate(
                manifests[row["experiment_id"]]
            ).model_dump(mode="json")
        )
        if (
            provenance.get("source_fingerprint") != current_source
            or provenance.get("manifest_fingerprint") != expected_hash
        ):
            raise RuntimeError(
                "Latest analysis trajectory does not match current source/manifest."
            )
        transcript = json.loads(Path(row["transcript_path"]).read_text())
        validate_publication_transcript(transcript, provenance)
        trace = manifest_fingerprint(
            {
                "samples": [
                    entry.get("samples_collection") for entry in transcript["entries"]
                ]
            }
        )
        acquisition_traces.setdefault(row["agent"], set()).add(trace)
    if any(len(traces) != len(targets) for traces in acquisition_traces.values()):
        raise RuntimeError(
            "Agent acquired identical data across purported independent SCM seeds."
        )
    output = args.runs_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_round_cache(output / "round_cache.json", records, qc)
    paths, contrasts = figure_treatment_effect_repaired(records, output)
    # The general evidence builder also emits graph claims; this study is effect-only.
    write_evidence_csv(output / "treatment_contrasts.csv", contrasts)
    provisional = 0
    missing = 0
    forced = 0
    for row in qc:
        transcript = json.loads(Path(row["transcript_path"]).read_text())
        forced += bool(transcript.get("forced_stop"))
        missing += transcript.get("final_score_status") != "finite"
        provisional += sum(
            entry.get("estimate_metadata", {}).get("backend", {}).get("fit_status")
            == "insufficient_data"
            for entry in transcript["entries"]
        )
    report = {
        "source_fingerprint": source_fingerprint(),
        "publication_eligible": True,
        "seed_precedence": "serialized SCM state > seeds.scm > seeds.global > default 911",
        "distinct_effective_scm_states": len(effective_states),
        "effective_state_by_manifest": effective_state_by_manifest,
        "unique_acquisition_traces_by_agent": {
            agent: len(traces) for agent, traces in acquisition_traces.items()
        },
        "manifests": len(targets),
        "agent_trajectories": len(qc),
        "rounds": len(records),
        "forced_stop_trajectories": forced,
        "unscored_final_responses": missing,
        "provisional_zero_estimate_rounds": provisional,
        "figure_paths": [str(path) for path in paths],
        "contrasts": contrasts,
        "scope": "Treatment effect only; historical graph results are not pooled.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
