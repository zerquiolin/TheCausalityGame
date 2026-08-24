"""Generate the matched-seed honest-causal-tree rebuttal manifests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "scripts"
    / "problem_instances"
    / "seeded"
    / "treatment_effect"
    / "honest_causal_tree"
)
OUTPUT_ROOT = (
    ROOT
    / "scripts"
    / "problem_instances_rebuttal"
    / "treatment_effect"
    / "honest_causal_tree"
)

REBUTTAL_BUDGET = {
    "rounds": 100,
    "time_s": 400,
    "samples": 400,
    "memory_mb": 2,
}


def build_manifest(source_path: Path) -> dict:
    """Return one source manifest with only its ID and budget changed."""
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    seed_suffix = source_path.stem.rsplit("_rs-", 1)[1]
    manifest["id"] = (
        "rebuttal/treatment_effect/honest_causal_tree/"
        f"hill_rs-{seed_suffix}"
    )
    manifest["run_plan"]["budget"] = dict(REBUTTAL_BUDGET)
    return manifest


def main() -> None:
    """Write all source seeds to the dedicated rebuttal directory."""
    source_paths = sorted(SOURCE_ROOT.glob("hill_rs-*.json"))
    if len(source_paths) != 21:
        raise RuntimeError(
            f"Expected 21 honest-causal-tree seed manifests, found {len(source_paths)}."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for source_path in source_paths:
        manifest = build_manifest(source_path)
        output_path = OUTPUT_ROOT / source_path.name
        output_path.write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
