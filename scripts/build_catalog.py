"""Migrate bundled SCMs and construct an auditable catalog of executable graph tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.generators import (
    SyntheticSCMGenerator,
    dataset_scm,
    migrate_legacy_scm,
)

ROOT = Path(__file__).resolve().parents[1]


def task(scm, identifier: str) -> ProblemInstanceSpec:
    """Build a small executable graph task using the empirical PC smoke configuration."""
    manifest = json.loads((ROOT / "examples/publication_smoke.json").read_text())
    manifest["id"] = identifier
    manifest["scm"] = scm.model_dump(mode="json")
    manifest["run_plan"]["budget"] = {"rounds": 2, "samples": 20}
    for agent in manifest["agents"]:
        agent["decider"]["params"] = {"num_obs": 5, "num_inter": 1}
    return ProblemInstanceSpec.model_validate(manifest)


def dataset_examples() -> dict:
    """Use documented complete-case slices of the three bundled dataset families."""
    import pandas as pd

    directory = ROOT / "scripts/data/datasets"
    ihdp = (
        pd.read_csv(directory / "ihdp/ihdp.csv")[["treat", "bw", "momage", "YC"]]
        .dropna()
        .head(500)
    )
    jobs = (
        pd.read_csv(directory / "jobs/nsw.csv")[
            ["treatment", "age", "education", "RE78"]
        ]
        .dropna()
        .head(500)
    )
    x = pd.read_csv(directory / "twins/twin_pairs_X_3years_samesex.csv", index_col=0)[
        ["mager8", "meduc6"]
    ]
    t = pd.read_csv(directory / "twins/twin_pairs_T_3years_samesex.csv", index_col=0)[
        ["dbirwt_0"]
    ]
    y = pd.read_csv(directory / "twins/twin_pairs_Y_3years_samesex.csv", index_col=0)[
        ["mort_0"]
    ]
    twins = x.join(t).join(y).dropna().head(500)
    return {
        "ihdp": dataset_scm(
            ihdp, outcomes=["YC"], controllable=["treat"], categorical=["treat"]
        ),
        "jobs": dataset_scm(
            jobs,
            outcomes=["RE78"],
            controllable=["treatment"],
            categorical=["treatment"],
        ),
        "twins": dataset_scm(
            twins,
            outcomes=["mort_0"],
            controllable=["dbirwt_0"],
            categorical=["mort_0"],
        ),
    }


def main() -> int:
    """Keep failed assets visible; include only constructed, sampled tasks in the count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/catalog-v2")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("Catalog verification needs a positive sample count.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    root = ROOT / "scripts/data/scms"
    for path in sorted(root.rglob("*.json")):
        identifier = "catalog/" + path.relative_to(root).with_suffix("").as_posix()
        record = {
            "id": identifier,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            scm = migrate_legacy_scm(json.loads(path.read_text()))
            samples = build_from_spec(scm).generate_samples(num_samples=args.samples)
            manifest = task(scm, identifier)
            output = args.output_dir / (identifier.removeprefix("catalog/") + ".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(manifest.model_dump_json(indent=2) + "\n")
            record.update(
                status="verified",
                manifest=str(output),
                nodes=len(scm.vars),
                samples=len(samples),
            )
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        inventory.append(record)
    factories = {
        f"synthetic-{kind}": SyntheticSCMGenerator(
            categorical_fraction=fraction
        ).generate()
        for kind, fraction in [("continuous", 0), ("hybrid", 0.5), ("categorical", 1)]
    }
    factories.update(dataset_examples())
    for name, scm in factories.items():
        identifier = "catalog/generated/" + name
        try:
            build_from_spec(scm).generate_samples(num_samples=args.samples)
            manifest = task(scm, identifier)
            output = args.output_dir / f"generated/{name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(manifest.model_dump_json(indent=2) + "\n")
            inventory.append(
                {
                    "id": identifier,
                    "status": "verified",
                    "manifest": str(output),
                    "nodes": len(scm.vars),
                }
            )
        except Exception as error:
            inventory.append(
                {
                    "id": identifier,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summary = {
        "verified": sum(row["status"] == "verified" for row in inventory),
        "failed": sum(row["status"] == "failed" for row in inventory),
        "disabled_assets": [
            str(path.relative_to(ROOT)) for path in root.rglob("*.disabled")
        ],
        "inventory": inventory,
        "limitations": [
            "Construction/sampling verifies schema and finite numerical sampling, not identifiability or dataset licenses.",
            "Dataset examples use the first 500 complete rows and configurable random forests; they are construction demonstrations, not benchmark comparisons.",
        ],
    }
    (args.output_dir / "catalog_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Catalog: {summary['verified']} verified tasks, {summary['failed']} failed assets. Report: {args.output_dir/'catalog_report.json'}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
