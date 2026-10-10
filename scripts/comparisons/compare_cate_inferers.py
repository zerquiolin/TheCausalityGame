"""Compare empirical CATE inferers with authoritative R implementations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.comparisons._common import (
    buffered_rounds,
    input_digest,
    observation,
    output_directory,
    write_reports,
)
from TheCausalityGame.agent.inferers import causal_tree as authoritative_tree
from TheCausalityGame.agent.inferers import outcome_regression as authoritative_outcome
from TheCausalityGame.agent.inferers import (
    transformed_outcome as authoritative_transformed,
)
from TheCausalityGame.agent.inferers.empirical import causal_tree as empirical_tree
from TheCausalityGame.agent.inferers.empirical import (
    outcome_regression as empirical_outcome,
)
from TheCausalityGame.agent.inferers.empirical import (
    transformed_outcome as empirical_transformed,
)
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec


def _hill_case(seed: int, n: int) -> tuple[str, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and sample the repository's Hill benchmark SCM with the shared seed."""
    path = Path("scripts/problem_instances/treatment_effect/hill.json")
    problem = ProblemInstanceSpec(**json.loads(path.read_text(encoding="utf-8")))
    scm = build_from_spec(problem.scm)
    sampled = scm.generate_samples(
        num_samples=n,
        interventions=None,
        random_state=np.random.RandomState(seed),
    )
    data = sampled[["Z", "Y"]].rename(columns={"Z": "T"})
    # Hill encodes its binary categories as strings; queries below use numeric arms.
    # Convert once before pairing so both implementations receive the same coding.
    data["T"] = pd.to_numeric(data["T"], errors="raise")
    return "benchmark_hill", data, np.zeros(25), np.full(25, 2.0)


def _cases(seed: int, n: int) -> list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    query = np.linspace(-1.0, 1.0, 101)
    cases: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for name, effect in (
        ("synthetic_linear", lambda z: 2.0 + z),
        ("synthetic_smooth", lambda z: 1.0 + np.sin(np.pi * z)),
        ("synthetic_piecewise", lambda z: np.where(z <= 0.0, 1.0, 4.0)),
    ):
        z = rng.uniform(-1.0, 1.0, n)
        treatment = rng.integers(0, 2, n).astype(float)
        tau = effect(z)
        outcome = 1.0 + z + treatment * tau + rng.normal(scale=0.1, size=n)
        cases.append(
            (
                name,
                pd.DataFrame({"T": treatment, "Z": z, "Y": outcome}),
                query,
                effect(query),
            )
        )
    cases.append(_hill_case(seed, n))
    return cases


def _query(
    values: np.ndarray, *, has_covariate: bool
) -> tuple[list[str], str, str, tuple[pd.DataFrame, pd.DataFrame]]:
    if has_covariate:
        control = pd.DataFrame({"T": np.zeros(len(values)), "Z": values})
        treated = pd.DataFrame({"T": np.ones(len(values)), "Z": values})
        return ["Z"], "T", "Y", (control, treated)
    control = pd.DataFrame({"T": np.zeros(len(values))})
    treated = pd.DataFrame({"T": np.ones(len(values))})
    return [], "T", "Y", (control, treated)


def compare(*, seed: int = 911, n: int = 500, rounds: int = 5) -> list[dict[str, Any]]:
    """Run CATE pairs on identical incremental buffers and return round records."""
    pairs = {
        "s_learner": (
            empirical_outcome.OutcomeRegressionInferer,
            authoritative_outcome.OutcomeRegressionInferer,
        ),
        "transformed_outcome": (
            empirical_transformed.TransformedOutcomeInferer,
            authoritative_transformed.TransformedOutcomeInferer,
        ),
        "honest_causal_tree": (
            empirical_tree.HonestCausalTreeInferer,
            authoritative_tree.HonestCausalTreeInferer,
        ),
    }
    rows: list[dict[str, Any]] = []
    for case_name, data, query_values, truth in _cases(seed, n):
        round_data = buffered_rounds(data, rounds)
        query = _query(query_values, has_covariate="Z" in data.columns)
        for method, implementations in pairs.items():
            for implementation, inferer_class in zip(
                ("empirical", "authoritative"), implementations, strict=True
            ):
                inferer = None
                cumulative_runtime = 0.0
                for round_number, chunk, cumulative_data in round_data:
                    shared_digest = input_digest(
                        cumulative_data, query_values.tolist(), truth.tolist()
                    )
                    started = time.perf_counter()
                    try:
                        if inferer is None:
                            inferer = inferer_class()
                        inferer.update(observation(chunk, round_number=round_number))
                        predictions = inferer.answer()(*query)[
                            "treatment_effect"
                        ].to_numpy(dtype=float)
                        rmse = float(np.sqrt(np.mean(np.square(predictions - truth))))
                        round_runtime = time.perf_counter() - started
                        cumulative_runtime += round_runtime
                        row = {
                            "case": case_name,
                            "method": method,
                            "implementation": implementation,
                            "seed": seed,
                            "round": round_number,
                            "samples_seen": len(cumulative_data),
                            "input_digest": shared_digest,
                            "status": "ok",
                            "runtime_s": round_runtime,
                            "cumulative_runtime_s": cumulative_runtime,
                            "pehe": rmse,
                            "rmse": rmse,
                            "ate_absolute_error": float(
                                abs(predictions.mean() - truth.mean())
                            ),
                            "backend": inferer.snapshot().summary.get("backend", {}),
                        }
                    except Exception as exc:  # noqa: BLE001
                        round_runtime = time.perf_counter() - started
                        cumulative_runtime += round_runtime
                        provenance = getattr(inferer, "provenance", None)
                        row = {
                            "case": case_name,
                            "method": method,
                            "implementation": implementation,
                            "seed": seed,
                            "round": round_number,
                            "samples_seen": len(cumulative_data),
                            "input_digest": shared_digest,
                            "status": "error",
                            "runtime_s": round_runtime,
                            "cumulative_runtime_s": cumulative_runtime,
                            "error": f"{type(exc).__name__}: {exc}",
                            "backend": (
                                provenance.to_dict() if provenance is not None else {}
                            ),
                        }
                    rows.append(row)
    return rows


def main() -> None:
    """Run CATE comparison rounds and write their artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = output_directory(args.output)
    write_reports(
        compare(seed=args.seed, n=args.samples, rounds=args.rounds),
        directory,
        "cate_comparison",
    )
    print(directory)


if __name__ == "__main__":
    main()
