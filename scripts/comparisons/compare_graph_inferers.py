"""Compare empirical DAG inferers with their authoritative upstream backends."""

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
from TheCausalityGame.agent.inferers import lingam as authoritative_lingam
from TheCausalityGame.agent.inferers import notears as authoritative_notears
from TheCausalityGame.agent.inferers import pc as authoritative_pc
from TheCausalityGame.agent.inferers.empirical import lingam as empirical_lingam
from TheCausalityGame.agent.inferers.empirical import notears as empirical_notears
from TheCausalityGame.agent.inferers.empirical import pc as empirical_pc
from TheCausalityGame.core.contracts.dto.causal_graph import (
    CausalGraphResult,
    graph_adjacencies,
    graph_directed_edges,
    graph_undirected_edges,
)
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec


def _synthetic_cases(
    seed: int, n: int
) -> list[tuple[str, pd.DataFrame, set[tuple[str, str]]]]:
    rng = np.random.default_rng(seed)
    gaussian_x = rng.normal(size=n)
    gaussian_y = 1.5 * gaussian_x + rng.normal(scale=0.3, size=n)
    gaussian_z = -2.0 * gaussian_y + rng.normal(scale=0.3, size=n)
    non_gaussian_x = rng.laplace(size=n)
    non_gaussian_y = 1.5 * non_gaussian_x + rng.laplace(scale=0.3, size=n)
    non_gaussian_z = -2.0 * non_gaussian_y + rng.laplace(scale=0.3, size=n)
    edges = {("X", "Y"), ("Y", "Z")}
    return [
        (
            "synthetic_gaussian_chain",
            pd.DataFrame({"X": gaussian_x, "Y": gaussian_y, "Z": gaussian_z}),
            edges,
        ),
        (
            "synthetic_nongaussian_chain",
            pd.DataFrame(
                {"X": non_gaussian_x, "Y": non_gaussian_y, "Z": non_gaussian_z}
            ),
            edges,
        ),
    ]


def _rc_circuit_case(
    seed: int, n: int
) -> tuple[str, pd.DataFrame, set[tuple[str, str]]]:
    path = Path("scripts/problem_instances/graph_discovery/rc_circuit.json")
    problem = ProblemInstanceSpec(**json.loads(path.read_text(encoding="utf-8")))
    scm = build_from_spec(problem.scm)
    data = scm.generate_samples(
        num_samples=n,
        interventions=None,
        random_state=np.random.RandomState(seed),
    )
    return "benchmark_rc_circuit", data, {tuple(edge) for edge in scm.dag.edges}


def _scores(result: CausalGraphResult, truth: set[tuple[str, str]]) -> dict[str, float]:
    truth_skeleton = {frozenset(edge) for edge in truth}
    predicted_skeleton = graph_adjacencies(result)
    directed = {(str(a), str(b)) for a, b in graph_directed_edges(result)}
    unresolved = graph_undirected_edges(result)
    true_positive = len(truth_skeleton & predicted_skeleton)
    precision = true_positive / len(predicted_skeleton) if predicted_skeleton else 0.0
    recall = true_positive / len(truth_skeleton) if truth_skeleton else 1.0
    adjacency_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    remaining_directed = set(directed)
    remaining_unresolved = set(unresolved)
    shd = 0
    for source, target in truth:
        adjacency = frozenset((source, target))
        if (source, target) in remaining_directed:
            remaining_directed.remove((source, target))
        elif adjacency in remaining_unresolved:
            remaining_unresolved.remove(adjacency)
        elif (target, source) in remaining_directed:
            remaining_directed.remove((target, source))
            shd += 1
        else:
            shd += 1
    shd += len(remaining_directed) + len(remaining_unresolved)
    directed_tp = len(truth & directed)
    directed_precision = directed_tp / len(directed) if directed else 0.0
    directed_recall = directed_tp / len(truth) if truth else 1.0
    directed_f1 = (
        2
        * directed_precision
        * directed_recall
        / (directed_precision + directed_recall)
        if directed_precision + directed_recall
        else 0.0
    )
    return {
        "adjacency_precision": precision,
        "adjacency_recall": recall,
        "adjacency_f1": adjacency_f1,
        "directed_precision": directed_precision,
        "directed_recall": directed_recall,
        "directed_f1": directed_f1,
        "shd": float(shd),
    }


def compare(
    *,
    seed: int = 911,
    n: int = 500,
    rounds: int = 5,
    include_benchmarks: bool = True,
) -> list[dict[str, Any]]:
    """Run graph pairs on identical incremental buffers and return round records."""
    cases = _synthetic_cases(seed, n)
    if include_benchmarks:
        cases.append(_rc_circuit_case(seed, n))
    pairs = {
        "pc": (empirical_pc.PCInferer, authoritative_pc.PCInferer),
        "notears": (
            empirical_notears.NOTEARSInferer,
            authoritative_notears.NOTEARSInferer,
        ),
        "direct_lingam": (
            empirical_lingam.LiNGAMInferer,
            authoritative_lingam.LiNGAMInferer,
        ),
    }
    rows: list[dict[str, Any]] = []
    for case_name, data, truth in cases:
        round_data = buffered_rounds(data, rounds)
        for method, implementations in pairs.items():
            for implementation, inferer_class in zip(
                ("empirical", "authoritative"), implementations, strict=True
            ):
                inferer = None
                cumulative_runtime = 0.0
                for round_number, chunk, cumulative_data in round_data:
                    shared_digest = input_digest(cumulative_data, sorted(truth))
                    started = time.perf_counter()
                    try:
                        if inferer is None:
                            inferer = inferer_class()
                        inferer.update(observation(chunk, round_number=round_number))
                        result = inferer.answer()
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
                            **_scores(result, truth),
                            "backend": inferer.snapshot().summary.get("backend", {}),
                        }
                    except Exception as exc:  # noqa: BLE001
                        round_runtime = time.perf_counter() - started
                        cumulative_runtime += round_runtime
                        backend = {}
                        provenance = getattr(inferer, "provenance", None)
                        if provenance is not None:
                            backend = provenance.to_dict()
                        elif implementation == "authoritative" and inferer is not None:
                            backend = inferer._provenance().to_dict()
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
                            "backend": backend,
                        }
                    rows.append(row)
    return rows


def main() -> None:
    """Run graph comparison rounds and write their artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = output_directory(args.output)
    rows = compare(
        seed=args.seed,
        n=args.samples,
        rounds=args.rounds,
        include_benchmarks=not args.synthetic_only,
    )
    write_reports(rows, directory, "graph_comparison")
    print(directory)


if __name__ == "__main__":
    main()
