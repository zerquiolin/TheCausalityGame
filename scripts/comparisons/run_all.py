"""Run graph and treatment-effect comparison suites into one directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.comparisons._common import output_directory, write_reports
from scripts.comparisons.compare_cate_inferers import compare as compare_cate
from scripts.comparisons.compare_graph_inferers import compare as compare_graph


def main() -> None:
    """Run all longitudinal comparison suites into one artifact directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = output_directory(args.output)
    graph_rows = compare_graph(
        seed=args.seed,
        n=args.samples,
        rounds=args.rounds,
        include_benchmarks=not args.synthetic_only,
    )
    cate_rows = compare_cate(seed=args.seed, n=args.samples, rounds=args.rounds)
    write_reports(graph_rows, directory, "graph_comparison")
    write_reports(cate_rows, directory, "cate_comparison")
    write_reports(graph_rows + cate_rows, directory, "combined_comparison")
    print(directory)


if __name__ == "__main__":
    main()
