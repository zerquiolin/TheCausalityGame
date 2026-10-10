"""Data buffering and validation shared by authoritative adapters."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.lib.errors.inferer import UnsupportedInfererInputError


def observational_frame(samples: list[Samples]) -> pd.DataFrame:
    """Combine only observational batches, as required by discovery backends."""
    frames = [
        sample.data
        for sample in samples
        if sample.kind == "observational" and sample.data is not None
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def combined_frame(samples: list[Samples]) -> pd.DataFrame:
    """Combine treatment-effect samples without changing their assignment labels."""
    frames = [sample.data for sample in samples if sample.data is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_matrix(frame: pd.DataFrame, *, algorithm: str) -> np.ndarray:
    """Validate a complete numeric frame and return a finite float matrix."""
    if any(
        not pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns
    ):
        raise UnsupportedInfererInputError(
            f"{algorithm} requires numeric columns only."
        )
    matrix = frame.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise UnsupportedInfererInputError(
            f"{algorithm} requires finite data without missing values."
        )
    return matrix


def adjacency_to_digraph(
    adjacency: np.ndarray,
    columns: list[str],
    *,
    transpose: bool = False,
) -> nx.DiGraph:
    """Convert an upstream weighted adjacency matrix to a NetworkX DAG estimate."""
    adjacency = np.asarray(adjacency, dtype=float)
    if (
        adjacency.shape != (len(columns), len(columns))
        or not np.isfinite(adjacency).all()
    ):
        raise ValueError(
            "Backend adjacency must be finite and square with one row per variable."
        )
    if len(set(columns)) != len(columns):
        raise ValueError("Backend adjacency variable names must be unique.")
    weights = adjacency.T if transpose else adjacency
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for source in range(len(columns)):
        for target in range(len(columns)):
            weight = float(weights[source, target])
            if source != target and not np.isclose(weight, 0.0):
                graph.add_edge(columns[source], columns[target], weight=weight)
    return graph


def zero_cate_answer() -> Any:
    """Return a validator-compatible zero CATE callable."""

    def estimate(
        X: list[str],  # noqa: N803
        treatment: str,
        outcome: str,
        covariate_values: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        del X, treatment, outcome
        return pd.DataFrame(
            np.zeros(len(covariate_values[0])),
            columns=["treatment_effect"],
        )

    return estimate
