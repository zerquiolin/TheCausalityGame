"""Shared helpers for DAG discovery inferers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.dto.environment import Samples


def observational_frame(samples: list[Samples]) -> pd.DataFrame:
    """Combine observational batches into one frame."""
    frames = [sample.data.copy() for sample in samples if sample.kind == "observational"]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a mixed data frame into numeric columns suitable for linear methods."""
    converted = df.copy()
    for column in converted.columns:
        series = converted[column]
        if pd.api.types.is_numeric_dtype(series):
            converted[column] = pd.to_numeric(series, errors="coerce").astype(float)
            continue
        if pd.api.types.is_bool_dtype(series):
            converted[column] = series.astype(float)
            continue
        converted[column] = pd.Categorical(series).codes.astype(float)
        converted.loc[series.isna(), column] = np.nan
    converted = converted.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return converted


def empty_graph(columns: list[str]) -> nx.DiGraph:
    """Create an empty directed graph over the provided variables."""
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    return graph


def graph_from_adjacency(
    adjacency: np.ndarray,
    columns: list[str],
    *,
    threshold: float,
) -> nx.DiGraph:
    """Convert a weighted adjacency matrix into a thresholded DAG."""
    graph = empty_graph(columns)
    d = len(columns)
    for parent_idx in range(d):
        for child_idx in range(d):
            if parent_idx == child_idx:
                continue
            if abs(float(adjacency[parent_idx, child_idx])) <= threshold:
                continue
            graph.add_edge(columns[parent_idx], columns[child_idx])

    if nx.is_directed_acyclic_graph(graph):
        return graph

    weighted_edges = sorted(
        (
            (abs(float(adjacency[parent_idx, child_idx])), columns[parent_idx], columns[child_idx])
            for parent_idx in range(d)
            for child_idx in range(d)
            if parent_idx != child_idx and graph.has_edge(columns[parent_idx], columns[child_idx])
        ),
        reverse=True,
    )

    dag = empty_graph(columns)
    for _weight, parent, child in weighted_edges:
        dag.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(parent, child)
    return dag


def fit_linear_coefficients(
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Solve a small ridge regression system."""
    if predictors.size == 0:
        return np.zeros(0, dtype=float)
    gram = predictors.T @ predictors
    penalty = np.eye(gram.shape[0], dtype=float) * ridge
    return np.linalg.solve(gram + penalty, predictors.T @ target)


def residualize(target: np.ndarray, source: np.ndarray, *, ridge: float) -> np.ndarray:
    """Remove the linear effect of one source variable from a target variable."""
    source_column = np.asarray(source, dtype=float).reshape(-1, 1)
    target_vector = np.asarray(target, dtype=float)
    coef = fit_linear_coefficients(source_column, target_vector, ridge=ridge)
    return target_vector - source_column @ coef


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Center and scale columns, guarding against zero-variance features."""
    centered = df.astype(float)
    means = centered.mean(axis=0)
    scales = centered.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (centered - means) / scales
