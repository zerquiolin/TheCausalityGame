"""Shared helpers for action selection among agent variants."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from TheCausalityGame.core.contracts.dto.environment import AvailableActions

ActionKey = tuple[tuple[str, int | float | str], ...]


def _hashable_value(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(round(float(value), 4))
    if isinstance(value, np.ndarray):
        return _hashable_value(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_hashable_value(v) for v in value)
    return value


def _treatment_value(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return tuple(_treatment_value(v) for v in value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_treatment_value(v) for v in value)
    return value


def _numeric_grid(domain: Iterable[int | float | np.integer | np.floating], num_points: int) -> list[float]:
    numeric_values = [float(v) for v in domain]
    if not numeric_values:
        return []
    low = float(min(numeric_values))
    high = float(max(numeric_values))
    if np.isclose(low, high):
        return [low]
    points = max(2, num_points)
    return [float(v) for v in np.linspace(low, high, num=points)]


def make_action_key(treatment: dict[str, int | float | str]) -> ActionKey:
    return tuple(sorted((name, _hashable_value(value)) for name, value in treatment.items()))


def collect_single_variable_candidates(
    available_actions: AvailableActions,
    *,
    grid_points: int = 5,
) -> dict[ActionKey, dict[str, int | float | str]]:
    candidates: dict[ActionKey, dict[str, int | float | str]] = {}
    for variable in available_actions.experiments:
        domain = variable.domain
        if not domain:
            continue

        first = domain[0]
        if isinstance(first, str):
            values: Iterable[int | float | str] = domain
        elif isinstance(first, (int, float, np.integer, np.floating)):
            values = _numeric_grid(domain, grid_points)
        else:
            values = domain

        for raw_value in values:
            treatment_value = _treatment_value(raw_value)
            treatment = {variable.name: treatment_value}
            key = make_action_key(treatment)
            candidates[key] = treatment

    return candidates
