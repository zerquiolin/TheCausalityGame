"""The Causality Game - Tests Utils."""

import inspect
from typing import Any

import numpy as np


def get_required_init_args(cls: Any) -> list[str]:  # noqa: ANN401
    """Return the required arguments for the class's __init__ method."""
    signature = inspect.signature(cls.__init__)
    required_params: list[str] = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        # Required if it has NO default value
        if param.default is inspect.Parameter.empty:
            required_params.append(name)
    return required_params


def assert_dicts_equal(
    d1: dict[str, Any],
    d2: dict[str, Any],
    path: str = "",
    msg: str = "",
    atol: float | None = None,
) -> None:
    """
    Assert that two dictionaries are equal.

    Parameters
    ----------
    d1 : dict[str, Any]
        First dictionary.
    d2 : dict[str, Any]
        Second dictionary.
    path : str, optional
        Path to the current key being compared, by default ""
    msg : str, optional
        Error message, by default ""
    atol : float | None, optional
        Absolute tolerance for comparing floating point values, by default None
    """
    for key, value in d1.items():
        assert key in d2, f"Key '{path + key}' missing in second dict"
        if isinstance(value, dict) and isinstance(d2[key], dict):
            # Explicitly cast to help type checkers
            sub_d1: dict[str, Any] = value
            sub_d2: dict[str, Any] = d2[key]
            assert_dicts_equal(sub_d1, sub_d2, path=path + key + ".", msg=msg)
        elif type(value) in [float, np.float64] and atol is not None:  # type: ignore
            assert np.isclose(
                value, d2[key], atol=atol
            ), f"{msg}\nValue mismatch at '{path + key}': {value} != {d2[key]}"
        else:
            assert (
                value == d2[key]
            ), f"{msg}\nValue mismatch at '{path + key}': {value} != {d2[key]}"
    for key in d2:
        assert key in d1, f"{msg}\nKey '{path + key}' missing in first dict"
