"""The Causality Game - Functions for serializing and deserializing NumPy RandomState."""

import json
from typing import Any

import numpy as np


def random_state_to_json(rs: np.random.RandomState) -> str:
    """
    Serialize a NumPy RandomState object to a JSON string.

    Parameters
    ----------
    rs : np.random.RandomState
        The random state to serialize.

    Returns
    -------
    str
        A JSON string representing the internal state of the random generator.
    """
    rng_state = rs.get_state()
    state_dict: dict[str, Any] = {
        "bit_generator": "RandomState",
        "state": {
            "key": rng_state[1].tolist(),  # type: ignore
            "pos": rng_state[2],  # type: ignore
            "has_gauss": rng_state[3],  # type: ignore
            "cached_gaussian": rng_state[4],  # type: ignore
            "state_name": rng_state[0],  # type: ignore
        },
    }
    return json.dumps(state_dict)


def random_state_from_json(json_str: str) -> np.random.RandomState:
    """
    Deserialize a NumPy RandomState object from a JSON string.

    Parameters
    ----------
    json_str : str
        A JSON-encoded string representing a RandomState.

    Returns
    -------
    np.random.RandomState
        The restored random state object.
    """
    state_dict = json.loads(json_str)
    return random_state_from_dict(state_dict)


def random_state_from_dict(state_dict: dict[str, Any]) -> np.random.RandomState:
    """
    Deserialize a NumPy RandomState object from a dictionary.

    Parameters
    ----------
    state_dict : dict[str, Any]
        A dictionary containing the state of a RandomState.

    Returns
    -------
    np.random.RandomState
        The restored random state object.
    """
    restored_rng = np.random.RandomState()
    restored_rng.set_state(
        (
            state_dict["state"]["state_name"],
            np.array(state_dict["state"]["key"], dtype=np.uint32),
            state_dict["state"]["pos"],
            state_dict["state"]["has_gauss"],
            state_dict["state"]["cached_gaussian"],
        )
    )
    return restored_rng
