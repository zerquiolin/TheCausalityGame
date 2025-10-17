"""The Causality Game - metrics utilities."""

import numpy as np


def log_penalty(x: float, alpha: float = 1.0, floor: float = 0.01) -> float:
    """
    Generate a log penalty function.

    The penalty decreases as x increases, but never reaches zero.
    The penalty is defined as 1 / (log(alpha * x + 2)) + floor.

    Args
    ----
        x (float): The input value.
        alpha (float): A scaling factor for the logarithm.
        floor (float): A minimum value for the penalty.
    """
    return 1 / (np.log((alpha / 100) * x + 2)) + floor
