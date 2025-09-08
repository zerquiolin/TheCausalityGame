from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytest

SCMNode = pytest.importorskip("TheCausalityGame.core.contracts.scm_node").SCMNode


@dataclass
class _ConstNode(SCMNode):
    """Minimal concrete node for testing the SCMNode interface."""

    const: int | float = 0

    def generate_values(
        self,
        parent_values: pd.DataFrame,
        random_state: np.random.RandomState,
        cancel_noise: bool = False,
    ):
        n = len(parent_values.index)
        return [self.const] * n

    # Provide serializable surface compatible with Serializable
    def to_spec(self) -> dict:
        return {
            "class": "tests._ConstNode",
            "params": {
                "name": self.name,
                "domain": list(self.domain) if self.domain is not None else None,
                "const": self.const,
            },
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "_ConstNode":
        p = spec["params"]
        return cls(
            name=p["name"],
            domain=p.get("domain"),
            noise_distribution=None,  # intentionally none
            random_state=np.random.RandomState(0),
            logger=logging.getLogger("test"),
            const=p["const"],
        )


def test_scmnode_basic_generation():
    node = _ConstNode(
        name="C",
        domain=[0, 1],
        noise_distribution=None,
        random_state=np.random.RandomState(123),
        logger=logging.getLogger("test"),
        const=7,
    )
    parents = pd.DataFrame(index=range(5))
    out = node.generate_values(parents, random_state=np.random.RandomState(999))
    assert out == [7, 7, 7, 7, 7]


def test_scmnode_serialization_roundtrip():
    node = _ConstNode(
        name="C",
        domain=[0, 1],
        noise_distribution=None,
        random_state=np.random.RandomState(123),
        logger=logging.getLogger("test"),
        const=3,
    )
    spec = node.to_spec()
    node2 = _ConstNode.from_spec(spec)
    assert node2.name == node.name
    assert node2.const == node.const
    assert node2.domain == node.domain
