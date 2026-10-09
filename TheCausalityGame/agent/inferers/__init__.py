"""Inferer implementations shipped with The Causality Game."""

from TheCausalityGame.agent.inferers.causal_tree import HonestCausalTreeInferer
from TheCausalityGame.agent.inferers.dag import DAGDiscoveryInferer
from TheCausalityGame.agent.inferers.lingam import LiNGAMInferer
from TheCausalityGame.agent.inferers.notears import NOTEARSInferer
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer
from TheCausalityGame.agent.inferers.pc import PCInferer
from TheCausalityGame.agent.inferers.scm import SCMDiscoveryInferer
from TheCausalityGame.agent.inferers.symbolic_scm import (
    SparseSymbolicSCMDiscoveryInferer,
)
from TheCausalityGame.agent.inferers.transformed_outcome import (
    TransformedOutcomeInferer,
)

__all__ = [
    "DAGDiscoveryInferer",
    "HonestCausalTreeInferer",
    "LiNGAMInferer",
    "NOTEARSInferer",
    "OutcomeRegressionInferer",
    "PCInferer",
    "SCMDiscoveryInferer",
    "SparseSymbolicSCMDiscoveryInferer",
    "TransformedOutcomeInferer",
]
