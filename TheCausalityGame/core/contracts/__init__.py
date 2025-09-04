"""Public contract exports for TheCausalityGame core."""

from __future__ import annotations

# Base contracts
from .agent import AgentContext, BaseAgent
from .decisions import Decision

# DTOs
from .dto import (
    Action,
    ActionOutcome,
    AvailableActions,
    ExperimentSpace,
    MetricScore,
    Observation,
    RoundInfo,
    Samples,
    SamplesBatch,
    StepRecord,
    TranscriptEntry,
)

# Enums
from .enum import HookEvent, StepKind
from .errors import (
    BudgetExceededError,
    ConfigurationError,
    DiscoveryError,
    InvalidAction,
    LoadError,
    SecurityViolation,
    TCGError,
    TimeoutExceeded,
)
from .metric import BaseMetric, BehaviorMetric, ResultMetric
from .mission import BaseMission
from .scm import BaseSCM
from .serializable import Serializable

__all__ = [
    # base contracts
    "AgentContext",
    "BaseAgent",
    "BaseMission",
    "BaseMetric",
    "BehaviorMetric",
    "ResultMetric",
    "BaseSCM",
    "Serializable",
    "Decision",
    # enums
    "HookEvent",
    "StepKind",
    # DTOs
    "Action",
    "ActionOutcome",
    "AvailableActions",
    "ExperimentSpace",
    "MetricScore",
    "Observation",
    "RoundInfo",
    "Samples",
    "SamplesBatch",
    "StepRecord",
    "TranscriptEntry",
    # errors
    "TCGError",
    "ConfigurationError",
    "DiscoveryError",
    "LoadError",
    "InvalidAction",
    "BudgetExceededError",
    "TimeoutExceeded",
    "SecurityViolation",
]
