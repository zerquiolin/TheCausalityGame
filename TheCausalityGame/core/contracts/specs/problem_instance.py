from contracts.types.common import JsonDict

from pydantic import BaseModel, ConfigDict

from .agent import AgentSpec
from .metric import MetricSpec, MetricSpecs
from .run import RunPlan
from .hook import HookSpec
from .settings import RuntimeSettingsSpec


class ProblemInstanceSpec(BaseModel):
    """Top-level manifest describing a benchmark run.

    Attributes:
        schema_version: Manifest schema version string.
        id: Run identifier (used as folder name under runs/).
        scm_spec: SCM specification {'class','config'}.
        mission_spec: Mission specification {'class','config'}.
        agent_specs: List of AgentSpec-like dicts (validated here).
        metric_specs: Pair of primary metrics (behavior & result).
        custom_metric_specs: Optional list of additional metrics.
        run_plan: Execution policy for the run.
        seeds: Seed hierarchy for determinism (global/scm/mission/agents).
        hook_plan: Hook subscriptions for lifecycle events.
        artifacts_policy: Global artifact toggles/policies.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    id: str

    scm_spec: JsonDict
    mission_spec: JsonDict
    agent_specs: list[AgentSpec]

    metric_specs: MetricSpecs
    custom_metric_specs: list[MetricSpec] = []

    run_plan: RunPlan

    seeds: JsonDict = {}
    hook_plan: list[HookSpec] = []
    artifacts_policy: JsonDict = {}
    runtime: RuntimeSettingsSpec = RuntimeSettingsSpec()
