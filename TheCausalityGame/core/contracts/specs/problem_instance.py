from TheCausalityGame.core.specs.agent import AgentSpec
from TheCausalityGame.core.specs.common import CommonSpec
from TheCausalityGame.core.specs.metric import MetricsSpec
from TheCausalityGame.core.specs.mission import MissionSpec
from TheCausalityGame.core.specs.run import RunPlanSpec
from TheCausalityGame.core.specs.scm import SCMSpec
from TheCausalityGame.core.specs.settings import RuntimeSettingsSpec


class ProblemInstanceSpec(CommonSpec):
    """Top-level manifest describing a benchmark run.

    Attributes
    ----------
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

    # TODO: Update the docstring above to match the actual attributes.

    id: str
    schema_version: str = "1.0.0"

    agents: list[AgentSpec]

    scm: SCMSpec
    mission: MissionSpec
    custom_metrics: list[MetricsSpec] = []

    run_plan: RunPlanSpec

    seeds: dict[str, int] = {}
    runtime: RuntimeSettingsSpec = RuntimeSettingsSpec()
