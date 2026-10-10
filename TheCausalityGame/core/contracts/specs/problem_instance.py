"""The Causality Game - Problem Instance Specification."""

from pydantic import model_validator

from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettingsSpec


class ProblemInstanceSpec(CommonSpec):
    """
    Specification for constructing a problem instance.

    A problem instance defines the complete configuration required to
    simulate and evaluate agents in a specific causal environment.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    id : str
        Unique identifier for the problem instance.
    schema_version : str, default='1.0.0'
        Version of the schema used to encode this spec.
    agents : list[AgentSpec]
        List of agent specifications to evaluate.
    scm : SCMSpec
        Specification for the Structural Causal Model (SCM).
    mission : MissionSpec
        Specification of the mission, including metrics and evaluation logic.
    custom_metrics : list[MetricSpec], optional
        Additional metrics for evaluation, beyond those required by the mission.
    run_plan : RunPlanSpec
        Description of the evaluation protocol
    seeds : dict[str, int], optional
        Dictionary of fixed random seeds for reproducibility.
    runtime : RuntimeSettingsSpec, optional
        Configuration for runtime behavior
    """

    id: str
    schema_version: str = "1.0.0"

    agents: list[AgentSpec]

    scm: SCMSpec
    mission: MissionSpec
    custom_metrics: list[MetricSpec] = []  # noqa: RUF012

    run_plan: RunPlanSpec

    seeds: dict[str, int] = {}  # noqa: RUF012
    runtime: RuntimeSettingsSpec = RuntimeSettingsSpec()

    @model_validator(mode="after")
    def validate_seeds_and_ids(self):
        """Require known seed namespaces and unique agent identifiers."""
        if any(not 0 <= seed < 2**32 for seed in self.seeds.values()):
            raise ValueError("Seeds must be unsigned 32-bit integers.")
        ids = [agent.id for agent in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Agent IDs must be unique.")
        return self
