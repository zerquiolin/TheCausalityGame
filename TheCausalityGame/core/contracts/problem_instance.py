"""The Causality Game - Problem Instance Definition."""

from __future__ import annotations

from typing import override

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettingsSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
)


class ProblemInstance(Serializable):
    """
    A fully resolved configuration to run a causal inference experiment.

    This includes a concrete SCM, mission, agents, metrics, execution plan, and runtime settings.

    Parameters
    ----------
    schema_version : str
        Schema version of the specification.
    id : str
        Unique identifier for the problem instance.
    scm : SCM
        The structural causal model.
    mission : Mission
        Mission containing evaluation logic and metrics.
    agents : list[Agent]
        List of agents to evaluate.
    custom_metrics : list[Metric]
        Additional user-defined metrics.
    run_plan : RunPlanSpec
        Execution plan including hooks, plots, and parallelism settings.
    seeds : dict[str, int]
        Mapping of component seeds for reproducibility.
    runtime : RuntimeSettingsSpec
        Execution configuration (logging, debug mode, etc.).
    """

    def __init__(  # noqa: PLR0913
        self,
        schema_version: str,
        id: str,
        scm: SCM,
        mission: Mission,
        agents: list[Agent],
        custom_metrics: list[Metric],
        run_plan: RunPlanSpec,
        seeds: dict[str, int],
        runtime: RuntimeSettingsSpec,
    ) -> None:
        """Initialize the ProblemInstance."""
        self.schema_version = schema_version
        self.id = id
        self.scm = scm
        self.mission = mission
        self.agents = agents
        self.custom_metrics = custom_metrics
        self.run_plan = run_plan
        self.seeds = seeds
        self.runtime = runtime

    @override
    def to_spec(self) -> ProblemInstanceSpec:
        """
        Serialize the problem instance to its specification form.

        Returns
        -------
        ProblemInstanceSpec
            The spec representation suitable for saving or exporting.
        """
        return ProblemInstanceSpec(
            class_=get_class_path(self.__class__),
            schema_version=self.schema_version,
            id=self.id,
            scm=self.scm.to_spec(),
            mission=self.mission.to_spec(),
            agents=[a.to_spec() for a in self.agents],
            custom_metrics=[m.to_spec() for m in self.custom_metrics],
            run_plan=self.run_plan,
            seeds=self.seeds,
            runtime=self.runtime,
        )

    @classmethod
    @override
    def from_spec(cls, spec: ProblemInstanceSpec) -> ProblemInstance:
        """
        Build a `ProblemInstance` from its specification.

        Parameters
        ----------
        spec : ProblemInstanceSpec
            The input specification defining the full setup.

        Returns
        -------
        ProblemInstance
            The fully constructed instance.
        """
        scm = build_from_spec(spec.scm)
        mission = build_from_spec(spec.mission)
        agents = [build_from_spec(a) for a in spec.agents]
        custom_metrics = [build_from_spec(m) for m in spec.custom_metrics]

        return cls(
            schema_version=spec.schema_version,
            id=spec.id,
            scm=scm,  # type: ignore
            mission=mission,  # type: ignore
            agents=agents,  # type: ignore
            custom_metrics=custom_metrics,  # type: ignore
            run_plan=spec.run_plan,
            seeds=spec.seeds,
            runtime=spec.runtime,
        )
