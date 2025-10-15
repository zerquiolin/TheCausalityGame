from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettingsSpec
from TheCausalityGame.core.infra.registry import build_from_spec, get_class_path


class ProblemInstance(Serializable):
    def __init__(
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
    ):
        self.schema_version = schema_version
        self.id = id
        self.scm = scm
        self.mission = mission
        self.agents = agents
        self.custom_metrics = custom_metrics
        self.run_plan = run_plan
        self.seeds = seeds
        self.runtime = runtime

    def to_spec(self) -> ProblemInstanceSpec:
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
    def from_spec(cls, spec: ProblemInstanceSpec) -> "SCM":
        # Build Components
        scm = build_from_spec(spec.scm)
        mission = build_from_spec(spec.mission)
        agents = [build_from_spec(a) for a in spec.agents]
        custom_metrics = [build_from_spec(m) for m in spec.custom_metrics]
        # Create Instance
        return cls(
            schema_version=spec.schema_version,
            id=spec.id,
            scm=scm,
            mission=mission,
            agents=agents,
            custom_metrics=custom_metrics,
            run_plan=spec.run_plan,
            seeds=spec.seeds,
            runtime=spec.runtime,
        )
