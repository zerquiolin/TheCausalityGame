import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.decisions import Decision
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    Feedback,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infra.registry import get_class_path


class ExhaustiveAgent(Agent):
    """
    An agent that enumerates all possible interventions exhaustively
    and then computes the requested causal or treatment‐effect task.
    """

    def __init__(self, id: str, num_obs: int = 1, num_inter: int = 1):
        super().__init__()
        self.id = id
        self.data: pd.DataFrame = pd.DataFrame()
        self._is_numeric = False
        self._num_obs = num_obs
        self._num_inter = num_inter
        self._model = None

    def set_context(self, ctx):
        self._context = ctx
        # TODO: Configure the task based on the game scenario

    def act(self, round_info: RoundInfo, available_actions: AvailableActions):
        # Create Decision
        decision = Decision.experiment()
        # Check available actions
        for experiment in available_actions.experiment:
            # Add observational data
            decision.add_experiment(treatment=None, n=self._num_obs)

            # Domain
            low, high = experiment.domain
            # Check domain type
            if type(low) is str:  # Categorical domain
                for value in experiment.domain:
                    decision.add_experiment(
                        treatment={experiment.name: value}, n=self._num_inter
                    )
            else:  # Numerical domain
                self._is_numeric = True
                for i in np.linspace(low, high, 10):  # 10 values uniformly spaced
                    decision.add_experiment(
                        treatment={experiment.name: i}, n=self._num_inter
                    )
        return decision

    def inform(self, samples_collection: SamplesCollection, feedback: Feedback) -> None:
        # Update samples / model
        # Update model
        if self._model is None:
            self._last = samples_collection
            return

        # Update the model
        for samples in samples_collection:
            data = samples.data
            X_train = data[self._features]
            y_train = data[self._target]
            # Partial Fit
            self._model.partial_fit(X_train, y_train)

    def answer(self):
        if self._model is None:

            def before(X, treatment, outcome, covariate_values):
                # Define features and target variable
                features = [treatment] + X
                target = outcome
                # Save the data
                self._features = features
                self._target = target

                # Define the treated and non-treated dataframes
                X_non_treated, X_treated = covariate_values

                # Define the training and prediction dataframes
                X_train = self._last[features]
                y_train = self._last[target]
                X_non_treated_pred = X_non_treated[features]
                X_treated_pred = X_treated[features]

                # Check both training and prediction dataframes have the same columns order
                assert (
                    list(X_train.columns)
                    == list(X_treated_pred.columns)
                    == list(X_non_treated_pred.columns)
                ), "Column order mismatch!"

                # Generate model
                # model = RandomForestRegressor(n_estimators=100, random_state=911)
                model = SGDRegressor(max_iter=1000, tol=1e-3)
                # Train the model
                model.partial_fit(X_train, y_train)
                # Save the model
                self._model = model

                # Predict on the last two appended rows
                Y_non_treated = model.predict(X_non_treated_pred)
                Y_treated = model.predict(X_treated_pred)

                # Calculate the difference for each pair of treated and non-treated
                if len(Y_non_treated) != len(Y_treated):
                    raise ValueError(
                        "Treated and non-treated predictions must have the same length."
                    )

                differences = Y_treated - Y_non_treated

                # Result
                result = pd.DataFrame(
                    differences, columns=["treatment_effect"], dtype=float
                )
                return result

            return before

        def after(X, treatment, outcome, covariate_values):
            # Define the treated and non-treated dataframes
            X_non_treated, X_treated = covariate_values

            # Define the prediction dataframes
            X_non_treated_pred = X_non_treated[self._features]
            X_treated_pred = X_treated[self._features]

            # Predict on the last two appended rows
            Y_non_treated = self._model.predict(X_non_treated_pred)
            Y_treated = self._model.predict(X_treated_pred)

            # Calculate the difference for each pair of treated and non-treated
            if len(Y_non_treated) != len(Y_treated):
                raise ValueError(
                    "Treated and non-treated predictions must have the same length."
                )

            differences = Y_treated - Y_non_treated

            # Result
            result = pd.DataFrame(
                differences, columns=["treatment_effect"], dtype=float
            )
            return result

        return after

    def to_spec(self) -> AgentSpec:
        return AgentSpec(
            class_=get_class_path(self.__class__),
            id=self.id,
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
            },
        )

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> "ExhaustiveAgent":
        return cls(
            id=spec.id,
            num_obs=spec.params["num_obs"],
            num_inter=spec.params["num_inter"],
        )
