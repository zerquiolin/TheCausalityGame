# Abstract
from abc import ABC, abstractmethod

# DAG
from TheCausalityGame.core.contracts.dag import DAG

# SCM
from TheCausalityGame.core.contracts.scm import SCM


class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self):
        """
        Generate an object (e.g., a DAG or SCM) based on provided parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AbstractSCMGenerator(AbstractGenerator):
    @abstractmethod
    def generate(self) -> DAG:
        """
        Generate a DAG based on provided configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AbstractSCMGenerator(AbstractGenerator):
    @abstractmethod
    def generate(self) -> SCM:
        """
        Generate a SCM based on provided configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
