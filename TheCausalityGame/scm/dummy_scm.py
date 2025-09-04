# scm/dummy_scm.py
from core.contracts.scm import SCMContract


class DummySCM(SCMContract):
    """A dummy SCM that just produces fixed samples."""

    def generate_samples(self, interventions: dict | None = None, n: int = 100):
        return {"X": [1] * n, "Y": [2] * n}
