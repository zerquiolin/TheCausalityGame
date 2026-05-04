# Contributing to The Causality Game

The Causality Game is a research framework for reproducible active causal inference experiments. Contributions should preserve the benchmark guarantees described in the thesis: explicit contracts, serializable specifications, deterministic execution, and clean separation between agent-visible data and evaluator-only ground truth.

## Development Setup

1. Create a virtual environment with Python 3.10 or newer.
2. Install the package in editable mode:

   ```bash
   pip install -e .[dev]
   ```

3. Run tests before submitting changes:

   ```bash
   pytest
   ```

## Contribution Guidelines

- Implement new SCMs, agents, missions, metrics, validators, hooks, and inferers against the relevant contract interfaces.
- Keep DTOs as the only agent-facing communication boundary. Do not add side channels that expose SCM internals or evaluator-only data.
- Make new components serializable through specs so experiments can be reproduced from configuration files and seeds.
- Add focused tests for new behavior, including deterministic replay when randomness is involved.
- Document new public components with minimal examples or manifest snippets.
- Avoid changing existing spec schemas without a migration path or explicit versioning.

## Pull Request Checklist

- The change has a clear motivation and scope.
- Unit or integration tests cover the new behavior.
- New public classes can be instantiated from specs.
- Randomness is seeded or explicitly documented.
- Documentation or examples are updated when user-facing behavior changes.

## License

By contributing, you agree that your contribution is licensed under the Apache-2.0 license used by this repository.
