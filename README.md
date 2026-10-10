# The Causality Game

Foundation for designing, simulating, and benchmarking causal inference agents against richly configurable structural causal models (SCMs). The project provides a modular runtime, a contract-driven component model, and a CLI for executing fully specified problem instances end to end.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI Reference](#cli-reference)
- [Problem Instance Specification](#problem-instance-specification)
- [Extending the Game](#extending-the-game)
- [Artifacts & Outputs](#artifacts--outputs)
- [Testing & Quality](#testing--quality)
- [Support & Further Reading](#support--further-reading)

## Overview
The Causality Game models an interactive loop in which an **agent** explores an SCM, receives **feedback** from a **mission**, and iteratively improves its causal estimate before submitting a final answer. The system is purpose-built for reproducible experimentation:

- Problem instances describe everything required to evaluate one or more agents: the SCM, mission, metrics, runtime budgets, hooks, and reproducibility settings.
- Components are defined through strongly typed Pydantic specs and instantiated dynamically at runtime, allowing teams to plug in custom implementations without modifying the core.
- The CLI (`tcg`) loads a manifest, orchestrates the run, and stores structured artifacts for later analysis.

**Interaction loop:** Agent → Decision → Environment → Samples → Agent. The environment records
metric feedback in the transcript after each round. `Agent.inform` receives observable samples;
the current interface does not deliver metric feedback to the agent. Answers are captured before
that round's samples are delivered, so a submission round records the estimate after the previous
update. See [publication validation](docs/PRE_SUBMISSION_VALIDATION.md) for the tested lifecycle
and remaining manuscript alignment work.

## Key Features
- Modular component contracts for agents, SCMs, missions, metrics, hooks, and validators.
- Parallel or sequential execution of multiple agents with resource budgets enforced per run.
- Deterministic replay via explicit seeding and serialized runtime settings.
- Development versus production runtime modes with scoped logging and artifact generation.
- Typer-based CLI with direct and subcommand usage for single-line execution of manifests.
- Comprehensive serialization layer enabling manifests to stay readable while remaining strict and self-validating.
- Published inferers use pinned, author-maintained implementations and record backend provenance in every belief snapshot; the historical handwritten implementations remain available under `agent.inferers.empirical`.

## Architecture

### Runtime flow
- `tcg` CLI parses a manifest into `ProblemInstanceSpec`.
- `Runner` bootstraps logging, artifact directories, and hook managers, then executes each agent either sequentially or via thread/process pools.
- `Game` constructs the agent, SCM, mission, metrics, and orchestrates the environment loop.
- `Environment` enforces budgets, triggers hooks, collects samples, evaluates answers, and records a transcript entry per round.
- Metrics, hooks, and artifact writers post-process results for reporting.

### Repository layout
| Path | Description |
| --- | --- |
| `TheCausalityGame/core/contracts` | Pydantic specs and abstract base classes for all runtime components. |
| `TheCausalityGame/core/runtime` | Runner, environment, and game orchestration logic. |
| `TheCausalityGame/core/infrastructure` | Registry, serialization helpers, artifact writer, decision helpers, logging utilities. |
| `TheCausalityGame/core/managers` | Budget and hook managers coordinating runtime policies. |
| `TheCausalityGame/agent` | Agent wrappers plus shipped inferers, deciders, and unified policies. |
| `TheCausalityGame/agent/inferers/empirical` | Reproducible snapshots of the original handwritten inferers. |
| `TheCausalityGame/mission`, `scm`, `metric`, `hook` | Domain-specific implementations shipped with the game. |
| `scripts/main.py` | Legacy helper replicating the CLI `run` command. |
| `scripts/comparisons` | Paired empirical-versus-authoritative benchmark runners and R lock/bootstrap files. |
| `tests/` | Unit tests and fixtures. |

## Installation

### Prerequisites
- Python 3.12 (the supported and tested publication environment)
- macOS, Linux, or Windows with POSIX-compatible shell
- Optional: `make`, `pipx`, or your preferred virtual environment tooling

### Steps
1. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. Install the project in editable mode along with runtime dependencies:
   ```bash
   pip install -e .
   ```
3. (Optional) Install development tooling:
   ```bash
   pip install -e .[dev]
   ```

### Authoritative inferers

The publication environment uses Python 3.12. `lingam==1.13.0` requires SciPy ≤ 1.13.1,
which has no Python 3.14 wheels. Python 3.13/3.14 support is not currently declared. Use
`python -m pip` from the activated environment to avoid installing into a different interpreter.
See [the reproducible environment instructions](docs/PUBLICATION_ENVIRONMENT.md) for tested
dependency constraints, R compatibility patches, clean installation and verification commands.

The lightweight install intentionally does not install research backends. Install the Python
backends with:

```bash
pip install -e '.[analysis,authoritative]'
```

The public PC, NOTEARS, and DirectLiNGAM paths then call `causal-learn==0.1.4.8`, the
authors' NOTEARS repository at commit `4a9ab19f`, and `lingam==1.13.0`, respectively.
They accept only their upstream-native parameters and never fall back to an empirical method.

The S-learner, transformed-outcome tree, and honest causal tree additionally require R and the
pinned R packages. Bootstrap the project-local R library with:

```bash
Rscript scripts/comparisons/bootstrap_r.R
export TCG_R_LIBS="$PWD/.r-library"
```

This installs `forestry-labs/causalToolbox` at commit `75acc74a` and
`susanathey/causalTree` at commit `51a8762b`; their GPL-licensed source remains an external
dependency. Exact citations, licenses, versions/commits, and adapter versions are included in
snapshot provenance. PC returns a serializable CPDAG; NOTEARS and DirectLiNGAM return directed
NetworkX graphs. These methods require numeric supported inputs, and the CATE adapters require a
binary treatment. Unsupported data and missing backends raise explicit errors.
See the [inferer reference table](TheCausalityGame/agent/inferers/REFERENCES.md) for the exact
upstream entrypoints, pins, licenses, and primary citations.

For an explicit historical run, change a published inferer path from, for example,
`TheCausalityGame.agent.inferers.pc:PCInferer` to
`TheCausalityGame.agent.inferers.empirical.pc:PCInferer`. The project-specific DAG, SCM, and sparse
symbolic SCM inferers have no authoritative upstream and remain empirical compatibility wrappers.

After installation the `tcg` CLI is available on your PATH.

## Quickstart
1. Use the runnable `examples/publication_smoke.json` graph example, or prepare a manifest using [Problem Instance Specification](#problem-instance-specification). The example uses the empirical PC implementation and needs no external research backend.
2. Run the manifest through the CLI:
   ```bash
   tcg run examples/publication_smoke.json --run-dir runs
   ```
   or use the shortcut form:
   ```bash
   tcg examples/publication_smoke.json
   ```
3. Inspect the generated artifacts under `runs/<problem_id>/<timestamp>` when the manifest requests development-mode outputs.

### Minimal manifest example
```json
{
  "class_": "TheCausalityGame.core.contracts.problem_instance:ProblemInstance",
  "schema_version": "1.0.0",
  "id": "demo_instance",
  "agents": [
    {
      "id": "exhaustive",
      "class_": "TheCausalityGame.agent.composable:ComposableAgent",
      "inferer": {
        "class_": "TheCausalityGame.agent.inferers.outcome_regression:OutcomeRegressionInferer"
      },
      "decider": {
        "class_": "TheCausalityGame.agent.deciders.exhaustive:ExhaustiveDecider",
        "params": {"num_obs": 2, "num_inter": 2}
      }
    }
  ],
  "scm": {
    "class_": "<SCM module path>:<SCMClass>",
    "vars": [
      {
        "class_": "<SCM node module>:<NodeClass>",
        "name": "Z",
        "accessibility": "controllable",
        "domain": [0, 1],
        "parents": null,
        "noise_distribution": {
          "class_": "<Noise module>:<NoiseClass>",
          "params": {"std": 1.0}
        }
      }
    ],
    "dag": {
      "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
      "nodes": ["Z"],
      "edges": []
    }
  },
  "mission": {
    "id": "cate_mission",
    "class_": "TheCausalityGame.mission.conditional_average_treatment_effect:ConditionalAverageTreatmentEffectMission",
    "behavior_metric": {
      "class_": "TheCausalityGame.metric.behavior.rounds:RoundsBehaviorMetric",
      "params": {"alpha": 0.1}
    },
    "result_metric": {
      "class_": "TheCausalityGame.metric.result.pehe:PEHEResultMetric"
    },
    "result_validator": {
      "class_": "TheCausalityGame.metric.result.result_validator.cate_function_validator:ConditionalAverageTreatmentEffectFunctionValidator"
    }
  },
  "custom_metrics": [],
  "run_plan": {
    "execution": "sequential",
    "parallel_backend": "thread",
    "max_workers": 1,
    "budget": {
      "rounds": 50,
      "samples": 5000
    },
    "hook_plan": []
  },
  "seeds": {"global": 12345},
  "runtime": {
    "mode": "dev",
    "debug_level": 20
  }
}
```
> **Note:** Replace every placeholder wrapped in angle brackets with the concrete class paths and parameters that match your scenario (see [Extending the Game](#extending-the-game)). The structure above mirrors the required nesting and field names enforced by the specs.

## CLI Reference

### `tcg`
- Direct invocation without a subcommand defaults to running a manifest.
- Global option `--run-dir / -o` controls the artifact root.

### `tcg run`
```
Usage: tcg run [OPTIONS] PROBLEM_PATH

  Execute a problem instance definition.

Options:
  -o, --run-dir PATH  Directory where run artifacts will be stored.
  --help              Show this message and exit.
```

Error handling includes clear diagnostics for missing files, unreadable JSON, and validation errors from `ProblemInstanceSpec`.

## Problem Instance Specification

Problem instances are validated against `ProblemInstanceSpec`. JSON documents must use `class_` for implementation paths; `CommonSpec` does not declare a `class` alias. Key top-level fields:

| Field | Description |
| --- | --- |
| `class_` | Required implementation path, normally `TheCausalityGame.core.contracts.problem_instance:ProblemInstance`. |
| `schema_version` | Schema identifier for compatibility management. |
| `id` | Unique identifier for the run; used in artifact folder names. |
| `agents` | List of [`AgentSpec`](TheCausalityGame/core/contracts/specs/agent.py) entries defining either `inferer + decider` or a unified `policy`. |
| `scm` | [`SCMSpec`](TheCausalityGame/core/contracts/specs/scm.py) describing nodes, DAG structure, and implementation class. |
| `mission` | [`MissionSpec`](TheCausalityGame/core/contracts/specs/mission.py) including behavior/result metrics and validators. |
| `custom_metrics` | Optional list of [`MetricSpec`](TheCausalityGame/core/contracts/specs/metric.py) evaluated in addition to mission metrics. |
| `run_plan` | [`RunPlanSpec`](TheCausalityGame/core/contracts/specs/run.py) covering execution mode, parallel backend, budgets, and hook plan. |
| `seeds` | Mapping of component-specific seeds for reproducibility (`global`, `agents`, `mission`, `scm`, ...). |
| `runtime` | [`RuntimeSettingsSpec`](TheCausalityGame/core/contracts/specs/settings.py) toggling DEV/PROD mode and debug verbosity. |

### Budgets and hooks
- Budgets (`run_plan.budget`) enforce per-agent limits on rounds, time (seconds), samples, and memory.
- Hooks (`run_plan.hook_plan`) subscribe to lifecycle events enumerated in `HookEvent` (e.g., `before_act`, `after_eval`, `benchmark_end`), enabling custom logging, visualizations, or integrations.

### Metrics and missions
- Missions encapsulate evaluation logic and must expose an ID, description, behavior metric, result metric, and validator.
- Metrics express support for mission deliverables, typically via `params` entries such as accepted deliverable kinds.
- Feedback returned to agents carries the latest metric evaluations so they can adapt before answering.

## Extending the Game

### Implement a new agent
1. For compositional agents, implement an `Inferer` and/or a `Decider` under `TheCausalityGame.core.contracts.inferer` and `TheCausalityGame.core.contracts.decider`.
2. For tightly coupled algorithms, implement an `AgentPolicy` under `TheCausalityGame.core.contracts.agent_policy`.
3. Use the shipped wrappers `TheCausalityGame.agent.composable:ComposableAgent` or `TheCausalityGame.agent.combined:CombinedAgent` in manifests.
4. Implement `from_spec`/`to_spec` for the new component and reference its `module:Class` path in the manifest.

### Add a mission or metric
- Missions inherit from `TheCausalityGame.core.contracts.mission.Mission` and must coordinate behavior/result metrics plus validation.
- Metrics extend `TheCausalityGame.core.contracts.metric.Metric`, consuming mission deliverables to compute scores.
- Register new components by referencing their `module:Class` path inside the manifest spec.

### Create a new SCM
- Derive from `TheCausalityGame.core.contracts.scm.SCM`, provide node definitions (`SCMNodeSpec`), and ensure deterministic sampling respecting `NodeAccessibility`.
- Update the manifest’s `scm` spec with the new class path and parameters (e.g., structural equations, noise models).

### Hook into runtime events
- Implement a hook by inheriting from the appropriate hook base in `TheCausalityGame.hook`.
- Add an entry to `run_plan.hook_plan` referencing the hook class path, events to subscribe to, and optional config.

### Validate custom deliverables
- Supply a `ResultValidator` class referenced by `mission.result_validator` to enforce schema-level or semantic checks before scoring.
- Combine with custom metrics to produce leaderboard-ready outputs.

## Artifacts & Outputs
- Artifacts are written under `runs/<problem_id>/<timestamp>/`.
- Development mode (`runtime.mode = "DEV"`) enables:
  - Agent-specific transcripts (`agents/<agent_id>/transcript.json`) with sanitized decisions and samples.
  - Structured provenance (`provenance.json`) describing the execution environment.
  - Scoped log directories when loggers are configured.
- Production mode minimizes disk usage by suppressing transcripts and extended logs.

## Testing & Quality
- Run unit tests: `pytest`
- Static typing: `mypy`
- Linting & formatting: `ruff check .`
- Continuous integration can combine these commands to enforce quality gates.

## Support & Further Reading
- Explore the concrete implementations under `TheCausalityGame/` to see working examples of SCMs, missions, metrics, agents, and hooks.
- The `scripts/main.py` helper mirrors `tcg run` for programmatic invocation inside notebooks or REPL sessions.
- For academic context and practical case studies, refer to materials in the `thesis/` directory.

---

Licensed under Apache-2.0 — see `pyproject.toml` for attribution details.
