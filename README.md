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

**Interaction loop:** Agent → Decision → Environment → Samples + Feedback → Agent. Feedback contains metric evaluations of the current answer before the agent informs its next action, ensuring it can adapt using the latest assessment.

## Key Features
- Modular component contracts for agents, SCMs, missions, metrics, hooks, and validators.
- Parallel or sequential execution of multiple agents with resource budgets enforced per run.
- Deterministic replay via explicit seeding and serialized runtime settings.
- Development versus production runtime modes with scoped logging and artifact generation.
- Typer-based CLI with direct and subcommand usage for single-line execution of manifests.
- Comprehensive serialization layer enabling manifests to stay readable while remaining strict and self-validating.

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
| `TheCausalityGame/mission`, `scm`, `metric`, `hook` | Domain-specific implementations shipped with the game. |
| `scripts/main.py` | Legacy helper replicating the CLI `run` command. |
| `tests/` | Unit tests and fixtures. |

## Installation

### Prerequisites
- Python ≥ 3.10
- macOS, Linux, or Windows with POSIX-compatible shell
- Optional: `make`, `pipx`, or your preferred virtual environment tooling

### Steps
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
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

After installation the `tcg` CLI is available on your PATH.

## Quickstart
1. Prepare a problem instance manifest (see [Problem Instance Specification](#problem-instance-specification)). Save it, for example, as `examples/hill_demo.json`.
2. Run the manifest through the CLI:
   ```bash
   tcg run examples/hill_demo.json --run-dir runs
   ```
   or use the shortcut form:
   ```bash
   tcg examples/hill_demo.json
   ```
3. Inspect the generated artifacts under `runs/<problem_id>/<timestamp>` when the manifest requests development-mode outputs.

### Minimal manifest example
```json
{
  "class": "TheCausalityGame.core.contracts.problem_instance:ProblemInstance",
  "schema_version": "1.0.0",
  "id": "demo_instance",
  "agents": [
    {
      "id": "exhaustive",
      "class": "TheCausalityGame.agent.composable:ComposableAgent",
      "inferer": {
        "class": "TheCausalityGame.agent.inferers.cate:CATEInferer"
      },
      "decider": {
        "class": "TheCausalityGame.agent.deciders.exhaustive:ExhaustiveDecider",
        "params": {"num_obs": 2, "num_inter": 2}
      }
    }
  ],
  "scm": {
    "class": "<SCM module path>:<SCMClass>",
    "vars": [
      {
        "class": "<SCM node module>:<NodeClass>",
        "name": "Z",
        "accessibility": "controllable",
        "domain": [0, 1],
        "parents": null,
        "noise_distribution": {
          "class": "<Noise module>:<NoiseClass>",
          "params": {"std": 1.0}
        }
      }
    ],
    "dag": {
      "class": "TheCausalityGame.scm.dag.core:CoreDAG",
      "nodes": ["Z"],
      "edges": []
    }
  },
  "mission": {
    "id": "cate_mission",
    "class": "TheCausalityGame.mission.conditional_average_treatment_effect:ConditionalAverageTreatmentEffectMission",
    "behavior_metric": {
      "class": "TheCausalityGame.metric.behavior.rounds:RoundsBehaviorMetric",
      "params": {"alpha": 0.1}
    },
    "result_metric": {
      "class": "TheCausalityGame.metric.result.pehe:PEHEResultMetric"
    },
    "result_validator": {
      "class": "TheCausalityGame.metric.result.result_validator.cate_function_validator:ConditionalAverageTreatmentEffectFunctionValidator"
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

Problem instances are validated against `ProblemInstanceSpec`. JSON documents may supply `class` instead of `class_` thanks to field aliases. Key top-level fields:

| Field | Description |
| --- | --- |
| `class` | Optional when using the spec-only workflow; set it to a `ProblemInstance` class path when you want `build_from_spec` to construct a concrete instance. |
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
