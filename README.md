
```cmd
ruff check .      # docstring coverage if D-rules enabled
mypy src          # type-check
pytest            # tests
```

---

# Important

---

The environment loop:

Agent -> Action -> Environment -> ActionOutcome -> Agent.

And inside the ActionOutcome, there is a feedback field that contains the metrics and other information about the action taken by the agent.

But here is an important note: The feedback yields the evaluation of the metrics given by the answer method before even informing the agent about the outcome of its current action. This is to ensure that the agent knows the current state of his answer, the result of his actions, and improves its next action / answer based on that feedback.

---

# JSON Game Instances

You’re thinking in exactly the right direction: keep the user-facing JSON simple, readable, and modular, and let code do the heavy lifting. Your sample is expressive but a bit “busy” for most users; the version below trims boilerplate while staying explicit where it matters (class to load + minimal config).

Below are two complete, copy-pasteable game manifests that fit our current foundation:
 • a Minimal manifest (cleanest UX),
 • a Full manifest (budgets, hooks, multiple metrics, per-component seeds).

They validate against our ProblemInstance Pydantic model and are friendly to tcg validate.

⸻

1) Minimal, clean manifest (great DX/UX)

```json
{
  "schema_version": "1.0.0",
  "id": "bn_cate_tiny",
  "scm_spec": {
    "class": "TheCausalityGame.scm.bayes:BayesianNetworkSCM",
    "config": {
      "nodes": [
        {
          "class": "TheCausalityGame.scm.bayes:BNBinaryNode",
          "name": "Z",
          "role": "controllable",
          "probs": [0.5, 0.5]
        },
        {
          "class": "TheCausalityGame.scm.bayes:BNBinaryNode",
          "name": "X",
          "role": "observable",
          "probs": [0.5, 0.5]
        },
        {
          "class": "TheCausalityGame.scm.bayes:BNNumericalChild",
          "name": "Y",
          "role": "observable",
          "parents": ["X", "Z"],
          "equation": "1.0 *X + 2.0* Z",
          "noise": {"class": "TheCausalityGame.scm.noise:Gaussian", "config": {"mean": 0, "std": 1}}
        }
      ],
      "edges": [["Z","Y"], ["X","Y"]]
    }
  },
  "mission_spec": {
    "class": "TheCausalityGame.missions.example:GenericPredictionMission",
    "config": {
      "mission_kind": "pred_v1",
      "accepted_deliverables": ["pred_grid_v1", "predict_fn_v1"],
      "eval_grid": {"n": 128, "seed": 42}
    }
  },
  "agent_specs": [
    {
      "id": "random_baseline",
      "class": "TheCausalityGame.agents.random:RandomAgent",
      "config": {"max_batch": 64}
    }
  ],
  "metric_specs": {
    "behavior": { "id": "rounds", "class": "TheCausalityGame.evaluators.behavior:RoundsUsed", "config": {} },
    "result":   { "id": "mse",    "class": "TheCausalityGame.evaluators.regression:MSEMetric", "config": { "supported": [["pred_v1","pred_grid_v1"], ["pred_v1","predict_fn_v1"]] } }
  },
  "custom_metric_specs": [],
  "run_plan": {
    "rounds": 5,
    "scheduler": "round_robin",
    "concurrency": 1,
    "budgets": { "time_s": 30, "samples": 5000, "memory_mb": 512 }
  },
  "seeds": {"global": 12345},
  "hook_plan": [],
  "artifacts_policy": {}
}
```

Why this is nice
 • Every component is just {"class": "module:Class", "config": {...}}.
 • Mission specifies what kinds of deliverables it accepts (IDs like "pred_grid_v1" or "predict_fn_v1"), but the agent doesn’t need to put that into JSON—agents return typed Python decisions; the runtime normalizes for persistence.
 • SCM node definitions are short (role + minimal params).

⸻

2) Full manifest (budgets, multiple metrics, hooks, per-component seeds)

```json
{
  "schema_version": "1.0.0",
  "id": "physics_ate_benchmark_v2",
  "scm_spec": {
    "class": "TheCausalityGame.scm.physics:SpringMassDamperSCM",
    "config": {
      "params": {"k": 1.2, "c": 0.1, "m": 1.0},
      "observable": ["x", "v"],
      "controllable": ["u"]
    }
  },
  "mission_spec": {
    "class": "TheCausalityGame.missions.ate:ATEMission",
    "config": {
      "mission_kind": "ate_v1",
      "accepted_deliverables": ["ate_scalar_v1", "ate_fn_v1"],
      "estimation_window": [0.0, 10.0],
      "seed": 2025
    }
  },
  "agent_specs": [
    {
      "id": "exhaustive",
      "class": "TheCausalityGame.agents.exhaustive:ExhaustiveAgent",
      "config": {"grid": [0.0, 0.5, 1.0], "per_round_limit": 500, "rng_seed": 7}
    },
    {
      "id": "causal_rl",
      "class": "TheCausalityGame.agents.rl:CausalRLAgent",
      "config": {"hidden": 128, "lr": 0.0003, "max_steps": 2000, "rng_seed": 99}
    }
  ],
  "metric_specs": [
    {
      "id": "abs_err",
      "class": "TheCausalityGame.evaluators.scalar:AbsoluteError",
      "config": {"supported": [["ate_v1","ate_scalar_v1"], ["ate_v1","ate_fn_v1"]]}
    },
    {
      "id": "sample_efficiency",
      "class": "TheCausalityGame.evaluators.behavior:SamplesUsed",
      "config": {"direction": "down"}
    }
  ],
  "run_plan": {
    "rounds": 20,
    "scheduler": "round_robin",
    "budgets": {"time_s": 120, "samples": 20000, "memory_mb": 1024}
  },
  "seeds": {
    "global": 424242,
    "agents": {"exhaustive": 7, "causal_rl": 99},
    "mission": 2025,
    "scm": 1337
  },
  "hook_plan": [
    {
      "id": "progress_bar",
      "class_path": "TheCausalityGame.core.hooks.progress:ConsoleProgress",
      "events": ["on_run_start", "on_round_finish", "on_run_finish"],
      "priority": 90,
      "config": {"show_eta": true}
    },
    {
      "id": "dataset_indexer",
      "class_path": "TheCausalityGame.core.hooks.datasets:IndexWriter",
      "events": ["after_env_generate_samples"],
      "priority": 80,
      "config": {"write_sha256": true}
    },
    {
      "id": "figures_after_scores",
      "class_path": "TheCausalityGame.core.hooks.outputs:MaterializeOutputs",
      "events": ["on_run_finish"],
      "priority": 70,
      "config": {
        "outputs": [
          {
            "class": "TheCausalityGame.outputs.plots:ScoreRadar",
            "config": {"out": "plots/score_radar.png"}
          },
          {
            "class": "TheCausalityGame.outputs.tables:Leaderboard",
            "config": {"out": "plots/leaderboard.md"}
          }
        ]
      }
    }
  ],
  "artifacts_policy": {
    "datasets": {"gzip": true, "shard_rows": 5000},
    "transcripts": {"gzip": false},
    "log_level": "INFO"
  }
}
```

Notes
 • Multiple agents → round-robin scheduler by default.
 • Two metrics: one deliverable metric, one behavior metric.
 • Hooks:
 • console progress,
 • dataset sharding/indexing after sample generation,
 • outputs rendered on run finish (plots/tables).
 • Artifacts policy gives users control over compression and sizes.

⸻

Why this UX is good
 • Users only specify what they want (components + minimal config).
They never have to construct framework-specific payloads.
 • Class paths are explicit (no magic). If they publish a new agent/mission, they just reference its import path.
 • Mission compatibility is clear: mission_kind and accepted_deliverables are declared in the mission config (the core uses this to validate metric compatibility and agent submissions).
 • No leakage of runtime details into the manifest:
 • No callable objects in JSON.
 • No episode subtleties — just rounds.
 • Budgets are simple and explicit.

⸻

CLI quick check

# validate shape

tcg validate examples/infra/manifest_template.json

# run (Phase 2 runtime will provide)

tcg run --manifest examples/infra/manifest_template.json --runs-dir runs/

⸻

Mapping to the runtime semantics
 • Agent API: Agents return typed decisions (Intervene, SubmitFinal(Deliverable)).
 • Normalize (internal): converts decision → Action (JSON-safe) + raw payload for mission validation.
 • Mission: validates the deliverable by type and/or schema/protocol id; returns a DeliverableHandle.
 • Metrics: declare support for (mission_kind, deliverable_kind); consume the handle, compute scores.
 • Artifacts:
 • transcripts.jsonl: JSON-only Actions/StepRecords
 • datasets/: gz JSONL shards from sampling
 • metrics_raw.jsonl + scores.json: per-metric results + aggregates
 • plots/: created by output plugins via hooks

⸻

If you want, I can also drop a tiny hello-world SCM/Mission/Agent/Metric set (10–20 lines each) so users can run a manifest end-to-end immediately.

# Run Manifest

## Usage examples

```bash
 • Dev mode (verbose + callables allowed):

tcg run examples/infra/manifest_template.json --mode dev --debug
```

```bash
 • Restricted mode (quiet, benchmark-safe):

tcg run examples/infra/manifest_template.json --mode restricted
```

# Usage of Decisions

```python
from TheCausalityGame.core.contracts.decisions import Decision, ExperimentSpec

# Build in one call with heterogeneous inputs
d1 = Decision.experiment((None, 300), ExperimentSpec({'X': 1}, 200))
# Add more experiments immutably
d1 = d1.add_experiment({'Z': 0}, 100)

# Build incrementally from empty experiment decision
d2 = Decision.experiment((None, 100))\
             .add_experiment({'X': 1}, 50)\
             .extend([({'Z': 1}, 25), (None, 25)])

# Submit answer
d3 = Decision.answer()
```
