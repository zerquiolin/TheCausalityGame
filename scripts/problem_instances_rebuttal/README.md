# Rebuttal problem instances

This directory contains the targeted NeurIPS rebuttal experiments. The
treatment-effect repair keeps the seven existing honest-causal-tree agent
configurations and the same 21 SCM seeds, but uses an explicit fixed horizon:

- 100 rounds
- 400 collected samples
- 400 seconds
- 2 MB measured memory

The fixed horizon is a forced termination condition. These runs must not be
described as evidence of adaptive stopping.

Generate or refresh the manifests:

```bash
python scripts/problem_instances_rebuttal/generate_manifests.py
```

Preview and execute:

```bash
python scripts/problem_instances_rebuttal/run_rebuttal.py --dry-run
python scripts/problem_instances_rebuttal/run_rebuttal.py --seed 47
python scripts/problem_instances_rebuttal/run_rebuttal.py --force --seed 47
python scripts/problem_instances_rebuttal/run_rebuttal.py
```

The runner resumes by skipping any manifest that already has a complete,
non-invalidated transcript for all seven agents. `--force` creates a new
timestamped execution after an implementation repair. It never removes earlier
run artifacts.
