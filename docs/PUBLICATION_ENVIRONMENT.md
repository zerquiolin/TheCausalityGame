# Publication environment

The supported Python range is `>=3.12,<3.13`. This is deliberately the version family
tested for the publication artifact; extending it requires a new dependency resolution
and the real-backend integration suite. The local reference run uses Python 3.12.13,
R 4.6.1 and macOS arm64. CI uses Python 3.12 and R 4.6.1 on Linux; the updated CI
configuration is pending an actual remote run.

`lingam==1.13.0` requires SciPy <=1.13.1. The attempted install into the machine's
Python 3.14 base environment selected a SciPy source build and failed on OpenBLAS
discovery. Use the project interpreter, not the base interpreter. Ruff and mypy now
target Python 3.12, matching package metadata and `.python-version`.

## Install from a clean environment

Run from the package repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-publication-py312.txt -e '.[authoritative]' pytest pytest-cov
python -m pip check
python -c 'import sys, TheCausalityGame; print(sys.executable); print(TheCausalityGame.__file__)'
```

`constraints-publication-py312.txt` records the 66 installed versions in the dependency
closure of the runtime, six Python/R adapter interfaces, plotting, tests and wheel tools.
It replaces machine-specific Conda build URLs with normal version requirements and
retains the NOTEARS Git commit. It is a version constraint set, not a hash-locked or
platform-independent binary archive. Optional notebooks and additional development
tools are outside that constrained closure; install `.[dev,analysis]` separately with
the same constraint file if needed. Matplotlib is a runtime dependency because core
DAG/plot contracts import it during ordinary CLI initialization.

## R backends and compatibility scope

```sh
Rscript scripts/comparisons/bootstrap_r.R .r-library
export TCG_R_LIBS="$PWD/.r-library"
R_LIBS_USER="$TCG_R_LIBS" Rscript -e 'print(.libPaths()); print(R.version.string)'
```

Python's bridge maps `TCG_R_LIBS` to `R_LIBS_USER` for its R subprocesses. R is not
installed by pip. Building the locked R sources needs Git/network access, the R
development toolchain and a C++ compiler; first setup can take minutes.

| Dependency | Locked source/version | Local compatibility modification |
| --- | --- | --- |
| jsonlite | 1.9.1 | None |
| remotes | 2.5.0 | None |
| causalTree | `51a8762bff0ab17b971bce852cb37bece2d8240f` | R 4.5+ public variable lookup, legacy `Free` compatibility, C17 callbacks |
| causalToolbox | `75acc74ae689fddebe63a43597ff21f3ec233bd4` | Preserve one-column features; omit unsupported `OOBhonest` in S_RF |
| Rforestry | `92b396d77d7ce2bbaba1c6d97c8861ea43ae1566` | Treat unknown `detectCores()` as unknown, retaining requested `nthread` |

The bootstrap verifies Git HEAD before local installation and writes `TCGSourceCommit`
and `TCGCompatibilityPatch` into DESCRIPTION. Repeated setup skips only a matching
source/patch installation. R result provenance checks the installed source stamp.
An older unstamped local library requires rerunning the bootstrap. These stamps record
the checked-out source and patch revision; they are not cryptographic verification of
an installed binary. Nested CRAN dependencies beyond the explicit lock entries are
not fully frozen. The S_RF/EstimateCate adapter does not provide OOB-honest intervals
or local-conformal inference.

Inspect installed source metadata with:

```sh
R_LIBS_USER="$TCG_R_LIBS" Rscript -e 'for (p in c("causalTree", "causalToolbox", "Rforestry")) { d <- packageDescription(p); print(d[c("Package", "Version", "TCGSourceCommit", "TCGCompatibilityPatch")]) }'
```

## Verify the installation

```sh
python -m pytest tests -p no:cacheprovider --cov=TheCausalityGame --cov-report=term
python -m scripts.comparisons.run_all --samples 120 --rounds 2
```

The normal suite must include all six `authoritative` integration cases; a skipped
backend is an incomplete publication-environment check. Do not interpret exit code 0
from the comparison runner as universal success: it deliberately writes failed rows.
Inspect JSONL statuses and run the strict report audit as described in
[PRE_SUBMISSION_VALIDATION.md](PRE_SUBMISSION_VALIDATION.md).

For the existing named Conda environment, activate `thecausalitygame` and install from
this checkout using `python -m pip install -c constraints-publication-py312.txt -e
'.[authoritative]'`. The audit discovered its earlier editable metadata pointed to an
old checkout; local source tests were valid because imports resolved from the current
working directory, but that did not prove the installed package was current. The audit has now refreshed that editable install to the current checkout.

## Wheel verification outside the source tree

```sh
python -m pip wheel --no-deps --wheel-dir dist .
python3.12 -m venv /tmp/tcg-wheel-check
/tmp/tcg-wheel-check/bin/python -m pip install -c constraints-publication-py312.txt 'dist/thecausalitygame-0.1.0-py3-none-any.whl[authoritative]' pytest
cd /tmp
/tmp/tcg-wheel-check/bin/python -m pip check
/tmp/tcg-wheel-check/bin/tcg --help
/tmp/tcg-wheel-check/bin/python -c "from importlib.resources import files; root=files('TheCausalityGame.agent.inferers'); assert root.joinpath('REFERENCES.md').is_file(); assert root.joinpath('r_backends/cate_backend.R').is_file()"
```

Confirm `TheCausalityGame.__file__` points into the new environment's site-packages.
Run the integration tests by their absolute repository path with
`--import-mode=importlib`, `TCG_R_LIBS` set to the absolute local R library, and a
writable `MPLCONFIGDIR` in restricted environments. Repository-relative benchmark
asset paths still require running comparisons from the checkout; those JSON assets
are not installed as standalone wheel data.
