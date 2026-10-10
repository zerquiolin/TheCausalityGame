# Uncommitted change review

No commits or staging were performed. The scientific code in the final wheel was tested outside the checkout; the root trackers are outside this Git repository.

## Review entry points

- `pyproject.toml`, `.python-version`, `constraints-publication-py312.txt`: supported Python and dependency versions.
- `scripts/comparisons/bootstrap_r.R`, `r-backends.lock.json`, `agent/inferers/r_backends/cate_backend.R`: R compatibility, shallow locked source fetches, install stamps and checked provenance.
- `TheCausalityGame/core/cli/main.py`, `README.md`, `examples/publication_smoke.json`: executable installation/example/CLI path.
- `tests/integration/conftest.py`, `tests/integration/test_runtime.py`, `tests/integration/test_authoritative_backends.py`: runtime isolation and actual backend checks.
- `scripts/comparisons/compare_cate_inferers.py`, `tests/unit/scripts/test_comparison_reports.py`: Hill binary coding before paired comparisons.
- `.github/workflows/ci.yml`: constrained Python/R integration and independent wheel verification; remote CI remains pending.
- `docs/PUBLICATION_ENVIRONMENT.md`, `docs/PRE_SUBMISSION_VALIDATION.md`, `docs/validation_summary.json`: setup, scientific limitations and final evidence.

## Outcomes

- Full suite: 162 passed, zero skips; 69.43% statement coverage.
- Clean installed wheel: 24 selected integration/CLI cases passed; seven-agent reduced Hill manifest valid.
- Separate base-only wheel: pip check/import/graph CLI example pass without research extras.
- Paired comparison: 105/105 upstream rows succeed; five historical empirical zero-covariate failures remain visible. Strict report audit: zero errors, five warnings.
- Full multi-seed manuscript rerun is pending; original manuscript figures and caches were preserved.

## File hashes for this review snapshot

These hashes identify the current uncommitted files, not a published or committed release. Changing a file after this report requires regenerating its hash.

| File | SHA-256 |
| --- | --- |
| `.github/workflows/ci.yml` | `9885650ab9ebb52c011aaefabc3d3de337bd9bbdf6b8ab4ce7207667130a7d63` |
| `.gitignore` | `abfdcc1f91274332910aeb4f77f03e164d24c1bd99464291423f953dc82e3e45` |
| `.python-version` | `7b55f8e67b5623c4bef3fa691288da9437d79d3aba156de48d481db32ac7d16d` |
| `README.md` | `426b454a289aa111927c42ab85e4fbaa86002fb71f88a3932c0b51859167dfa6` |
| `TheCausalityGame/agent/inferers/REFERENCES.md` | `935fa259beffb1dd49151e99ae316cc70067ae93d7ff38e429e087c0a0ae3a8f` |
| `TheCausalityGame/agent/inferers/r_backends/cate_backend.R` | `c430baebb94776d7fe7805f82abed6d7ce4b93e028484fe3258cc54898135994` |
| `TheCausalityGame/core/cli/main.py` | `0800ab0a74bf7e332f6e1aed188b0c170a36d4319b04228ebd32201fdfe0c560` |
| `constraints-publication-py312.txt` | `4769976e71744db6d525ac05d941643fb6b36ef14078fd76577bd732f58d7237` |
| `docs/PRE_SUBMISSION_VALIDATION.md` | `19bf8580b1f0facf521f7786e884f33357f592854693118f6867341549aec89b` |
| `docs/PUBLICATION_ENVIRONMENT.md` | `0bf825facd33e801aed26abdd4124b3edfcbae17c67357870a2839d12247a62a` |
| `docs/validation_summary.json` | `1f7dd13a1c4cd7f2e8002050049af451670be1125d5b1fe4837b2e633a322590` |
| `examples/publication_smoke.json` | `cb8135b832e72fa2c90a66a4ad81eb158c0e0c0b88cbdde946118f03ad2a72ef` |
| `pyproject.toml` | `27a97b88e0670cf1585a42683ce84f61ba1b80ac7a28961661a7feb7e7135f6c` |
| `scripts/comparisons/README.md` | `15824b4b5b212b1ed3a3ee75e58c17ab7c1134cfc075055e02002845d08f0e8d` |
| `scripts/comparisons/bootstrap_r.R` | `3bf9943481c325d384d1a1f2c7964f978bd62824599fab32b83e52681fd34a55` |
| `scripts/comparisons/compare_cate_inferers.py` | `74e6f7c3ac98f7a1f09aa001b1e5130ef7dedfc0b633f1601ac894d70fbe3a9e` |
| `scripts/comparisons/r-backends.lock.json` | `dfa61cf5cedfc595abc554a5d2c6e7bd3dc574a8f8c3da1d07d01eaa8380a446` |
| `tests/integration/conftest.py` | `a312d26f540643cbd657b9ea260d755a98b22e059ecc41efc2de709b6b02ac9d` |
| `tests/integration/test_authoritative_backends.py` | `6d3006bcbaa51edf8419192bef35d5e3745efde4b4936b492f9e361a35ec1f18` |
| `tests/integration/test_runtime.py` | `977df849153f4054461d5804289915404652f46312e8c4d8ce8699d7c48a429f` |
| `tests/unit/infra/test_cli.py` | `8a6e9b8b3dbc2d987157f33211dd1ee6abe46b093ccfe619e51ca3df1dddd75c` |
| `tests/unit/scripts/test_comparison_reports.py` | `ac71ebcb8f5a0c2e2a0a0ee09ff9c7d0696cb9ba063717ad0c25d2accbee9d96` |
