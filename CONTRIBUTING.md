# Contributing

This repository accompanies a master's thesis and its companion paper
(`paper/PaperCIE_LaTeX/`, submitted to Computers & Industrial Engineering). It is not
under active feature development, but issues and pull requests are welcome for:

- Reproducibility problems (a command in `README.md` that doesn't work as
  documented, a pinned dependency that no longer resolves, a platform-specific
  bug).
- Bugs in `code/` (the `thesis_fjsp_drl` package) found while reusing it —
  environments, baselines, or the stats pipeline.
- Corrections to the thesis text, figures, or tables, if you spot a
  discrepancy between what's reported and what the code/data actually produce.

## Before opening an issue

Check ["Quick smoke test"](README.md#quick-smoke-test-5-10-minutes) in the root
`README.md` first — it validates the whole pipeline (env, PDR/GA/CP-SAT/RL,
disruption rollout, a real training step) in 5-10 minutes and is the fastest
way to tell whether a problem is environmental or a real bug.

## Development setup

```bash
cd code
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest -m "not slow" -q
```

See `code/README.md` for the package's internal module layout and design
decisions, and the root `README.md` for the full reproduction guide (E0-E8).

## Pull requests

- Keep changes scoped — this codebase has 752 tests and a frozen, pre-registered
  experimental grid (`thesis/PreRegistration.md`); changes that would alter
  already-reported thesis numbers need an explicit note in the PR description,
  not a silent recomputation.
- Run `pytest -m "not slow"` locally before submitting; CI (`.github/workflows/tests.yml`)
  runs the same check plus `scripts/verify_optima.py` on every push/PR.
- Match the existing code style (typed, module-docstring-documented — see any
  file under `code/envs/` or `code/models/` for the convention).

## License

MIT — see [`LICENSE`](LICENSE). By contributing, you agree your contribution
is licensed under the same terms.
