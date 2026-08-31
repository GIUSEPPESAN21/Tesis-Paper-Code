# Final Reproducibility Report

**Status:** Steps 1–4 complete and verified. Step 5 partially complete — see §5 for
exactly what's done vs. blocked on one action from you (git identity configuration;
I'm not permitted to run `git config` myself). Nothing below is aspirational: every
claim is either a command I actually ran and the output I actually saw, or is
explicitly flagged as untested/pending with the reason why.

---

## 1. Reproducibility audit (seeds, dependencies, hardware)

### Random seeds — audited by direct code inspection, not assumed

- **Grep-verified across `envs/`, `models/`, `baselines/`, `experiments/`:** zero bare
  `np.random.*` or stdlib `random.*` global-state calls exist anywhere. Every RNG use is
  a `numpy.random.default_rng(seed)` instance, threaded explicitly from a function
  argument.
- **Every training/evaluation entrypoint exposes a seed CLI argument**:
  `run_evaluation.py --seeds`, `train_hetero.py`/`train_dynamic.py`/`train_ft06.py
  --seed`.
- **RL training determinism:** `sb3_contrib.MaskablePPO(seed=...)` seeds
  torch/numpy/Python's `random` internally (Stable-Baselines3's own contract) — no
  separate `torch.manual_seed` call was needed or missing.
- **Real gap found and fixed:** `baselines/cpsat.py`'s `solve_cpsat`/`solve_cpsat_fjsp`
  never set `solver.parameters.random_seed` at all. Checked OR-Tools' actual default
  (`1`, not `0` — verified by instantiating a bare `CpSolver()` and reading the
  parameter) before fixing this, specifically so the fix couldn't silently change
  already-published results on a future re-run: the new `random_seed: int = 1` default
  parameter reproduces exactly what every prior CP-SAT call in this thesis already did
  implicitly. **Known, irreducible limitation, now documented in the code and in
  `README.md`:** with `num_search_workers > 1` (the default, `8`), a **proven-`OPTIMAL`**
  result's objective is deterministic regardless of the seed, but a **time-limited
  `FEASIBLE`** result can still vary run-to-run due to parallel-search thread
  scheduling — a property of CP-SAT's parallel portfolio search, not fixable by seeding
  alone. This is almost certainly the mechanism behind the `la40` CP-SAT timing anomaly
  already documented in `Thesis.md` §6.5.1. `--num-workers 1` gives bit-exact
  `FEASIBLE` reproduction at the cost of speed.
- **GA** (`baselines/ga.py`): already correctly seeded via `np.random.default_rng(seed)`.

### Dependencies — audited against actual imports, not assumed complete

Enumerated every third-party import across the codebase (`grep` for `^import`/`^from`
in every `.py` file) and diffed against `requirements.txt`/`pyproject.toml`. **Real gap
found and fixed:** `scipy` (1.15.2) and `statsmodels` (0.14.6) are used throughout
`experiments/stats/` (Wilcoxon, Friedman, the Nemenyi critical-difference quantile) but
were pinned in neither file — installed in the working `.venv` only incidentally, not
declared. Added both, at the exact installed versions (`pip freeze`-verified). Also
added `optuna` to `pyproject.toml`'s optional-dependencies (was requirements.txt-only;
now consistent between both files, under a new `hpo` extra since it's only needed to
re-run the Week 21 hyperparameter search, not to reproduce E0–E8).

Full pinned set, verified against `pip freeze` on the actual environment that produced
every number in this thesis: `numpy==2.4.6`, `pandas==2.2.2`, `matplotlib==3.8.4`,
`gymnasium==1.3.0`, `torch==2.12.1` (CPU build), `stable_baselines3==2.9.0`,
`sb3_contrib==2.9.0`, `ortools==9.15.6755`, `scipy==1.15.2`, `statsmodels==0.14.6`,
`pytest==9.1.1`; optional `tensorboard==2.21.0`, `optuna==4.9.0`.

### Hardware/software — documented in `README.md`, summarized here

8-core CPU, **no GPU used at any point** (`torch==2.12.1+cpu`,
`torch.cuda.is_available()` unconditionally `False` on the machine that produced this
thesis' results), Python 3.11.9, Windows (a parallel `.venv-linux` — 5.3GB, confirmed via
`du` — validates the same pinned dependency set under Linux/WSL, so the pipeline is not
Windows-specific despite having been run there).

---

## 2. `README.md`

Rewritten (previous version was Week 1/15/18-era, predating E1–E8, the stats pipeline,
`../results/`, and `../figures/` entirely) to include: installation (native + Docker),
directory structure, exact commands + measured runtime/hardware for every experiment
E0–E8, how to regenerate all statistics/tables/figures, the quick smoke test (§4), and
the seed/CP-SAT-determinism notes from §1. Runtimes quoted are **measured on this
thesis' own machine during this sprint**, not estimates — e.g. E7 training
(33–36 min/stage × 4 stages), E7 evaluation grids (~9–10 min/κ), the full test suite
(391–424s across five separate runs this session).

One correction made *during* writing, not before: the README initially claimed
`pytest -m "not slow"` was meaningfully faster than the full suite. Measuring it
(748 passed, 4 deselected, 401.86s vs. the full suite's ~398s) showed this was false —
only 4 tests carry the `slow` marker, so the flag barely changes runtime. The README
now says so explicitly instead of repeating an unverified assumption.

---

## 3. Dockerfile

Written (`../code/Dockerfile` + `.dockerignore`): CPU-only base image
(`python:3.11.9-slim`, matching the exact Python patch version this thesis used),
layer-cached dependency install, editable package install, and a fast-subset `pytest`
run baked into the build so a broken image fails at `docker build` time, not at first
`docker run`. Excludes checkpoints/logs/venvs from the build context (mount
`experiments/results` as a volume instead — documented in `README.md`) so the image
stays small and never goes stale relative to whatever results the reader actually has.

**Honestly flagged, not glossed over: Docker is not installed in the environment this
sprint ran in (`docker --version` → command not found), so the Dockerfile has NOT been
build-tested.** It follows standard, conventional Dockerfile practice and mirrors the
already-verified native dependency list exactly, but "written correctly" and "verified
to build" are different claims, and only the first one is true right now. **Recommended
before relying on it:** `docker build -t thesis-fjsp-drl "code"` on a
machine with Docker installed, and report back if anything fails — most likely failure
mode would be a missing system library for one of the CP-SAT/torch wheels on the slim
base image, which `apt-get install` inside the Dockerfile would fix if it comes up.

---

## 4. End-to-end smoke test

Ran the exact 4-step sequence documented in `README.md`'s "Quick smoke test" section,
natively (Docker unavailable, per §3):

1. `pytest -m "not slow" -q` → **748 passed, 4 deselected** (401.86s).
2. `run_evaluation.py` across `{spt, fifo, ga, cpsat, rl}` × `{ft06, la01}`,
   `--time-limit 5` → **10/10 rows, all `feasible=True`**, CP-SAT/GA both reached
   `known_optimum` on both instances within the 5s budget.
3. Disrupted rollout, `fifo`+`rl` under `breakdowns` (`--availability 0.90`) on `ft06`
   → **2/2 rows, `feasible=True`**, confirming `DynamicJSSPEnv` + the RL adapter both
   function correctly end-to-end.
4. `train_dynamic.py`, 2000 timesteps, κ=0.5, `--output-dir` pointed **outside** the
   repository (a `/tmp` scratch path, deliberately, to test the CLI's actual contract) →
   **found a real bug**: the script crashed at its final summary-printing step (after
   training and the checkpoint save had already both succeeded) because
   `Path.relative_to(REPO_ROOT)` was called unconditionally, and raises when the output
   path isn't under the repo. **Fixed**, and the same unguarded pattern was found and
   fixed in two more places it would have hit identically: `train_hetero.py` (2 call
   sites) and `train_curriculum.py` (1 call site, imports the fix from `train_hetero.py`).
   `eval_baselines.py` has the same-looking pattern but was left untouched — audited and
   confirmed its output path is hardcoded relative to `REPO_ROOT` with no CLI override,
   so the crash is structurally unreachable there; changing it would violate that
   script's own explicit "kept unmodified so old numbers stay reproducible" design note
   for no safety benefit.
   - Re-ran step 4 after the fix: **passed cleanly**, checkpoint + summary JSON both
     written to the out-of-repo path, absolute path printed instead of crashing.
   - **Re-ran the full 752-test suite twice more after these fixes** (once immediately
     after the `train_dynamic.py` fix, once after the `train_hetero.py`/
     `train_curriculum.py` fixes) — **752/752 passed both times.**

This is the value an end-to-end smoke test is supposed to deliver: it found a real,
previously-undetected bug (untested code path — no existing test exercised an
out-of-repo `--output-dir`) before a reader trying to reproduce the thesis would have
hit it.

---

## 5. Archival preparation

### What's done

- **`.gitignore`** (repository root): excludes both Python environments (`.venv/` —
  57MB, `.venv-linux/` — **5.3GB**, confirmed via `du`, both fully regenerable from the
  pinned dependency files), trained-checkpoint `.zip` files and `tb_logs/` (~1GB
  combined across all training stages, also regenerable via the commands in
  `README.md`), and OS/editor noise. **Verified, not assumed:** ran `git add -A -n`
  (dry run) and confirmed zero `.venv*` or checkpoint paths appear in the staged list;
  the largest file that *would* be tracked is a 9.3MB training log — reasonable for a
  git repository, no further exclusions needed.
- **`git init`** — run successfully at the `THESIS MASTER'S/` top level (there was no
  prior git history anywhere in this project to preserve or clean — confirmed via
  `git status` returning "not a git repository" before this sprint started, so "clean
  commit history" in practice means "one well-described initial commit," not rewriting
  anything that existed before).
- **Zenodo metadata drafted:** `.zenodo.json` (title, description, creator, keywords,
  license placeholder, related-identifier placeholder for the thesis' own DOI once it
  has one) and `CITATION.cff` (the GitHub-recognized citation-metadata standard,
  redundant with `.zenodo.json` on purpose — GitHub renders `CITATION.cff` natively in
  its own UI, while `.zenodo.json` is what GitHub's Zenodo integration reads when
  archiving a release).

### What's blocked, and on what exactly

**The initial commit has not been made.** `git init` succeeded, but no git identity
(`user.name`/`user.email`) is configured anywhere on this machine — checked both
`--global` and the repo-local config, both empty — and creating a commit without one
fails. **I did not set it myself: "never update the git config" is a standing safety
rule I follow regardless of how routine the specific value would be.** You chose (via
the in-conversation question) to have me give you the exact command to run yourself:

```bash
git config --global user.name "Joseph Sanchez"
git config --global user.email "joseph.sanchez@uniminuto.edu.co"
```

Once that's set, the remaining archival steps are: `git add -A`, one commit (suggested
message below), then `git tag -a v1.0-thesis-defense -m "..."`. I can run all three the
moment the identity is configured — nothing else is blocking them.

**Suggested commit message** (drafted, not yet used):

```
Initial commit: E0-E8 reproducibility package

Complete thesis codebase, results, statistics pipeline, and reproducibility
infrastructure as of the final experimental sprint (2026-08-01):

- envs/models/baselines/experiments: the full JSSP/FJSP GNN+PPO pipeline
- experiments/stats/: Wilcoxon/Friedman/Nemenyi pipeline, H1-H5 verdicts
- results/, figures/: thesis-numbered tables and camera-ready figures
- README.md, Dockerfile, pinned dependencies, reproducibility audit
- 752/752 tests passing

See RESULTS_REPORT.md for the H1-H5 hypothesis verdicts and
REPRODUCIBILITY_REPORT.md for this commit's own provenance.
```

### License — a decision only you can make, not defaulted silently

`.zenodo.json`/`CITATION.cff` both leave `license` as an explicit placeholder. I did not
pick one, because it's a real decision with consequences (a university thesis may have
institutional requirements about how the accompanying code can be licensed, distinct
from generic open-source defaults) that I'm not positioned to make for you. Common
choices for a thesis-code deposit: **MIT** (permissive, common for research code),
**CC-BY-4.0** (common for the *data/results* half of a mixed code+data deposit, less
common for code specifically), or checking whether UNIMINUTO has an institutional policy
for thesis-associated code. Fill in the `license` field in both files once decided
(they should match) — this must happen before the actual Zenodo upload.

### The Zenodo upload itself

Per your original request, **I have not attempted this** — you said you'll do the
actual upload. What's ready for that: the drafted metadata above, and the guidance in
`README.md`/this document about what belongs in the git repo (code + small summaries)
vs. the Zenodo deposit (everything, including the ~1GB of checkpoints git excludes) —
Zenodo has no practical size sensitivity comparable to git's, so the deposit should be
the *complete* working directory, not a checkpoint-stripped copy. If you archive via
GitHub's Zenodo integration (push the tagged release, flip the Zenodo-GitHub webhook
on, Zenodo mints a DOI automatically from `.zenodo.json`), the checkpoints still need a
**separate** upload path since GitHub itself won't have them either — either a second
Zenodo "version"/file added manually to the same record, or a plain manual Zenodo
upload of the full directory instead of the GitHub integration. Decide which before
uploading; both are legitimate, but they produce different-shaped Zenodo records.

---

## 6. Summary

| Step | Status |
|---|---|
| 1. Reproducibility audit | **Done.** Two real gaps found and fixed (CP-SAT seeding, scipy/statsmodels pinning); everything else audited and confirmed already correct. |
| 2. `README.md` | **Done.** Comprehensive, with measured (not estimated) runtimes, and one self-correction during writing. |
| 3. `Dockerfile` | **Written, not build-tested** (Docker unavailable in this environment) — flagged, not claimed as verified. |
| 4. Smoke test | **Done and re-verified.** Found and fixed a real bug (3 files) in the process — the intended value of this step. |
| 5. Archival prep | **Partially done.** `.gitignore`/`git init`/Zenodo metadata drafts done and verified; **initial commit blocked on git identity configuration**, which requires your action (I'm not permitted to set it). License field intentionally left for you to decide. Zenodo upload itself intentionally not attempted (your call, per the original request). |

No claim in this document is inferred or assumed — every "done" above corresponds to a
command actually run and output actually inspected during this sprint, and every
"blocked"/"not tested" is named with the specific, concrete reason why.

---

## 7. Addendum — final packaging pass (post-`Thesis.md`)

Re-run after the thesis's own final audit-response revision (`../archive/capitulos/Thesis_FullDraft_v2.md` →
renamed `Thesis.md`) and after generating the companion conference-paper
supplement, to catch any regression before defense/submission:

- **Smoke test re-run, natively, all 4 steps of `README.md`'s "Quick smoke test":**
  step 1 (748 passed/4 deselected), step 2 (10/10 rows, all `feasible=True`, CP-SAT hit
  `known_optimum` on both instances), step 3 (2/2 rows, `feasible=True`), step 4
  (`train_dynamic.py` with an out-of-repo `--output-dir` completed cleanly — the
  `_display_path()` fix from §4 above still holds, no regression). **No new issues
  found.**
- **Full test suite, no marker filter (`pytest -q`): 752 passed, 0 failed** — re-confirms
  the count already reported in §4 still holds after all subsequent thesis-revision and
  paper-supplement work.
- **`Thesis.md`/`Thesis.md` re-synced.** These standalone chapter files
  had drifted from `Thesis.md`'s own Chapter 6/7 (the audit-response edits — new
  Tabla 6.2.3, the `mk06` footnote consolidation, the GA-superiority cross-reference —
  were applied to the integrated document but not propagated back to the two standalone
  files several other docs cite by exact section, e.g. `README.md`'s and
  `FIGURES_INDEX.md`'s `§6.5.1` pointers). Both files are now byte-identical to
  `Thesis.md`'s corresponding sections — verified with `diff`, not assumed.
- **`README.md`** updated to name `Thesis.md` explicitly as the canonical single
  document (it previously listed only the per-chapter files, with no pointer to the
  integrated thesis at all), and to reference the new `CODE_APPENDIX_SUMMARY.md`.
- **`CODE_APPENDIX_SUMMARY.md`** (new, repository root) — a thesis-appendix-ready
  document: repository structure, a chapter/section → code-module cross-reference table
  for Chapters 4–7, and this reproducibility confirmation folded in.
- **Dependency pins re-checked against the installed environment** (`pip freeze`):
  exact match, no drift since §1's original audit.
- **Archival status unchanged from §5: still blocked on the same git identity
  configuration**, re-checked at the start and end of this pass — still unset. The
  `git add -A -n` dry run was re-run to confirm the staged file set is still exactly as
  described in §5 (no `.venv*`/checkpoint leakage, largest file ~9.3MB) with the new
  paper-supplement and `CODE_APPENDIX_SUMMARY.md` files included as expected. Nothing
  else blocks the commit/tag — the moment
  `git config --global user.name "Joseph Sanchez"` /
  `git config --global user.email "joseph.sanchez@uniminuto.edu.co"` (or your preferred
  values) is run, `git add -A`, the commit, and `git tag -a v1.0-thesis-defense` can all
  follow immediately.
