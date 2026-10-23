# GNN+PPO for Dynamic FJSP under Stochastic Perturbations — thesis reproducibility package

[![tests](https://github.com/GIUSEPPESAN21/Tesis-Paper-Code/actions/workflows/tests.yml/badge.svg)](https://github.com/GIUSEPPESAN21/Tesis-Paper-Code/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](code/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CPU-only](https://img.shields.io/badge/hardware-CPU--only-lightgrey.svg)](#hardware--this-thesis-actual-environment)

This is the top-level entry point for reproducing this master's thesis end-to-end: the
code, the trained checkpoints, the raw experimental results (E0–E8), the statistical
analysis pipeline, the camera-ready figures, and the thesis chapters that cite them all.

**If you only want to check that the pipeline runs at all, skip to
["Quick smoke test"](#quick-smoke-test-5-10-minutes) — that's the fastest path to
confidence, and what an examiner should run first.**

## What's in this repository

```
Tesis-Paper-Code/
├── README.md                         <- you are here
├── LICENSE, CITATION.cff, .zenodo.json, .gitignore, .gitattributes, CONTRIBUTING.md
├── .github/                           <- CI workflow (tests.yml)
├── thesis/                           <- the thesis itself and everything that cites/supports it directly
│   ├── Thesis.md                      <- THE thesis: single integrated document (all 8 chapters,
│   │                                      references, changelog); cite this by section number
│   │                                      (§6.5.1, Table 6.0, etc.) — there is no separate
│   │                                      per-chapter file anymore (see archive/README.md)
│   ├── RESULTS_REPORT.md              <- E1-E8 results, H1-H5 verdicts, full traceability
│   ├── REPRODUCIBILITY_REPORT.md      <- seed/hardware/dependency audit, archival/DOI plan (Section 5)
│   ├── FIGURES_INDEX.md               <- every figure -> thesis section, caption, data source
│   ├── PreRegistration.md             <- frozen (2026-07-29) hypotheses H1-H5, decision rules
│   ├── StatsAnalysis_TechSpec.md      <- spec for experiments/stats/ (Wilcoxon/Friedman/Nemenyi)
│   ├── E7_E8_Plan.md                  <- E7/E8 scope, preconditions, schedule
│   ├── CODE_APPENDIX_SUMMARY.md       <- thesis-appendix-ready: repo structure, chapter<->code map
│   └── defense/                       <- active defense-prep material
│       ├── DefensePresentation_Outline.md
│       └── QA_Prep.md
├── paper/                            <- the live paper submissions
│   ├── COR_GNN-PPO/                    <- Computers & Operations Research submission (this study, H1-H5):
│   │                                      manuscript, highlights, cover letter (submitted 2026-10-23)
│   ├── JSched_PriceOfOptimality/       <- Journal of Scheduling manuscript and cover letter (companion study)
│   └── PaperCIE_LaTeX/                 <- companion Computers & Industrial Engineering (Elsevier, Q1)
│                                          submission on the RI-artifact finding: bilingual EN/ES
│                                          manuscripts, the analysis code behind every number, and
│                                          Paper_CIE_DecisionRecord.md (methodology/correction log)
├── figures/                          <- camera-ready figures (Spanish, colorblind-safe, PNG+PDF)
├── docs/                             <- active, non-duplicated supporting documentation — see
│   │                                     docs/README.md for the map; cite thesis/Thesis.md, not these
│   ├── especificaciones/               <- TechSpec / architecture-spec documents (incl. Anexos E-G)
│   ├── planificacion/                  <- ablation/baseline plans, V&V protocol, planning documents
│   ├── auditorias/                     <- consistency-check and pre-defense audit reports
│   ├── logs/                           <- weekly experiment logs (Week 17/21/24)
│   └── future_work/                    <- ROADMAP.md: pending work sourced from the thesis's own
│                                           §8.6 and the paper's own "Threats to validity"
├── archive/                          <- superseded/historical material, moved not deleted (git mv,
│                                          history intact) — see archive/README.md for what's here
│                                          and why each item was archived, not part of the active thesis
├── results/
│   ├── tablas/                       <- thesis-numbered tables (Tabla_6.X.Y.{csv,md,tex})
│   ├── datos_crudos/                  <- raw per-row CSVs, by experiment batch
│   └── README.md                      <- what's here vs. what stays in code/
└── code/                             <- the actual Python package: envs, models, baselines,
    │                                      experiments, tests, and the trained checkpoints
    └── docs/                           <- weekly progress/audit reports for the code package
                                            itself (training, validation, refactor, HPO logs)
```

**Mejoras del tutor (2026-08-20):** diez recomendaciones formales de revisión,
auditadas y en su mayoría implementadas en código/tests esta sesión — nada ejecutado
(ver `code/docs/AUDIT_PREFLIGHT.md` para los hallazgos de la auditoría,
`code/docs/RUN_PLAN.md` para los comandos exactos y estimaciones de tiempo
por fase, `code/docs/MEJORAS_TUTOR_IMPLEMENTADAS.md` para el estado
recomendación-por-recomendación). Nuevo código relevante:
`code/envs/rescheduling.py` (políticas de reoptimización periódica/por
evento), `code/experiments/slack_analysis.py` (métricas de holgura CPM),
`code/experiments/stats/comparators.py` (comparadores PDR separados),
`code/experiments/stats/seeds.py` (agregación multi-semilla). Todos los
resultados numéricos que estas piezas producirían quedan `[PENDIENTE-CORRIDA]` hasta
que se corran.

`code/` is a self-contained, installable Python package
(`thesis_fjsp_drl`) with its own `README.md` covering the code's internal design
decisions. This top-level README is the reproduction guide; the code README is the
API/module reference.

**Where the trained checkpoints live:** this git repository intentionally does **not**
track `experiments/results/**/checkpoints/*.zip` or `tb_logs/` (~1GB, regenerable via
the training commands below — see `.gitignore`). The **complete** archive, checkpoints
included, is the Zenodo deposit described in `thesis/REPRODUCIBILITY_REPORT.md` §5 — that
is the canonical, full-reproduction artifact; download it if you want to reproduce
E1–E8 without retraining. This git repository is the code + small result-summary layer
(CSVs, tables, figures — everything actually cited by the thesis text).

---

## Installation

### Option A — native Python (recommended for development)

```bash
cd "code"
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS/Linux:    source .venv/bin/activate
pip install -e ".[dev]"
```

This installs every pinned dependency from `pyproject.toml`/`requirements.txt` and puts
`envs`/`models`/`baselines`/`experiments` on the import path via an editable install.
**Requires Python 3.11** (pinned — `requires-python = ">=3.11"` in `pyproject.toml`;
the environment this thesis was produced on used exactly 3.11.9). CPU-only is fully
supported and is what every result in this thesis was actually produced on — see
["Hardware"](#hardware--this-thesis-actual-environment) below.

### Option B — Docker (recommended for long-term reproducibility)

```bash
cd "code"
docker build -t thesis-fjsp-drl .
docker run --rm -it -v "$(pwd)/experiments/results:/app/experiments/results" thesis-fjsp-drl
```

See [`code/Dockerfile`](code/Dockerfile) — CPU-only, pinned
base image, reproduces the exact dependency set below without depending on whatever
Python/OS the reader happens to have in three years. This is the environment to use if
"native" installation ever breaks due to upstream package/OS drift.

### Pinned dependencies

Every dependency is pinned to an exact version (`requirements.txt`/`pyproject.toml`),
audited against what's actually imported (see `thesis/REPRODUCIBILITY_REPORT.md` §1 for
the audit method): `numpy==2.4.6`, `pandas==2.2.2`, `matplotlib==3.8.4`,
`gymnasium==1.3.0`, `torch==2.12.1` (CPU build), `stable_baselines3==2.9.0`,
`sb3_contrib==2.9.0`, `ortools==9.15.6755`, `scipy==1.15.2`, `statsmodels==0.14.6`,
`pytest==9.1.1`. Optional: `tensorboard==2.21.0` (`pip install -e ".[monitoring]"`,
training-time logging only — nothing downstream depends on it), `optuna==4.9.0`
(`pip install -e ".[hpo]"`, only needed to *re-run* the Week 21 hyperparameter search,
not to reproduce E0–E8 against the already-chosen hyperparameters).

---

## Hardware — this thesis' actual environment

Documented honestly, not idealized, because H5 (decision latency) and E8's timing
numbers are hardware-dependent and every other experiment's *wall-clock* (not
correctness) depends on it too:

| | |
|---|---|
| CPU | 8 logical cores (no GPU-class machine was used) |
| GPU | **None used.** `torch==2.12.1+cpu` — a CUDA build was never installed. Every training run and every evaluation in this thesis is CPU-only. (`thesis/PreRegistration.md` Amendment 2 — "training hardware" and "CPU-only" collapse to one measurement in this codebase; there is no GPU number to report as primary.) |
| OS | Windows 10/11, Python run via a Git-Bash shell (a parallel `.venv-linux` was also validated under WSL/Linux — the codebase and dependency pins are OS-agnostic) |
| RAM | No experiment in this thesis is memory-bound; a modern 8+ GB machine is more than sufficient |
| Python | 3.11.9 |

**A GPU is not required and was never used to produce any number in this thesis.** If
you have one, `torch`/`sb3_contrib` will still work, but no result here was validated
against GPU execution, and the codebase's own evaluation path
(`baselines.interface.TrainedRLBaseline`) is deliberately pinned to CPU regardless
(see `code/models/policy.py`'s `device` parameter).

---

## Reproducibility — random seeds

Audited end-to-end (`thesis/REPRODUCIBILITY_REPORT.md` §1) — every source of randomness
in this codebase is either seeded explicitly or documented as a known, bounded
exception:

- **Environments, PDRs, GA, the FJSP synthetic generator, disruption generators**: every
  RNG is a `numpy.random.default_rng(seed)` instance threaded explicitly from a `seed`/
  `--seed`/`--seeds` argument. No bare `np.random.*` or stdlib `random.*` global-state
  call exists anywhere in `envs/`, `models/`, `baselines/`, or `experiments/`
  (grep-verified, not assumed).
- **RL training** (`experiments/train_hetero.py`, `train_dynamic.py`, `train_curriculum.py`):
  `--seed` is passed to `sb3_contrib.MaskablePPO(seed=...)`, which seeds
  `torch`/`numpy`/`random` internally (Stable-Baselines3's own contract). CUDA-specific
  determinism flags (`torch.use_deterministic_algorithms`, `cudnn.deterministic`) are
  **not** set — moot in this thesis (CPU-only throughout) but would need attention
  before trusting bit-exact reproduction on a GPU.
- **CP-SAT** (`baselines/cpsat.py`): `solver.parameters.random_seed` is now set
  explicitly (default `1`, matching OR-Tools' own default exactly — this default was
  *audited and fixed to match*, not newly chosen, precisely so this change doesn't
  silently alter any already-published thesis number on re-run). **Known, documented,
  irreducible limitation:** CP-SAT runs with `num_search_workers=8` (parallel portfolio
  search) by default; a **proven-`OPTIMAL`** result's objective value is deterministic
  regardless of thread scheduling (branch-and-bound correctness doesn't depend on it),
  but a **time-limited `FEASIBLE`** result's exact makespan can vary run-to-run even
  with a fixed seed, because of thread race conditions inherent to parallel search —
  this is a property of OR-Tools, not a bug in this codebase. Pass `--num-workers 1` to
  `run_evaluation.py` for bit-exact `FEASIBLE`-result reproduction, at the cost of much
  slower solves. **This is almost certainly the mechanism behind the `la40` CP-SAT
  timing anomaly documented in `thesis/Thesis.md` §6.5.1** (a `FEASIBLE`, time-limit-
  exhausted result on an otherwise-comparable instance) — flagged there, not hidden.

---

## Reproducing the thesis results: E0–E8

All commands below assume `cd "code"` and an activated environment (native
or `docker run -it thesis-fjsp-drl bash`). **Trained checkpoints are already included in
this repository** (`experiments/results/checkpoints/`,
`experiments/results/curriculum_full/checkpoints/`, `experiments/results/dynamic/checkpoints/`)
— reproducing E1–E8 evaluates against these, it does **not** require retraining from
scratch. Retraining commands are given separately at the end of this section for anyone
who wants to verify the training process itself.

Runtimes below are **measured on this thesis' own 8-core CPU-only machine**, not
estimates — see `thesis/REPRODUCIBILITY_REPORT.md` §1 for how each was captured. Expect
meaningful variance on different hardware, and expect CP-SAT specifically to vary run-
to-run per the seeding note above.

### E0 — environment validation

```bash
pytest                                    # full suite: 752 passed, ~7 min (measured: 398s)
pytest -m "not slow"                      # 748 passed, 4 deselected -- measured ~6-7 min too
                                           # (only 4 tests actually carry the "slow" marker;
                                           # most of the suite's time is many small CP-SAT-
                                           # light tests, not a handful of slow outliers --
                                           # don't expect this flag to meaningfully speed
                                           # things up, despite the name)
python scripts/verify_optima.py           # re-prove data/known_optima.csv with fresh CP-SAT solves, ~2-5 min
```

**Hardware:** any machine meeting the install requirements. **Expected result:**
752/752 tests pass, 0 failures (this exact count as of this thesis' final sprint —
`thesis/Thesis.md` Table 6.1.3). `verify_optima.py` reproduces every optimum in
`data/known_optima.csv` to 0.00% error.

### E1 — static competitiveness (JSSP), the primary input to H1

```bash
CKPT=experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip
INSTANCES="ft06 ft10 ft20 la01 la02 la03 la04 la05 la06 la07 la08 la09 la10 la11 la12 la13 la14 la15 la16 la17 la18 la19 la20 la21 la22 la23 la24 la25 la26 la27 la28 la29 la30 la31 la32 la33 la34 la35 la36 la37 la38 la39 la40"

# Deterministic rollout (seed 0, the confirmatory row) + PDR/GA/CP-SAT, all 43 instances
python run_evaluation.py --instances $INSTANCES --methods pdr ga cpsat rl \
    --seeds 0 --rl-policy "$CKPT" --out experiments/results/e1_seed0_baselines.csv

# Stochastic RL rollouts (seeds 1-4, descriptive-only per thesis/PreRegistration.md Sec 2)
python run_evaluation.py --instances $INSTANCES --methods rl --rl-sample \
    --seeds 1 2 3 4 --rl-policy "$CKPT" --out experiments/results/e1_rl_stochastic.csv
```

**Hardware/runtime:** CP-SAT dominates — most instances solve to `OPTIMAL` in
under a second, but a handful (notably `la40`) can hit the 60s default time limit or,
per the CP-SAT non-determinism note above, occasionally run far longer (the `la40`
anomaly reached ~7216s in the run this thesis reports — see `thesis/Thesis.md` §6.5.1).
**Budget several hours for a from-scratch run of all 43 instances**; the deterministic
pass alone took the majority of that. Use `--num-workers 1` for a slower but variance-
free repeat, or restrict `--instances` for a faster partial check (see the smoke test).

### E2 — static competitiveness (FJSP), Arm A (synthetic) and Arm B (Brandimarte)

```bash
FJSP_CKPT=experiments/results/curriculum_full/checkpoints/curriculum_stage24_mk05_final.zip

# Arm A: synthetic FJSP over the same 43 base instances
python run_evaluation.py --fjsp --instances $INSTANCES --methods ga cpsat rl \
    --seeds 0 --rl-policy "$FJSP_CKPT" --out experiments/results/e2_armA_synth_seed0.csv

# Arm B: real Brandimarte benchmarks (mk01-mk07)
python run_evaluation.py --fjsp --instances mk01 mk02 mk03 mk04 mk05 mk06 mk07 \
    --methods ga cpsat rl --seeds 0 --rl-policy "$FJSP_CKPT" \
    --out experiments/results/e2_armB_mk_seed0.csv
```

**Known limitation, not a bug:** `mk06`'s `rpd_pct` is excluded downstream (upstream
machine-count mismatch in the `SchedulingLab/fjsp-instances` mirror — raw makespan is
still recorded). **Hardware/runtime:** GA is the slower method here (fixed 60s budget
per instance by default); expect on the order of an hour for the full 50-instance
(43+7) sweep across both arms.

### E3 — size generalization (H2), zero-shot on `la21`–`la40`

E3 reuses E1's already-computed rows for `la21`–`la40` (RL + PDR) rather than
re-running them (`experiments/results/week24/build_e3_from_e1.py` shows the exact
extraction) — **no new RL/PDR evaluation is needed once E1 has run.** This is
intentional, not a shortcut: E3 is E1's `la21`–`la40` subset, re-labeled and re-tiered.

### E4/E5/E6 — robustness under disruption (H3), 9 regimes × 30 seeds

```bash
SEEDS=$(seq 0 29)
CKPT=experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip

# E4 (breakdowns), E5 (stochastic times), E6 (arrivals) -- PDR arm
for avail in 0.95 0.90 0.80; do
  python run_evaluation.py --instances ft06 la01 --methods pdr \
    --disruption breakdowns --availability "$avail" --seeds $SEEDS --quiet \
    --out "experiments/results/e4_breakdowns_avail${avail}.csv"
done
# ... same pattern for --disruption stochastic_times --proc-sigma {0.1,0.25,0.5}
# ... and --disruption arrivals --utilization {0.7,0.85,0.95} --methods lwkr mor lor fifo

# RL arm (same 9 regimes, --rl-policy instead of --methods pdr)
for avail in 0.95 0.90 0.80; do
  python run_evaluation.py --instances ft06 la01 --methods rl \
    --disruption breakdowns --availability "$avail" --seeds $SEEDS --quiet \
    --rl-policy "$CKPT" --out "experiments/results/e4_breakdowns_avail${avail}_rl.csv"
done
```

Full copy-pasteable scripts (all 9 regimes × both arms, exactly as run for this thesis):
[`experiments/results/week24/run_e4e5e6.sh`](code/experiments/results/week24/run_e4e5e6.sh)
(PDR arm) and
[`run_e4e5e6_rl.sh`](code/experiments/results/week24/run_e4e5e6_rl.sh)
(RL arm).

**Hardware/runtime:** the RL arm (9 × 30 × 2 = 540 rollouts) took **~9-10 minutes**
measured directly on this thesis' machine (comparable to E7's grid, below — PDR
rollouts are cheaper still, no neural-network forward pass).

### E7 — disruption-aware reward, κ ∈ {0.5, 1.0} (H4)

κ=0 is the **same** checkpoint E1–E6 already used — not a third training run. Two new
trainings (κ=0.5, κ=1.0), each a two-stage curriculum (`ft06`→`la01`):

```bash
# Stage D1 (ft06), then D2 (la01), for kappa=0.5:
python experiments/train_dynamic.py --instance ft06 --kappa 0.5 \
    --init-checkpoint experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip \
    --stage D1 --timesteps 100000 --seed 0
python experiments/train_dynamic.py --instance la01 --kappa 0.5 \
    --init-checkpoint experiments/results/dynamic/checkpoints/dynamic_kappa0.5_D1_ft06_final.zip \
    --stage D2 --timesteps 100000 --seed 0
# Repeat both stages for --kappa 1.0

# Evaluation grid (9 regimes x 30 seeds x 2 instances + nominal pass), per kappa:
bash experiments/results/week25/run_e7_grid.sh 0.5 \
    experiments/results/dynamic/checkpoints/dynamic_kappa0.5_D2_la01_final.zip
bash experiments/results/week25/run_e7_grid.sh 1.0 \
    experiments/results/dynamic/checkpoints/dynamic_kappa1.0_D2_la01_final.zip
```

**Hardware/runtime (measured, this thesis' machine):** each training stage,
**33–36 minutes** (100k timesteps, `n_envs=4`, CPU-only) — **~135 minutes for all four
stages**. Each evaluation grid, **~9-10 minutes**. **Total E7: ~2.5-3 hours.**
**Environment note if you background these:** launch long training runs via
`nohup ... & disown` from a script file, not a bare backgrounded shell command — this
thesis' own reproduction hit two silent mid-training terminations when relying on a
tool-tracked background process across a session boundary; the `nohup`+`disown` pattern
survived reliably (see `PROGRESS_SUMMARY.md` for the exact incident).

### E8 — decision latency (H5)

Shares E1's execution pass (`wall_clock_s` column) by design — see
`thesis/StatsAnalysis_TechSpec.md` §2.3. The "clean" re-measurement this thesis reports ran
on an **otherwise-idle machine**, deliberately sequenced before any other compute that
day:

```bash
python run_evaluation.py --instances $INSTANCES --methods pdr ga cpsat rl \
    --seeds 0 --rl-policy "$CKPT" \
    --out experiments/results/e1_e8_clean_seed0_baselines.csv --quiet
python run_evaluation.py --instances $INSTANCES --methods rl --rl-sample \
    --seeds 1 2 3 4 --rl-policy "$CKPT" \
    --out experiments/results/e1_e8_clean_rl_stochastic.csv --quiet
```

**Hardware/runtime:** dominated by CP-SAT exactly as in E1 (same pass) — budget several
hours, and run on a machine doing nothing else if you want a timing-clean repeat (H5's
verdict specifically depends on *timing*, not just correctness).

### Retraining from scratch (optional — not required to reproduce E1–E8)

```bash
python experiments/train_curriculum.py --help    # the 19/24-stage curricula behind the shipped checkpoints
python experiments/train_hetero.py --instance ft06 --timesteps 400000
```

Retraining the full curricula from scratch was **not re-timed for this reproducibility
sprint** (out of scope — E1–E8 reproduce against the shipped checkpoints, which is the
thesis' actual evaluation protocol) and would take substantially longer than any single
E1–E8 step above; see `docs/logs/TrainingPlan_Week17.md` / `docs/planificacion/AblationPlan.md` for that history if
you need it.

---

## Regenerating the statistics, tables, and figures

Once E1–E8's CSVs exist (either freshly reproduced above, or using the ones already
checked into `experiments/results/` / `results/datos_crudos/`):

```bash
cd "code"

# H1-H5 verdicts, RPD/RI/latency summary tables, CD diagrams -- writes
# experiments/results/stats/{tables,figures}/
python -m experiments.stats.cli --results-dir experiments/results \
    --experiments E1 E2a E2b E3 E4 E5 E6 E7 E8 --out-dir experiments/results/stats --skip-missing

# Thesis-numbered copies (format conversion only, zero recomputation) -> ../results/tablas/
python scripts/export_thesis_tables.py --out-dir ../results/tablas

# Camera-ready Spanish/colorblind-safe figures (PNG 300dpi + vector PDF) -> ../figures/
python scripts/generate_thesis_figures.py --out-dir ../figures
```

All three are deterministic given the same `experiments/results/` state — no
recomputation happens outside what's described above, and `generate_thesis_figures.py`
reuses the exact same `experiments.stats` functions the pipeline itself uses, so a
regenerated figure cannot numerically drift from what `experiments/results/stats/tables/`
or `thesis/Thesis.md` already cite. Expected runtime: under 2 minutes combined
(measured; scipy/statsmodels statistics on already-small CSVs, matplotlib rendering).

`python -m experiments.stats.cli` prints the full H1–H5 verdict table to stdout; compare
against Table 6.0 in `thesis/Thesis.md` as the fastest correctness check available.

---

## Quick smoke test (5-10 minutes)

**For an examiner who wants to confirm the pipeline actually runs, without waiting hours
for full E1/E8.** This is a reduced configuration — 2 small instances, few seeds, short
CP-SAT/GA time limits — chosen to exercise every code path (PDR, GA, CP-SAT, RL,
disruption rollout, the stats pipeline) in minutes, not to reproduce a specific
thesis number.

```bash
cd "code"

# 1. Environment sanity (skips the slow CP-SAT-heavy tests)
pytest -m "not slow" -q

# 2. All four method families, two small instances, one seed, short solver budgets
CKPT=experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip
python run_evaluation.py --instances ft06 la01 --methods spt fifo ga cpsat rl \
    --seeds 0 --time-limit 5 --rl-policy "$CKPT" --out /tmp/smoke_e1.csv

# 3. One disrupted rollout (breakdowns), confirming DynamicJSSPEnv + the RL adapter both work
python run_evaluation.py --instances ft06 --methods fifo rl \
    --disruption breakdowns --availability 0.90 --seeds 0 \
    --rl-policy "$CKPT" --out /tmp/smoke_disrupted.csv

# 4. A tiny real training step (proves train_dynamic.py's full path -- env, policy,
#    checkpoint warm-start, PPO update loop -- without committing to a real budget)
python experiments/train_dynamic.py --instance ft06 --kappa 0.5 \
    --init-checkpoint "$CKPT" --stage SMOKE --timesteps 2000 --n-envs 2 \
    --output-dir /tmp/smoke_dynamic

echo "Smoke test passed if all four steps above exited 0 and /tmp/smoke_*.csv have rows."
```

**Expected result:** step 1 passes (748 passed, 4 deselected — despite the flag's name,
this is not meaningfully faster than the full suite on this codebase, see the note
above); steps 2-3 write CSVs with `feasible=True` in every row and
`status in {OK, OPTIMAL, FEASIBLE}`; step 4 prints a `[done]` line with a checkpoint
path. **This does not validate any thesis number** — it validates that the code paths
run end-to-end on the reader's machine. This exact sequence is what
`thesis/REPRODUCIBILITY_REPORT.md` §4 reports having actually run, **including a real bug
this smoke test caught and a fix that's already applied**: `train_dynamic.py` (and,
audited and fixed at the same time, `train_hetero.py`/`train_curriculum.py`) crashed at
their final summary-printing step — *after* training and the checkpoint save had
already succeeded — whenever `--output-dir` pointed outside the repository (exactly
what step 4 above does, deliberately, to avoid littering the working tree). Fixed by
falling back to an absolute path in the display string instead of assuming
`Path.relative_to(REPO_ROOT)` always succeeds; re-verified after the fix, and the full
752-test suite still passes. This is exactly the kind of issue an end-to-end smoke test
is supposed to catch — it's recorded here rather than silently fixed, matching this
project's practice of documenting what was found and corrected, not just claiming a
clean result.

---

## Design decisions (code-level)

Most of the "why", not just the "what", lives next to the code as module docstrings.
See [`code/README.md`](code/README.md) for the architectural
decisions (hand-rolled GIN vs. `torch_geometric`, `MaskablePPO` vs. vanilla PPO,
`SchedulingEnv`'s shared base, CP-SAT as ground truth, etc.) and the module-by-module
directory structure. For a thesis-chapter-by-thesis-chapter map of which code module
produced which result, see [`thesis/CODE_APPENDIX_SUMMARY.md`](thesis/CODE_APPENDIX_SUMMARY.md)
(written to be included directly as a thesis appendix).

## License and citation

See `code/pyproject.toml` for the package metadata. If you use this code or
its results, please cite the thesis (citation details to be finalized at defense/
publication — see `thesis/REPRODUCIBILITY_REPORT.md` §5 for the archival/DOI plan).
