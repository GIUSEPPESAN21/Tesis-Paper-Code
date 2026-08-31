# thesis_fjsp_drl

Deep reinforcement learning for job-shop scheduling (JSSP → FJSP), master's thesis
codebase. The pipeline: instance loading → static/dynamic Gymnasium environments →
priority-dispatching-rule / GA / CP-SAT baselines → a unified evaluation harness → a
GIN+PPO agent trained and compared against those baselines under a range of disruption
scenarios.

Week 18 landed the **heterogeneous GNN encoder + size-agnostic PPO policy** and the
first trained agent, including a flexible (FJSP) environment with a joint
`(operation, machine)` action space and the real Brandimarte `MK*` benchmarks. See
[`TRAINING_REPORT_WEEK18.md`](docs/TRAINING_REPORT_WEEK18.md) for the training curves,
the makespan comparison against the PDR/GA/CP-SAT baselines, and an honest assessment
against the success criteria in `../docs/logs/TrainingPlan_Week17.md`.

Week 15 was a **code-health sprint** (no new research features): consolidate four
sprints' worth of environments/baselines/instance loaders behind a clean, tested,
documented interface so next month's GNN+PPO work has a stable foundation to plug into.
See [`REFACTOR_NOTES_WEEK15.md`](docs/REFACTOR_NOTES_WEEK15.md) for exactly what changed and
why. Earlier milestones: [`VALIDATION_WEEK1.md`](docs/VALIDATION_WEEK1.md) (first running
pipeline), [`VALIDATION_WEEK6.md`](docs/VALIDATION_WEEK6.md) (dynamic/disruption extension),
[`BASELINES_RESULTS_WEEK9.md`](docs/BASELINES_RESULTS_WEEK9.md) /
[`BASELINES_RESULTS_WEEK12.md`](docs/BASELINES_RESULTS_WEEK12.md) (PDR/CP-SAT, then GA,
baseline sweeps).

## Structure

```
thesis_fjsp_drl/
├── run_evaluation.py       # unified CLI: instances x methods x disruption -> results CSV
├── envs/
│   ├── base_env.py          # SchedulingEnv: shared base of JSSPEnv/DynamicJSSPEnv
│   ├── instance.py          # JSSPInstance (fixed routing)
│   ├── fjsp_instance.py     # FJSPInstance (machine eligibility sets) + synthetic generator
│   ├── loader.py            # OR-Library/Taillard parsers, instance registry, known optima
│   ├── fjsp_loader.py       # Brandimarte MK* parser + FJSP known-optima registry
│   ├── disjunctive_graph.py # adjacency builders, lower bounds, feasibility checks
│   ├── jssp_env.py          # static JSSP env (+ machine nodes / eligibility edges)
│   ├── fjsp_env.py          # flexible JSSP env: joint (operation, machine) action space
│   ├── dynamic_jssp_env.py  # event-driven env: breakdowns, stochastic times, arrivals
│   ├── disruptions.py       # breakdown/processing-time/arrival generators
│   └── events.py            # future-event-list simulation core
├── baselines/
│   ├── interface.py         # Baseline protocol + adapters (PDR, GA, CP-SAT, CP-SAT-FJSP, RL placeholder)
│   ├── dispatching_rules.py # nine PDRs (SPT/LPT/MWKR/LWKR/MOR/LOR/FIFO/EDD/CR) + random
│   ├── ga.py                 # genetic algorithm (JSSP + FJSP-ready assignment-vector chromosome)
│   ├── cpsat.py              # CP-SAT exact solver, JSSP and FJSP (OptionalIntervalVar)
│   └── due_dates.py          # TWK due-date synthesis for EDD/CR
├── models/                  # GIN + heterogeneous GIN encoders, MaskablePPO wiring,
│                             # per-node (JSSP) and per-pair (FJSP) actor heads
├── data/instances/          # ft06/ft10/ft20 (Fisher-Thompson), la01-la40 (Lawrence)
├── data/fjsp_instances/     # Brandimarte MK* (flexible), fetched not transcribed
├── data/known_optima.csv    # literature reference makespans, re-provable via
│                             # scripts/verify_optima.py
├── data/fjsp_known_optima.csv # same, for the flexible benchmarks
├── scripts/                  # one-off provenance scripts (fetch/verify instances)
├── experiments/              # training scripts, historical eval script, results
│   └── results/               # week24/, week25/, dynamic/, stats/, curriculum_full/,
│                                 ablations_full/, ablations_la16/ -- the E1-E8 batches
│                                 that back the thesis; week18/, hpo_smoke/ -- earlier
│                                 (Week 9-21) hetero-training/HPO dev artifacts, archived
│                                 not cited; baselines_week9.{csv,md}, baselines_week12.
│                                 {csv,md} stay at this root (not archived) because
│                                 ../thesis/Thesis.md / ../thesis/PreRegistration.md cite them by
│                                 this exact path
├── tests/                    # pytest suite (unit + run_evaluation.py integration tests)
└── notebooks/                 # exploratory analysis (empty for now)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the pinned versions in `pyproject.toml` (also mirrored in
`requirements.txt` for a plain `pip install -r requirements.txt` workflow) and
puts `envs`, `models`, `baselines` on the import path via an editable install,
so `import envs` works from anywhere without `sys.path` hacks.

If you already have a large PyTorch install elsewhere and don't want to
re-download it, create the venv with `--system-site-packages` instead and
`pip install --no-deps -e .` — that's what was done to build this codebase
(see "Deviations" below).

## Running things

```bash
pytest                                    # full suite (~600 tests, ~7 min -- CP-SAT/GA-heavy)
pytest -m "not slow"                      # skip the exact-solver tests
python scripts/fetch_instances.py         # (re-)populate data/instances/ from OR-Library
python scripts/fetch_fjsp_instances.py    # (re-)populate data/fjsp_instances/ (Brandimarte)
python scripts/verify_optima.py           # re-prove data/known_optima.csv with CP-SAT
python experiments/train_ft06.py          # Week 1 homogeneous GIN+MaskablePPO smoke run
python experiments/train_hetero.py --instance ft06 --timesteps 400000
python experiments/train_hetero.py --instance mk01 --fjsp --timesteps 300000
```

`train_ft06.py` (the Week 1 homogeneous path, kept as the ablation configuration) and
`train_hetero.py` (the Week 18 heterogeneous, size-agnostic policy) both accept
`--timesteps`, `--seed`, `--hidden-dim`, `--num-layers`, etc. — see `--help`. Both write
a checkpoint, curve PNGs, and a per-episode metrics CSV to `experiments/results/`;
`train_hetero.py` additionally writes TensorBoard logs and SB3's `progress.csv` under
`experiments/results/tb_logs/<run>/`, plus a JSON run summary.

To evaluate a trained policy through the unified harness, point `--rl-policy` at its
checkpoint — the `rl` method slot then uses the real policy instead of the schema-stable
placeholder, with no other CLI or CSV change:

```bash
python run_evaluation.py --instances ft06 --methods pdr ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageA_ft06_final.zip
python run_evaluation.py --fjsp --instances mk01 --methods ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageF_mk01_final.zip
```

### `run_evaluation.py` — the unified evaluation harness

```bash
python run_evaluation.py --help

# Default sweep: nine PDRs excluded by default (pass --methods pdr for all nine),
# a few PDRs + GA + CP-SAT + the RL placeholder, on ft/la instances, no disruption.
python run_evaluation.py

# Pick your own instances, methods (rule names, "ga", "cpsat", "rl", or the "pdr"/"all"
# group shorthands), and seeds.
python run_evaluation.py --instances ft06 la01 la02 --methods pdr ga cpsat rl --seeds 0 1 2

# Evaluate under a disruption profile: none (default) / breakdowns / stochastic_times /
# arrivals / combined. PDRs roll out live against DynamicJSSPEnv; GA/CP-SAT (offline
# solvers) use predictive-reactive replay -- see the module docstring in
# run_evaluation.py and REFACTOR_NOTES_WEEK15.md for what that means and why.
python run_evaluation.py --disruption breakdowns --seeds 0 1 2 3 4

# FJSP (machine eligibility sets): synthesizes routing flexibility on top of the real
# bundled instances (see envs/fjsp_instance.py); only ga/cpsat support it.
python run_evaluation.py --fjsp --methods ga cpsat --instances ft06 la01
```

Every run writes a long-format CSV (`experiments/results/run_evaluation.csv` by
default, or `--out <path>`) with one row per `(instance, method, disruption, seed)` —
columns: `instance, family, num_jobs, num_machines, method, disruption, seed, makespan,
known_optimum, rpd_pct, wall_clock_s, feasible, status, is_optimal, violations_count,
timestamp`. This shape is what a paired Wilcoxon test, a Friedman test (rank methods
per instance-block), or Nemenyi's post-hoc comparison need directly, with no reshaping.

`experiments/eval_baselines.py` (Week 9/12) still exists but is no longer the
recommended entry point — it is kept unmodified so the numbers in
`BASELINES_RESULTS_WEEK9.md`/`BASELINES_RESULTS_WEEK12.md` stay reproducible from the
exact script that produced them.

## Design decisions

Most of the "why", not just the "what", lives next to the code as module
docstrings — this section is a pointer, not a duplicate. See
[`REFACTOR_NOTES_WEEK15.md`](docs/REFACTOR_NOTES_WEEK15.md) for the Week 15 decisions
specifically (the shared `SchedulingEnv` base, FJSP completion, the unified harness,
predictive-reactive replay under disruption, the RL placeholder).

- **Hand-rolled GIN (`models/gin.py`) instead of `torch_geometric`.** The
  disjunctive graph is small; a dense `A @ H` layer is simpler and faster at
  this scale, and drops a dependency whose wheels are pinned to an exact
  torch+CUDA build — the single most common reason a published DRL-scheduling
  repo fails to install three years later.
- **`sb3_contrib.MaskablePPO` instead of vanilla `stable_baselines3.PPO`**
  (`models/policy.py`). At most one operation per job is ever legal, so most
  of `Discrete(N)` is illegal at every step. `MaskablePPO` masks the action
  distribution before sampling and before the PPO log-prob; vanilla PPO has no
  equivalent, and reward-penalty workarounds corrupt the on-policy gradient.
  `JSSPEnv.action_masks()` was written to match `MaskablePPO`'s auto-discovery
  convention directly, so no extra wrapper is needed.
- **Custom Dict-observation features extractor** instead of SB3's default
  `CombinedExtractor`. The default processes each observation key
  independently; here `node_features` is only meaningful jointly with the two
  adjacency matrices (that's what message passing *is*), so
  `JSSPGraphFeaturesExtractor` consumes the whole dict in one pass.
- **List-scheduling transition + dense critical-path-lower-bound reward**
  (`envs/jssp_env.py`, `envs/disjunctive_graph.py`). Dispatching in some order
  under earliest-feasible placement can always reproduce or improve on an
  optimal schedule, so nothing is lost by restricting the action space to
  "pick a ready operation." The reward telescopes exactly to
  `H(s_0) - C_max`, so maximizing return under `gamma=1` is minimizing
  makespan up to a constant.
- **CP-SAT as ground truth, not a literature table** (`baselines/cpsat.py`,
  `scripts/verify_optima.py`). `data/known_optima.csv` is regenerable by
  solving every bundled instance with OR-Tools rather than trusting a
  hand-transcribed number, and the same solver is reused inside
  `tests/test_env.py` to derive a *verified* optimal action sequence for ft06
  instead of hand-authoring one.
- **`SchedulingEnv` (`envs/base_env.py`) shared by `JSSPEnv`/`DynamicJSSPEnv`.**
  The static and dynamic environments' *internal* state representations are
  genuinely different (fixed-size list-scheduling arrays vs. an event-driven
  future-event list) and stay separate; what they share — instance resolution,
  the `reward_scale` convention, `static_lb`/`dispatch_order` — now lives in
  one base class instead of being an accident of `DynamicJSSPEnv` subclassing
  `JSSPEnv` and overriding almost everything it inherited.
- **FJSP is a first-class instance type, not a solver-local fixture**
  (`envs/fjsp_instance.py`). Machine eligibility sets used to live only inside
  `baselines/cpsat.py` as a test fixture; `FJSPInstance` now has the same
  validation rigor as `JSSPInstance` and a synthetic generator
  (`generate_fjsp_instance`) that derives reproducible routing flexibility from
  real bundled benchmark data, so CP-SAT's `OptionalIntervalVar` model and the
  GA's assignment-vector chromosome are both exercised against real processing
  times, not only a hand-built toy. No real FJSP benchmark file loader
  (Brandimarte `MK*`, Hurink) is bundled — that is a research-scope decision
  for a later sprint, not a refactor-sprint blocker.

### Known limitation (by design, for now)

`JSSPGraphFeaturesExtractor` flattens per-node embeddings into a fixed-size
feature vector sized for one specific instance's operation count. That's
sufficient for this week's single-instance smoke test (`ft06`) but does not
transfer across instance sizes. Multi-instance training needs a size-agnostic,
per-node action head (a shared scorer applied to each node embedding, as in
Zhang et al. 2020's L2D) — planned once the current pipeline's correctness is
established, not before.

Priority dispatching rules have no FJSP counterpart this sprint: a PDR ranks
*among the legal frontier operations*, but has no rule for *which eligible
machine* to run one on once that set has more than one member. `PDRBaseline`
raises a clear `TypeError` on an `FJSPInstance` rather than guessing; use
`GABaseline`/`CPSATFJSPBaseline` for FJSP. See `baselines/interface.py`'s
module docstring.

### Deviations from the plan / environment notes

- **No isolated venv with a fresh PyTorch install.** The development machine
  already had `torch==2.12.1`, `gymnasium`, `ortools`, `numpy`, `pandas`,
  `matplotlib`, `pytest` installed globally. Re-downloading a ~2GB PyTorch
  wheel into a from-scratch venv wasn't worth it, so `.venv` was created with
  `--system-site-packages` (inherits the heavy packages already validated on
  this machine) and only `sb3_contrib`/`stable_baselines3` (lightweight, not
  previously installed) were added on top. `requirements.txt` /
  `pyproject.toml` pin the exact versions used either way, so a from-scratch
  environment builds identically; it just costs a larger download.
- **`baselines/cpsat.py` fixed for the installed OR-Tools version.**
  `CpSolver.BestObjectiveBound`/`.WallTime` are deprecated no-call properties
  in `ortools==9.15.6755` (accessing them returns a bound method, not a
  value); the snake_case replacements `best_objective_bound`/`wall_time` are
  used instead. Found via the CP-SAT-optimal-sequence test in
  `tests/test_env.py`, which exercises the whole `solve_cpsat` return path.

(GA landed in Week 12, machine eligibility/FJSP completion in Week 15 — see the
weekly docs linked at the top of this file for the full history instead of an
ever-growing list here.)
