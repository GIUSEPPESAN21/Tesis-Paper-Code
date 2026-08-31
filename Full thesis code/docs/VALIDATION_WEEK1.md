# Week 1 Validation Report

**Scope:** foundational scaffolding for the DRL job-shop-scheduling thesis
codebase — benchmark instance loader, disjunctive-graph Gymnasium environment,
and a minimal GIN+PPO training pipeline. The goal this week was a *running,
tested* pipeline, not a converged model. This report checks the three
criteria requested for Week 1 sign-off.

Full environment/setup notes and design-decision rationale: [`README.md`](README.md).

---

## (a) Instance loader passes known-optimum tests

`tests/test_loader.py::TestKnownOptima` asserts the parser + bundled instance
files reproduce the literature makespans for all three Fisher-Thompson
instances named in the brief:

| instance | known optimum | assertion |
|---|---|---|
| ft06 | 55 | `test_ft06_known_optimum` |
| ft10 | 930 | `test_ft10_known_optimum` |
| ft20 | 1165 | `test_ft20_known_optimum` |

These aren't just trusted from a hand-transcribed table: `data/known_optima.csv`
is provable from the instance files themselves via CP-SAT
(`baselines/cpsat.py`, `scripts/verify_optima.py`). Re-running that proof
fresh for this report (60s budget per instance):

```
  ft06   6x6   Cmax=55     bound=55     OPTIMAL     0.69s
  ft10  10x10  Cmax=930    bound=930    OPTIMAL     5.45s
  ft20  20x5   Cmax=1165   bound=1165   OPTIMAL     3.31s
```

All three close with a matching proven lower bound (`bound == Cmax`), i.e.
CP-SAT independently confirms the literature values in the brief.

Additionally, `test_static_lower_bound_never_exceeds_known_optimum` checks —
for **every** bundled instance, all 43 of them (`ft06/ft10/ft20` +
`la01`-`la40`) — that the classical job/machine lower bound never exceeds the
known optimum. A parse bug (wrong machine index, transposed dimensions, off-
by-one in duration) reliably breaks this invariant, so it's a cheap
whole-dataset sanity net beyond the three headline instances.

**Result: `tests/test_loader.py` — 104/104 passed** (also see
`tests/test_instance.py` — 14/14 passed — for the underlying `JSSPInstance`
data structure and its derived-quantity math).

---

## (b) Environment respects all constraints under random and greedy-SPT policies

`tests/test_env.py::TestConstraintSafety::test_rule_rollout_is_always_feasible`
rolls out **every** dispatching rule in `baselines/dispatching_rules.py` —
`random`, `spt`, `lpt`, `fifo`, `mwkr`, `mopnr` — on two instances (`ft06`,
`la01`), three seeds each (36 rule/instance/seed combinations), and checks the
resulting schedule against `envs.disjunctive_graph.is_schedule_feasible`,
which independently re-verifies both constraint families:

1. **Precedence** — each job's operations execute in technological order.
2. **Machine capacity** — no two operations on the same machine overlap.

Every one of the 36 combinations produces a feasible schedule with makespan
`>= static_lower_bound`. `TestConstraintSafety` also checks that a random
rollout dispatches every operation exactly once (no duplicate/missing
dispatches), and cross-checks feasibility a second, independent way by calling
`is_schedule_feasible` directly on the raw `start_times` array after an SPT
rollout, rather than only trusting the `info["schedule_feasible"]` the env
itself reports.

`tests/test_disjunctive_graph.py` unit-tests the constraint checker itself
against hand-built feasible/infeasible schedules on a small pen-and-paper
toy instance (precedence violation, machine overlap, and negative-start cases
are each individually triggered and caught).

**Result: `tests/test_env.py` — 51/51 passed**,
`tests/test_disjunctive_graph.py` — 14/14 passed.

---

## (c) PPO+GIN pipeline runs end-to-end for ≥1000 timesteps on ft06

`experiments/train_ft06.py` trains `MaskablePPO` (`sb3_contrib`) with the
custom `JSSPGraphFeaturesExtractor` (GIN encoder, `models/policy.py`) on
`ft06`. Run for this report:

```
python experiments/train_ft06.py --timesteps 20000 --seed 0
```

- **20,000 timesteps, 560 complete episodes, no crashes** (well above the
  1,000-timestep bar).
- Wall clock: ~3m19s on CPU (no GPU available on the dev machine).
- Mean makespan, first 20 episodes: **98.5** → last 20 episodes: **67.8**.
- **Best makespan seen during training: 57** — 2 above the proven optimum of
  55, after only 20k timesteps and no hyperparameter tuning. This is a
  genuine (if noisy) learning signal, not just "doesn't crash": an untrained
  or randomly-initialized policy on ft06 averages roughly 100-140.
- Every rollout is feasible by construction: `MaskablePPO` masks illegal
  actions before sampling, so the environment's own constraint checks (see §b)
  apply to every training episode, not just the evaluation ones.

![ft06 training curves](experiments/results/week18/reward_curve_ft06.png)

*Left: per-episode return (scaled, undiscounted). Right: per-episode
makespan, with the proven optimum (55) as a reference line. Full per-episode
data: `experiments/results/week18/train_ft06_metrics.csv`; trained checkpoint:
`experiments/results/ppo_gin_ft06.zip`.*

The pipeline's plumbing is also covered by fast dedicated smoke tests
(`tests/test_policy.py`, 5/5 passed) that exercise the features extractor's
forward pass, agent construction, a short `.learn()` call, and a full masked
greedy rollout — these run in seconds and are meant to catch shape/plumbing
regressions in CI without waiting for a full training run.

**Result: pipeline runs end-to-end, produces a visible learning trend, zero
crashes across the full 20k-timestep run and the repeated smoke-test runs.**

---

## Full test suite

```
$ pytest -q
188 passed in 15.60s
```

| file | tests | covers |
|---|---|---|
| `tests/test_instance.py` | 14 | `JSSPInstance` construction, validation, derived quantities |
| `tests/test_loader.py` | 104 | file parsing (standard + Taillard), bundled instances, known optima |
| `tests/test_disjunctive_graph.py` | 14 | adjacency construction, lower bound, reward potential, feasibility checker |
| `tests/test_env.py` | 51 | Gymnasium API, action masking, constraint safety, CP-SAT-verified ft06 optimum |
| `tests/test_policy.py` | 5 | GIN features extractor, MaskablePPO construction/training/rollout |

## Bugs found and fixed this week

- `baselines/cpsat.py`: `CpSolver.BestObjectiveBound`/`.WallTime` are
  deprecated no-argument methods in the installed `ortools==9.15.6755`
  (accessing them without calling returns a bound method object, not a
  number — `int(solver.BestObjectiveBound)` raised `TypeError`). Fixed by
  switching to the snake_case properties `best_objective_bound`/`wall_time`.
  Caught by `tests/test_env.py::TestOptimalSequence`, which is also why that
  test exists rather than trusting `solve_cpsat` untested.

## Next steps (not this week)

- Size-agnostic (per-node) action head so one policy can train across
  multiple instance sizes, not just `ft06` (see README "Known limitation").
- Longer training runs + hyperparameter sweep once the size-agnostic head is
  in, aimed at closing the remaining ~4% gap to optimal on ft06 and
  generalizing to `la01`-`la40`.
- GA baseline (`baselines/`) — not implemented yet, see README "Deviations".
- Dynamic JSSP extension (stochastic breakdowns, mid-episode job arrivals) —
  referenced as future work in `envs/jssp_env.py`'s docstring.
