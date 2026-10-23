# Week 15 Refactor Notes

**Scope:** a code-health sprint, explicitly *not* a research-features sprint. Four
sprints' worth of environments, baselines, and instance loaders (Weeks 1, 6, 9, 12) had
accumulated real inconsistencies that would have made next month's GNN+PPO work harder
to build on top of. This report documents what changed and why, grouped by the four
steps of the sprint, plus what was deliberately left alone or out of scope.

---

## STEP1a: `DynamicJSSPEnv`/`JSSPEnv` now share an honest base class

**Before:** `class DynamicJSSPEnv(JSSPEnv)`, but its `__init__` called
`gym.Env.__init__(self)` directly (not `super().__init__()`) and every other method
(`reset`, `step`, `action_masks`, `makespan`, `remaining_work_per_op`,
`schedule_dataframe_records`, `_observation`, `_info`, `render`) was fully overridden.
The only things actually inherited-and-reused were the `static_lb` and `dispatch_order`
properties — two property definitions bought an entire misleading `is-a` relationship: a
reader seeing `DynamicJSSPEnv(JSSPEnv)` would reasonably assume far more sharing than
existed, and the deliberate `__init__` bypass (with its own explanatory comment) was a
sign the inheritance was fighting the design rather than expressing it.

**After:** `envs/base_env.py` introduces `SchedulingEnv(gym.Env)`, and both
`JSSPEnv(SchedulingEnv)` and `DynamicJSSPEnv(SchedulingEnv)` inherit from it as siblings.
`SchedulingEnv` provides exactly what both environments compute identically:

- `_init_common(instance, reward_scale, include_machine_edges)` — instance resolution
  (`load_by_name` if given a string), `static_lb` via `static_lower_bound`,
  `reward_scale` defaulting to `1 / static_lb`, and the feature-normalization `_horizon`.
  Both `__init__` methods now call this instead of duplicating five lines apiece.
- `static_lb` / `dispatch_order` — read-only properties every `Baseline` adapter and PDR
  relies on to treat either environment interchangeably.
- Documented (but not `abc.ABC`-enforced — see below) stubs for `action_masks`,
  `makespan`, `remaining_work_per_op`, `schedule_dataframe_records`, so the contract
  `baselines/` code is written against has one readable home instead of only existing as
  convention spread across two files.

**Design decision (flagged): no `abc.ABC`.** `gym.Env` is a plain class, not `ABCMeta`;
mixing it with `abc.ABC` is *usually* safe (Python computes the most-derived metaclass
automatically when there's no conflict) but adds a failure mode with no real benefit
here — every consumer of `SchedulingEnv` in this codebase already fully implements the
contract, and `NotImplementedError`-raising stub methods communicate the same intent to
a reader without the metaclass risk.

**What deliberately did *not* change:** every test-visible internal attribute name
(`self.instance`, `self.reward_scale`, `self._static_lb`, `self._horizon`,
`self.include_machine_edges`, `self._dispatch_order`, and every `DynamicJSSPEnv`-only
attribute like `op_state`/`machine_state`/`_fel`/`_breakdown_gen`) is unchanged, because
`tests/test_dynamic_env.py` reaches into most of them directly (e.g.
`env._breakdown_gen.is_enabled(k)`, `env._fel.is_empty`,
`env._machine_busy_intervals`). The refactor's safety net was exactly this: run
`tests/test_env.py` + `tests/test_dynamic_env.py` (346 tests) after every edit, not just
at the end. `step()`/`reset()`/`_observation()` were **not** unified — the static env's
fixed-size list-scheduling arrays and the dynamic env's growable event-driven state are
genuinely different representations, and forcing them into one shape would have
distorted one design for no benefit (the sprint's own instruction: "clean interface over
new functionality," not "identical implementation").

---

## STEP1b: FJSP support is now complete, not just scaffolded

Before this sprint, "FJSP support" was real in pieces but never assembled:
`baselines/cpsat.py` had a private `FJSPInstance` dataclass with **zero validation**,
described in its own docstring as "a placeholder just complete enough to exercise the
CP-SAT `OptionalIntervalVar` path against a synthetic fixture"; `baselines/ga.py`'s
assignment-vector chromosome was implemented and tested only against that same
hand-built fixture; and `baselines/interface.py`'s `Baseline` adapters — the one uniform
call surface everything else (including the new `run_evaluation.py`) uses — wrapped the
JSSP path only, with a comment noting "there is no FJSP adapter here yet."

What changed:

1. **`FJSPInstance` moved to `envs/fjsp_instance.py`.** Machine eligibility is instance
   data, not a solver detail — it now sits next to `JSSPInstance`, with the same
   validation rigor (`__post_init__` checks job/position counts, that every operation has
   at least one eligible machine, no duplicate machine per operation, machine indices in
   range, non-negative durations). `baselines/cpsat.py` and `baselines/ga.py` both import
   it from there now; nothing constructs an unvalidated FJSP instance anymore.
   - **Consequence:** `solve_cpsat_fjsp`'s own "no eligible machine" check became dead
     code (an `FJSPInstance` can no longer exist in that state) and was deleted —
     `tests/test_cpsat.py::test_no_eligible_machine_raises` now asserts the `ValueError`
     at construction time instead.
2. **`generate_fjsp_instance(base, ...)`** (`envs/fjsp_instance.py`) derives a
   reproducible synthetic FJSP instance from a real bundled JSSP instance: every
   operation's original `(machine, duration)` survives as its first eligible
   alternative (so a degenerate run — forced to alternative 0 everywhere — reproduces
   the base instance's proven optimum exactly, checked in both
   `tests/test_fjsp_instance.py` and `tests/test_baselines_interface.py`), plus 0..N
   further alternatives with perturbed durations. This is *not* a real benchmark loader
   (no Brandimarte `MK*`/Hurink parser is bundled — fetching and parsing a new benchmark
   family is a research-scope decision for a later sprint) but it means every FJSP code
   path is now exercised against real `ft`/`la` processing-time data, not only a
   hand-built 2x2 toy.
3. **`CPSATFJSPBaseline`** added to `baselines/interface.py`, wrapping
   `solve_cpsat_fjsp` with the same call signature as `CPSATBaseline` (name
   `"cpsat_fjsp"` so a results CSV can tell the two apart).
4. **`GABaseline`** now accepts either instance type. The one real gap this exposed:
   `GeneticAlgorithm.run()` computed `machine_of_op` internally (via `decode()`) but
   never returned it, so there was no way to build an FJSP-aware feasibility check from
   the adapter. Fixed by adding `machine_of_op` to `run()`'s return dict (additive; no
   existing test read the dict exhaustively) and routing `GABaseline.solve()` through
   `baselines.cpsat.is_fjsp_schedule_feasible` (promoted from `_is_fjsp_schedule_feasible`
   — it now has two callers, not one, so the leading underscore was actively misleading)
   when the instance is FJSP.
5. **`PDRBaseline` explicitly rejects `FJSPInstance`** with a `TypeError`, rather than
   either crashing confusingly inside `JSSPEnv(instance)` or silently producing a
   meaningless result. This is a **deliberate scope boundary, not an oversight**: a PDR
   ranks *among the legal frontier operations*, but once an operation has more than one
   eligible machine it also needs a *machine-selection* rule, and none of Part B6's nine
   PDRs define one. Inventing a machine-selection heuristic is a research decision (which
   one? SPT-among-eligible? load-balanced?), explicitly out of scope for "no new
   research features." `run_evaluation.py --fjsp` enforces the same rule at the CLI
   argument-resolution level, so it fails fast with a clear message instead of raising
   deep inside a sweep.

---

## STEP2: `run_evaluation.py` — the unified evaluation harness

One CLI, one CSV schema, four method families (PDR / GA / CP-SAT / RL placeholder),
five disruption profiles (`none` / `breakdowns` / `stochastic_times` / `arrivals` /
`combined`), optional FJSP mode. `experiments/eval_baselines.py` (Week 9/12) is left
untouched rather than rewritten on top of this module, so the numbers already cited in
`BASELINES_RESULTS_WEEK9.md`/`BASELINES_RESULTS_WEEK12.md` stay reproducible from the
exact script that produced them.

### CSV schema

Long format, one row per `(instance, method, disruption, seed)`:

```
instance, family, num_jobs, num_machines, method, disruption, seed, makespan,
known_optimum, rpd_pct, wall_clock_s, feasible, status, is_optimal,
violations_count, timestamp
```

This shape is what the planned Wilcoxon/Friedman/Nemenyi analysis needs directly: a
paired Wilcoxon test between two methods pivots on `(instance, seed)`; a Friedman test
ranks all methods within each instance-block; Nemenyi's post-hoc follows from the same
average ranks. No reshaping step is needed between this CSV and any of the three.
`known_optimum`/`rpd_pct` are left blank for FJSP rows — a synthetically generated
instance has no literature-proven optimum, and its true optimum is generally *better*
than the base JSSP instance's (routing flexibility only helps), so reusing the JSSP
optimum as a stand-in would be actively misleading, not just approximate.

### Design decision (flagged): predictive-reactive replay for GA/CP-SAT under disruption

A PDR is already an online `(env, rng) -> op_id` policy, so "evaluate it under
disruption" is a direct rollout against a live `DynamicJSSPEnv` — no new machinery,
exactly what `tests/test_dynamic_env.py` already exercises. GA and CP-SAT are *offline*
solvers: they only ever see the nominal instance and have no notion of "solve while a
machine might break." Making them evaluable under disruption at all needs some bridge,
and the standard one (used throughout the dynamic-scheduling literature) is
**predictive-reactive execution**: solve the nominal instance once to get a planned
dispatch order, then execute that order against a live, disrupted `DynamicJSSPEnv`,
dispatching the next still-legal planned operation whenever possible.

The one gap a fixed offline plan cannot cover by construction: a dynamically arrived
job's operations were never in the plan, because the job didn't exist when the plan was
computed. `run_predictive_reactive` falls back to the already-tested `lwkr` (Least Work
Remaining) rule for any legal operation the plan doesn't anticipate. `lwkr` was chosen
deliberately over `mwkr` (which the codebase already uses elsewhere as a "strong single
rule" default) because `mwkr_rule` indexes the base instance's fixed-shape arrays
directly and is **not arrival-safe** — using it as the fallback would crash the instant
an arrived job's operation needed it, which is exactly the situation the fallback exists
to handle. This surfaced as a real bug during manual smoke-testing (see "Bugs found"
below), not a hypothetical.

This is glue necessary to make four fundamentally different method families comparable
under the same disruption axis at all — not a new scheduling algorithm, and not scored
or reported as if it were "the GA under breakdowns" in some novel sense beyond "the GA's
plan, executed reactively."

### Design decision (flagged): RL placeholder, not a stub method skipped in the CSV

`RLPlaceholderBaseline` (`baselines/interface.py`) satisfies the same `Baseline`
protocol as every real adapter and always returns a `BaselineResult` with
`status="NOT_IMPLEMENTED"`, `makespan=nan`, `feasible=False` — never raises. The
alternative (silently omitting `"rl"` rows, or raising `NotImplementedError`) would mean
the CSV's shape changes the day a real policy is swapped in, which is exactly the
instability this sprint exists to prevent. Swapping the placeholder for a trained
GIN+PPO policy next month changes zero downstream code: not the CLI, not the CSV
columns, not whatever reads the CSV for Wilcoxon/Friedman/Nemenyi.

### Bug found and fixed during manual verification

Manually smoke-testing every `--disruption` value (not just the automated test suite —
see STEP3) surfaced that `spt`/`lpt`/`mwkr`/`edd`/`cr` all fail once arrivals are enabled
(three with a raw `IndexError`, two with `baselines/dispatching_rules.py`'s own
`NotImplementedError` guard) — a known, already-documented limitation of those five
rules (`baselines/dispatching_rules.py`'s module docstring), but `run_evaluation.py`
didn't check for it before this fix, so `--disruption arrivals --methods mwkr` crashed
mid-sweep instead of failing fast. `validate_methods_for_disruption` now rejects
`ARRIVAL_UNSAFE_PDR_NAMES = {spt, lpt, mwkr, edd, cr}` for `arrivals`/`combined` at
argument-resolution time, with a message naming the six arrival-safe alternatives.

---

## STEP3: integration tests (`tests/test_run_evaluation.py`)

17 tests, all invoking `run_evaluation.main()` directly (no subprocess — faster, and
exceptions surface as normal Python exceptions rather than needing subprocess
return-code/stderr parsing), covering:

- End-to-end row count and CSV schema (`EXPECTED_COLUMNS` matches the module
  docstring's documented column list exactly).
- Every row well-formed for the `none` disruption case (types parse, `known_optimum`
  matches the literature value, `feasible` is `True`, `violations_count` is `0`).
- The RL placeholder row is explicit (`status == "NOT_IMPLEMENTED"`), not a crash.
- All four non-`none` disruption profiles actually run end-to-end for both a PDR
  (online rollout) and GA (predictive-reactive replay) — this is the test class that
  would have caught the arrival-unsafe-PDR bug above, had it existed before the fix.
- `--fjsp` produces blank `known_optimum`/`rpd_pct` and the `fjsp-synthetic` family
  label.
- Every documented `SystemExit` guard actually fires: `--fjsp` + disruption,
  `--fjsp` + a PDR method, an arrival-unsafe PDR + `arrivals`/`combined`, an unknown
  method name.
- `write_csv([], ...)` raises `ValueError` rather than silently writing a headerless
  file that would make a schema test pass vacuously.

All `time_limit_s` values in this file are small (0.5-1.0s; ft06 is 36 operations, and
CP-SAT/GA both converge usefully well under a second at that size) so the file stays
fast enough to run as part of the default suite alongside the unit tests.

---

## STEP4: this file + `README.md`

`README.md` was rewritten (not incrementally patched) to reflect the current
repository shape: the structure tree now lists every module that exists today
(`base_env.py`, `fjsp_instance.py`, `dispatching_rules.py`, `ga.py`, `due_dates.py`,
`dynamic_jssp_env.py`, `disruptions.py`, `events.py`, `run_evaluation.py`), the
"Running things" section documents `run_evaluation.py`'s CLI with one example per
disruption profile and FJSP, and a "Deviations" bullet claiming "GA baseline not
implemented this week" (true in Week 1, false since Week 12) was removed rather than
left to mislead a future reader — the project's own documentation rule
("nothing should go stale silently") applies just as much to a README.

---

## Test suite status

Baseline (before this sprint's changes, project `.venv`): **469 passed** in 370.07s
(0:06:10).

After this sprint (base class refactor + FJSP completion + `run_evaluation.py` +
integration tests): **510 passed** in 381.95s (0:06:22), zero failures. Every
pre-existing test still passes unmodified except the one noted above
(`test_no_eligible_machine_raises`, updated to match `FJSPInstance` now validating at
construction time, a strictly earlier and clearer failure than before). The 41 new
tests: `tests/test_fjsp_instance.py` (17), `tests/test_run_evaluation.py` (17), plus
FJSP-path additions to `tests/test_baselines_interface.py` (5) and
`tests/test_ga.py` (2, `TestRunReturnsMachineOfOp`). Runtime is essentially unchanged
(+12s over +41 tests) because the new tests reuse small time limits (0.5-3s) rather
than adding new CP-SAT/GA-heavy cases at the scale of the existing `la05`
five-seed-at-60s regression test, which continues to dominate the suite's wall clock.

---

## What was explicitly left alone (scope boundaries)

- **No real FJSP benchmark file loader.** `envs/fjsp_instance.py`'s synthetic generator
  is real-benchmark-*derived*, not a substitute for actually parsing Brandimarte `MK*`
  or Hurink files. That is new research surface area (a new format, new provenance
  questions exactly like `scripts/fetch_instances.py`'s own "don't hand-transcribe
  benchmark data" precedent), not a refactor.
- **No PDR machine-selection rule for FJSP.** See STEP1b point 5 above.
- **`DynamicJSSPEnv` still does not model machine eligibility.** It remains fixed-routing
  only, same as before this sprint. `run_evaluation.py --fjsp` and `--disruption
  <anything but none>` are mutually exclusive for exactly this reason, enforced at the
  CLI level rather than left to fail deep inside a rollout.
- **`experiments/eval_baselines.py` was not deleted or rewritten.** It stays as the
  exact provenance trail for two already-published result tables.
- **No changes to `models/`, `experiments/train_ft06.py`, or the reward/observation
  design of either environment.** This sprint's instruction was explicit: a clean,
  well-documented interface for the RL model to plug into next month, not new
  functionality — the GNN+PPO architecture itself is next month's work.
