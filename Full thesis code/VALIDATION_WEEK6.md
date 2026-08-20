# Week 6 Validation Report

**Scope:** the dynamic/stochastic JSSP extension --- machine breakdowns, stochastic
processing times, and mid-episode job arrivals --- implemented per
`DynamicEnv_TechSpec.md` on top of the Week 1 static environment (`envs/jssp_env.py`,
`envs/disjunctive_graph.py`, `envs/instance.py`, unchanged; see `VALIDATION_WEEK1.md`).
New modules: `envs/events.py`, `envs/disruptions.py`, `envs/dynamic_jssp_env.py`, plus two
additive functions in `envs/disjunctive_graph.py` (`estimated_makespan_dynamic`,
`is_schedule_feasible_dynamic`). This report checks the three criteria requested for
Week 6 sign-off (spec §9.3 a/b/c) plus the additive-not-rewritten regression contract
(§9.1/§9.2).

---

## Design clarifications made explicit (spec §10 culture: flag, don't silently bake in)

The spec resolves almost every implementation choice, but four places required a concrete
engineering decision this report flags for committee visibility, same spirit as
§10.1–§10.5:

1. **Action mask needs a "machine is free" clause, not just "not broken."** §6.2 lists
   "unscheduled AND frontier AND not preempted AND machine not broken." Read literally,
   two different jobs' frontier operations both requiring the same *busy* (not broken)
   machine would both show as legal, and dispatching either would double-book that
   machine the instant it happened. §3.5's decision-epoch definition already requires a
   free machine, so `action_masks()` uses "machine is FREE (neither busy nor broken)" as
   the single machine clause, which subsumes the two clauses in §6.2's bullet list rather
   than contradicting them.
2. **`baselines/dispatching_rules.py` is frozen and only partially arrival-safe.**
   `spt_rule`, `lpt_rule`, and `mwkr_rule` index the *base* `JSSPInstance`'s fixed-shape
   arrays (`instance.flat_durations`, `instance.remaining_work()`) directly by operation
   id, which cannot cover an id belonging to a dynamically arrived job. Rather than
   editing the frozen baseline module, the six-PDR rollout protocol
   (`test_rule_rollout_is_always_feasible_dynamic`) runs with breakdowns and stochastic
   processing times enabled but **arrivals disabled**; dynamic arrivals are validated
   separately (`TestDynamicArrivals`, §c below) with tests that don't route through that
   module. `random_rule`, `fifo_rule`, and `mopnr_rule` are in fact arrival-safe (they
   only use `action_masks()`, `job_ready_time`, and pure `job_of`/`position_of`
   arithmetic), so this restriction is specifically about the other three.
3. **"Byte-identical to the static env" needed a precise definition.** §3.2 (E2)
   requires a job's next operation to enter the ready set only when its predecessor's
   `OP_COMPLETE` actually fires (real completion). The static env advances the same
   frontier pointer at *dispatch* time instead, by design (list scheduling decouples
   decision order from real time). Both are correct for their own model, but it means
   `is_ready`/the action mask, `is_job_complete`, and `earliest_start_estimate` --- the
   three features keyed on completion rather than dispatch --- structurally and
   correctly diverge mid-episode even with every disruption axis off. Verified
   empirically (not just asserted): for all 6 PDRs x 2 instances (`ft06`, `la01`),
   reward, `terminated`, `adj_precedence`, and every dispatch-status-dependent node
   feature match the static env exactly at *every* step; the three completion-gated
   features converge back to an exact match at the terminal step, and total return /
   final makespan always match. `test_default_config_matches_static_env_exactly` checks
   precisely this (12 rule x instance combinations).
4. **Precedence must be checked against realised completion, not nominal duration, even
   under breakdowns alone (no stochastic processing times).** A preempted-then-resumed
   operation's real completion is later than `start + base_duration` --- the pause is
   invisible to the static `is_schedule_feasible(instance, start_times)` checker, which
   only knows the base instance's nominal durations. Blindly reusing it for the
   precedence check (as a literal reading of the spec's test description for
   `test_precedence_preserved_under_breakdown` might suggest) would be a correctness bug
   in the *test*, not just missing coverage, so both the precedence and machine-capacity
   checks use the new `is_schedule_feasible_dynamic`, which checks precedence against
   `end_times` (realised) and capacity against the environment's own recorded busy/broken
   intervals.

---

## (a) Statistical validation of the three disruption generators

All numbers below are a fresh run of `tests/test_disruptions.py`'s exact calibration
formulas, `N = 5000` draws per assertion, fixed master seeds (deterministic, not a
p-value-threshold hypothesis test --- see spec §8), computed independently of the
generator's own reported target so a calibration bug in the generator itself would still
be caught.

**Breakdown generator** (`ft06`, `mttr_multiplier=2.0`, `ttr_shape_sigma=0.5`):

| `A_k` target | MTBF target | empirical mean TTF | error | MTTR target | empirical mean TTR | error | KS p-value (TTF vs Exp) | empirical availability |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 207.944 | 207.006 | -0.45% | 10.944 | 10.887 | -0.53% | 0.929 | 0.9500 |
| 0.90 | 98.500 | 98.055 | -0.45% | 10.944 | 10.887 | -0.53% | 0.929 | 0.9001 |
| 0.80 | 43.778 | 43.580 | -0.45% | 10.944 | 10.887 | -0.53% | 0.929 | 0.8001 |

MTTR and its empirical estimate are identical across rows, as expected: `MTTR` depends
only on `mttr_multiplier * mean_processing_time`, not on `A_k` (checked directly by
`test_mean_ttr_independent_of_availability`). The KS test (`scipy.stats.kstest` against
`Exp(lambda_k)`, one dedicated "shape" test rather than run on every assertion, per spec
§8) does not reject the exponential hypothesis (`p = 0.929 >> 0.01`). The renewal-process
availability recovery (drawing 5000 TTF/TTR pairs in sequence and taking
`sum(TTF)/(sum(TTF)+sum(TTR))`) lands within 0.01% of every target `A_k`.

**Processing time sampler** (mean-1 Lognormal multiplier):

| `s` | empirical mean | target mean | empirical variance | theoretical variance `exp(s²)-1` | error |
|---|---|---|---|---|---|
| 0.10 | 0.9980 | 1.0 | 0.00991 | 0.01005 | -1.38% |
| 0.25 | 0.9949 | 1.0 | 0.06339 | 0.06449 | -1.71% |
| 0.50 | 0.9894 | 1.0 | 0.27667 | 0.28403 | -2.59% |

Mean is within 1.1% of 1.0 at every `s` (well inside the 5%-relative / 0.02-absolute test
tolerance --- an absolute floor is needed here because 5% of a mean-1 quantity is a very
tight band). Variance increases monotonically with `s` and matches the closed form
`exp(s²) - 1` (derived in the `ProcessingTimeSampler.draw_multiplier` docstring: since
`mu = -s²/2` is chosen so `E[X] = 1` exactly, `Var(X) = (exp(s²)-1)*exp(2*mu+s²) =
(exp(s²)-1)*exp(0)`) within 2.6% at every tested value, comfortably inside the 10%
tolerance spec §8 allows for this noisier statistic.

**Arrival process** (Poisson, calibrated to target utilization `rho`, `ft06`):

| `rho` | `lambda_a` target | empirical `lambda_a` | error |
|---|---|---|---|
| 0.70 | 0.12792 | 0.12691 | -0.79% |
| 0.85 | 0.15533 | 0.15410 | -0.79% |
| 0.95 | 0.17360 | 0.17223 | -0.79% |

(Cutoff effectively disabled and `max_dynamic_jobs=100000` for this specific rate
measurement, so the 5000-draw renewal average isn't truncated by either termination
mechanism.) Separately, `test_arrival_process_stops_at_cutoff` and
`test_arrival_process_stops_at_max_dynamic_jobs` confirm the two termination-safety
mechanisms independently: no arrival time is ever generated past
`arrival_cutoff_time`, and generation stops at exactly `max_dynamic_jobs` jobs.

`test_generators_independent_of_seed_spawn` confirms the third calibration property
required by spec §4: toggling `processing_time_sigma` from `0.0` to `0.5` leaves the
breakdown generator's draw sequence byte-identical (both are seeded from the same
`spawn(3)` call, so they are independent RNG streams by construction, not by coincidence).

**Result: `tests/test_disruptions.py` --- 25/25 passed.** `tests/test_events.py` (FEL
ordering, the fixed tie-break priority, `cancel()`'s heap-validity preservation) ---
14/14 passed.

---

## (b) Hard constraints hold under disruption, across the Week 1 rollout protocol

Mirroring Week 1's protocol (all six PDRs x multiple instances x multiple seeds), now
under disruption: `tests/test_dynamic_env.py::TestConstraintSafetyUnderBreakdown` runs
all six dispatching rules (`random`, `spt`, `lpt`, `fifo`, `mwkr`, `mopnr`) on `ft06` and
`la01`, three seeds each (36 combinations), with a `DisruptionConfig(target_availability=
0.8, mttr_multiplier=2.0, processing_time_sigma=0.25)` (breakdowns + stochastic
processing times; arrivals off for this specific harness --- flagged decision #2 above)
and checks every resulting schedule against `is_schedule_feasible_dynamic`. **All 36
combinations are feasible**, with makespan `>= static_lower_bound` in every case.

Two further parametrized checks (`random`/`spt`/`mwkr` x 4 seeds = 12 each) isolate the
two constraint families specifically requested by spec §8:

- **Precedence** (`test_precedence_preserved_under_breakdown`): zero
  `"precedence violated"` entries in the violation list, checked against *realised*
  completion times (see flagged decision #4).
- **No machine double-booking, including no work during a broken interval**
  (`test_no_machine_double_booking_under_breakdown`): zero `"overlap"` or
  `"broken interval"` entries. Each of these 12 runs is also asserted to have actually
  triggered at least one real breakdown (`len(machine_broken_intervals) > 0`), so the
  check is not vacuously passing on a rollout where nothing broke down.

Preemption bookkeeping itself is checked directly (`TestPreemption`, not statistically ---
a mocked/injected `BREAKDOWN` event mid-operation, per spec §8's suggested method): the
operation's `remaining_duration` after preemption, the two split machine-busy segments
(`[dispatch, breakdown)` and `[repair_complete, repair_complete + remaining)`), and the
resumed `OP_COMPLETE` firing at exactly `repair_complete_time + remaining_duration` are
all asserted exactly, both end-to-end through the normal event loop and with a
manually-driven step-by-step variant that inspects the intermediate `PREEMPTED` state
before any resume happens. `TestActionMasking` separately confirms a frontier operation
whose machine is broken is masked out (`mask == 0`), where the static-only masking rule
would have said `1`, and that a `PREEMPTED` operation is never legal to (re-)dispatch.

An **additional stress sweep** beyond the pinned test seeds --- 60 seeds, `ft06`, all
three disruption axes simultaneously on (`target_availability=0.85`,
`processing_time_sigma=0.25`, `target_utilization=0.8`, `max_dynamic_jobs=6`), random
legal-action rollouts --- produced **zero infeasible schedules and zero runaway
episodes**, including scenarios (verified by inspection of one transcript, see below)
where the same operation is preempted by a fresh breakdown more than once before it
finally completes.

**Result: `tests/test_dynamic_env.py` --- 88/88 passed.**

---

## (c) Episodes terminate correctly with dynamic arrivals

`TestDynamicArrivals::test_episode_terminates_with_dynamic_arrivals` runs `ft06` with
`target_utilization=0.85`, `max_dynamic_jobs=10` to completion under random legal
actions: the episode terminates only once `info["num_scheduled"] ==
info["num_operations"]` (every arrived job's operations, not just the original six, are
`DONE`) and at least one arrival actually occurred
(`info["num_jobs_arrived"] > base_num_jobs`); the resulting schedule is feasible.
`test_episode_terminating_system_bounded_length` repeats a similar rollout across 8 seeds
and asserts step count never exceeds `max_possible_ops` --- a regression guard against the
arrival cutoff/cap silently being bypassed (§4.3's mandatory termination safety), since an
unbounded arrival process would otherwise eventually write past the pre-allocated
growable arrays rather than merely looping forever.

`test_dynamic_job_extends_graph_correctly` confirms the graph-growth mechanics
independently of termination: after a forced `ARRIVAL` event, the new job's operations
appear with `UNSCHEDULED` state, the correct precedence chain edges (and *no* stray edge
leaking into a neighboring job's block), and correct machine-sharing edges to every
currently-unscheduled pre-existing operation on the same machine (checked both
directions, since `adj_machine` is symmetric).

### Example event log (transcribed for human sanity-checking)

A short episode (toy 2-job instance, `target_availability=0.7`, `mttr_multiplier=1.5`,
`processing_time_sigma=0.2`, `target_utilization=0.6`, `max_dynamic_jobs=2`, seed 4, random
legal actions) --- chosen because it happens to exercise every event type, including the
same operation (op 4, one of the two dynamically arrived jobs) being preempted by four
separate breakdowns before it finally completes:

```
dispatch order: [2, 0, 4, 6, 3, 7, 5, 1]

('arrival',        0.871468, 2)          # job 2 (first dynamic arrival) instantiated
('arrival',        1.547821, 3)          # job 3 (second dynamic arrival, hits max_dynamic_jobs=2)
('op_complete',     3.419432, 2, 1)      # op 2 done on machine 1
('op_complete',     3.698360, 0, 0)      # op 0 done on machine 0
('breakdown',       5.994076, 1, 4)      # machine 1 breaks down mid-op-4; op 4 preempted
('op_complete',     6.519217, 6, 0)      # op 6 done on machine 0 (unaffected: different machine)
('op_complete',     7.569696, 3, 0)      # op 3 done on machine 0
('repair_complete', 8.472827, 1)         # machine 1 repaired; op 4 resumes
('breakdown',       9.394005, 1, 4)      # short TTF draw -- machine 1 breaks again, op 4 re-preempted
('repair_complete',13.736095, 1)
('breakdown',      15.041507, 1, 4)      # and again
('repair_complete',17.409477, 1)
('breakdown',      18.917675, 1, 4)      # and again -- op 4 has now survived 4 breakdowns
('repair_complete',20.540791, 1)
('op_complete',    23.555357, 4, 1)      # op 4 *finally* completes
('op_complete',    24.237968, 7, 1)
('op_complete',    25.747108, 1, 1)
('op_complete',    25.901373, 5, 0)      # last operation done -- episode terminates
```

Final makespan: **25.90**, `schedule_feasible: True`, zero violations from
`is_schedule_feasible_dynamic` (including the repeated-preemption interval bookkeeping for
op 4, which the checker validates against four separate busy segments on machine 1, none
overlapping any of the four broken intervals). This is a genuinely unlucky sample path
(four short TTF draws in a row on the same machine) rather than a cherry-picked easy case,
which is exactly the kind of edge case a hand-transcribed log is meant to make visible to
a reviewer.

---

## Regression contract (additive, not a rewrite)

`test_zero_disruption_config_disables_every_axis` confirms `DynamicJSSPEnv(instance,
DisruptionConfig())` seeds no `BREAKDOWN`/`ARRIVAL` events at `reset()` and both
generators report themselves disabled. `test_default_config_matches_static_env_exactly`
(12 combinations: 6 rules x 2 instances) is the empirical proof described in flagged
decision #3 above. No changes were made to `models/gin.py`, `models/policy.py`,
`experiments/train_ft06.py`, or any existing signature in `envs/instance.py`,
`envs/jssp_env.py`, or `envs/disjunctive_graph.py` --- only two new functions were added
to the last file.

---

## Full test suite

```
$ pytest -q
315 passed in ~15s
```

| file | tests | covers |
|---|---|---|
| `tests/test_instance.py` | 14 | unchanged (Week 1) |
| `tests/test_loader.py` | 104 | unchanged (Week 1) |
| `tests/test_disjunctive_graph.py` | 14 | unchanged (Week 1); new functions covered by `test_dynamic_env.py` instead, since they need dynamic-env state to exercise meaningfully |
| `tests/test_env.py` | 51 | unchanged (Week 1) |
| `tests/test_policy.py` | 5 | unchanged (Week 1); training pipeline untouched this sprint |
| `tests/test_events.py` | 14 | FEL ordering, tie-break priority, `cancel()` |
| `tests/test_disruptions.py` | 25 | statistical validation of the 3 disruption generators, RNG independence |
| `tests/test_dynamic_env.py` | 88 | regression vs static, constraint safety under breakdown, preemption bookkeeping, action masking, dynamic arrivals/termination |

Full suite run 3 times consecutively with no flakiness observed (all seeds fixed; no
`Date.now()`/wall-clock dependence anywhere in the new code).

## Bugs found and fixed this week

- Initial regression-test draft ran each rule independently against the static and
  dynamic envs' own masks and compared observations step-by-step; this produces
  legitimate divergence almost immediately (see flagged decision #3) and is not a
  meaningful test. Fixed by deriving the dispatch sequence from the dynamic env's own
  (causally valid) decision epochs and replaying that exact sequence through the static
  env, comparing only the quantities that are actually invariant between the two models.
- `_run_until_decision_epoch`'s original drain condition (`while not
  has_decision_epoch(): drain`) would loop forever once every operation is `DONE` but
  breakdowns remain enabled, since breakdown/repair is an unbounded renewal process that
  never leaves the FEL empty. Fixed by also exiting the drain loop once
  `_check_termination()` holds, regardless of what's still pending in the FEL.

## Next steps (not this week)

- Wire `DynamicJSSPEnv` into the PPO+GIN training pipeline (explicitly deferred, §0)
  once the size-agnostic action head mentioned in `VALIDATION_WEEK1.md`'s "Next steps"
  is in place --- the growable-array design here (`max_possible_ops`,
  `observation_space` fixed at its maximum size) is exactly what that head needs to
  consume.
- Implement the `kappa * Delta_inest` robustness-aware reward term (Part D3); this
  sprint only wired the plumbing (`info["planned_start_deviation"]`), per spec §7.
- Run the `preemption_policy="restart"` ablation (implemented, config-gated, not
  exercised by the required test suite) if the committee requests a sensitivity check
  against preempt-resume.
- A `sigma_r` (TTR shape) sensitivity sweep, per §10.1's suggestion, time permitting.
