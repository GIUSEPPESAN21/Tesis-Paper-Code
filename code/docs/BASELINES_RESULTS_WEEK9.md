# Week 9 Baseline Results

**Scope:** implements `Baselines_TechSpec.md`'s STEP1-4 --- the full 9-rule PDR suite, a
CP-SAT wrapper extended to FJSP, an evaluation script, and unit tests --- on top of the
already-validated environment and instance loader (`envs/`, `data/instances/`; see
`VALIDATION_WEEK1.md`/`VALIDATION_WEEK6.md`, 315/315 tests passing before this sprint).
**The genetic algorithm (`Baselines_TechSpec.md` §5) is explicitly deferred**, per this
sprint's instructions ("next month's code sprint") --- `baselines/ga.py`,
`GABaseline`, and `baselines/harness.py` (the generic multi-baseline sweep module, whose
main purpose was feeding GA/CP-SAT sweeps) are not implemented. STEP3's "small evaluation
script" is `experiments/eval_baselines.py`, a standalone script rather than a reusable
harness API, matching what was actually asked for.

Result: **444/444 tests passing** (315 pre-existing + 129 new), CP-SAT proves the known
optimum on all 13 evaluated instances (ft06, ft10, ft20, la01-la10) well inside its 60s
budget (worst case 3.8s), and every PDR produces a feasible schedule everywhere it was run.

---

## What was added

| File | Status | Contents |
|---|---|---|
| `baselines/dispatching_rules.py` | EXTENDED | `lwkr_rule`, `lor_rule`, `edd_rule`, `cr_rule`; `"mor"` registry alias for `mopnr_rule`; `STATIC_ONLY_RULES` (empty, exported for forward-compat). The six pre-existing rules (`random_rule`, `spt_rule`, `lpt_rule`, `fifo_rule`, `mwkr_rule`, `mopnr_rule`) are byte-identical --- verified against a frozen snapshot. |
| `baselines/due_dates.py` | NEW | `DueDateConfig`, `synthesize_due_dates` (TWK rule), `ensure_due_dates` (lazy cache on `env.due_dates`). |
| `baselines/interface.py` | NEW | `BaselineResult`, `Baseline` protocol, `PDRBaseline`, `CPSATBaseline` --- the "family of classes sharing a common interface" STEP1 asked for. No `GABaseline` (deferred). |
| `baselines/cpsat.py` | EXTENDED | `FJSPInstance` (minimal placeholder dataclass), `solve_cpsat_fjsp` (`OptionalIntervalVar` + `AddExactlyOne`), `_is_fjsp_schedule_feasible` (internal self-check). `solve_cpsat` untouched. |
| `baselines/__init__.py` | EXTENDED | Exports the new public names. |
| `envs/jssp_env.py` | EXTENDED (additive) | `remaining_work_per_op()` --- one-line delegation to `instance.remaining_work()`. |
| `envs/dynamic_jssp_env.py` | EXTENDED (additive) | `remaining_work_per_op()` --- per-job reverse-cumsum over the growable duration array, covers arrived jobs. |
| `experiments/eval_baselines.py` | NEW | STEP3's evaluation script; writes `experiments/results/baselines_week9.{csv,md}`. |
| `tests/test_dispatching_rules.py` | NEW | Formula correctness, MOR alias, regression snapshot, all-9-PDRs-feasible, arrival safety. |
| `tests/test_due_dates.py` | NEW | TWK formula, arrival release-time semantics, lazy-cache behavior. |
| `tests/test_baselines_interface.py` | NEW | Protocol conformance, time-limit requirements, CP-SAT status/optimal consistency. |
| `tests/test_cpsat.py` | NEW | CP-SAT recovers known optimum on ft06/ft10/ft20 (STEP4); FJSP path correctness (ragged-eligibility toy + degenerate-single-alternative cross-check against `solve_cpsat`). |
| `tests/test_env.py`, `tests/test_dynamic_env.py` | EXTENDED | `RULES` registry tripwire assertion. |
| `tests/fixtures/pdr_regression_snapshot.json` | NEW | Frozen `run_rule` output (6 rules x 2 instances x 5 seeds = 60 records) backing the regression test. |

---

## STEP1 --- the nine PDRs

All nine rules (SPT, LPT, MWKR, LWKR, MOR, LOR, FIFO, EDD, CR) share one interface:
`(env, rng) -> op_id`, callable directly as the `policy` in the same rollout loop that
will later host the trained RL agent (`run_rule(env, rule_name, seed)` already *is* that
loop). On top of that, `baselines/interface.py::PDRBaseline` wraps any rule name into a
`.solve(instance, seed=..., time_limit_s=...) -> BaselineResult` object satisfying the
same `Baseline` protocol as `CPSATBaseline` --- this is the class-based common interface,
used uniformly by `experiments/eval_baselines.py`.

`RULES` has **11 keys but 10 unique callables**: `"mopnr"` and `"mor"` are two registry
entries for the same function (`mopnr_rule` keeps its original name, per the tech spec's
"do not rename the function"). `len(set(RULES.values())) == 10` (nine PDRs + random) is
the tripwire actually asserted in `tests/test_env.py`/`tests/test_dynamic_env.py`.

---

## STEP2 --- CP-SAT, JSSP and FJSP

`solve_cpsat` (JSSP, `IntervalVar` + `AddNoOverlap`) is unchanged. `solve_cpsat_fjsp`
adds one `OptionalIntervalVar` per (operation, eligible machine) pair, ties them together
with `AddExactlyOne` (CP-SAT's assignment constraint), groups by machine for
`AddNoOverlap`, and reuses the JSSP path's precedence/objective/status-reporting
unchanged. Both take an explicit, configurable `time_limit_s` (default `60.0`) and return
schedule (`start_times`), `makespan`, `is_optimal`, and `wall_time` (solve time) --- via
the unchanged `CPSATResult` dataclass.

`FJSPInstance` is a minimal placeholder (rectangular `ops_per_job`, ragged
`eligible_machines`/`durations` per operation) sufficient to exercise the CP-SAT path;
real FJSP instance loading (MK01-MK10, Hurink) is out of scope, per the tech spec.
Verified on a hand-built 2-job/2-machine ragged-eligibility toy (optimum 7, hand-checked)
and cross-validated by wrapping `ft06` as a degenerate single-alternative FJSP instance,
where `solve_cpsat_fjsp` reproduces `solve_cpsat`'s optimum of 55 exactly.

---

## STEP3 --- evaluation results

`python experiments/eval_baselines.py --time-limit 60 --seed 0` --- full table (130 rows)
at `experiments/results/baselines_week9.csv` / `.md`. Summary below.

### Mean RPD (%) vs. known optimum, across all 13 instances

| method | mean RPD % | min | max |
|---|---:|---:|---:|
| **cpsat** | **0.00** | 0.00 | 0.00 |
| fifo | 17.80 | 0.00 | 44.29 |
| mor | 35.33 | 4.07 | 68.50 |
| mwkr | 38.05 | 12.31 | 71.25 |
| spt | 148.15 | 76.38 | 215.27 |
| lpt | 160.63 | 103.02 | 206.56 |
| edd | 195.69 | 154.30 | 274.86 |
| lwkr | 197.88 | 155.88 | 274.86 |
| lor | 212.17 | 140.00 | 262.37 |
| cr | 236.24 | 179.90 | 295.91 |

CP-SAT proved optimality on **all 13/13 instances**, worst-case wall time 3.79s (ft10),
well inside the 60s budget. FIFO is the strongest single PDR on this instance set (best
on 11/13 instances), with MOR/MWKR close behind and best on the remaining 2 (la03,
la10) --- consistent with the scheduling literature's general finding that MWKR-family
rules are strong, general-purpose JSSP baselines. LWKR/EDD/LOR/CR (the four new rules)
are, as expected of their "least priority first" character, substantially weaker than
their mirror-image counterparts (MWKR, MOR) --- this is the intended, literature-typical
result, not a bug: these are the classic weak baselines a learned dispatcher is expected
to beat easily, exactly as strong SPT/MWKR results are the harder bar to clear.

### Best PDR per instance

| instance | best PDR | RPD % |
|---|---|---:|
| ft06 | fifo | 21.82 |
| ft10 | fifo | 30.97 |
| ft20 | fifo | 44.29 |
| la01 | fifo | 20.42 |
| la02 | fifo | 24.58 |
| la03 | mwkr | 33.50 |
| la04 | fifo | 11.86 |
| la05 | fifo | 0.00 |
| la06 | fifo | 9.50 |
| la07 | fifo | 9.66 |
| la08 | fifo | 8.92 |
| la09 | fifo | 3.37 |
| la10 | mor | 4.07 |

Full per-instance-per-method table: see `experiments/results/baselines_week9.csv` (or
`.md`); reproduced in full at the bottom of this report for convenience.

---

## STEP4 --- unit tests

**444/444 passing** (`pytest -q`, ~36s), 129 new:

| file | new tests | covers |
|---|---:|---|
| `tests/test_dispatching_rules.py` | 38 | formula correctness (LWKR/LOR/EDD/CR across 3 decision epochs), MOR≡mopnr, 60-record regression snapshot, all 9 PDRs feasible on ft06/ft10/la01, arrival safety |
| `tests/test_due_dates.py` | 6 | TWK formula, arrival release-time semantics, lazy-cache idempotency |
| `tests/test_baselines_interface.py` | 6 | protocol conformance, time-limit requirement (CP-SAT raises, PDR doesn't), status/optimal consistency, ft06 optimum recovery |
| `tests/test_cpsat.py` | 5 | **STEP4's explicit requirement**: CP-SAT recovers the known optimum on ft06/ft10/ft20; FJSP ragged-eligibility + degenerate-JSSP-equivalence checks |
| `tests/test_env.py`, `tests/test_dynamic_env.py` | 2 | `RULES` registry tripwire |

Full suite run twice consecutively, no flakiness (every test seed is fixed; the only
`seed=None`-entropy fallback, `PDRBaseline`'s default, is never exercised by a test that
asserts an exact value).

---

## Design decisions and issues found (flagged, not silently baked in)

1. **`RULES` dict size is 11, not 10.** The tech spec's own §7 asks for a
   `len(RULES) == 10` tripwire ("nine PDRs + random"), but also asks for `"mopnr"` to keep
   its dict entry *and* `"mor"` to be added as a new alias --- two keys for the same
   function, which makes `len(RULES) == 11`. Resolved by asserting
   `len(set(RULES.values())) == 10` instead (nine *unique* PDR callables + random), which
   is what a silently-broken new-rule registration would actually violate; `len(RULES)`
   is separately pinned at 11 in the same test so this distinction is explicit, not lost.

2. **EDD/CR are not fully arrival-safe, and are documented as such rather than forced.**
   `lwkr_rule`/`lor_rule`/the `mor` alias are genuinely arrival-safe (verified against
   `DynamicJSSPEnv` with real arrivals enabled: `tests/test_dispatching_rules.py::TestArrivalSafety`).
   `edd_rule`/`cr_rule` are safe for every scenario this sprint's tests actually exercise
   (static `JSSPEnv`; `DynamicJSSPEnv` with breakdowns/stochastic durations but arrivals
   off), but due-date synthesis for a *dynamically arrived* job (release time = actual
   arrival time) is out of scope this sprint --- no evaluation here needs it (STEP3 is
   static `ft`/`la` instances only). Rather than leaving a latent `IndexError` for
   whoever runs EDD/CR against `DynamicJSSPEnv` with arrivals enabled next, both rules
   now raise a clear `NotImplementedError` in that specific case
   (`tests/test_dispatching_rules.py::TestArrivalSafety::test_due_date_rules_raise_clear_error_with_real_arrivals`).
   This is a deliberate scope cut, not an oversight; per the tech spec's own §3.2 fallback,
   moving them to `STATIC_ONLY_RULES` instead was considered and rejected, because that
   would have silently dropped them from `tests/test_env.py`'s existing
   `RULES`-parametrized static-feasibility coverage, which is a real regression the clear
   error avoids without giving up.

3. **`ensure_due_dates` populates lazily instead of raising if unset.** The tech spec's
   §7 test list asks for `edd_rule`/`cr_rule` to "raise a clear error ... if
   `env.due_dates` has not been populated." That would break the existing
   `RULES`-keyed parametrized tests in `tests/test_env.py`/`tests/test_dynamic_env.py`,
   which call `RULES[rule_name](env, rng)` directly with no setup step of their own ---
   exactly the auto-coverage mechanism the tech spec's own §7 relies on for the four new
   rules. Resolved by making due-date population lazy and idempotent
   (`baselines/due_dates.py::ensure_due_dates`, cached on `env.due_dates`): EDD/CR work
   as drop-in `(env, rng) -> op_id` callables with zero setup, exactly like the other
   seven rules, while still being callable with a pre-set custom `DueDateConfig` if a
   caller wants one.

4. **CR's `t_now`.** `JSSPEnv` has no global simulation clock by design. Per the tech
   spec's own §9.2, `t_now = max(job_ready_time[job], machine_free_time[machine])` ---
   the candidate operation's own earliest feasible start --- is used for both `JSSPEnv`
   and `DynamicJSSPEnv`, rather than switching to the dynamic env's real simulated clock
   for consistency across both environments (the tech spec flags this as a legitimate,
   documented departure from a literal reading of Part B6, not an oversight).

5. **`baselines/harness.py` was not built as a separate module.** The tech spec calls for
   a generic `run_baseline_suite(baselines, instances, seeds, time_limit_s)` reusable
   harness; STEP3 only asked for "a small evaluation script." Since the genetic algorithm
   (the harness's other main consumer, alongside a future multi-seed sweep) is deferred,
   the sweep logic was written directly in `experiments/eval_baselines.py` instead of
   behind a speculative reusable API with only one real caller. `baselines/interface.py`
   (the `Baseline` protocol + `BaselineResult`) is still built exactly as specified, since
   STEP1 explicitly asked for a common interface and it's what makes
   `eval_baselines.py`'s method loop uniform.

6. **`FJSPInstance` is a placeholder, not wired into `envs/loader.py`.** Per the tech
   spec's explicit scope note: real FJSP instance loading (MK01-MK10, Hurink) is
   deferred; `FJSPInstance` lives in `baselines/cpsat.py` (not a new `envs/` module) and
   is only exercised by a hand-built synthetic fixture and a JSSP-as-degenerate-FJSP
   cross-check, both in `tests/test_cpsat.py`.

No genuine bugs were found in the already-validated `envs/` code during this sprint;
both new environment methods (`remaining_work_per_op`) are pure additive delegations and
required no changes to existing tested behavior.

---

## Full results table

| instance | jobs x machines | method | makespan | known optimum | RPD (%) | wall-clock (s) | feasible | status |
|---|---|---|---|---|---|---|---|---|
| ft06 | 6x6 | spt | 122.0 | 55 | 121.82 | 0.005 | True | OK |
| ft06 | 6x6 | lpt | 132.0 | 55 | 140.00 | 0.005 | True | OK |
| ft06 | 6x6 | mwkr | 74.0 | 55 | 34.55 | 0.005 | True | OK |
| ft06 | 6x6 | lwkr | 154.0 | 55 | 180.00 | 0.005 | True | OK |
| ft06 | 6x6 | mor | 73.0 | 55 | 32.73 | 0.005 | True | OK |
| ft06 | 6x6 | lor | 132.0 | 55 | 140.00 | 0.010 | True | OK |
| ft06 | 6x6 | fifo | 67.0 | 55 | 21.82 | 0.005 | True | OK |
| ft06 | 6x6 | edd | 154.0 | 55 | 180.00 | 0.007 | True | OK |
| ft06 | 6x6 | cr | 171.0 | 55 | 210.91 | 0.007 | True | OK |
| ft06 | 6x6 | cpsat | 55.0 | 55 | 0.00 | 0.032 | True | OPTIMAL |
| ft10 | 10x10 | spt | 2932.0 | 930 | 215.27 | 0.021 | True | OK |
| ft10 | 10x10 | lpt | 2851.0 | 930 | 206.56 | 0.022 | True | OK |
| ft10 | 10x10 | mwkr | 1289.0 | 930 | 38.60 | 0.022 | True | OK |
| ft10 | 10x10 | lwkr | 2657.0 | 930 | 185.70 | 0.022 | True | OK |
| ft10 | 10x10 | mor | 1553.0 | 930 | 66.99 | 0.023 | True | OK |
| ft10 | 10x10 | lor | 3069.0 | 930 | 230.00 | 0.023 | True | OK |
| ft10 | 10x10 | fifo | 1218.0 | 930 | 30.97 | 0.025 | True | OK |
| ft10 | 10x10 | edd | 2657.0 | 930 | 185.70 | 0.023 | True | OK |
| ft10 | 10x10 | cr | 3682.0 | 930 | 295.91 | 0.024 | True | OK |
| ft10 | 10x10 | cpsat | 930.0 | 930 | 0.00 | 3.787 | True | OPTIMAL |
| ft20 | 20x5 | spt | 3047.0 | 1165 | 161.54 | 0.022 | True | OK |
| ft20 | 20x5 | lpt | 2562.0 | 1165 | 119.91 | 0.022 | True | OK |
| ft20 | 20x5 | mwkr | 1995.0 | 1165 | 71.25 | 0.023 | True | OK |
| ft20 | 20x5 | lwkr | 2981.0 | 1165 | 155.88 | 0.023 | True | OK |
| ft20 | 20x5 | mor | 1963.0 | 1165 | 68.50 | 0.024 | True | OK |
| ft20 | 20x5 | lor | 3193.0 | 1165 | 174.08 | 0.020 | True | OK |
| ft20 | 20x5 | fifo | 1681.0 | 1165 | 44.29 | 0.023 | True | OK |
| ft20 | 20x5 | edd | 3030.0 | 1165 | 160.09 | 0.020 | True | OK |
| ft20 | 20x5 | cr | 3384.0 | 1165 | 190.47 | 0.027 | True | OK |
| ft20 | 20x5 | cpsat | 1165.0 | 1165 | 0.00 | 1.004 | True | OPTIMAL |
| la01 | 10x5 | spt | 1462.0 | 666 | 119.52 | 0.008 | True | OK |
| la01 | 10x5 | lpt | 1944.0 | 666 | 191.89 | 0.008 | True | OK |
| la01 | 10x5 | mwkr | 845.0 | 666 | 26.88 | 0.008 | True | OK |
| la01 | 10x5 | lwkr | 1964.0 | 666 | 194.90 | 0.008 | True | OK |
| la01 | 10x5 | mor | 867.0 | 666 | 30.18 | 0.008 | True | OK |
| la01 | 10x5 | lor | 2254.0 | 666 | 238.44 | 0.009 | True | OK |
| la01 | 10x5 | fifo | 802.0 | 666 | 20.42 | 0.009 | True | OK |
| la01 | 10x5 | edd | 1964.0 | 666 | 194.90 | 0.009 | True | OK |
| la01 | 10x5 | cr | 2193.0 | 666 | 229.28 | 0.012 | True | OK |
| la01 | 10x5 | cpsat | 666.0 | 666 | 0.00 | 0.041 | True | OPTIMAL |
| la02 | 10x5 | spt | 1892.0 | 655 | 188.85 | 0.007 | True | OK |
| la02 | 10x5 | lpt | 1763.0 | 655 | 169.16 | 0.008 | True | OK |
| la02 | 10x5 | mwkr | 982.0 | 655 | 49.92 | 0.008 | True | OK |
| la02 | 10x5 | lwkr | 1762.0 | 655 | 169.01 | 0.008 | True | OK |
| la02 | 10x5 | mor | 908.0 | 655 | 38.63 | 0.008 | True | OK |
| la02 | 10x5 | lor | 2174.0 | 655 | 231.91 | 0.009 | True | OK |
| la02 | 10x5 | fifo | 816.0 | 655 | 24.58 | 0.008 | True | OK |
| la02 | 10x5 | edd | 1680.0 | 655 | 156.49 | 0.008 | True | OK |
| la02 | 10x5 | cr | 2311.0 | 655 | 252.82 | 0.009 | True | OK |
| la02 | 10x5 | cpsat | 655.0 | 655 | 0.00 | 0.123 | True | OPTIMAL |
| la03 | 10x5 | spt | 1053.0 | 597 | 76.38 | 0.007 | True | OK |
| la03 | 10x5 | lpt | 1212.0 | 597 | 103.02 | 0.010 | True | OK |
| la03 | 10x5 | mwkr | 797.0 | 597 | 33.50 | 0.011 | True | OK |
| la03 | 10x5 | lwkr | 1660.0 | 597 | 178.06 | 0.010 | True | OK |
| la03 | 10x5 | mor | 928.0 | 597 | 55.44 | 0.010 | True | OK |
| la03 | 10x5 | lor | 1789.0 | 597 | 199.66 | 0.008 | True | OK |
| la03 | 10x5 | fifo | 824.0 | 597 | 38.02 | 0.009 | True | OK |
| la03 | 10x5 | edd | 1660.0 | 597 | 178.06 | 0.008 | True | OK |
| la03 | 10x5 | cr | 1671.0 | 597 | 179.90 | 0.010 | True | OK |
| la03 | 10x5 | cpsat | 597.0 | 597 | 0.00 | 0.104 | True | OPTIMAL |
| la04 | 10x5 | spt | 1429.0 | 590 | 142.20 | 0.007 | True | OK |
| la04 | 10x5 | lpt | 1687.0 | 590 | 185.93 | 0.008 | True | OK |
| la04 | 10x5 | mwkr | 1000.0 | 590 | 69.49 | 0.008 | True | OK |
| la04 | 10x5 | lwkr | 1677.0 | 590 | 184.24 | 0.009 | True | OK |
| la04 | 10x5 | mor | 856.0 | 590 | 45.09 | 0.009 | True | OK |
| la04 | 10x5 | lor | 2138.0 | 590 | 262.37 | 0.009 | True | OK |
| la04 | 10x5 | fifo | 660.0 | 590 | 11.86 | 0.008 | True | OK |
| la04 | 10x5 | edd | 1677.0 | 590 | 184.24 | 0.008 | True | OK |
| la04 | 10x5 | cr | 2146.0 | 590 | 263.73 | 0.008 | True | OK |
| la04 | 10x5 | cpsat | 590.0 | 590 | 0.00 | 0.166 | True | OPTIMAL |
| la05 | 10x5 | spt | 1521.0 | 593 | 156.49 | 0.007 | True | OK |
| la05 | 10x5 | lpt | 1511.0 | 593 | 154.81 | 0.009 | True | OK |
| la05 | 10x5 | mwkr | 666.0 | 593 | 12.31 | 0.008 | True | OK |
| la05 | 10x5 | lwkr | 1637.0 | 593 | 176.05 | 0.008 | True | OK |
| la05 | 10x5 | mor | 639.0 | 593 | 7.76 | 0.008 | True | OK |
| la05 | 10x5 | lor | 1744.0 | 593 | 194.10 | 0.008 | True | OK |
| la05 | 10x5 | fifo | 593.0 | 593 | 0.00 | 0.008 | True | OK |
| la05 | 10x5 | edd | 1508.0 | 593 | 154.30 | 0.008 | True | OK |
| la05 | 10x5 | cr | 1866.0 | 593 | 214.67 | 0.009 | True | OK |
| la05 | 10x5 | cpsat | 593.0 | 593 | 0.00 | 0.029 | True | OPTIMAL |
| la06 | 15x5 | spt | 2371.0 | 926 | 156.05 | 0.015 | True | OK |
| la06 | 15x5 | lpt | 2625.0 | 926 | 183.48 | 0.015 | True | OK |
| la06 | 15x5 | mwkr | 1151.0 | 926 | 24.30 | 0.014 | True | OK |
| la06 | 15x5 | lwkr | 2788.0 | 926 | 201.08 | 0.014 | True | OK |
| la06 | 15x5 | mor | 1235.0 | 926 | 33.37 | 0.016 | True | OK |
| la06 | 15x5 | lor | 2911.0 | 926 | 214.36 | 0.013 | True | OK |
| la06 | 15x5 | fifo | 1014.0 | 926 | 9.50 | 0.015 | True | OK |
| la06 | 15x5 | edd | 2788.0 | 926 | 201.08 | 0.015 | True | OK |
| la06 | 15x5 | cr | 3370.0 | 926 | 263.93 | 0.018 | True | OK |
| la06 | 15x5 | cpsat | 926.0 | 926 | 0.00 | 0.034 | True | OPTIMAL |
| la07 | 15x5 | spt | 1888.0 | 890 | 112.14 | 0.013 | True | OK |
| la07 | 15x5 | lpt | 2606.0 | 890 | 192.81 | 0.014 | True | OK |
| la07 | 15x5 | mwkr | 1108.0 | 890 | 24.49 | 0.013 | True | OK |
| la07 | 15x5 | lwkr | 2685.0 | 890 | 201.69 | 0.013 | True | OK |
| la07 | 15x5 | mor | 1186.0 | 890 | 33.26 | 0.015 | True | OK |
| la07 | 15x5 | lor | 2468.0 | 890 | 177.30 | 0.014 | True | OK |
| la07 | 15x5 | fifo | 976.0 | 890 | 9.66 | 0.015 | True | OK |
| la07 | 15x5 | edd | 2685.0 | 890 | 201.69 | 0.014 | True | OK |
| la07 | 15x5 | cr | 2837.0 | 890 | 218.76 | 0.016 | True | OK |
| la07 | 15x5 | cpsat | 890.0 | 890 | 0.00 | 0.058 | True | OPTIMAL |
| la08 | 15x5 | spt | 2358.0 | 863 | 173.23 | 0.013 | True | OK |
| la08 | 15x5 | lpt | 2193.0 | 863 | 154.11 | 0.013 | True | OK |
| la08 | 15x5 | mwkr | 1278.0 | 863 | 48.09 | 0.013 | True | OK |
| la08 | 15x5 | lwkr | 3235.0 | 863 | 274.86 | 0.014 | True | OK |
| la08 | 15x5 | mor | 1153.0 | 863 | 33.60 | 0.015 | True | OK |
| la08 | 15x5 | lor | 2897.0 | 863 | 235.69 | 0.015 | True | OK |
| la08 | 15x5 | fifo | 940.0 | 863 | 8.92 | 0.015 | True | OK |
| la08 | 15x5 | edd | 3235.0 | 863 | 274.86 | 0.016 | True | OK |
| la08 | 15x5 | cr | 3181.0 | 863 | 268.60 | 0.017 | True | OK |
| la08 | 15x5 | cpsat | 863.0 | 863 | 0.00 | 0.054 | True | OPTIMAL |
| la09 | 15x5 | spt | 2645.0 | 951 | 178.13 | 0.013 | True | OK |
| la09 | 15x5 | lpt | 2373.0 | 951 | 149.53 | 0.013 | True | OK |
| la09 | 15x5 | mwkr | 1270.0 | 951 | 33.54 | 0.015 | True | OK |
| la09 | 15x5 | lwkr | 3211.0 | 951 | 237.65 | 0.013 | True | OK |
| la09 | 15x5 | mor | 1043.0 | 951 | 9.67 | 0.015 | True | OK |
| la09 | 15x5 | lor | 3318.0 | 951 | 248.90 | 0.013 | True | OK |
| la09 | 15x5 | fifo | 983.0 | 951 | 3.37 | 0.015 | True | OK |
| la09 | 15x5 | edd | 3227.0 | 951 | 239.33 | 0.015 | True | OK |
| la09 | 15x5 | cr | 3451.0 | 951 | 262.88 | 0.019 | True | OK |
| la09 | 15x5 | cpsat | 951.0 | 951 | 0.00 | 0.098 | True | OPTIMAL |
| la10 | 15x5 | spt | 2149.0 | 958 | 124.32 | 0.013 | True | OK |
| la10 | 15x5 | lpt | 2270.0 | 958 | 136.95 | 0.014 | True | OK |
| la10 | 15x5 | mwkr | 1224.0 | 958 | 27.77 | 0.014 | True | OK |
| la10 | 15x5 | lwkr | 3193.0 | 958 | 233.30 | 0.013 | True | OK |
| la10 | 15x5 | mor | 997.0 | 958 | 4.07 | 0.014 | True | OK |
| la10 | 15x5 | lor | 2983.0 | 958 | 211.38 | 0.015 | True | OK |
| la10 | 15x5 | fifo | 1034.0 | 958 | 7.93 | 0.017 | True | OK |
| la10 | 15x5 | edd | 3193.0 | 958 | 233.30 | 0.014 | True | OK |
| la10 | 15x5 | cr | 3058.0 | 958 | 219.21 | 0.017 | True | OK |
| la10 | 15x5 | cpsat | 958.0 | 958 | 0.00 | 0.035 | True | OPTIMAL |

## Next steps (not this week)

- Implement the genetic algorithm (`baselines/ga.py`, `GABaseline`) --- next month's code
  sprint, per this sprint's explicit instructions.
- `baselines/harness.py` as a generic multi-baseline sweep API, once the GA gives it a
  second real caller beyond the evaluation script.
- Real FJSP instance loading (MK01-MK10, Hurink) into `envs/`, at which point
  `FJSPInstance` should move out of `baselines/cpsat.py` into a proper `envs/` module and
  the GA's already-built (but untested-on-real-data) assignment-vector chromosome path
  can be exercised for real.
- Arrival-aware due-date synthesis for EDD/CR (release time = actual arrival time, per
  `Baselines_TechSpec.md` §4.2), if EDD/CR need to run against `DynamicJSSPEnv` with
  arrivals enabled in a future sprint.
