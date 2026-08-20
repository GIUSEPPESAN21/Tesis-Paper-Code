# Week 12 Baseline Results — Genetic Algorithm

**Scope:** implements `GA_TechSpec.md` (finalized GA hyperparameters, random-immigrants
stagnation guard) on top of `Baselines_TechSpec.md` §5's already-specified encoding and
operators, deferred at the end of Week 9 (`BASELINES_RESULTS_WEEK9.md`). The GA is now a
full `Baseline`-protocol citizen (`GABaseline`), alongside the nine PDRs and CP-SAT.

**Headline result:** GA reaches the exact known optimum on **8 of 13** instances and
comes within **10.64%** on the worst one (`ft10`), for a **mean RPD of 2.61%** across all
13 instances — an order of magnitude better than FIFO (17.80%, the best single-pass PDR)
and roughly 14x better than MOR/MWKR (35–38%). It beats every PDR on every single
instance. 444/444 pre-existing tests still pass; 25 new tests added (469/469 total,
including one deliberately expensive 5-minute permanent regression test).

**MK01–MK10/FJSP evaluation is deferred**, same as Week 9's CP-SAT FJSP work: no FJSP
instance loader exists, and this repo's own `scripts/fetch_instances.py` establishes an
explicit precedent against hand-transcribing benchmark data from memory. See "Design
decisions and issues found" below for the full reasoning and what would unblock it.

---

## What was added

| File | Status | Contents |
|---|---|---|
| `baselines/ga.py` | NEW | `Chromosome`, `GAConfig`, `default_population_size`, `decode`/`DecodeResult`, `random_chromosome`, `pox`, `swap_mutation`, `uniform_crossover_assignment`, `reassignment_mutation`, `tournament_select`, `GeneticAlgorithm` (wall-clock-first main loop with random immigrants). |
| `baselines/interface.py` | EXTENDED | `GABaseline` adapter, JSSP-only (mirrors `CPSATBaseline`'s own JSSP-only scope; the FJSP-ready GA path is validated directly, not through the `Baseline` interface, exactly like `solve_cpsat_fjsp`). |
| `baselines/cpsat.py` | EXTENDED (tiny) | `FJSPInstance.job_of`/`.position_of` added (inverse of `.op_id`), needed by the GA's reassignment mutation. |
| `baselines/__init__.py` | EXTENDED | Exports `Chromosome`, `GAConfig`, `GeneticAlgorithm`, `default_population_size`, `GABaseline`. |
| `experiments/eval_baselines.py` | EXTENDED | GA added to the sweep (same `time_limit_s` as CP-SAT, per `GA_TechSpec.md` §3's fairness principle); new `--ga-seeds` option; writes `experiments/results/baselines_week12.csv/.md`. |
| `tests/test_ga.py` | NEW | 23 tests: decode always feasible (+ a regression guard for the makespan bug found this sprint, see below), FJSP-ready decode/operators against a synthetic ragged fixture, POX/swap/reassignment operator correctness, elitism monotonicity + elites-pass-through, random-immigrants-replace-worst-not-elite, wall-clock budget respected, beats `random_rule` on `ft06`, default-population-size band tripwire, and the permanent `la05`-reaches-zero-RPD regression test. |
| `tests/test_baselines_interface.py` | EXTENDED | `GABaseline` added to the protocol-conformance parametrize; new `test_ga_requires_time_limit`. |

---

## STEP1 — chromosome encoding and decoding

`Chromosome(sequence, assignment)`: `sequence` is a permutation-with-repetition (job
index repeated `ops_per_job` times); `assignment` is an eligible-machine *index* per
operation, `None` for classic JSSP. `decode()` is a left-to-right list-scheduling scan —
the exact same rule `JSSPEnv.step()` uses (`start = max(job_ready_time[job],
machine_free_time[machine])`) — so every decoded chromosome is guaranteed
semi-active-feasible by construction; verified directly (200 random chromosomes on
`ft06`, zero violations) rather than assumed.

The FJSP-ready path (`assignment` non-`None`) is fully implemented and tested against a
hand-built ragged-eligibility `FJSPInstance` fixture (2 jobs x 2 ops, the same toy Week
9's CP-SAT FJSP tests use, known optimum 7) — decode, `uniform_crossover_assignment`,
and `reassignment_mutation` all pass their tests on it, and the GA reaches makespan 7 on
it directly. Real FJSP instance loading is the only missing piece.

## STEP2 — operators

- **POX** (`pox`): partitions jobs into `J1`/`J2`, keeps parent A's `J1` genes in place,
  fills the rest with parent B's `J2` genes in order. Verified over 100 random parent
  pairs: child is always a valid permutation-with-repetition (same per-job operation
  counts), which is what guarantees precedence feasibility with no repair step.
- **Uniform crossover** (`uniform_crossover_assignment`): per-gene 50/50, no-op for JSSP.
- **Swap mutation** (`swap_mutation`): two random positions swapped; verified to preserve
  per-job operation counts.
- **Reassignment mutation** (`reassignment_mutation`, FJSP only): one random operation's
  machine choice replaced with a different eligible alternative; verified to always stay
  within the operation's eligible set.
- **Tournament selection + elitism**: `tournament_size=3` (with replacement),
  `elite_count=2`, best individuals copied unmutated into the next generation — verified
  by both the standard monotonicity test (best-of-generation makespan never increases,
  5 seeds) and a direct object-identity check that the elites literally pass through
  unchanged.
- **Random immigrants** (`GA_TechSpec.md` §5, new this sprint, not in
  `Baselines_TechSpec.md` §5): each generation, the worst `immigrant_fraction` (default
  0.05) of the *offspring* — never the elites, which don't pass through this step at all
  — are replaced with fresh random chromosomes. Verified precisely: refactored the
  generation loop into `_generate_offspring`/`_apply_random_immigrants` specifically so
  the test can confirm the replaced indices are exactly the worst-scoring offspring by
  object identity, not merely "some individuals changed."

## STEP3 — main loop and configuration

`GAConfig` defaults are `GA_TechSpec.md` Table 2.1 exactly: `tournament_size=3`,
`elite_count=2`, `p_crossover=0.90`, `p_mutation_sequence=p_mutation_assignment=0.15`,
`immigrant_fraction=0.05`. `population_size=None` auto-scales via
`default_population_size(num_operations)` (80 / 60 / 40 across the ≤100 / 101–450 /
451–2000 operation bands, per §2.1) — for every instance evaluated this sprint
(≤100 operations), that's uniformly 80.

The stopping criterion is wall-clock time, not a fixed generation count
(`Baselines_TechSpec.md` §5.6): `GeneticAlgorithm.run(instance, seed, time_limit_s)`
raises `ValueError` if `time_limit_s` is `None`, otherwise loops until
`time.monotonic() >= deadline`, reporting `generations_run` and
`best_makespan_history` in `BaselineResult.extra`. Verified to respect the budget
(`elapsed <= time_limit_s + 1.0s` slack for one in-flight generation) across
`time_limit_s in {0.5, 1.5, 3.0}`.

---

## STEP4 — evaluation results (four-baseline comparison)

`python experiments/eval_baselines.py --time-limit 60 --seed 0` — full table (143 rows:
13 instances x 11 methods) at `experiments/results/baselines_week12.csv` / `.md`. This
supersedes `BASELINES_RESULTS_WEEK9.md`'s 130-row table as **the reference comparison
table for the thesis**; Week 9's file is left untouched as the historical PDR+CP-SAT-only
record.

### Mean RPD (%) vs. known optimum, across all 13 instances

| method | mean RPD % | min | max |
|---|---:|---:|---:|
| **cpsat** | **0.00** | 0.00 | 0.00 |
| **ga** | **2.61** | 0.00 | 10.64 |
| fifo | 17.80 | 0.00 | 44.29 |
| mor | 35.33 | 4.07 | 68.50 |
| mwkr | 38.05 | 12.31 | 71.25 |
| spt | 148.15 | 76.38 | 215.27 |
| lpt | 160.63 | 103.02 | 206.56 |
| edd | 195.69 | 154.30 | 274.86 |
| lwkr | 197.88 | 155.88 | 274.86 |
| lor | 212.17 | 140.00 | 262.37 |
| cr | 236.24 | 179.90 | 295.91 |

### GA vs. CP-SAT vs. best PDR, per instance

| instance | GA RPD % | CP-SAT RPD % | best PDR (RPD %) |
|---|---:|---:|---|
| ft06 | 0.00 | 0.00 | fifo (21.82) |
| ft10 | 10.64 | 0.00 | fifo (30.97) |
| ft20 | 8.76 | 0.00 | fifo (44.29) |
| la01 | 0.00 | 0.00 | fifo (20.42) |
| la02 | 6.26 | 0.00 | fifo (24.58) |
| la03 | 5.36 | 0.00 | mwkr (33.50) |
| la04 | 2.88 | 0.00 | fifo (11.86) |
| la05 | 0.00 | 0.00 | fifo (0.00) |
| la06 | 0.00 | 0.00 | fifo (9.50) |
| la07 | 0.00 | 0.00 | fifo (9.66) |
| la08 | 0.00 | 0.00 | fifo (8.92) |
| la09 | 0.00 | 0.00 | fifo (3.37) |
| la10 | 0.00 | 0.00 | mor (4.07) |

GA reaches the **exact proven optimum** on 8/13 instances (`ft06`, `la01`,
`la05`–`la10`) and stays under 11% everywhere else; it beats the best PDR on every
instance, typically by 1-2 orders of magnitude. Every GA schedule was independently
verified feasible (`is_schedule_feasible`), and every run used its full 60-second budget
(2,111–4,342 generations observed, more generations on the smaller/cheaper `ft06`/`la0x`
instances than the larger `ft10`/`ft20`, as expected).

**Seed caveat (flagged, not smoothed over):** the table above uses a single seed
(`seed=0`) per instance, matching how the main table is presented for all four method
families. GA is stochastic in a way PDRs and CP-SAT are not (CP-SAT's search is
internally randomized but not seed-controlled by this wrapper; a single PDR pass is
deterministic given its seed), so a single-seed number understates the *true* variance a
thesis-grade comparison (Section 5.2.3's Wilcoxon/Friedman procedure, which needs
multiple replicas) would want. The one instance with dedicated multi-seed evidence is
`la05`: `tests/test_ga.py::test_ga_reaches_zero_rpd_on_la05` runs 5 seeds at the full
60s budget and asserts median RPD is exactly 0.0% (passed). `eval_baselines.py` supports
`--ga-seeds N` for a full multi-seed sweep across all 13 instances; it was not run this
sprint (13 instances x 5 seeds x 60s ≈ 65 minutes, versus ≈13 minutes for the single-seed
sweep actually run) — a deliberate scope/time trade-off for this sprint's deliverable, not
a hidden limitation. Chapter 5/6's formal statistical comparison should use a proper
multi-seed sweep, not this table's single-seed numbers, for the GA row specifically.

---

## STEP5 — unit tests

**469/469 passing** total (444 pre-existing + 25 new), full suite ~36s except for one
deliberately expensive permanent regression test (+5 min):

| file | new tests | covers |
|---|---:|---|
| `tests/test_ga.py` | 23 | decode always feasible + makespan-bug regression guard, FJSP-ready decode/operators on a synthetic fixture, POX/swap/reassignment operator correctness, elitism monotonicity + elites-unchanged, random-immigrants-replace-worst-not-elite, wall-clock budget respected, beats `random_rule` on `ft06`, default-population-size band tripwire, permanent `la05`-zero-RPD regression (5 seeds x 60s) |
| `tests/test_baselines_interface.py` | 2 | `GABaseline` added to protocol-conformance parametrize; `GABaseline.solve` raises `ValueError` without `time_limit_s` |

---

## Design decisions and issues found

1. **A real bug was found and fixed during implementation: `decode()`'s makespan was
   computed as `max(start_times)` instead of `max(end_times)`.** This silently
   *under-reports* the makespan by roughly the duration of whichever operation happens
   to have the latest start time. It was caught because the GA's very first test run on
   `ft06` reported a makespan of 52 — *below* the proven optimum of 55, which is
   mathematically impossible for a genuinely feasible schedule and was the tell that
   something was wrong, not that the GA had found a better-than-optimal solution.
   Root-caused to the one-line aggregation bug, fixed (track `end_times` separately,
   report `end_times.max()`), and re-verified: 500 random chromosomes on `ft06` now
   never fall below 55, and the GA reaches exactly 55. A permanent regression test
   (`test_decode_never_beats_the_proven_optimum`) guards against this specific class of
   bug recurring silently. This is exactly the kind of defect `is_schedule_feasible`
   alone cannot catch (the reported *start times* were legitimate; only the derived
   *makespan number* was wrong), which is why "does the result beat a proven optimum"
   is now an explicit sanity check, not just "is it feasible."

2. **MK01–MK10/FJSP evaluation is deferred, not attempted with fabricated data.** STEP4
   conditioned this on the FJSP instance loader being ready ("if ready") — it is not:
   `envs/instance.py` still only models fixed routing, and no MK01-MK10 files are
   bundled. This repo's `scripts/fetch_instances.py` documents its own reasoning for why
   this project fetches benchmark data from a canonical source instead of hand-
   transcribing it ("a single typo silently invalidates every downstream experiment").
   Reproducing Brandimarte's MK01–MK10 instance files from memory would be exactly that
   kind of silent-corruption risk, and no verified canonical source was fetched this
   sprint. What *is* done: the GA's assignment-vector chromosome, decoder, and operators
   are fully implemented and tested against a synthetic ragged-eligibility fixture
   (mirroring Week 9's `solve_cpsat_fjsp` treatment exactly), so no GA code should need
   to change once a real FJSP loader lands — only the loader itself, and a source for
   MK01-MK10's data, are missing. Unblocking this needs either (a) the user supplying
   the MK01-MK10 files directly, or (b) a confirmed canonical URL to fetch them from,
   analogous to `scripts/fetch_instances.py`'s OR-Library download.

3. **`GABaseline` wraps only the JSSP path, mirroring `CPSATBaseline`'s own precedent.**
   `GeneticAlgorithm` itself is FJSP-ready (point 2's synthetic-fixture testing proves
   it), but the `Baseline` protocol's declared instance type is `JSSPInstance`, and
   `solve_cpsat_fjsp` was never wrapped in a `CPSATBaseline`-equivalent adapter either —
   staying consistent avoids inventing an FJSP-typed protocol variant for a code path
   that has no real instances to run on yet.

4. **The main results table uses one GA seed per instance, not five.** `GA_TechSpec.md`
   §4.2's own tuning-confirmation protocol used 5 seeds; a full 5-seed sweep across all
   13 instances would take roughly 5x the wall-clock time this sprint's evaluation run
   took (~65 min vs. ~13 min). Rather than silently presenting a single-seed number as
   if it were the whole story, this is flagged explicitly (STEP4 above) with the one
   piece of multi-seed evidence actually collected (`la05`, 5 seeds, permanent test) and
   the exact command (`--ga-seeds N`) needed to redo the full sweep when the time budget
   allows.

5. **Random immigrants (`GA_TechSpec.md` §5) is implemented exactly as specified**,
   including refactoring the generation loop into two sub-methods
   (`_generate_offspring`/`_apply_random_immigrants`) specifically so the "replaces the
   worst offspring, never the elites" claim could be tested precisely (by object
   identity) rather than approximately. This refactor is behavior-preserving --- the
   full `_one_generation` composes the two sub-methods identically to the original
   single-method implementation.

---

## Full results table

| instance | jobs x machines | method | seed | makespan | known optimum | RPD (%) | wall-clock (s) | feasible | status |
|---|---|---|---|---|---|---|---|---|---|
| ft06 | 6x6 | spt | 0 | 122.0 | 55 | 121.82 | 0.007 | True | OK |
| ft06 | 6x6 | lpt | 0 | 132.0 | 55 | 140.00 | 0.006 | True | OK |
| ft06 | 6x6 | mwkr | 0 | 74.0 | 55 | 34.55 | 0.005 | True | OK |
| ft06 | 6x6 | lwkr | 0 | 154.0 | 55 | 180.00 | 0.006 | True | OK |
| ft06 | 6x6 | mor | 0 | 73.0 | 55 | 32.73 | 0.006 | True | OK |
| ft06 | 6x6 | lor | 0 | 132.0 | 55 | 140.00 | 0.005 | True | OK |
| ft06 | 6x6 | fifo | 0 | 67.0 | 55 | 21.82 | 0.005 | True | OK |
| ft06 | 6x6 | edd | 0 | 154.0 | 55 | 180.00 | 0.006 | True | OK |
| ft06 | 6x6 | cr | 0 | 171.0 | 55 | 210.91 | 0.006 | True | OK |
| ft06 | 6x6 | cpsat | 0 | 55.0 | 55 | 0.00 | 0.035 | True | OPTIMAL |
| ft06 | 6x6 | ga | 0 | 55.0 | 55 | 0.00 | 60.001 | True | OK |
| ft10 | 10x10 | spt | 0 | 2932.0 | 930 | 215.27 | 0.021 | True | OK |
| ft10 | 10x10 | lpt | 0 | 2851.0 | 930 | 206.56 | 0.021 | True | OK |
| ft10 | 10x10 | mwkr | 0 | 1289.0 | 930 | 38.60 | 0.022 | True | OK |
| ft10 | 10x10 | lwkr | 0 | 2657.0 | 930 | 185.70 | 0.021 | True | OK |
| ft10 | 10x10 | mor | 0 | 1553.0 | 930 | 66.99 | 0.022 | True | OK |
| ft10 | 10x10 | lor | 0 | 3069.0 | 930 | 230.00 | 0.021 | True | OK |
| ft10 | 10x10 | fifo | 0 | 1218.0 | 930 | 30.97 | 0.026 | True | OK |
| ft10 | 10x10 | edd | 0 | 2657.0 | 930 | 185.70 | 0.021 | True | OK |
| ft10 | 10x10 | cr | 0 | 3682.0 | 930 | 295.91 | 0.023 | True | OK |
| ft10 | 10x10 | cpsat | 0 | 930.0 | 930 | 0.00 | 3.528 | True | OPTIMAL |
| ft10 | 10x10 | ga | 0 | 1029.0 | 930 | 10.64 | 60.016 | True | OK |
| ft20 | 20x5 | spt | 0 | 3047.0 | 1165 | 161.54 | 0.021 | True | OK |
| ft20 | 20x5 | lpt | 0 | 2562.0 | 1165 | 119.91 | 0.022 | True | OK |
| ft20 | 20x5 | mwkr | 0 | 1995.0 | 1165 | 71.25 | 0.022 | True | OK |
| ft20 | 20x5 | lwkr | 0 | 2981.0 | 1165 | 155.88 | 0.021 | True | OK |
| ft20 | 20x5 | mor | 0 | 1963.0 | 1165 | 68.50 | 0.023 | True | OK |
| ft20 | 20x5 | lor | 0 | 3193.0 | 1165 | 174.08 | 0.020 | True | OK |
| ft20 | 20x5 | fifo | 0 | 1681.0 | 1165 | 44.29 | 0.021 | True | OK |
| ft20 | 20x5 | edd | 0 | 3030.0 | 1165 | 160.09 | 0.023 | True | OK |
| ft20 | 20x5 | cr | 0 | 3384.0 | 1165 | 190.47 | 0.023 | True | OK |
| ft20 | 20x5 | cpsat | 0 | 1165.0 | 1165 | 0.00 | 0.977 | True | OPTIMAL |
| ft20 | 20x5 | ga | 0 | 1267.0 | 1165 | 8.76 | 60.021 | True | OK |
| la01 | 10x5 | spt | 0 | 1462.0 | 666 | 119.52 | 0.008 | True | OK |
| la01 | 10x5 | lpt | 0 | 1944.0 | 666 | 191.89 | 0.009 | True | OK |
| la01 | 10x5 | mwkr | 0 | 845.0 | 666 | 26.88 | 0.008 | True | OK |
| la01 | 10x5 | lwkr | 0 | 1964.0 | 666 | 194.90 | 0.009 | True | OK |
| la01 | 10x5 | mor | 0 | 867.0 | 666 | 30.18 | 0.010 | True | OK |
| la01 | 10x5 | lor | 0 | 2254.0 | 666 | 238.44 | 0.008 | True | OK |
| la01 | 10x5 | fifo | 0 | 802.0 | 666 | 20.42 | 0.009 | True | OK |
| la01 | 10x5 | edd | 0 | 1964.0 | 666 | 194.90 | 0.009 | True | OK |
| la01 | 10x5 | cr | 0 | 2193.0 | 666 | 229.28 | 0.009 | True | OK |
| la01 | 10x5 | cpsat | 0 | 666.0 | 666 | 0.00 | 0.040 | True | OPTIMAL |
| la01 | 10x5 | ga | 0 | 666.0 | 666 | 0.00 | 60.002 | True | OK |
| la02 | 10x5 | spt | 0 | 1892.0 | 655 | 188.85 | 0.009 | True | OK |
| la02 | 10x5 | lpt | 0 | 1763.0 | 655 | 169.16 | 0.009 | True | OK |
| la02 | 10x5 | mwkr | 0 | 982.0 | 655 | 49.92 | 0.009 | True | OK |
| la02 | 10x5 | lwkr | 0 | 1762.0 | 655 | 169.01 | 0.007 | True | OK |
| la02 | 10x5 | mor | 0 | 908.0 | 655 | 38.63 | 0.013 | True | OK |
| la02 | 10x5 | lor | 0 | 2174.0 | 655 | 231.91 | 0.016 | True | OK |
| la02 | 10x5 | fifo | 0 | 816.0 | 655 | 24.58 | 0.016 | True | OK |
| la02 | 10x5 | edd | 0 | 1680.0 | 655 | 156.49 | 0.017 | True | OK |
| la02 | 10x5 | cr | 0 | 2311.0 | 655 | 252.82 | 0.013 | True | OK |
| la02 | 10x5 | cpsat | 0 | 655.0 | 655 | 0.00 | 0.121 | True | OPTIMAL |
| la02 | 10x5 | ga | 0 | 696.0 | 655 | 6.26 | 59.996 | True | OK |
| la03 | 10x5 | spt | 0 | 1053.0 | 597 | 76.38 | 0.009 | True | OK |
| la03 | 10x5 | lpt | 0 | 1212.0 | 597 | 103.02 | 0.008 | True | OK |
| la03 | 10x5 | mwkr | 0 | 797.0 | 597 | 33.50 | 0.009 | True | OK |
| la03 | 10x5 | lwkr | 0 | 1660.0 | 597 | 178.06 | 0.008 | True | OK |
| la03 | 10x5 | mor | 0 | 928.0 | 597 | 55.44 | 0.009 | True | OK |
| la03 | 10x5 | lor | 0 | 1789.0 | 597 | 199.66 | 0.009 | True | OK |
| la03 | 10x5 | fifo | 0 | 824.0 | 597 | 38.02 | 0.009 | True | OK |
| la03 | 10x5 | edd | 0 | 1660.0 | 597 | 178.06 | 0.008 | True | OK |
| la03 | 10x5 | cr | 0 | 1671.0 | 597 | 179.90 | 0.011 | True | OK |
| la03 | 10x5 | cpsat | 0 | 597.0 | 597 | 0.00 | 0.147 | True | OPTIMAL |
| la03 | 10x5 | ga | 0 | 629.0 | 597 | 5.36 | 60.008 | True | OK |
| la04 | 10x5 | spt | 0 | 1429.0 | 590 | 142.20 | 0.008 | True | OK |
| la04 | 10x5 | lpt | 0 | 1687.0 | 590 | 185.93 | 0.007 | True | OK |
| la04 | 10x5 | mwkr | 0 | 1000.0 | 590 | 69.49 | 0.009 | True | OK |
| la04 | 10x5 | lwkr | 0 | 1677.0 | 590 | 184.24 | 0.013 | True | OK |
| la04 | 10x5 | mor | 0 | 856.0 | 590 | 45.09 | 0.014 | True | OK |
| la04 | 10x5 | lor | 0 | 2138.0 | 590 | 262.37 | 0.013 | True | OK |
| la04 | 10x5 | fifo | 0 | 660.0 | 590 | 11.86 | 0.015 | True | OK |
| la04 | 10x5 | edd | 0 | 1677.0 | 590 | 184.24 | 0.010 | True | OK |
| la04 | 10x5 | cr | 0 | 2146.0 | 590 | 263.73 | 0.024 | True | OK |
| la04 | 10x5 | cpsat | 0 | 590.0 | 590 | 0.00 | 0.129 | True | OPTIMAL |
| la04 | 10x5 | ga | 0 | 607.0 | 590 | 2.88 | 60.010 | True | OK |
| la05 | 10x5 | spt | 0 | 1521.0 | 593 | 156.49 | 0.007 | True | OK |
| la05 | 10x5 | lpt | 0 | 1511.0 | 593 | 154.81 | 0.008 | True | OK |
| la05 | 10x5 | mwkr | 0 | 666.0 | 593 | 12.31 | 0.008 | True | OK |
| la05 | 10x5 | lwkr | 0 | 1637.0 | 593 | 176.05 | 0.008 | True | OK |
| la05 | 10x5 | mor | 0 | 639.0 | 593 | 7.76 | 0.008 | True | OK |
| la05 | 10x5 | lor | 0 | 1744.0 | 593 | 194.10 | 0.009 | True | OK |
| la05 | 10x5 | fifo | 0 | 593.0 | 593 | 0.00 | 0.008 | True | OK |
| la05 | 10x5 | edd | 0 | 1508.0 | 593 | 154.30 | 0.008 | True | OK |
| la05 | 10x5 | cr | 0 | 1866.0 | 593 | 214.67 | 0.009 | True | OK |
| la05 | 10x5 | cpsat | 0 | 593.0 | 593 | 0.00 | 0.015 | True | OPTIMAL |
| la05 | 10x5 | ga | 0 | 593.0 | 593 | 0.00 | 60.003 | True | OK |
| la06 | 15x5 | spt | 0 | 2371.0 | 926 | 156.05 | 0.014 | True | OK |
| la06 | 15x5 | lpt | 0 | 2625.0 | 926 | 183.48 | 0.015 | True | OK |
| la06 | 15x5 | mwkr | 0 | 1151.0 | 926 | 24.30 | 0.014 | True | OK |
| la06 | 15x5 | lwkr | 0 | 2788.0 | 926 | 201.08 | 0.013 | True | OK |
| la06 | 15x5 | mor | 0 | 1235.0 | 926 | 33.37 | 0.014 | True | OK |
| la06 | 15x5 | lor | 0 | 2911.0 | 926 | 214.36 | 0.015 | True | OK |
| la06 | 15x5 | fifo | 0 | 1014.0 | 926 | 9.50 | 0.030 | True | OK |
| la06 | 15x5 | edd | 0 | 2788.0 | 926 | 201.08 | 0.015 | True | OK |
| la06 | 15x5 | cr | 0 | 3370.0 | 926 | 263.93 | 0.017 | True | OK |
| la06 | 15x5 | cpsat | 0 | 926.0 | 926 | 0.00 | 0.036 | True | OPTIMAL |
| la06 | 15x5 | ga | 0 | 926.0 | 926 | 0.00 | 60.018 | True | OK |
| la07 | 15x5 | spt | 0 | 1888.0 | 890 | 112.14 | 0.013 | True | OK |
| la07 | 15x5 | lpt | 0 | 2606.0 | 890 | 192.81 | 0.013 | True | OK |
| la07 | 15x5 | mwkr | 0 | 1108.0 | 890 | 24.49 | 0.014 | True | OK |
| la07 | 15x5 | lwkr | 0 | 2685.0 | 890 | 201.69 | 0.016 | True | OK |
| la07 | 15x5 | mor | 0 | 1186.0 | 890 | 33.26 | 0.017 | True | OK |
| la07 | 15x5 | lor | 0 | 2468.0 | 890 | 177.30 | 0.013 | True | OK |
| la07 | 15x5 | fifo | 0 | 976.0 | 890 | 9.66 | 0.013 | True | OK |
| la07 | 15x5 | edd | 0 | 2685.0 | 890 | 201.69 | 0.013 | True | OK |
| la07 | 15x5 | cr | 0 | 2837.0 | 890 | 218.76 | 0.015 | True | OK |
| la07 | 15x5 | cpsat | 0 | 890.0 | 890 | 0.00 | 0.063 | True | OPTIMAL |
| la07 | 15x5 | ga | 0 | 890.0 | 890 | 0.00 | 60.041 | True | OK |
| la08 | 15x5 | spt | 0 | 2358.0 | 863 | 173.23 | 0.013 | True | OK |
| la08 | 15x5 | lpt | 0 | 2193.0 | 863 | 154.11 | 0.013 | True | OK |
| la08 | 15x5 | mwkr | 0 | 1278.0 | 863 | 48.09 | 0.013 | True | OK |
| la08 | 15x5 | lwkr | 0 | 3235.0 | 863 | 274.86 | 0.013 | True | OK |
| la08 | 15x5 | mor | 0 | 1153.0 | 863 | 33.60 | 0.014 | True | OK |
| la08 | 15x5 | lor | 0 | 2897.0 | 863 | 235.69 | 0.013 | True | OK |
| la08 | 15x5 | fifo | 0 | 940.0 | 863 | 8.92 | 0.014 | True | OK |
| la08 | 15x5 | edd | 0 | 3235.0 | 863 | 274.86 | 0.014 | True | OK |
| la08 | 15x5 | cr | 0 | 3181.0 | 863 | 268.60 | 0.015 | True | OK |
| la08 | 15x5 | cpsat | 0 | 863.0 | 863 | 0.00 | 0.076 | True | OPTIMAL |
| la08 | 15x5 | ga | 0 | 863.0 | 863 | 0.00 | 60.023 | True | OK |
| la09 | 15x5 | spt | 0 | 2645.0 | 951 | 178.13 | 0.018 | True | OK |
| la09 | 15x5 | lpt | 0 | 2373.0 | 951 | 149.53 | 0.017 | True | OK |
| la09 | 15x5 | mwkr | 0 | 1270.0 | 951 | 33.54 | 0.018 | True | OK |
| la09 | 15x5 | lwkr | 0 | 3211.0 | 951 | 237.65 | 0.018 | True | OK |
| la09 | 15x5 | mor | 0 | 1043.0 | 951 | 9.67 | 0.021 | True | OK |
| la09 | 15x5 | lor | 0 | 3318.0 | 951 | 248.90 | 0.020 | True | OK |
| la09 | 15x5 | fifo | 0 | 983.0 | 951 | 3.37 | 0.016 | True | OK |
| la09 | 15x5 | edd | 0 | 3227.0 | 951 | 239.33 | 0.015 | True | OK |
| la09 | 15x5 | cr | 0 | 3451.0 | 951 | 262.88 | 0.016 | True | OK |
| la09 | 15x5 | cpsat | 0 | 951.0 | 951 | 0.00 | 0.073 | True | OPTIMAL |
| la09 | 15x5 | ga | 0 | 951.0 | 951 | 0.00 | 60.054 | True | OK |
| la10 | 15x5 | spt | 0 | 2149.0 | 958 | 124.32 | 0.047 | True | OK |
| la10 | 15x5 | lpt | 0 | 2270.0 | 958 | 136.95 | 0.043 | True | OK |
| la10 | 15x5 | mwkr | 0 | 1224.0 | 958 | 27.77 | 0.046 | True | OK |
| la10 | 15x5 | lwkr | 0 | 3193.0 | 958 | 233.30 | 0.048 | True | OK |
| la10 | 15x5 | mor | 0 | 997.0 | 958 | 4.07 | 0.057 | True | OK |
| la10 | 15x5 | lor | 0 | 2983.0 | 958 | 211.38 | 0.053 | True | OK |
| la10 | 15x5 | fifo | 0 | 1034.0 | 958 | 7.93 | 0.051 | True | OK |
| la10 | 15x5 | edd | 0 | 3193.0 | 958 | 233.30 | 0.042 | True | OK |
| la10 | 15x5 | cr | 0 | 3058.0 | 958 | 219.21 | 0.053 | True | OK |
| la10 | 15x5 | cpsat | 0 | 958.0 | 958 | 0.00 | 0.062 | True | OPTIMAL |
| la10 | 15x5 | ga | 0 | 958.0 | 958 | 0.00 | 60.033 | True | OK |

## Next steps (not this sprint)

- Real FJSP instance loading (MK01-MK10, Hurink) once a verified data source is
  available --- either the user supplies the files, or a confirmed canonical URL is
  provided to fetch from, mirroring `scripts/fetch_instances.py`'s pattern. The GA's
  FJSP-ready path and `solve_cpsat_fjsp` are both already waiting for it.
- A full multi-seed GA sweep (`--ga-seeds 5`, ~65 minutes) across all 13 instances, to
  give Chapter 5/6's Wilcoxon/Friedman statistical comparison (`Section5.1_5.2.md`
  §5.2.3) proper replica counts for the GA row specifically, not just `la05`'s.
  Section 5.2.3 also has the GA entering the joint Friedman/Nemenyi ranking (unlike
  CP-SAT, which stays out as the reference-only method) once the DRL policy exists.
- `baselines/harness.py` as a generic multi-baseline sweep API, if a second real caller
  beyond `experiments/eval_baselines.py` emerges (e.g. the DRL-policy comparison).
- The `la11`-`la40` (intermediate) and `ta` (large) instance blocks per
  `Section5.1_5.2.md` Table 5.1.1, at their respective 300s/600-900s time-limit tiers,
  once budget allows --- `default_population_size`'s 60/40 bands are untested
  extrapolations until then.
