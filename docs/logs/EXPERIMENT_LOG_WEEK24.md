# Experiment Log — Week 24 (2026-07-31)

**Scope:** Execution of E1–E6 per `../docs/planificacion/FinalExperimentMatrix.md`, on a single-machine Linux
dev environment (`code/.venv-linux`, Python 3.12, PyTorch 2.12.1+cu130,
16 logical CPU cores, GPU present but not used by the evaluation code path — see the
device note under E1). No significance testing performed this window, per STEP4 of the
task brief — that is explicitly deferred to next month's dedicated analysis script.

**Status legend:** ✅ complete · ⚠️ complete with a flagged anomaly · ⛔ blocked/deferred

---

## Environment / infrastructure notes (apply across all experiments)

1. **Test suite regression.** `pytest -q -m "not slow"`: 589 passed, 3 failed (all
   `tests/test_hetero_policy.py`, a device-placement bug in the *training*-only
   `build_hetero_maskable_ppo` helper — confirmed NOT present in the actual evaluation
   path `baselines.interface.TrainedRLBaseline` uses, see E0 close-out for the full
   root-cause). Does not block any result in this log.
2. **`--rl-sample` CLI flag added** to `run_evaluation.py` (previously `TrainedRLBaseline`
   always ran deterministic argmax regardless of `--seeds`, which cannot produce the
   "multiple stochastic policy rollouts" the task brief and `../../thesis/PreRegistration.md`
   call for). 1 deterministic (argmax) + 4 stochastic (`--rl-sample`, seeds 1–4) rollouts
   per instance, matching the matrix's "5/instance" RL convention. PDR/GA/CP-SAT run once
   (seed 0) per the matrix's own "single run suffices" instruction.
3. **`tests/test_fjsp_loader.py` written and passing** (43 tests, <1s incl. the CP-SAT
   padding-neutrality proof) — the precondition `../../thesis/PreRegistration.md` Amendment 1
   requires before Arm B may be used confirmatively. See finding #2 below for one
   exclusion this test surfaced.
4. **Checkpoints used:**
   - JSSP (E1, E3, E8): `experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip`
   - FJSP (E2 Arm A + Arm B): `experiments/results/curriculum_full/checkpoints/curriculum_stage24_mk05_final.zip`
     (a *different*, per-pair-head checkpoint — the JSSP per-node-head checkpoint cannot
     score FJSP's (operation, machine) action space; confirmed by a hard `ValueError` the
     first time this was tried with the wrong checkpoint, see finding #3).

---

## Findings / anomalies

### Finding #1 — RL policy device path is CPU-only regardless of GPU availability
`baselines.interface.TrainedRLBaseline._policy_for` constructs
`PerNodeMaskableActorCriticPolicy` raw (not via `MaskablePPO`), and nothing in that path
calls `.to(device)`; `nn.Module` parameters default to CPU at construction regardless of
GPU availability, and `load_state_dict` preserves the destination tensor's device. So in
the current code, **the trained policy's forward pass always runs on CPU**, whether or
not a GPU is present. This does not affect correctness (confirmed by a direct
before-committing-compute smoke test: deterministic and stochastic rollouts on `ft06`
both produced feasible, sensible makespans), but it does affect what "E8 GPU-primary vs
CPU-only-secondary" can honestly mean here — see the E8 section.

### Finding #2 — `mk06` upstream data inconsistency (Arm B)
The `SchedulingLab/fjsp-instances` mirror's own reference index records `mk06` as 15
machines (the canonical Brandimarte 1993 figure, and what `data/fjsp_known_optima.csv`
carries), but the mirror's own `brandimarte/mk06.txt` data file uses only machine indices
0–9 (10 distinct machines) and its own header line reads `10 10`. Re-fetching with
`--force` reproduces the identical file — not a transient download glitch, a genuine
mismatch between the mirror's metadata and its data file for this one instance.
**Consequence:** `mk06`'s makespan is NOT comparable to the literature's 58-optimum RPD
denominator, because the loaded instance is not actually the canonical 15-machine mk06.
`tests/test_fjsp_loader.py::TestParsing::test_mk06_known_upstream_machine_count_mismatch_is_still_present`
pins this so a future upstream fix is caught automatically. **Reporting decision:** mk06's
raw makespan/feasibility is still recorded (all methods ran successfully on it), but its
`rpd_pct` column must be treated as invalid/excluded from any RPD-based confirmatory
comparison or mean until this is resolved upstream. mk01–mk05, mk07 are unaffected.

### Finding #3 — FJSP checkpoint mismatch caught by an explicit guard, not silently
The first Arm A/Arm B launch used the JSSP checkpoint (`curriculum_stage19_la20_final`)
for FJSP instances by mistake; `baselines.interface.TrainedRLBaseline._policy_for` raised
a clear `ValueError` ("carries a 'per_node' actor head but mk01 needs a 'per_pair' one")
rather than silently producing garbage or crashing opaquely. **Consequence:** the GA and
CP-SAT rows already computed before the crash (mk01: GA=5.00% RPD, CP-SAT=0.00%/proven
optimal) were lost because `run_evaluation.py` only calls `write_csv` once, at the very
end, after every row is collected in memory — a single exception anywhere in the sweep
discards the entire run's output, not just the failing row. Re-run with the correct
`curriculum_stage24_mk05_final` checkpoint from the start. **Flag for later hardening**
(not fixed this sprint, out of scope for a Week 24 evaluation pass): `run_evaluation.py`
could write incrementally or catch-and-continue per method, so a late failure in a long
sweep doesn't discard hours of earlier, valid results.

### Finding #4 (UPDATED, same day) — E4–E6's RL arm was wired and executed; it is not blocked, but comes with a quantified, honest limitation
Originally reported blocked (below is the original diagnosis, kept for the record).
**Update: fixed and executed the same day, at the user's explicit request, after a
careful diagnosis + gated verification process.** See the dedicated section after E4/E5/E6
below ("E4/E5/E6 RL arm — wiring fix and results") for the full account: what was
diagnosed, what was fixed, what the verification gate found (a genuine, non-fixable
semantic limitation, not a shape bug), and the H3 verdict this produced.

**Original diagnosis (still accurate as far as it goes):** `DynamicJSSPEnv`'s
observation dict had no `adj_eligibility` key at all (which `models/policy.py`'s
hetero-GNN encoder hard-requires), and its `machine_features`/`node_features` widths
differed from the static env's (6 vs 7, 9 vs 8 columns respectively) — a real
architecture-level mismatch, not a shallow shape bug, and `run_evaluation.py`'s own
`validate_methods_for_disruption` hard-blocked `--methods rl` under any `--disruption`
other than `none` with an explicit, intentional `SystemExit`. This part **was** fixable
same-day without retraining (see below), contrary to the more pessimistic initial read.

**What was true regardless:** PDRs already rolled out live against `DynamicJSSPEnv` via
`run_evaluation.py::run_pdr_online` (fully implemented, exercised by
`tests/test_dynamic_env.py`, no dependency on the RL wiring at all) — and per
`../../thesis/PreRegistration.md` §4, **the PDR subset, not RL, is H3's frozen comparator** for
E4–E6's confirmatory decision rule. So the load-bearing part of E4–E6 was never blocked.

---

## E0 — Validación DES

**Status: ✅ complete.** See `../docs/logs/E0_CLOSEOUT_WEEK24.md` for the full consolidated narrative.
No new compute scheduled, per the matrix's own §2.0. One test regression found and
root-caused (Finding #1 context) but confirmed non-blocking.

---

## E1 — Estático JSSP

**Status: ✅ complete.** 688 rows total (516 deterministic + 172 RL-stochastic), merged
into `experiments/results/week24/e1_merged.csv`. **100% feasible, zero timeouts, zero
crashes.**

- Instances: 43 (`ft06`, `ft10`, `ft20`, `la01`–`la40`), `--disruption none`.
- Methods: 9 PDRs + GA + CP-SAT (seed 0 each, 516 rows = 43×12) + RL (1 deterministic +
  4 stochastic seeds 1–4 via `--rl-sample`, 215 rows total across the merged file).
- Outputs: `e1_seed0_baselines.csv` (PDR/GA/CP-SAT/RL-deterministic) +
  `e1_rl_stochastic.csv` (RL seeds 1–4) → merged as `e1_merged.csv`.

**Descriptive stats (mean/std RPD% vs known optimum, across all 43 instances):**

| method | n | mean RPD% | std RPD% | mean wall-clock (s) |
|---|---|---|---|---|
| cpsat | 43 | 0.02 | 0.13 | 4.667 |
| ga | 43 | 5.52 | 5.31 | 60.012 |
| fifo | 43 | 19.16 | 11.10 | 0.023 |
| **rl** | **215** | **29.08** | **14.73** | **0.398** |
| mor | 43 | 39.83 | 16.91 | 0.024 |
| mwkr | 43 | 42.59 | 16.19 | 0.023 |
| lpt | 43 | 281.88 | 116.91 | 0.023 |
| spt | 43 | 268.23 | 114.01 | 0.027 |
| lwkr | 43 | 391.91 | 177.61 | 0.023 |
| lor | 43 | 401.15 | 174.56 | 0.023 |
| edd | 43 | 387.34 | 177.00 | 0.022 |
| cr | 43 | 430.31 | 183.62 | 0.024 |

(Full per-instance breakdown: `e1_merged_descriptive.csv`, grouped by `method` only above
for the log; a per-instance × method table is in the CSV for the dedicated stats pass.)

**Descriptive read:** CP-SAT essentially reaches proven optimum everywhere (0.02% mean
RPD — the tiny residual is instances where it hit the 60s budget without a formal proof,
not an incorrect result). GA is close behind (5.5%). RL (29.08%, 0.40ms wall-clock,
i.e. ~150× faster than GA's 60s budget) beats 7 of the 9 PDRs but loses to `fifo` — a
believable, reportable result, not a red flag: this checkpoint is the *size-agnostic
curriculum* policy (see infrastructure note #4), and per §2.1 of the matrix, `ft06` sits
at the edge of its curriculum exposure while everything past `la20` is genuine zero-shot,
so a real gap to `fifo`/GA/CP-SAT on some instances is expected and should be reported as
such in Chapter 6.2, not smoothed over.

---

## E2 — Estático FJSP

### Arm A (synthetic, 43 instances)

**Status: ✅ complete.** 301 rows (129 deterministic + 172 RL-stochastic), merged into
`experiments/results/week24/e2_armA_merged.csv`. **100% feasible, zero crashes.**

- Same 43 instances, synthetic FJSP conversion (`--fjsp`, `--fjsp-seed 0`).
- Methods: GA + CP-SAT (seed 0) + RL (1 deterministic + 4 stochastic), via the FJSP
  checkpoint (`curriculum_stage24_mk05_final`).
- No `known_optimum`/RPD column (synthetic instances have no literature optimum, by
  design — see `run_evaluation.py`'s own docstring); raw makespan and cross-method ratios
  are the reportable comparison instead.

**Descriptive stats (raw makespan, across 43 instances):**

| method | n | mean makespan | std | mean wall-clock (s) | max wall-clock (s) |
|---|---|---|---|---|---|
| cpsat_fjsp | 43 | 893.19 | 320.78 | 39.21 | 60.03 |
| ga | 43 | 980.56 | 355.17 | 60.02 | 60.05 |
| rl | 215 | 1295.78 | 464.54 | 0.92 | 38.52 |

CP-SAT reached a proven optimum on 17/43 instances (`status=OPTIMAL`) and the 60s-budget
best-feasible on the remaining 26 (`status=FEASIBLE`, not proven) — expected, since these
synthetic FJSP conversions of `la31`-scale instances are considerably harder to close
than the small `ft06`-scale ones. Per-instance ratios: RL/CP-SAT mean = **1.456**
(σ=0.217) — RL runs ~46% above CP-SAT on average on this synthetic set; GA/CP-SAT mean =
**1.094** (σ=0.064) — GA is much closer to CP-SAT, as expected for an offline solver
given the same 60s budget CP-SAT gets. RL's wall-clock (0.92s mean, but up to 38.5s on
the largest FJSP instances — the per-pair action head scores every (operation, machine)
combination, so cost scales with `num_ops × num_machines`, unlike JSSP's per-node head)
is still far cheaper than GA/CP-SAT's fixed 60s budget.

### Arm B (real Brandimarte mk01–mk07)

**Status: ✅ complete**, with Finding #2's mk06 exclusion noted.

- `tests/test_fjsp_loader.py` written and passing (precondition met).
- mk01–mk05: in-sample (seen during curriculum training stages 20–24).
- mk06, mk07: zero-shot held-out.
- CP-SAT/GA baselines freshly measured (none existed before this run).

Results (deterministic pass, seed 0) — complete:

| instance | GA makespan | GA RPD | CP-SAT makespan | CP-SAT status | RL (det.) makespan | RL RPD | known optimum | status |
|---|---|---|---|---|---|---|---|---|
| mk01 | 42 | 5.00% | 40 | OPTIMAL (proven) | 60 | 50.00% | 40 | proven, in-sample |
| mk02 | 32 | 23.08% | 26 | OPTIMAL (proven) | 63 | 142.31% | 26 | best_known, in-sample |
| mk03 | 204 | 0.00% | 204 | OPTIMAL (proven) | 374 | 83.33% | 204 | proven, in-sample |
| mk04 | 76 | 26.67% | 60 | OPTIMAL (proven) | 78 | 30.00% | 60 | proven, in-sample |
| mk05 | 181 | 5.23% | 173 | FEASIBLE (60s cutoff, not proven) | 201 | 16.86% | 172 | best_known, in-sample |
| mk06 | 81 | 39.66%† | 59 | FEASIBLE (60s cutoff)† | 170 | 193.10%† | 58† | best_known, **zero-shot, RPD invalid — Finding #2** |
| mk07 | 164 | 17.99% | 142 | FEASIBLE (60s cutoff, not proven) | 309 | 122.30% | 139 | best_known, zero-shot |

† mk06's `known_optimum`/RPD column is not a valid comparison — see Finding #2. Raw
makespans (81/59/170) are still genuine, feasible results for whatever 10-job/10-machine
instance the mirror's `mk06.txt` actually encodes; they are simply not comparable to the
canonical 15-machine mk06's literature optimum of 58.

**Descriptive read:** CP-SAT reaches the proven optimum on all 3 `status=proven`
instances (mk01, mk03, mk04) with time to spare (well under the 60s budget on mk01/mk04;
mk03 needed the fewest seconds of all despite being the largest job count, likely because
its flexibility is low enough to prune quickly). CP-SAT does *not* close mk02 to proven
(only mk01/03/04 are `status=proven`in the reference table to begin with, so this is
expected, not a solver failure) and hits the 60s wall-clock cutoff without a proof on
mk05/mk06/mk07 (`status=FEASIBLE`, not `OPTIMAL`) — consistent with best-known (not
proven) reference values for those three. RL's RPD is worst on mk02 and mk06 (the
smallest-job/most-machine-heavy shapes) and mk03 (the largest job count) — worth
flagging for Chapter 6.2 as a shape-sensitivity pattern to discuss, not just a single
aggregate FJSP number.

RL stochastic pass (28 rows = 7 instances × 4 seeds) complete — see
`experiments/results/week24/e2_armB_mk_stochastic.csv`. Notable: RL's in-sample RPD on
mk01/mk04 (50%, 30%) is markedly worse than the FJSP curriculum's own Week 21 report
(5.0% RPD on mk01 with the Week 18 1M-step standalone checkpoint) — expected, since this
run intentionally uses the *generalist* curriculum checkpoint (`stage24_mk05`, 300k
steps/stage after a head reset) for consistency with Arm A/E3's "one size-agnostic
policy" framing, not the specialized single-instance checkpoint. Flag for Chapter 6.2
discussion: report both checkpoints' numbers, do not conflate them.

**mk06 RPD values (203–293% in the stochastic pass) are invalid** per Finding #2 —
report raw makespan only, exclude from any RPD-based statistic until the upstream data
issue is resolved.

---

## E3 — Generalización tamaño

**Status: ✅ complete.** Per matrix §2.3/§2.8, reused E1's own `la21`–`la40` rows rather
than re-running (E3 wants exactly 1 deterministic zero-shot rollout/instance, which is
precisely E1's RL-deterministic + 9-PDR + GA/CP-SAT seed-0 pass) —
`experiments/results/week24/build_e3_from_e1.py` extracts these 20 instances × 12 methods
= 240 rows into `e3_generalization_la21_la40.csv`.

**Cross-check against `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`'s own zero-shot table (same
checkpoint, same methodology) — the single most important sanity check in this whole
sprint, since it validates the entire eval pipeline end-to-end, not just one number:**

| instance | Week 21 RL RPD | Week 24 RL RPD | delta |
|---|---|---|---|
| la21 | 35.6% | 35.56% | -0.04pp |
| la25 | 27.4% | 27.43% | +0.03pp |
| la30 | 19.9% | 19.85% | -0.05pp |
| la35 | 26.1% | 26.11% | +0.01pp |
| la40 | 21.4% | 21.36% | -0.04pp |

**All 5 match within 0.05 percentage points.** This confirms the Week 24 evaluation
harness, checkpoint loading, and RPD computation are all consistent with Week 21's
independently-produced numbers — strong evidence the pipeline is correct, not just that
it runs without crashing.

**Descriptive stats by tier** (RL method, RPD%, n=5 instances/tier):

| tier | instances | mean RPD% | std RPD% |
|---|---|---|---|
| 1.5–2× (`la21`–`la25`) | 5 | 32.55 | 6.52 |
| 2× (`la26`–`la30`) | 5 | 32.92 | 9.97 |
| 3× (`la31`–`la35`) | 5 | 23.86 | 4.12 |
| 2.25× (`la36`–`la40`) | 5 | 32.47 | 7.99 |

(Computed from `e3_generalization_la21_la40.csv`; full per-instance/method table there
for the Friedman/Nemenyi pass next month.) **Note the non-monotonic pattern**: RPD does
not increase steadily with instance size — the 3× tier (`la31`–`la35`) actually shows the
*lowest* mean RPD (23.9%) of the four tiers, lower than even the smallest 1.5–2× tier
(32.6%). This is a real, reportable finding, not an artifact (it is also visible in
Week 21's own 5-instance table: `la35`=26.1% and `la40`=21.4% are both lower than
`la21`=35.6%) — the generalization curve is not simply monotonic in instance size, and
Chapter 6.2 should describe it as such rather than assume a strictly worsening trend.
Fresh 9-PDR + GA + CP-SAT baseline sweep is included in the same CSV for all 20 instances
(Week 12 only went up to `la10`, so this is new coverage, not a re-run).

---

## E4/E5/E6 — Robustez (PDR arm + RL arm, both complete)

**Status: ✅ complete (PDR arm).** ✅ **RL arm also complete** (wired same-day after the
PDR arm; see the dedicated section immediately following this one for the full
diagnosis/fix/verification/results account and the H3 verdict). GA/CP-SAT scoped to one
descriptive regime per matrix guidance (§2.4–2.6), not full replication.

- E4 (breakdowns, `A_k` ∈ {0.95, 0.90, 0.80}): 3 × 540 rows (`ft06`+`la01` × 9 PDRs × 30 seeds).
- E5 (stochastic times, `σ` ∈ {0.1, 0.25, 0.5}): 3 × 540 rows.
- E6 (arrivals, `ρ` ∈ {0.7, 0.85, 0.95}): 3 × 240 rows (4 arrival-safe PDRs only: `lwkr`,
  `mor`, `lor`, `fifo` — `ARRIVAL_UNSAFE_PDR_NAMES` excludes `spt`/`lpt`/`mwkr`/`edd`/`cr`).
- Descriptive-only: GA + CP-SAT via predictive-reactive replay, `ft06`, breakdowns @ 0.90,
  5 seeds (NOT the full 9×30×2 grid — per the matrix's own explicit instruction that
  GA/CP-SAT are not H3's frozen comparator and full replication there would cost ~9h for
  zero decision-rule contribution).
- **All 3,420 PDR rows across every sub-run: 100% feasible.** Zero timeouts, zero crashes.
- Full per-instance/level/method mean+std RPD tables: `experiments/results/week24/e4e5e6/summary_tables.txt`.

**E4 (breakdowns) descriptive summary** (mean RPD% vs static-nominal optimum, n=30/cell):
best-performing PDR is consistently `mwkr`/`mor` (~23–40% RPD depending on availability),
worst is `spt` (~67–103%). Monotonic trend confirmed: RPD decreases as `A_k` increases
(e.g. `mwkr` on `la01`: 59.3%→30.3%→22.5% for `A_k`=0.80/0.90/0.95) — the expected
direction, a basic sanity check the disruption generator and PDR rollout are behaving
correctly.

**E5 (stochastic times) descriptive summary:** same ranking (`mwkr`/`mor` best,
`spt`/`edd` worst), RPD increases monotonically with `σ` as expected (e.g. `mwkr` on
`ft06`: 13.2%→14.6%→26.5% for `σ`=0.10/0.25/0.50).

**E6 (arrivals) descriptive summary — metric caveat, not an anomaly:** RPD values here
are large (200–293%) and mildly *decrease* as `utilization` increases, which looks
counter-intuitive at first glance. This is an artifact of the descriptive metric choice,
not a bug: `rpd_pct` here is computed against the *base, no-arrivals* instance's static
optimum, but under arrivals the realized schedule covers many more operations than the
base instance ever had — dividing a much-larger-scope makespan by a small-scope optimum
inflates RPD mechanically and isn't a fair like-for-like comparison. This is exactly why
`../../thesis/PreRegistration.md`/the matrix specify **RI and `E[ΣT_j]` (expected total
tardiness/flow time)**, not raw RPD, as E6's primary metric — computing those properly is
in scope for next month's dedicated analysis script, not this descriptive pass. `mor` is
still the best-ranked PDR and `lor`/`lwkr` the worst, consistently across all 3 levels.

**Descriptive GA/CP-SAT run** (`ft06`, breakdowns @ `A_k`=0.90, 5 seeds, predictive-reactive
replay): GA and CP-SAT's replanned makespans coincide exactly on 4/5 seeds and differ by
only 1 unit on the 5th (59 vs 60) — expected, since `ft06` is small enough that GA's
offline solve is already at or near the CP-SAT-proven optimum, so the reactive-replay
outcome converges regardless of which offline planner produced the nominal schedule.

---

---

## E4/E5/E6 RL arm — wiring fix, verification, and results (same-day follow-up)

**Status: ✅ complete.** Requested and completed the same day as the rest of this log,
after the RL arm was initially reported blocked. Full account below, including a real
limitation surfaced during verification that the user was consulted on directly before
proceeding.

### What was diagnosed

Compared `DynamicJSSPEnv`'s observation contract against the trained checkpoint's actual
weight shapes (not just current source — read directly from the saved `state_dict`,
which confirmed `in_dim_ope=8`, `in_dim_mach=7`, hard-wired):

| Field | Static (checkpoint expects) | `DynamicJSSPEnv` (as found) | Fix needed |
|---|---|---|---|
| `node_features` | 8 columns | 9 (extra `is_preempted`, genuinely new dynamic-only info) | Adapter must drop it — cannot be un-invented, it's real information the static env never had |
| `machine_features` | 7 columns incl. `utilization_accum` | 6 (missing `utilization_accum` entirely) | Add the missing column |
| `adj_eligibility` | Required key | Absent entirely | Add the key |
| `action_space` size | `Discrete(num_ops)` | `Discrete(max_possible_ops)` (padded) | **Not actually a problem** — `PerNodeActorHead` is a shared per-node MLP with no hardcoded N, and padding ops are correctly masked out |

### What was fixed (no retraining)

1. **`envs/dynamic_jssp_env.py`**: added `utilization_accum` (busy-time-so-far / realised
   schedule span), computed **dispatch-gated** (using `effective_duration`/`start_times`
   set at dispatch time, not `end_times` set at completion) specifically so it matches
   the static env exactly at every decision epoch, not just by convergence at the end —
   `NUM_MACHINE_FEATURES` 6→7. Added `adj_eligibility` (one-hot per operation row over
   `_flat_machines`, structural under this environment's fixed-routing-only scope,
   zero-padded for not-yet-arrived ops matching the existing `adj_machine`/
   `adj_precedence` convention).
2. **`baselines/interface.py`**: new `_DynamicNodeFeatureAdapter(gym.ObservationWrapper)`
   that drops the extra `is_preempted` column before the frozen policy sees it, and a new
   `TrainedRLBaseline.solve_dynamic()` method (mirrors `solve()`, built on the same
   checkpoint-caching machinery) that rolls the policy out live against a wrapped
   `DynamicJSSPEnv`.
3. **`run_evaluation.py`**: removed the hard `SystemExit` block on `--methods rl` +
   `--disruption`; added `run_rl_online()` (mirrors `run_pdr_online()`) and wired it into
   `evaluate_one()`'s dispatch logic.

### Verification gate — a real limitation was found and the user was consulted before proceeding

The user's instruction was explicit: verify identical actions/makespan between a static
rollout and the adapted-dynamic rollout (disruptions off) before running anything
further, and only proceed if that check passed **exactly**.

**It did not pass exactly**, for a reason that turned out to be structural, pre-existing,
and *not* fixable by further environment work: `DynamicJSSPEnv`'s operation-readiness is
**completion-gated** (an op becomes legal only once its predecessor's completion event
actually fires — correct and necessary for modeling breakdowns/preemption), while the
static env's is **dispatch-gated** (an op becomes legal the instant its predecessor is
dispatched — a deliberate simplification the static env's own docstring already
documents). This divergence in *when an action becomes legal* is not something the
observation-contract fix touches at all — column-for-column, the new fields
(`utilization_accum`, `adj_eligibility`) were verified numerically exact against the
static env (see below), but a **policy making its own adaptive decisions** can still end
up choosing a different, still-legal action at a step where the two environments'
legal-action sets happen to differ, cascading into a different final makespan — even with
every disruption axis off. Measured magnitude: `ft06` dynamic makespan ≈3–5 units
*better* than static; `la01` ≈18 units *worse* (≈2.5%) — consistent and reproducible
across seeds (not noise), but a modest few-percent effect, not a breakdown.

**This was reported to the user directly, with three explicit options, before any
further action was taken.** The user chose: accept a weaker but still real proof
(environment-mechanics correctness via replay-equivalence, extending the existing
`test_default_config_matches_static_env_exactly` regression test) in place of the
impossible-to-satisfy literal "identical actions" criterion, document the quantified
divergence honestly, and proceed to the full grid.

**Revised gate, executed:**
- Extended `tests/test_dynamic_env.py::TestRegressionAgainstStatic::test_default_config_matches_static_env_exactly`
  (24 parametrized cases: 12 rules × 2 instances) to directly check `adj_eligibility` and
  `machine_features[:, :6]` match the static env exactly at every step (not just
  non-interference) — **24/24 passed**, confirming the new fields are numerically
  correct, not just shaped correctly.
- New `tests/test_baselines_interface.py::TestTrainedRLBaselineDynamicWiring`: replays a
  fixed PDR-derived dispatch sequence through the actual wrapped environment
  `solve_dynamic` uses, confirms identical makespan vs. the static env, and a basic
  end-to-end sanity check of `solve_dynamic` itself — **3/3 passed**.
- Full suite re-run after all changes: **632 passed**, same pre-existing 3 failures as
  Finding #1 (unrelated, confirmed non-blocking), **zero new regressions**.

**Honest limitation, stated plainly for Chapter 6.4/7:** the RL-under-disruption numbers
below come from a policy that was never trained against the dynamic environment's
completion-gated legality semantics. It is not blind to the disruptions themselves (it
correctly sees breakdown/stochastic-time/arrival state via the machine features and
node/adjacency growth), but its decisions are made with a systematic, small-but-real
mismatch versus what a policy retrained on the dynamic contract directly would produce.
This is exactly the caveat the original Finding #4 anticipated as the cost of "adapt the
environment" vs. "retrain the policy" — now measured, not just hypothesized.

### E4/E5/E6 RL-arm results

Same grid as the PDR arm: `ft06` + `la01`, 30 seeds, all 9 regimes (E6 has no
arrival-safety restriction for RL — unlike the unsafe PDRs, `DynamicJSSPEnv`'s growable
array handles arrivals natively). **All 540 RL-under-disruption rows: 100% feasible.**
Combined PDR+RL CSVs per regime: `experiments/results/week24/e4e5e6/*_with_rl.csv`
(4,500 rows total across all 9 regimes).

### H3 verdict (per `../../thesis/PreRegistration.md` §4's frozen decision rule)

`RI = (E[Cmax_pert] − Cmax_nom) / Cmax_nom`, `Cmax_nom` = each method's own E1
deterministic (seed==0) nominal makespan (`../../thesis/StatsAnalysis_TechSpec.md` §5.2's pinned
source — reused, not recomputed). Paired Wilcoxon signed-rank test on
`(RI_PDR − RI_RL)` per (instance × seed), computed separately per regime, best-performing
(lowest-RI) PDR as comparator, `α=0.05`.

| regime | best PDR (by RI) | RI (best PDR) | RI (RL) | median(RI_PDR − RI_RL) | p-value | accept H3? |
|---|---|---|---|---|---|---|
| E4 breakdowns, A_k=0.95 | cr | −0.547 | +0.010 | −0.609 | 1.6e-11 | No |
| E4 breakdowns, A_k=0.90 | cr | −0.504 | +0.115 | −0.630 | 1.6e-11 | No |
| E4 breakdowns, A_k=0.80 | cr | −0.402 | +0.405 | −0.818 | 1.6e-11 | No |
| E5 stochastic, σ=0.10 | cr | −0.598 | −0.031 | −0.560 | 1.6e-11 | No |
| E5 stochastic, σ=0.25 | cr | −0.583 | −0.017 | −0.572 | 1.6e-11 | No |
| E5 stochastic, σ=0.50 | cr | −0.577 | +0.067 | −0.635 | 1.6e-11 | No |
| E6 arrivals, ρ=0.70 | lwkr | +0.271 | +1.727 | −1.501 | 1.6e-11 | No |
| E6 arrivals, ρ=0.85 | lwkr | +0.277 | +1.706 | −1.424 | 1.6e-11 | No |
| E6 arrivals, ρ=0.95 | lwkr | +0.257 | +1.631 | −1.414 | 1.6e-11 | No |

**H3 regimes accepted: 0/9. Overall verdict: REJECT H3** (the trained policy's robustness
index is not lower than the best PDR's in any of the 9 regimes; the effect is
statistically significant but in the opposite direction of the H3 claim in every case).

**Methodological transparency note — the "best PDR" identification has a real quirk,
disclosed rather than silently worked around:** the frozen rule selects the
lowest-*median-RI* PDR per regime, and for E4/E5 that mechanically picks `cr` — one of
the *worst absolute performers* in the nominal case (E1: `cr` on `ft06` has RPD ≈211%,
the single worst PDR measured). `cr`'s RI looks artificially favorable because its own
nominal baseline is so bad that almost any perturbed run looks like an "improvement"
relative to it — an artifact of RI's method-specific-baseline design, not a computation
error (this is the frozen D6 formula, applied exactly as specified; the quirk is
inherent to the metric, not to this analysis). **Sensitivity check performed:**
re-running the same test against `mwkr` (E1's actual best-performing PDR by absolute
RPD, excluded from E6 as arrival-unsafe) gives the *same* qualitative result — H3
rejected in every E4/E5 regime tested (p ranging 7.6e-6 to 2.3e-11, median diff still
negative in all 6 cases) — confirming **the REJECT H3 verdict is not an artifact of
which PDR the frozen rule happens to select.**

**Interpretation:** RL's robustness index is worst, by a wide margin, under **arrivals**
(RI ≈+1.7, i.e., the perturbed makespan averages ~2.7× the nominal) — far worse than
under breakdowns or stochastic times (RI in the −0.03 to +0.4 range). This is consistent
with, and gives a quantitative shape to, the verification-gate limitation above: arrivals
present the policy with a genuinely novel, never-trained-on scenario (new jobs entering
the schedule), whereas breakdowns/stochastic-times perturb the timing of operations the
policy already knows exist. **Full support for H3 was never a realistic prospect given a
policy deployed without retraining against the dynamic contract; this result quantifies
exactly how large that gap is, per regime, for the thesis to report honestly** — a
negative-but-informative confirmatory result, not a null one.

---

## E8 — Tiempo de decisión

**Status: ✅ complete.**

| instance | num_ops | ms/decision (E1, `--num-workers 8` in effect, 5 rollouts) | ms/decision (dedicated pass, multi-thread) | ms/decision (dedicated pass, single-thread) | CP-SAT wall-clock (s) |
|---|---|---|---|---|---|
| ft06 | 36 | 9.52 ± 17.78 | 4.857 | 1.061 | 0.103 |
| la05 | 50 | 36.54 ± 78.74 | 1.078 | 1.036 | 0.006 |
| la20 | 100 | 2.21 ± 1.93 | 1.299 | 1.313 | 0.177 |
| la35 | 300 | 2.76 ± 0.20 | 3.448 | 3.357 | 0.172 |

**Methodological caveat (important for any latency claim in Chapter 6.5):** the E1 column
above was measured while E2 Arm A/B and E4–E6 were running concurrently in separate
background processes on the same 16-core machine, competing for CPU — hence its large
variance (σ up to 8× the mean on `ft06`/`la05`) compared to the dedicated E8 pass (run
with far less concurrent load). **Treat the dedicated-pass column as the more trustworthy
number**; the E1 column is included for completeness but should not be over-interpreted
as a precision benchmark. If Chapter 6.5 needs a tight, publishable latency figure, a
clean re-measurement on an otherwise-idle machine is recommended rather than reusing E1's
noisy numbers.

**Descriptive read (dedicated pass):** for the smallest instance (`ft06`, 36 ops),
single-threaded inference is *faster* than the default multi-threaded configuration
(ratio 0.22) — a known PyTorch effect where thread-spawn/synchronization overhead
dominates for tiny tensors and swamps any parallelism benefit. For larger instances
(`la05`/`la20`/`la35`), the ratio is close to 1.0: this network is simply too small for
multi-threading to help or hurt meaningfully at this scale. In every case, RL's
per-decision cost (1–5 ms) is 1–2 orders of magnitude below CP-SAT's total wall-clock for
a full solve (6–177 ms) — consistent with the matrix's own expectation that RL inference
is a genuinely fast, near-real-time decision rule relative to an exact solver. Also see
Finding #1: both columns are actually CPU inference in this codebase's current
implementation, so the "GPU-primary" framing in the original matrix doesn't literally
apply — the multi-thread/single-thread contrast is the closest honest analogue available
without a code change.

## What was NOT attempted this window (explicit, per the task's own priority order)

- **E7**: out of scope for this window by the master plan's own schedule (Sprint 2,
  Semana 27). Now additionally informed by a real, measured number: E4–E6's RL-arm
  wiring fix + H3 verdict (see above) means E7's `κ`-reward experiments, when they run,
  should account for the same completion-gated-vs-dispatch-gated policy-behavior gap
  quantified here, not just the reward term itself.
- **Statistical significance testing beyond H3** (Wilcoxon/Friedman/Nemenyi/CD diagrams
  for E1/E2/E3): explicitly deferred per STEP4 of the task brief; descriptive stats only
  for those experiments in this log. (H3's Wilcoxon test *was* run — it's part of E4–E6's
  own frozen decision rule, not the general "next month" statistical pipeline the task
  brief was referring to for the other experiments.)
- **M/M/1 analytic-queue unit test**: optional per the matrix's own E0 note; not built
  this window to protect compute time for E1–E6.
- **Retraining the policy against `DynamicJSSPEnv`'s completion-gated contract**: the
  E4–E6 RL-arm follow-up used an environment-side adapter instead (see above), per the
  user's explicit choice after being shown the tradeoff. Retraining remains the
  principled fix for the quantified policy-behavior gap this adapter approach cannot
  close, and is the natural Sprint-2/E7 follow-up.

---

## Summary

Every experiment in E1–E6, plus E0 and E8, is now complete — including E4–E6's **RL**
arm, wired and executed the same day as a deliberate follow-up after the rest of this
log was written. That fix required a real diagnosis (confirmed against the trained
checkpoint's actual weight shapes, not just source), a genuine environment change
(`utilization_accum`, `adj_eligibility` added to `DynamicJSSPEnv`, verified numerically
exact against the static env via an extended regression test), and — critically — an
honest mid-implementation pause: the user's own requested verification gate (identical
actions/makespan, disruptions off) **failed**, for a structural reason (completion-gated
vs. dispatch-gated operation legality) that turned out not to be fixable without
undermining the dynamic environment's correct breakdown/preemption modeling. That
finding was reported in full, with three options, before any further action was taken;
the user chose to accept a weaker-but-real replay-equivalence proof and proceed, with the
resulting policy-behavior gap measured and documented rather than hidden. **This is the
one place in the whole Week 24 log where a result should be read with an explicit
caveat attached** (see the dedicated RL-arm section above) — not because the engineering
was rushed, but because the caveat is a genuine, quantified property of evaluating a
statically-trained policy on a more realistic dynamic environment without retraining.

**H3 verdict: REJECT** (0/9 regimes accept; robust to which comparator PDR is used —
checked against both the frozen rule's mechanical choice and a genuinely strong PDR by
absolute performance). This is a negative but informative confirmatory result: the
trained policy is measurably *less* robust than the best PDRs under all three disruption
types, most severely under arrivals (a scenario qualitatively unlike anything in its
training data), consistent with the wiring fix's own verification finding that the
policy was never trained to handle the dynamic environment's completion-gated action
legality.

**Total rows from freshly-run compute this window:** 688 (E1) + 301 (E2 Arm A) + 49
(E2 Arm B: 21 deterministic + 28 stochastic) + 3,420 (E4/E5/E6 PDR grid) + 540 (E4/E5/E6
RL grid) + 10 (E4 descriptive GA/CP-SAT) + 4 (E8) = **5,012 evaluation rows** — plus E3's
240-row CSV, which reuses E1's own `la21`–`la40` rows rather than new compute (by design,
per matrix §2.3/§2.8). **100% feasible** everywhere a feasibility check applies, zero
unrecovered crashes (one recovered crash, Finding #3, re-run correctly from the correct
checkpoint).

**Five findings surfaced that need committee/advisor visibility**, none of which block
Chapter 6 from being written, but all of which change what it should honestly claim:
1. A real (training-only, not evaluation-path) device bug in `test_hetero_policy.py`'s
   helper — cosmetic to this sprint, worth a follow-up fix before Sprint 2's E7 training.
2. `mk06`'s upstream data mismatch — excludes one zero-shot FJSP data point from RPD
   reporting until resolved upstream; pinned by a regression test so it self-corrects.
3. A checkpoint-mismatch guard worked exactly as designed, but exposed that
   `run_evaluation.py` discards an entire sweep's results on any single exception —
   worth hardening before a longer/riskier sweep is run unattended.
4. E4–E6's RL arm was successfully wired same-day (see dedicated section above) — the
   PDR arm (H3's actual frozen comparator) was never blocked in the first place, and the
   RL arm's initial "blocked" assessment turned out to be more pessimistic than
   warranted for the shape/contract issue, but correctly anticipated that a
   never-retrained policy would carry a real behavioral cost once actually measured.
5. The RL-arm fix surfaced a genuine, non-fixable-without-retraining limitation: a
   policy trained on static list-scheduling's dispatch-gated legality cannot be made to
   behave identically on the dynamic environment's (correct) completion-gated legality.
   Quantified at a few percent of makespan on `ft06`/`la01` with disruptions off; the
   H3 verdict above shows the cost is much larger once real disruptions are added,
   especially for arrivals.

**Recommended next steps (Semana 25 review, per the matrix's own schedule):** (a) decide
whether Finding #2's mk06 exclusion needs an upstream GitHub issue filed against
`SchedulingLab/fjsp-instances`; (b) for E7 (Sprint 2), budget for retraining against the
dynamic environment's actual contract rather than reusing this adapter — the H3 result
above is a strong, measured argument that the adapter path is not sufficient for a
confirmatory robustness claim, only a descriptive one; (c) note for Sprint 2 that a
clean, isolated E8 timing re-measurement would strengthen the H5 latency claim beyond
this sprint's concurrency-contaminated numbers.
