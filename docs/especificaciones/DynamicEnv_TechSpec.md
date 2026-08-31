# DynamicEnv Technical Specification v1.0
### Extending the static JSSP Gymnasium environment to dynamic job arrivals, stochastic processing times, and machine breakdowns

**Audience:** a coding agent implementing this directly on top of the existing, already-tested repository (`envs/`, `models/`, `baselines/`, `tests/`; see `VALIDATION_WEEK1.md` for the current state — 188/188 tests passing on the static environment).

**Prime directive: EXTEND, DO NOT REWRITE.** `envs/jssp_env.py`, `envs/disjunctive_graph.py`, `envs/instance.py`, `models/gin.py`, `models/policy.py` are validated and frozen for this sprint. Every new behavior in this spec is additive: new modules, a new environment subclass/wrapper, and backward-compatible extensions to existing observation/action spaces gated behind a config flag so `JSSPEnv(instance)` with no dynamic config continues to behave exactly as today (this is the regression contract — see §9).

Source-of-truth cross-reference: this spec formalizes Part D1/D5 of the thesis framework document ("../docs/planificacion/Step 4 Thesis.md") and is the basis for thesis Section 4.2 ("../../archive/capitulos/Section4.2.md", companion document). Where this spec makes a concrete engineering choice not fully pinned down by D1/D5 (e.g., the exact shape parameter of the repair-time distribution, or the content model for newly arriving jobs), that choice is flagged explicitly in §10 as an open decision this spec is resolving, not restating.

---

## 0. Scope

**In scope for this sprint:**
- Discrete-event core: a real future-event list (FEL), event dataclasses, `heapq`-based priority queue.
- Three independent, unit-testable disruption generators: machine breakdowns (E3/E4), stochastic processing-time multipliers, dynamic job arrivals (E1).
- Wiring these into a `DynamicJSSPEnv` that wraps/extends `JSSPEnv`.
- Preemption handling (one policy, justified — §5).
- Observation/action space extensions (machine-state block, masking of broken machines).
- New unit tests + a `VALIDATION_WEEK6.md` validation report.

**Out of scope for this sprint (explicitly deferred, do not implement):**
- FJSP routing flexibility (choice of machine per operation) — still fixed routing per `envs/instance.py`.
- Any change to `models/gin.py`, `models/policy.py`, or `experiments/train_ft06.py` (training pipeline). The dynamic environment must be steppable and testable standalone (e.g., under the existing dispatching-rule baselines) before it is wired into training in a later sprint.
- The robustness-aware reward term (`κ·Δ_inest`, Part D3) — implement only the plumbing needed to compute it later (i.e., track realized start-time deviation from the last plan), not the term itself.

---

## 1. Module layout

```
envs/
  events.py            # NEW — Event dataclass, FutureEventList (priority queue)
  disruptions.py        # NEW — BreakdownGenerator, ProcessingTimeSampler, ArrivalProcess
  dynamic_jssp_env.py    # NEW — DynamicJSSPEnv(JSSPEnv)
  instance.py            # UNCHANGED (reused for both the seed instance and generated arrivals)
  jssp_env.py            # UNCHANGED (static env stays as-is; DynamicJSSPEnv composes/extends it)
  disjunctive_graph.py    # UNCHANGED, but see §6.3 — `estimated_makespan` needs a disruption-aware
                          # variant; add it as a NEW function `estimated_makespan_dynamic`, do not
                          # modify the existing signature (used by tests/test_disjunctive_graph.py).
tests/
  test_events.py          # NEW — FEL ordering, tie-breaking, event dataclass invariants
  test_disruptions.py     # NEW — statistical validation of the three generators in isolation
  test_dynamic_env.py     # NEW — Gymnasium API + constraint safety under disruption
VALIDATION_WEEK6.md       # NEW — statistical + constraint validation report (see §9)
```

---

## 2. Data structures

### 2.1 `Event` (in `envs/events.py`)

```python
@dataclass(order=True)
class Event:
    time: float
    priority: int = field(compare=True)   # tie-break order, see table below
    seq: int = field(compare=True)        # insertion sequence, final tie-break (FIFO for equal time+priority)
    kind: EventKind = field(compare=False)
    payload: dict = field(compare=False)
```

- `EventKind` is an `enum.IntEnum` with members `ARRIVAL` (E1), `OP_COMPLETE` (E2), `BREAKDOWN` (E3), `REPAIR_COMPLETE` (E4). (E5, the decision epoch, is not queued — it is *derived*: after processing all events at the current time, check `env._pending_decision()`; see §3.5.)
- **Tie-break priority when `time` is equal** (ties will happen: e.g., a repair completing at the same instant a new job arrives): process in this fixed order — `REPAIR_COMPLETE` (1) → `OP_COMPLETE` (2) → `BREAKDOWN` (3) → `ARRIVAL` (4). Rationale: resource-freeing events must resolve before resource-consuming events, so a machine that becomes available and one that becomes unavailable at the same instant don't race; repairs before breakdowns because a machine should not be considered "newly broken" if it was mid-repair-completion in the same instant (edge case, but must be deterministic for test reproducibility).
- `seq` is a monotonically increasing counter assigned at `push()` time, guaranteeing a total order (required because `heapq` needs a tiebreaker to avoid comparing non-comparable `payload` dicts).

### 2.2 `FutureEventList` (in `envs/events.py`)

Thin wrapper around `heapq` (`push(event)`, `pop_min() -> Event`, `peek_min_time() -> float | None`, `is_empty`, `cancel(predicate)` — the last one is required because a machine breakdown can invalidate an already-scheduled `OP_COMPLETE` event for the operation it preempts; implement `cancel` as O(n) filter + `heapq.heapify`, not a true decrease-key structure — FEL sizes here are small (≤ a few hundred events per episode), no need for a fancier structure).

### 2.3 `DisruptionConfig` (in `envs/disruptions.py`)

```python
@dataclass(frozen=True)
class DisruptionConfig:
    # Breakdowns
    target_availability: float | dict[int, float] = 1.0   # A_k; 1.0 = breakdowns disabled.
    mttr_multiplier: float = 2.0                           # MTTR = mttr_multiplier * mean_processing_time
    ttr_shape_sigma: float = 0.5                           # sigma_r of the lognormal TTR (see §4.1)
    # Stochastic processing times
    processing_time_sigma: float = 0.0                     # s; 0.0 = deterministic (static-env-equivalent)
    # Dynamic arrivals
    target_utilization: float | None = None                # rho; None = arrivals disabled
    arrival_cutoff_time: float | None = None                # see §4.3; None -> auto = 0.8 * static horizon
    max_dynamic_jobs: int = 50                              # hard safety cap, see §4.3
    # Reproducibility
    seed: int | None = None
```

All three disruption axes are independently switchable (set the corresponding field to its "off" value) so that `DynamicJSSPEnv(instance, DisruptionConfig())` (all defaults) is **numerically identical** to today's `JSSPEnv(instance)` — this is the regression test in §9.

---

## 3. Event routines

Notation: `t` = current simulated clock; `k` = machine index; `o` = operation (global id); `j` = job index.

### 3.1 (E1) `ARRIVAL` — payload: `{job_template: JSSPInstance-like row}`

1. Instantiate a new job `j_new` (see §4.3 for how its operations/durations/machine-order are generated) with a fresh, previously-unused range of global operation ids appended after the current maximum (i.e., extend, don't renumber — renumbering would invalidate any in-flight `Event.payload` referencing old op ids).
2. Extend the disjunctive graph in place: append rows/columns to `adj_precedence` (chain edges for `j_new`'s own operations) and `adj_machine` (edges to every currently-unscheduled operation sharing a machine with any operation of `j_new`). This requires `JSSPEnv`'s fixed-size numpy arrays to become growable — see §6.1 for the exact mechanism (pre-allocate to `max_dynamic_jobs` and slice by an active-op count, rather than reallocating every arrival, to keep this O(1) amortized).
3. Mark `j_new`'s first operation as `ready` (added to the ready set) if its machine is currently free; otherwise it waits like any other ready-but-machine-busy operation.
4. Record `info["event_log"]` entry `("arrival", t, j_new)` for the validation report (§9).

### 3.2 (E2) `OP_COMPLETE` — payload: `{op_id, machine}`

Identical semantics to today's synchronous transition in `JSSPEnv.step()`, but now scheduled as a *future* event at push time (`t_now + duration`) rather than resolved instantaneously:
1. Mark `op_id` scheduled/complete, set `end_times[op_id] = t`.
2. Free `machine`: `machine_free_time[machine] = t`.
3. Activate the job's next operation (if any) into the ready set.
4. This is the only event type that can make the environment trigger a decision epoch (§3.5).

### 3.3 (E3) `BREAKDOWN` — payload: `{machine}`

1. If `machine` is currently idle (no in-process operation): mark it `broken`; done — no preemption bookkeeping needed. Draw the next `REPAIR_COMPLETE` event at `t + TTR` (§4.1) and push it.
2. If `machine` is currently processing operation `o` (started at `s_o`, scheduled to complete at `s_o + p_o`):
   - `cancel()` the pending `OP_COMPLETE` event for `o` from the FEL.
   - Compute `elapsed = t - s_o`, `remaining = p_o - elapsed` (must be `> 0`; if a breakdown and an `OP_COMPLETE` are generated for the same instant, the tie-break order in §2.1 guarantees `OP_COMPLETE` fires first, so this case cannot arise as a race — assert it defensively anyway).
   - Store `remaining` on the operation record (new field `remaining_duration[o]`, default `-1` meaning "not preempted").
   - Mark `machine` `broken`; mark `o` state `preempted` (not `ready`, not `in process`, not `done` — a fourth op-state; update `NODE_FEATURE` derivations accordingly, §6.2).
   - Draw TTR and push `REPAIR_COMPLETE` at `t + TTR`.
3. Draw the *next* `BREAKDOWN` event for this machine at `t + TTF_next` (§4.1) and push it — breakdowns are a renewal process, so the next TTF is drawn fresh after each repair, not at breakdown time; see the note at the end of §4.1 about where exactly the next TTF gets drawn (it is drawn in `REPAIR_COMPLETE`, not here — a machine cannot break down again while already broken).
4. Record `info["event_log"]` entry `("breakdown", t, machine, preempted_op=o|None)`.

### 3.4 (E4) `REPAIR_COMPLETE` — payload: `{machine}`

1. Mark `machine` `free`.
2. If an operation `o` was preempted on this machine (`remaining_duration[o] != -1`): per the preemption policy (§5), either (resume) push `OP_COMPLETE` for `o` at `t + remaining_duration[o]` and mark `o` `in process` again with `machine_free_time[machine] = t + remaining_duration[o]`, or (restart) treat `o` as newly ready-but-unscheduled again (its earlier partial start/elapsed time is discarded) so it re-enters the normal dispatch flow and the agent must (or the environment must, per policy — see §5) re-dispatch it.
3. Draw this machine's *next* `BREAKDOWN` event at `t + TTF` (§4.1) and push it — this is the renewal point referenced in §3.3 step 3.
4. Record `info["event_log"]` entry `("repair_complete", t, machine)`.

### 3.5 (E5) Decision epoch — derived, not queued

After draining all events at the current `t` (there can be several simultaneous events — process all of them before checking for a decision epoch, per the tie-break order), check: does any machine have `state == free` **and** is there at least one `ready` operation eligible for that machine? If yes, expose an observation to the agent and block the event loop until `step(action)` is called with a legal action (which itself pushes a new `OP_COMPLETE` event, per §3.2). If no machine/ready-operation pair exists, keep draining the FEL without yielding control to the agent (this can happen, e.g., immediately after a breakdown leaves no free machines).

**Termination check** (replaces `terminated = self._num_scheduled == self.instance.num_operations` from the static env): `terminated` when (a) the FEL contains no more pending `ARRIVAL` events (i.e., `t > arrival_cutoff_time` or `max_dynamic_jobs` reached) **and** (b) every operation of every job instantiated so far (original + all arrived) is `done`. Both conditions are necessary — do not terminate early just because condition (b) holds transiently with more arrivals still pending in the FEL.

---

## 4. Disruption generators (`envs/disruptions.py`)

Each generator is a standalone class with its own `numpy.random.Generator` instance (seeded independently, derived via `np.random.default_rng(seed).spawn(3)` from the top-level `DisruptionConfig.seed` so that toggling one disruption axis on/off does not perturb the random stream consumed by the others — this matters for reproducible ablations across E4/E5/E6 in the experimental matrix). Each must be usable and unit-testable **without** instantiating a full `DynamicJSSPEnv`.

### 4.1 `BreakdownGenerator` — TTF ~ Exp(λ_k), TTR ~ Lognormal(μ_r, σ_r²)

**Calibration target:** per-machine availability `A_k = MTBF_k / (MTBF_k + MTTR_k) ∈ {0.95, 0.90, 0.80}` (or a per-machine dict if heterogeneous machines are desired; default is one shared `A_k` value applied to every machine).

**Step 1 — fix MTTR.** The spec (Part D5) leaves `σ_r` as a free design choice ("MTTR as a fraction of mean processing time"); this spec pins it down concretely:

```
p_bar = instance.durations.mean()          # mean processing time across the base instance
MTTR  = config.mttr_multiplier * p_bar      # default multiplier = 2.0 (Part D5 example)
sigma_r = config.ttr_shape_sigma            # default 0.5 -> CV = sqrt(exp(sigma_r**2) - 1) ≈ 0.53
mu_r  = ln(MTTR) - sigma_r**2 / 2.0         # so that E[Lognormal(mu_r, sigma_r^2)] == MTTR exactly
```

**Step 2 — solve for MTBF and λ_k, given target A_k:**

```
MTBF_k = A_k * MTTR / (1 - A_k)             # from A_k = MTBF/(MTBF+MTTR)
lambda_k = 1.0 / MTBF_k                      # rate of the Exp(lambda_k) TTF distribution
```

`A_k = 1.0` is the disabled/off state (`MTBF_k -> inf`, `lambda_k -> 0`); implement as "do not schedule any BREAKDOWN event for this machine" rather than a literal zero rate.

**Sampling:**
```
draw_ttf(k) -> np_random.exponential(scale=1/lambda_k)
draw_ttr(k) -> np_random.lognormal(mean=mu_r, sigma=sigma_r)
```

**Where the first TTF per machine is drawn:** at `DynamicJSSPEnv.reset()`, draw one initial `BREAKDOWN` event per machine at `draw_ttf(k)` and push all of them to the FEL before the episode starts (matches "programar llegadas y primeras averías" in the D1 main-loop pseudocode). Subsequent TTFs are drawn at `REPAIR_COMPLETE` time (§3.4 step 3), making breakdowns a proper renewal process per machine.

### 4.2 `ProcessingTimeSampler` — multiplier ~ Lognormal(−s²/2, s²)

Exactly as specified in Part D5, no design gap to fill here:

```
draw_multiplier() -> np_random.lognormal(mean=-s**2/2, sigma=s)   # E[multiplier] == 1.0 by construction
effective_duration(op) = base_duration[op] * draw_multiplier()
```

`s ∈ {0.1, 0.25, 0.5}` per the experimental matrix (Part D4, E5). **Design decision (flagged, §10):** draw the multiplier **once per operation, at the moment it is dispatched** (not once per instance, and not redrawn if preempted/resumed under the resume policy — see §5), so that `effective_duration` is realized-and-fixed the instant `OP_COMPLETE` is scheduled. This matches how `static_lower_bound` / `estimated_makespan` reasoning in the static env already treats duration as fixed-at-dispatch-time; only the *value* being fixed changes (sampled vs. deterministic).

### 4.3 `ArrivalProcess` — Poisson process calibrated to target utilization ρ

**Calibration:**
```
E_load_per_job = instance.job_totals.mean()      # mean total processing time per job, base instance
m = instance.num_machines
lambda_a = rho * m / E_load_per_job               # solves rho = lambda_a * E_load_per_job / m
```

Inter-arrival times ~ `Exp(lambda_a)` (equivalent to a Poisson counting process with rate `lambda_a`); draw the next arrival time incrementally (`next_arrival = t + np_random.exponential(scale=1/lambda_a)`) rather than pre-generating a fixed count, and push each `ARRIVAL` event only once the previous one has fired (so the FEL never holds more than one pending `ARRIVAL` event at a time).

**Termination safety (mandatory — a Poisson process alone never naturally stops):** stop generating new `ARRIVAL` events once **either** `t > arrival_cutoff_time` **or** `max_dynamic_jobs` arrivals have already occurred, whichever comes first. Default `arrival_cutoff_time = 0.8 * H_nominal`, where `H_nominal` is the static env's existing horizon normalizer (`max(instance.total_processing_time, 1)`, already computed in `JSSPEnv.__init__`) — this leaves the last 20% of the nominal horizon as a "drain" period with no new arrivals, ensuring the episode is a **terminating system** in the DES sense (§4.2.1 of ../../archive/capitulos/Section4.2.md) and bounding worst-case episode length.

**Content model for arriving jobs (design decision, flagged §10 — not specified by D1/D5):** each new job reuses `instance.ops_per_job` (rectangularity is preserved — FJSP-style ragged jobs are out of scope, per §0) and is generated as: (a) a uniformly random permutation of `range(instance.num_machines)`, truncated/sampled without replacement to length `ops_per_job` if `ops_per_job < num_machines` (matches the "no repeated machine within a job" invariant already enforced in `JSSPInstance.__post_init__`); (b) durations drawn i.i.d. by bootstrap resampling from `instance.flat_durations` (the empirical duration distribution of the base instance) — not a parametric fit, to avoid introducing an extra distributional assumption not present in D1/D5. Flag this choice explicitly in the validation report (§9) as a modeling assumption, since a reviewer may reasonably ask why arriving jobs "look like" the base instance's jobs rather than being drawn from an independent distribution.

---

## 5. Preemption policy: **preempt-resume** (chosen and justified)

**Decision:** when a machine breaks down while processing an operation, the operation is **paused, not discarded**. Its remaining processing time (`remaining = p_o - elapsed`, computed at breakdown time under the realized/effective duration from §4.2) is preserved, and the operation resumes for exactly `remaining` more time units once the machine is repaired (§3.4 step 2). This is **preempt-resume**, not **preempt-restart**.

**Justification (three independent reasons, so the choice survives a committee challenge on any one of them):**

1. **Physical realism for the target domain.** This thesis models discrete-part machining/assembly job shops (the `ft`/`la` benchmark family), not chemical batch processes where a partially completed batch is typically spoiled by an interruption. For machining and assembly operations, resuming from the point of interruption (after a tool change, a repair, a re-clamp) is the standard real-world practice; discarding all partial progress on every breakdown would systematically bias the simulated system toward pessimism relative to any real shop floor it is meant to represent.
2. **Consistency with the closest related work identified in Chapter 3.** Lassoued et al. [38] and Zhang et al. [40] — the two works in the literature review whose disruption-handling mechanics are closest to this thesis's — both model breakdown recovery as resumption of interrupted work (reactive rescheduling that preserves partial progress), not restart. Adopting restart here would make direct comparison against those baselines in Experiment E4 (Part D4) apples-to-oranges without a compensating adjustment.
3. **Tractable, unambiguous reward and lower-bound accounting.** Under resume, `remaining_duration[o]` is a single well-defined scalar propagated through `estimated_makespan_dynamic` (§6.3) exactly like a normal operation's duration, just starting from a partial value — no need to decide how to re-sample a "fresh" duration for the restarted attempt (which would be an extra, unmotivated random draw under stochastic processing times, §4.2) or how to retroactively revise `H(s)` for time already "spent" on a now-discarded partial operation.

**Preempt-restart is retained as a documented, config-gated alternative** (`DisruptionConfig` field `preemption_policy: Literal["resume", "restart"] = "resume"`), specifically so that a later sprint can run it as a sensitivity ablation if a reviewer requests one — but `"resume"` is the default and the only mode required to pass the acceptance criteria in §9. Under `"restart"`: on `REPAIR_COMPLETE`, discard `remaining_duration[o]`, return `o` to the ready set as if never dispatched (it re-enters normal frontier-based masking, §6.2), and — if `processing_time_sigma > 0` — redraw a fresh multiplier for its next attempt (a restarted operation is a new realization, not a continuation).

---

## 6. Observation / action space extensions

### 6.1 Growable graph arrays (supports E1)

Pre-allocate `node_features`, `adj_precedence`, `adj_machine` at `reset()` time to `max_possible_ops = instance.num_operations + max_dynamic_jobs * instance.ops_per_job` rows/columns, and track `active_ops` (an `int`, or a boolean mask) marking which rows are "live." All existing per-op computations in `JSSPEnv` (which assume a fixed `n_ops`) must be sliced to `[:active_ops]` (or masked) before being returned in the observation — **do not** resize numpy arrays on every arrival (expensive, and invalidates SB3's fixed-shape `observation_space` contract mid-episode, which is exactly the "known limitation" already flagged in `models/policy.py` for changing instance size). `observation_space` itself must be declared once, at its maximum size, at `reset()` time; unused (not-yet-arrived) rows are zero-padded and excluded from the action mask.

### 6.2 New/updated node features

Add a fourth `is_preempted` boolean alongside the existing `is_scheduled`/`is_ready`/`is_job_complete` (Section 4.3.1 of the thesis), and extend `action_masks()`:

```
mask[i] = 1  iff  op i is unscheduled
               AND op i is the frontier operation of its (already-arrived) job
               AND op i is NOT currently preempted
               AND the machine required by op i is NOT currently broken
```

The last two clauses are the only change versus the static env's `action_masks()` (Section 4.3.2); everything else is unchanged. A preempted operation automatically resumes via `REPAIR_COMPLETE` (§3.4) — the agent never re-dispatches it, so it correctly stays outside the action mask (mask bit stays 0) for the entire preemption window, not just "not ready."

### 6.3 Machine-state features (new `d_m`-wide machine node block, per Part D2)

For each machine `k`: one-hot `{free, busy, broken}`; `time_to_available` (0 if free, else `machine_free_time[k] - t` if busy, else the drawn repair completion time minus `t` if broken — normalized by the horizon, same convention as existing features); `assigned_pending_load` (sum of durations of ready-or-later operations already known to require this machine, normalized by `W_max`); and `estimated_availability` (a running empirical estimate of `A_k`, useful as a feature for the disruption-aware variant, Part D2's "para la variante consciente de averías"). Wire this as a new `observation_space["machine_features"]` key, additive to the existing `Dict` — do not remove or rename any existing key, to keep the static-mode-off configuration byte-for-byte compatible (§9).

### 6.4 `estimated_makespan_dynamic` (new function in `disjunctive_graph.py`, do not modify `estimated_makespan`)

Same job-chain optimistic-propagation principle as the existing `estimated_makespan`, extended with: (a) a preempted operation's optimistic completion is `t_breakdown + remaining_duration[o] + expected_repair_remaining` rather than `running + duration`; (b) machines currently broken contribute their expected repair completion time as a lower bound on when they can next process anything, folded into the propagation for any unscheduled operation still waiting on that machine. Keep this function pure and independently unit-testable (§8) — it is the basis for the dense reward, and any bug here silently corrupts every training signal, exactly the class of bug the Sprint-1 CP-SAT ground truth was built to catch for the static case.

---

## 7. Reward function — plumbing only this sprint

Implement and expose `info["planned_start_deviation"]` per dispatched operation (the difference between the start time assigned at the *previous* decision epoch's plan, if any, and the realized start time after a disruption) — this is the raw signal Part D3's `Δ_inest` term will consume in a later sprint. **Do not implement the `κ·Δ_inest` term itself or change the reward formula this sprint** — the reward returned by `step()` remains exactly `(H_dynamic(s_t) - H_dynamic(s_{t+1})) * reward_scale` using `estimated_makespan_dynamic` (§6.4) in place of `estimated_makespan`, with `reward_scale` computed identically to today (`1 / static_lower_bound(instance)`, using the *base* instance only — do not attempt to recompute a "dynamic lower bound" that accounts for arrivals/breakdowns, which is not well-defined before they occur).

---

## 8. Required unit tests

All new tests live in `tests/test_events.py`, `tests/test_disruptions.py`, `tests/test_dynamic_env.py`. Statistical tolerances below use `N ≥ 1000` samples per assertion unless stated otherwise, and a two-sided check against the theoretical value with a tolerance wide enough to keep flake rate negligible (prefer a fixed relative tolerance, e.g. 5%, combined with a fixed master seed per test, over a statistical hypothesis test with a p-value threshold, to keep CI deterministic — reserve an actual Kolmogorov–Smirnov goodness-of-fit test, e.g. `scipy.stats.kstest`, for one dedicated "shape" test per distribution rather than every test, since KS tests are slower and their false-positive rate at α=0.05 makes them noisy as a CI gate run every commit).

**`test_events.py`**
- `test_fel_orders_by_time` — events pop in non-decreasing `time` order.
- `test_fel_tie_break_order` — three events at identical `time` with different `kind` pop in the fixed priority order of §2.1.
- `test_fel_cancel_removes_matching_event` — `cancel(predicate)` removes exactly the matching event(s) and preserves heap validity for subsequent pops.

**`test_disruptions.py`**
- `test_breakdown_generator_mean_ttf_matches_target` — for each `A_k ∈ {0.95, 0.90, 0.80}`, empirical mean of `draw_ttf` over N draws within 5% of `MTBF_k` computed per §4.1.
- `test_breakdown_generator_mean_ttr_matches_target` — empirical mean of `draw_ttr` within 5% of `MTTR` (independent of `A_k`, since MTTR is fixed by `mttr_multiplier` alone).
- `test_breakdown_generator_ttf_shape_ks` — one KS test confirming `draw_ttf` samples are consistent with `Exp(lambda_k)` (not just matching the mean).
- `test_breakdown_generator_availability_recovers_target` — simulate a long renewal process (many TTF/TTR draws in sequence) and confirm empirical `MTBF/(MTBF+MTTR)` recovers `A_k` within tolerance.
- `test_processing_time_sampler_mean_is_one` — empirical mean of `draw_multiplier` within 5% of `1.0`, for each `s ∈ {0.1, 0.25, 0.5}`.
- `test_processing_time_sampler_variance_scales_with_s` — empirical variance increases monotonically with `s` and is within 10% of the theoretical `Lognormal` variance `(exp(s**2)-1)*exp(s**2)` (since mean is fixed at 1, `exp(2*mu+s^2)=1` reduces this to `(exp(s**2)-1)`... — verify the exact closed form in the implementation docstring rather than hardcoding it in two places).
- `test_arrival_process_rate_matches_target_utilization` — over a long simulated horizon, empirical `lambda_a` (inter-arrival rate) within 5% of the value computed by the calibration formula in §4.3, for each `rho ∈ {0.70, 0.85, 0.95}`.
- `test_arrival_process_stops_at_cutoff` — no `ARRIVAL` events generated past `arrival_cutoff_time`, and never more than `max_dynamic_jobs` total.
- `test_generators_independent_of_seed_spawn` — toggling `processing_time_sigma` from 0 to nonzero does not change the sequence of breakdown times drawn (confirms the independent RNG streams from `spawn()`, §4 intro).

**`test_dynamic_env.py`**
- `test_default_config_matches_static_env_exactly` — **the regression test**: `DynamicJSSPEnv(instance, DisruptionConfig())` (all-defaults/off) run against a fixed seed produces byte-identical `(obs, reward, terminated, info["makespan"])` trajectories to `JSSPEnv(instance)` under the same dispatch sequence. This is the single most important test in this sprint — it proves the extension is additive, not a silent behavior change.
- `test_precedence_preserved_under_breakdown` — reuse `is_schedule_feasible` (existing, unchanged) against the final schedule of episodes run with breakdowns enabled; must still report zero violations.
- `test_no_machine_double_booking_under_breakdown` — same, for the machine-capacity constraint, specifically checking that no operation is ever scheduled to run *during* its machine's broken interval (a check not covered by the existing `is_schedule_feasible`, since that function has no notion of broken intervals — extend it or add a dedicated dynamic-aware feasibility checker here; do not silently rely on the static checker being sufficient).
- `test_preempted_operation_resumes_with_correct_remaining_duration` — force a breakdown mid-operation via a mocked/injected event, assert `remaining_duration` bookkeeping and the resumed `OP_COMPLETE` time are exactly as computed in §3.3–§3.4.
- `test_action_mask_excludes_broken_machine_frontier_ops` — a frontier operation whose machine is broken must have `mask == 0`, even though it would be `1` under the static-only masking rule.
- `test_dynamic_job_extends_graph_correctly` — after an `ARRIVAL` event, the new job's operations appear in `node_features`/`adj_precedence`/`adj_machine` with correct precedence chains and correct machine-sharing edges to pre-existing unscheduled operations.
- `test_episode_terminates_with_dynamic_arrivals` — an episode with arrivals enabled still reaches `terminated=True` in finite steps, and only after all arrived jobs (not just the original instance's jobs) are complete.
- `test_episode_terminating_system_bounded_length` — episode step count stays under a generous fixed upper bound across `N` random seeds, confirming the cutoff/cap in §4.3 prevents runaway episodes (guards against a future regression where cutoff logic is accidentally bypassed).
- `test_rule_rollout_is_always_feasible_dynamic` — dynamic-environment analogue of the existing static test of the same name (`tests/test_env.py`): run all six dispatching rules (`baselines/dispatching_rules.py`, unchanged) against `DynamicJSSPEnv` with a fixed disruption config and seed, and confirm every resulting schedule is feasible per the dynamic-aware checker above. This is the cheapest whole-environment sanity net, mirroring the design rationale already documented in `baselines/dispatching_rules.py`.

---

## 9. Acceptance criteria / definition of done

1. All tests in §8 pass, plus the full existing suite (188 tests, `tests/test_instance.py` through `tests/test_policy.py`) continues to pass unmodified — **zero regressions** in the static path.
2. `test_default_config_matches_static_env_exactly` passes — the additive-not-rewritten contract is empirically proven, not just asserted in a docstring.
3. `VALIDATION_WEEK6.md` is written, following the same structure as `VALIDATION_WEEK1.md`, and must report: (a) the statistical validation results from `test_disruptions.py` with actual empirical-vs-theoretical numbers (not just "passed"); (b) confirmation that all hard constraints (precedence, no double-booking, no scheduling during a broken interval) hold across at least the same rollout protocol used in Week 1 (all six PDRs × multiple instances × multiple seeds); (c) confirmation that episodes terminate correctly with dynamic arrivals, including at least one example event log (`info["event_log"]`) transcribed for a short episode so a human reviewer can sanity-check the event sequence by eye.
4. No changes to `models/gin.py`, `models/policy.py`, `experiments/train_ft06.py`, or the public signatures of anything in `envs/instance.py`, `envs/jssp_env.py`, `envs/disjunctive_graph.py` (only additions, e.g. the new `estimated_makespan_dynamic` function, are permitted in the last file).

---

## 10. Open design decisions this spec resolves (flagged for thesis committee visibility)

Part D1/D5 of the framework document specifies the event *types*, the disruption *distributions*, and the *calibration targets* (`A_k`, `s`, `rho` value sets) precisely, but leaves several implementation-level choices genuinely open. This spec makes a concrete, justified choice for each so the coding agent has no ambiguity; each should be called out explicitly in `../../archive/capitulos/Section4.2.md`/Chapter 4 prose as a stated modeling assumption, not silently baked in:

1. **TTR shape parameter `σ_r`** — not specified by D5 beyond "Lognormal"; this spec fixes `σ_r = 0.5` by default (configurable), justified only by "moderate variability," not by a cited empirical source. Flag as an assumption; a sensitivity check across a couple of `σ_r` values would strengthen the eventual robustness experiments (Part D4, E4) if time permits.
2. **Content model for dynamically arriving jobs** (§4.3) — D1/D5 specify the arrival *process*, not what an arriving job's operations look like. This spec's bootstrap-resampling choice is a reasonable default but is not derived from the framework document and should be defended on its own merits in the thesis text.
3. **Arrival cutoff / termination safety** (§4.3) — required for the system to be terminating (§4.2.1 of ../../archive/capitulos/Section4.2.md) but not explicitly specified in D1/D5; this spec's `0.8 × H_nominal` default is an engineering choice, not a value derived from the framework document.
4. **Preemption policy** (§5) — D1 explicitly left this open ("según supuesto"); this spec resolves it to preempt-resume, with the justification in §5 intended to be reused directly in the thesis prose.
5. **Once-per-dispatch sampling of the processing-time multiplier** (§4.2) — a timing choice (when exactly the random draw happens) not made explicit in D5's distributional specification.

None of these five choices contradicts anything explicitly stated in Part D1/D5 — they fill genuine gaps, and are flagged here precisely so that filling them isn't mistaken, during the thesis defense, for having been dictated by the original framework document.
