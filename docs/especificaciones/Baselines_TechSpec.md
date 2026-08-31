# Baselines Technical Specification v1.0
### A common `Baseline` interface, the full 9-rule PDR suite, a genetic algorithm, and a CP-SAT wrapper for JSSP/FJSP

**Audience:** a coding agent implementing this directly on top of the existing, already-tested repository (`envs/`, `baselines/`, `tests/`; see `VALIDATION_WEEK1.md` and `VALIDATION_WEEK6.md` for current state — 315/315 tests passing, static + dynamic environments both validated).

**Prime directive: EXTEND, DO NOT SILENTLY REWRITE.** `envs/jssp_env.py`, `envs/dynamic_jssp_env.py`, `envs/disjunctive_graph.py`, `envs/instance.py`, `models/gin.py`, `models/policy.py` are validated and frozen for behavior already covered by existing tests. `baselines/dispatching_rules.py` carries its own docstring note — *"Week 9 of the plan expands this module with the full PDR suite and a benchmarking harness"* — so this spec **is** that expansion and editing it is in scope, subject to one hard constraint: **every existing public name (`random_rule`, `spt_rule`, `lpt_rule`, `fifo_rule`, `mwkr_rule`, `mopnr_rule`, `RULES`, `run_rule`, `DispatchingRule`) must keep byte-identical behavior**, verified by a regression test (§7) before anything else is added. `baselines/cpsat.py` (`solve_cpsat`, `CPSATResult`, `dispatch_order_from_starts`) keeps its existing signature; FJSP support is additive (new function/parameter, not a rewrite of the JSSP path).

**Source-of-truth cross-reference:** this spec formalizes Part B6 (dispatching rules) and Part B7 (GA, CP-SAT) of the thesis framework document (`../docs/planificacion/Step 4 Thesis.md`) and is the implementation basis for thesis Section 4.6 (`../../archive/capitulos/Section4.6.md`, companion document, already written). Where this spec makes a concrete engineering choice not pinned down by B6/B7 (e.g., how to synthesize due dates for EDD/CR on benchmarks that have none, or how to define "current time" for CR inside a clockless constructive environment), that choice is flagged explicitly in §9 as an open decision this spec resolves, not restates.

---

## 0. Scope

**In scope for this sprint:**
- A common `Baseline` protocol + `BaselineResult` dataclass (`baselines/interface.py`, NEW) that every method below implements.
- The remaining four PDRs (LWKR, LOR, EDD, CR) needed to complete the nine required by Part B6, added to `baselines/dispatching_rules.py`, **arrival-safe by construction** (unlike the three flagged in `VALIDATION_WEEK6.md` decision #2).
- A due-date synthesis module (`baselines/due_dates.py`, NEW) required by EDD/CR, since the benchmark families used in this thesis (`ft`/`la`/`ta`) carry no due dates.
- A genetic algorithm solver (`baselines/ga.py`, NEW) for JSSP (and FJSP-ready via the machine-assignment chromosome, even though FJSP instance loading itself is out of scope — see `envs/instance.py`'s own extension note).
- FJSP-capable CP-SAT support added to `baselines/cpsat.py` (`OptionalIntervalVar` path), gated so the existing JSSP path (`solve_cpsat`) is untouched.
- A benchmarking harness (`baselines/harness.py`, NEW) that runs any `Baseline` over a set of instances/seeds and emits a results table feeding directly into the Section D6 statistical procedure.
- New unit tests (`tests/test_dispatching_rules.py`, `tests/test_due_dates.py`, `tests/test_ga.py`, `tests/test_baselines_interface.py`) + extension of `tests/test_env.py`/`tests/test_dynamic_env.py`'s existing `RULES`-parametrized tests to cover the new rules automatically.

**Out of scope for this sprint (explicitly deferred):**
- Loading actual FJSP instance files (`MK01`–`MK10`, Hurink) — `envs/instance.py` models fixed routing only; the GA's machine-assignment chromosome and CP-SAT's `OptionalIntervalVar` path are built now so they are ready the moment FJSP instance loading lands, but are only exercised in this sprint's tests via a synthetic ragged-eligibility fixture, not real MK/Hurink files.
- Wiring the benchmarking harness into the Chapter 5/6 experimental-matrix runner (Parts D4/D6) — this sprint produces the harness and its output schema; the runner that sweeps the full E1–E8 matrix is a later sprint.
- Any change to the reward function, GNN encoder, or PPO training pipeline.

---

## 1. Module layout

```
baselines/
  __init__.py             # EXTEND — export the new public names (see §8)
  interface.py             # NEW — Baseline Protocol, BaselineResult dataclass
  dispatching_rules.py     # EXTEND — add lwkr_rule, lor_rule, edd_rule, cr_rule; extend RULES;
                            # do not change any existing function's behavior (regression-tested, §7)
  due_dates.py              # NEW — synthesize d_j (TWK rule), needed by edd_rule/cr_rule
  ga.py                      # NEW — GeneticAlgorithm solver, POX/uniform crossover, swap/reassignment mutation
  cpsat.py                   # EXTEND — add solve_cpsat_fjsp (OptionalIntervalVar), keep solve_cpsat unchanged
  harness.py                  # NEW — run_baseline(s) over instances x seeds, results DataFrame/records
envs/
  jssp_env.py                # EXTEND (additive only) — add JSSPEnv.remaining_work_per_op(), see §9.3
  dynamic_jssp_env.py          # EXTEND (additive only) — add matching DynamicJSSPEnv.remaining_work_per_op()
tests/
  test_dispatching_rules.py    # NEW — formula-level correctness + arrival-safety for all 9 PDRs
  test_due_dates.py             # NEW — TWK synthesis correctness, arrival-safety
  test_ga.py                     # NEW — encoding/decoding round-trip, operator correctness, convergence sanity
  test_baselines_interface.py     # NEW — every Baseline implementation satisfies the common protocol
  test_env.py                      # EXTEND — RULES-parametrized tests auto-cover the 4 new rules (no code change
                                     # needed beyond what RULES already triggers, but verify explicitly, §7)
  test_dynamic_env.py               # EXTEND — same auto-coverage check for the dynamic env
```

---

## 2. Common `Baseline` interface (`baselines/interface.py`)

### 2.1 `BaselineResult`

```python
@dataclass(frozen=True)
class BaselineResult:
    """Uniform output of any baseline method, feeding directly into the Section D6 metrics."""
    instance: str                      # instance.name
    method: str                        # e.g. "spt", "ga", "cpsat", "cpsat_fjsp"
    start_times: np.ndarray            # shape (num_operations,), same convention as JSSPEnv.start_times
    makespan: float
    wall_clock_time: float             # seconds, wall-clock, measured by the adapter itself
    feasible: bool                     # result of an independent is_schedule_feasible(_dynamic) check
    violations: list[str]              # empty iff feasible
    status: str                        # "OK" (PDR/GA always), or CP-SAT's solver status ("OPTIMAL"/"FEASIBLE"/...)
    is_optimal: bool                   # True only when status == "OPTIMAL" (CP-SAT); False for PDR/GA always
    extra: dict = field(default_factory=dict)  # method-specific detail (GA: generations run, best-fitness curve;
                                                 # PDR: dispatch_order; CP-SAT: best_bound)
```

**Design rule:** `start_times` is the single canonical representation of a schedule across all four methods (PDR, GA, CP-SAT, and — for comparison purposes only — the DRL policy). Every adapter, regardless of its internal representation (a chromosome, a CP-SAT interval model, an in-env dispatch sequence), must convert to this array before returning. This is what lets `harness.py` call `is_schedule_feasible(instance, result.start_times)` identically for every method, exactly mirroring how `baselines/cpsat.py::solve_cpsat` already self-verifies today.

### 2.2 `Baseline` protocol

```python
class Baseline(Protocol):
    name: str  # short identifier used as BaselineResult.method and in result tables, e.g. "mwkr", "ga", "cpsat"

    def solve(
        self,
        instance: JSSPInstance,
        *,
        seed: int | None = None,
        time_limit_s: float | None = None,
    ) -> BaselineResult:
        """Solve `instance` and return a BaselineResult.

        Args:
            instance: The instance to solve (base instance only this sprint; a DynamicJSSPEnv
                variant is out of scope for GA/CP-SAT, see ../../archive/capitulos/Section4.6.md §4.6.4 point 2).
            seed: Governs any stochastic tie-breaking (PDR) or population initialization (GA).
                Must be None-safe (falls back to entropy) exactly like envs/disruptions.py's
                spawn_disruption_rngs.
            time_limit_s: Wall-clock budget. Required (non-None) for GA and CP-SAT — solve()
                must raise ValueError if time_limit_s is None for those two. Ignored by PDR
                adapters (a single deterministic pass is not meaningfully time-limited) but
                still accepted in the signature so every Baseline is call-compatible.
        """
        ...
```

Every concrete baseline below is implemented as a small adapter class satisfying this protocol — `PDRBaseline(rule_name: str)`, `GABaseline(**ga_config)`, `CPSATBaseline(fjsp: bool = False, **solver_config)` — so `harness.py` never special-cases a method by type.

---

## 3. The nine priority dispatching rules (`baselines/dispatching_rules.py`)

### 3.1 Formal definitions (Part B6, verbatim)

For the candidate frontier operation $O_{j,h}$ at decision time, with $p_{j,h}$ its duration, $\sum_{h'\ge h} p_{j,h'}$ its job's remaining work, and $n_j$ its job's total operation count:

| Rule | Key (minimized unless noted) | Status |
|---|---|---|
| SPT | $p_{j,h}$ | existing (`spt_rule`) — **do not touch** |
| LPT | $-p_{j,h}$ (i.e. maximize $p_{j,h}$) | existing (`lpt_rule`) — **do not touch** |
| MWKR | $-\sum_{h'\ge h}p_{j,h'}$ | existing (`mwkr_rule`) — **do not touch** |
| LWKR | $\sum_{h'\ge h}p_{j,h'}$ | **NEW** (`lwkr_rule`) |
| MOR | $-(n_j-h+1)$ | existing under the name `mopnr_rule` — **keep the name, add `"mor"` as a registry alias, do not rename the function** |
| LOR | $(n_j-h+1)$ | **NEW** (`lor_rule`) |
| FIFO | `job_ready_time[job]` | existing (`fifo_rule`) — **do not touch** |
| EDD | $d_j$ (synthesized, §4) | **NEW** (`edd_rule`) |
| CR | $(d_j - t_{\text{now}}) / \sum_{h'\ge h}p_{j,h'}$ | **NEW** (`cr_rule`) — see §9.2 for the definition of $t_{\text{now}}$ |

Random tie-breaking uses the existing `_argmin_with_tiebreak` helper unchanged for every new rule — do not introduce a second tie-breaking implementation.

### 3.2 Mandatory arrival-safety (fixes `VALIDATION_WEEK6.md` decision #2 at the root, not per-rule)

`spt_rule`, `lpt_rule`, and `mwkr_rule` are arrival-unsafe because they index `env.instance.flat_durations` / `env.instance.remaining_work()` — fixed-shape arrays sized to the **base** instance — directly by global operation id, which cannot cover an operation belonging to a dynamically arrived job. `random_rule`, `fifo_rule`, and `mopnr_rule` are safe because they only use `env.action_masks()`, `env.job_ready_time`, and pure `job_of`/`position_of` arithmetic (no fixed-shape array indexed by op id).

**All four new rules (LWKR, LOR, EDD, CR) must follow the safe pattern, not the unsafe one.** LOR is trivial (pure `job_of`/`position_of` arithmetic, exactly like `mopnr_rule`). LWKR and CR need "remaining work of operation `op`", which today only exists as `env.instance.remaining_work()` (base-instance-only). Resolve this by adding one small, purely additive public method to **both** environments (§9.3):

```python
# envs/jssp_env.py (JSSPEnv) — additive method, delegates to the existing instance method
def remaining_work_per_op(self) -> np.ndarray:
    """Remaining work of every operation, indexed by global op id. Shape (num_operations,)."""
    return self.instance.remaining_work().ravel()

# envs/dynamic_jssp_env.py (DynamicJSSPEnv) — additive method, computed from the growable arrays
def remaining_work_per_op(self) -> np.ndarray:
    """Same contract as JSSPEnv.remaining_work_per_op(), but covers arrived jobs too.
    Shape (active_ops,); computed as a per-job reverse-cumsum over self._flat_durations
    grouped by self._job_index, mirroring JSSPInstance.remaining_work()'s semantics exactly."""
```

`lwkr_rule` and `cr_rule` then call `env.remaining_work_per_op()[legal]` — **never** `env.instance.remaining_work()` directly — which makes them work unmodified on both `JSSPEnv` and `DynamicJSSPEnv`, including for dynamically-arrived jobs. This is a one-line delegation on the static side (zero behavior change, so it does not violate the "frozen" status of `jssp_env.py` in any way that matters — it is exactly the same kind of pure-additive extension already used for `estimated_makespan_dynamic` in `disjunctive_graph.py` during the previous sprint) and a genuinely new but small computation on the dynamic side.

**Verification requirement:** add `lwkr`, `lor`, `edd`, `cr`, and `mor` (the new alias) to `RULES`, then confirm — as an explicit assertion in `tests/test_dynamic_env.py`, not just incidentally — that `TestDynamicArrivals`-style rollouts with all nine rules produce feasible schedules with at least one arrival actually occurring (mirroring `test_episode_terminates_with_dynamic_arrivals`). If any new rule cannot be made arrival-safe this way, do **not** add it to the shared `RULES` dict (which is auto-parametrized by `test_env.py`/`test_dynamic_env.py`, §1) — register it instead in a separate `STATIC_ONLY_RULES` dict and document why, following the exact precedent of decision #2.

---

## 4. Due-date synthesis (`baselines/due_dates.py`)

### 4.1 The gap

Part B2.1's MILP formulation carries a due date $d_j$ per job, and Part B6 defines EDD/CR in terms of it, but `envs/instance.py::JSSPInstance` has no due-date field — the `ft`/`la`/`ta` benchmark families are pure-makespan instances with none published. EDD and CR are therefore **not directly computable** from the data as-is; this module fills that gap explicitly, rather than silently.

### 4.2 Total Work Content (TWK) rule

```python
def synthesize_due_dates(
    job_totals: np.ndarray,          # per-job total processing time, e.g. instance.job_totals
    release_times: np.ndarray,        # per-job release date r_j; zeros for the base instance
    tightness: float = 1.5,           # k; configurable, see §9.1 for the default's justification
) -> np.ndarray:
    """d_j = r_j + tightness * job_totals[j]  (Total Work Content / TWK rule)."""
```

- For the base (static) instance, `release_times` is all-zeros (every job is available at $t=0$), so $d_j = k\cdot\sum_h p_{j,h}$.
- For a dynamically arrived job, `release_times[j]` is that job's actual arrival time (`info["event_log"]` entry `("arrival", t, j)`, Section 4.5.3), so the synthesized due date is relative to when the job actually became available, not to the episode start — this is what keeps `edd_rule`/`cr_rule` well-defined and arrival-safe: due dates are computed **incrementally**, once per job (at the moment it becomes known — `reset()` for base jobs, the `ARRIVAL` event for dynamic ones), not precomputed once against a fixed-size array.
- `tightness=1.5` is this spec's default (moderately tight — a job's due date is 50% later than its total processing time, in the same "reasonable default, not a cited empirical value" spirit as `envs/disruptions.py`'s `ttr_shape_sigma=0.5`). Flag it explicitly in `../../archive/capitulos/Section4.6.md`/Chapter 4 prose as a modeling assumption (§9.1), and expose it as a configurable parameter on `edd_rule`/`cr_rule` (via a small `DueDateConfig` dataclass, not a hardcoded constant) so a sensitivity sweep over `tightness` is a one-line change later.

### 4.3 Caching

Compute the due date for a job exactly once (at `reset()` for base jobs, at the `ARRIVAL` event handler for dynamic ones — the natural extension point, symmetric with how `ArrivalProcess.generate_job()` is called exactly once per arrival) and cache it on the environment (e.g. `env.due_dates: np.ndarray`, sized like `job_ready_time`), rather than recomputing it from `job_totals` on every `edd_rule`/`cr_rule` call — recomputation is not wrong, just wasteful, and caching keeps the due date stable across the life of the episode even if `job_totals`-equivalent bookkeeping changes shape as jobs arrive.

---

## 5. Genetic algorithm (`baselines/ga.py`)

### 5.1 Chromosome encoding

```python
@dataclass
class Chromosome:
    sequence: np.ndarray   # shape (num_operations,); job-index repeated ops_per_job[j] times;
                            # the r-th occurrence of job j designates that job's r-th operation
    assignment: np.ndarray | None  # shape (num_operations,); eligible-machine index per operation;
                                     # None for classic JSSP (fixed routing — omit the gene entirely,
                                     # do not store a length-0-choice array)
```

- **JSSP (this sprint's actual test data, `ft`/`la`/`ta`):** `assignment=None`; the decoder reads the machine straight from `instance.machines`.
- **FJSP-ready path:** `assignment` indexes into `instance.eligible_machines[j][h]` (or equivalent ragged structure once FJSP instance loading lands) — implement the decoder and the assignment-vector operators now (against a small synthetic fixture with a hand-built ragged eligibility list in `tests/test_ga.py`) so no GA code needs to change when FJSP instances arrive; only the instance loader is out of scope.

### 5.2 Decoding

Left-to-right scan of `sequence`; for each occurrence of job `j`, dispatch its next operation on the machine given by `instance.machines[j, pos]` (JSSP) or `eligible_machines[j][pos][assignment[op_id]]` (FJSP) at `start = max(job_ready_time[j], machine_free_time[machine])` — **the same list-scheduling rule `JSSPEnv.step()` already uses** (Section 4.3.2), so every decoded chromosome is guaranteed semi-active-feasible by construction and cross-checkable against `envs.disjunctive_graph.is_schedule_feasible` with zero special-casing.

### 5.3 Crossover

- **POX (Precedence Operation Crossover)** on `sequence`: partition `range(num_jobs)` uniformly at random into $J_1, J_2$; offspring inherits, in original position, every gene belonging to $J_1$ from parent A, and fills the remaining positions with $J_2$'s genes in the order they appear in parent B. Implement as a pure function `pox(seq_a, seq_b, j1, rng) -> child_seq`, unit-tested for the invariant that the child is always a valid permutation-with-repetition (same per-job operation counts as the parents) — this invariant is what guarantees precedence feasibility without a repair step.
- **Uniform crossover** on `assignment` (FJSP only, no-op when `assignment is None`): per-gene independent 50/50 inheritance from either parent.

### 5.4 Mutation

- **Swap mutation** on `sequence`: pick two positions uniformly at random and swap their values.
- **Reassignment mutation** on `assignment` (FJSP only): pick an operation uniformly at random, replace its assigned machine with a different uniformly-random eligible machine.
- Apply each independently with its own mutation probability (`p_mutation_sequence`, `p_mutation_assignment`, both configurable, defaults `0.1`–`0.2` per standard GA practice — not derived from B6/B7, flag as a tunable default like the others in §9).

### 5.5 Selection

Tournament selection (`tournament_size: int = 3`, configurable) for the bulk of the next generation, plus **elitism** (`elite_count: int = 2`, configurable): the `elite_count` best individuals of generation $g$ are copied unmutated into generation $g+1$, guaranteeing the best-found makespan is monotone non-increasing across generations — assert this monotonicity directly in `tests/test_ga.py` as a regression guard (a bug that lets the elite slip is a correctness bug, not a tuning issue).

### 5.6 Population size, generations, and the wall-clock-first stopping rule

Per `../../archive/capitulos/Section4.6.md` §4.6.2 and the thesis's own scope constraint (single student, single GPU, nine months — `../docs/planificacion/Step 4 Thesis.md` §1.5), **the primary stopping criterion is wall-clock time (`time_limit_s`, required by the `Baseline` protocol), not a fixed generation count.** Implement the GA loop as:

```python
def solve(self, instance, *, seed=None, time_limit_s=None) -> BaselineResult:
    if time_limit_s is None:
        raise ValueError("GABaseline.solve requires an explicit time_limit_s")
    deadline = time.monotonic() + time_limit_s
    ... initialize population ...
    generations_run = 0
    while time.monotonic() < deadline:
        ... one generation: selection, crossover, mutation, elitism ...
        generations_run += 1
    # report generations_run in BaselineResult.extra, so the Chapter 6 discussion can
    # correlate "how many generations fit in the time budget" with instance size
```

**Recommended defaults** for population size, to be treated as a starting configuration validated empirically against the chosen `time_limit_s`, not a fixed law: `population_size=80` for training-scale instances (6×6, 10×10), reduced toward `population_size=40` for large generalization-test instances (Part D4/E3, up to 100×20) where each fitness evaluation (a full chromosome decode) costs more — the exact crossover point should be calibrated once against the actual `time_limit_s` chosen for CP-SAT (§6), not guessed twice independently.

### 5.7 Fitness

`fitness(chromosome) = -makespan(decode(chromosome))`. No partial/incremental fitness evaluation this sprint — full decode per individual per generation; flag incremental decoding as a future optimization only if profiling shows the GA is generation-starved relative to CP-SAT's time budget on the largest instances (Part D4/E3).

---

## 6. CP-SAT wrapper (`baselines/cpsat.py`)

### 6.1 JSSP path — unchanged

`solve_cpsat` keeps its exact current signature and model (`IntervalVar` per operation, `AddNoOverlap` per machine, chained `start(next) >= end(prev)` per job, `AddMaxEquality` + `Minimize` on makespan, self-verification against `is_schedule_feasible`). Do not modify it; wrap it behind `CPSATBaseline(fjsp=False)` in `interface.py` without touching its internals.

### 6.2 FJSP path — new, additive

```python
def solve_cpsat_fjsp(
    instance: "FJSPInstance",   # ragged eligible-machines structure; see §0 scope note
    time_limit_s: float = 60.0,
    num_workers: int = 8,
    log_progress: bool = False,
) -> CPSATResult:
    """Solve an FJSP instance (routing + sequencing) with CP-SAT."""
```

For each operation $O_{j,h}$ and each eligible machine $k\in M_{j,h}$: create an `OptionalIntervalVar(start, duration, end, presence_literal)` bound to a fresh boolean `x_{j,h,k}`. Constrain `sum(x_{j,h,k} for k in M_{j,h}) == 1` (exactly one alternative present — the CP-SAT translation of Part B2.1's assignment constraint, without needing the MILP's explicit big-M disjunctive pair). Group the optional intervals by machine and call `AddNoOverlap` per machine exactly as in the JSSP path — CP-SAT automatically ignores intervals whose presence literal is false, so the same constraint call handles routing flexibility with no extra bookkeeping. Precedence (`start(next) >= end(prev)`) uses each operation's *actual* start/end variables (defined regardless of which alternative is chosen), unchanged from the JSSP path. Reuse `CPSATResult` unchanged as the return type; the extra routing decision (`x_{j,h,k}`) is not part of the reported schedule, only the resulting `start_times`, since `BaselineResult.start_times` (§2.1) is the canonical cross-method representation and does not need per-alternative bookkeeping once a solution is found — recover the chosen machine per operation, if needed for diagnostics, via `solver.Value(x_{j,h,k})` and store it in `BaselineResult.extra["machine_assignment"]`, not in the core schedule representation.

**`AddCumulative` note (documented extension point, not implemented this sprint):** for the capacity-1-per-machine case used throughout this thesis, `AddNoOverlap` and `AddCumulative(intervals, demands=[1]*n, capacity=1)` are equivalent; this spec uses `AddNoOverlap` (simpler, and consistent with the already-validated JSSP path) and only documents `AddCumulative` as the natural generalization if a future extension groups identical machines into a capacity->1 pooled resource — do not implement a `AddCumulative` code path this sprint; a stub docstring note in `solve_cpsat_fjsp` pointing to this spec section is sufficient.

### 6.3 Time limit and status reporting

Both `solve_cpsat` and `solve_cpsat_fjsp` require an explicit `time_limit_s` (already the case for the former; keep the same default `60.0` for consistency) and must report `status` (`"OPTIMAL"`/`"FEASIBLE"`/`"INFEASIBLE"`/`"UNKNOWN"`) and `is_optimal` distinctly — the `CPSATBaseline` adapter in `interface.py` copies these straight into `BaselineResult.status`/`is_optimal` without collapsing the distinction, since Section D6's RPD calculation depends on knowing whether the comparison denominator is a proven optimum or only a proven lower bound (`best_bound`).

---

## 7. Required unit tests

**`tests/test_dispatching_rules.py`** (NEW)
- `test_new_rules_formula_correctness` — for a small hand-built instance with known remaining-work/operation-count values, assert `lwkr_rule`/`lor_rule` pick the operation the closed-form definition (§3.1) predicts, for at least two decision epochs (not just the first).
- `test_mor_alias_matches_mopnr` — `RULES["mor"]` and `RULES["mopnr"]` (or the renamed canonical entry, whichever direction is chosen) produce byte-identical dispatch sequences on the same seed/instance.
- `test_existing_rules_byte_identical` — **the regression test**: `random_rule`, `spt_rule`, `lpt_rule`, `fifo_rule`, `mwkr_rule`, `mopnr_rule` produce byte-identical dispatch sequences, makespans, and returns to a snapshot recorded from the pre-change module (fixture: a frozen JSON/CSV of `run_rule` output for `ft06`/`la01`, seeds 0–4, generated once before starting this sprint's edits). This is the single most important test in this sprint, mirroring `test_default_config_matches_static_env_exactly`'s role in the previous one.
- `test_edd_cr_use_synthesized_due_dates` — `edd_rule`/`cr_rule` raise a clear error (not a silent `AttributeError`) if `env.due_dates` has not been populated, and produce the expected ranking on a hand-built instance with known synthesized due dates.
- `test_new_rules_arrival_safe` — parametrized over `{"lwkr", "lor", "edd", "cr", "mor"}`: run each through `DynamicJSSPEnv` with arrivals enabled (mirroring `TestDynamicArrivals::test_episode_terminates_with_dynamic_arrivals`) and assert termination + feasibility + at least one arrival occurred.

**`tests/test_due_dates.py`** (NEW)
- `test_twk_formula` — `synthesize_due_dates` matches the closed form exactly for hand-computed inputs.
- `test_due_dates_use_actual_release_time_for_arrivals` — a job's synthesized due date equals `arrival_time + tightness * job_total`, not `tightness * job_total` alone, for a dynamically arrived job.

**`tests/test_ga.py`** (NEW)
- `test_decode_is_always_precedence_feasible` — for 200 random chromosomes on `ft06`, decoding never violates precedence (checked via `is_schedule_feasible`).
- `test_pox_preserves_per_job_operation_counts` — POX offspring is always a valid permutation-with-repetition.
- `test_elitism_makespan_monotone_non_increasing` — best-of-generation makespan never increases across generations, over 5 seeds.
- `test_ga_respects_time_limit` — `solve()` returns within `time_limit_s + <small fixed slack>` for a range of `time_limit_s` values.
- `test_ga_beats_random_rule` — sanity floor: GA's makespan on `ft06` with a generous time budget (e.g. 10s) is no worse than `random_rule`'s median over 20 seeds — a GA that cannot beat random dispatch indicates a decoder or operator bug, not a tuning issue.

**`tests/test_baselines_interface.py`** (NEW)
- `test_all_baselines_satisfy_protocol` — `PDRBaseline`, `GABaseline`, `CPSATBaseline` each produce a `BaselineResult` with `start_times` that passes `is_schedule_feasible`, for `ft06`.
- `test_ga_cpsat_require_time_limit` — `GABaseline.solve`/`CPSATBaseline.solve` raise `ValueError` when `time_limit_s=None`; `PDRBaseline.solve` does not (accepts and ignores it).
- `test_cpsat_status_and_optimal_flag_consistent` — `is_optimal=True` implies `status=="OPTIMAL"`, and never the reverse implication's negation (i.e. no combination where `is_optimal=True` but `status!="OPTIMAL"`).

**Extensions to existing files**
- `tests/test_env.py` / `tests/test_dynamic_env.py`: no new test functions strictly required (the existing `@pytest.mark.parametrize("rule_name", list(RULES.keys()))` tests already pick up new `RULES` entries automatically), but add one explicit assertion per file confirming `len(RULES) == 10` (nine PDRs + `random`) after this sprint, as a tripwire against a new rule silently failing to register.

---

## 8. `baselines/__init__.py` — updated public exports

```python
from baselines.cpsat import CPSATResult, dispatch_order_from_starts, solve_cpsat, solve_cpsat_fjsp
from baselines.dispatching_rules import RULES, STATIC_ONLY_RULES, run_rule
from baselines.due_dates import synthesize_due_dates, DueDateConfig
from baselines.ga import GeneticAlgorithm, Chromosome, GAConfig
from baselines.interface import Baseline, BaselineResult, PDRBaseline, GABaseline, CPSATBaseline
from baselines.harness import run_baseline_suite

__all__ = [
    "RULES", "STATIC_ONLY_RULES", "CPSATResult", "dispatch_order_from_starts", "run_rule",
    "solve_cpsat", "solve_cpsat_fjsp", "synthesize_due_dates", "DueDateConfig",
    "GeneticAlgorithm", "Chromosome", "GAConfig",
    "Baseline", "BaselineResult", "PDRBaseline", "GABaseline", "CPSATBaseline",
    "run_baseline_suite",
]
```

(`STATIC_ONLY_RULES` is only exported if §3.2's arrival-safety check forces any rule out of the shared registry; if all nine pass, this dict is empty and the export is kept for forward-compatibility rather than removed.)

---

## 9. Open design decisions this spec resolves (flagged for thesis committee visibility)

Part B6/B7 specifies the rule *formulas*, the GA's *operator families*, and CP-SAT's *modeling style* precisely, but leaves several implementation-level choices genuinely open — same spirit as `../docs/especificaciones/DynamicEnv_TechSpec.md` §10. Each is called out here so `../../archive/capitulos/Section4.6.md`/Chapter 4 prose can state it as a documented modeling assumption, not something silently baked in:

1. **Due-date tightness factor (`tightness=1.5` in the TWK rule).** B6 assumes $d_j$ exists; it does not specify how to synthesize one for benchmarks that lack it. This spec's TWK rule and its default tightness are a reasonable, literature-standard convention (total-work-content-based due dates are a common device in scheduling experiments that need EDD/CR on due-date-free benchmarks), not a value derived from B6 itself. A sensitivity sweep over `tightness` would strengthen the eventual EDD/CR results if time permits.
2. **Definition of "current time" $t_{\text{now}}$ for CR inside the static, clockless constructive environment.** `JSSPEnv` has no global simulation clock by design (each `step()` *is* a decision epoch; Section 4.3.2/`../../archive/capitulos/Section4.2.md`), so Part B6's $(d_j-t)$ is ill-defined as written for the static path. This spec resolves it as $t_{\text{now}} = \max(\text{job\_ready\_time}[j], \text{machine\_free\_time}[m])$ — the candidate operation's own earliest feasible start, i.e. the instant it *would* begin if dispatched — which is well-defined, deterministic, and requires no new environment state. Note this is a genuine departure from a literal reading of B6 (which implicitly assumes a live discrete-event clock, which the *dynamic* environment actually has); document both variants and, for `DynamicJSSPEnv`, prefer the real simulated clock (`env.clock`/equivalent) over this proxy where available, flagging the asymmetry explicitly in the thesis text.
3. **`remaining_work_per_op()` as a new additive method on both environments (§3.2).** Not required by B6/B7 directly, but the cleanest fix for the arrival-safety gap `VALIDATION_WEEK6.md` flagged in the PDR module; touches `envs/jssp_env.py`, nominally frozen, but as a pure zero-behavior-change delegation, consistent with the precedent set by `estimated_makespan_dynamic`'s addition to `disjunctive_graph.py` in the previous sprint. If the committee or a stricter reading of the freeze policy disallows touching `jssp_env.py` at all, the documented fallback is a module-level helper in `dispatching_rules.py` that duck-types on `hasattr(env, "remaining_work_per_op")` vs. falling back to `env.instance.remaining_work().ravel()` — functionally equivalent, slightly less clean.
4. **GA population size / generation count as a function of a wall-clock budget rather than a fixed schedule.** B7 does not specify population size or generation count at all; this spec ties both to the same fairness principle already established for CP-SAT (`../docs/planificacion/Step 4 Thesis.md`'s explicit "limitar CP-SAT y GA por tiempo de pared para comparaciones justas"), rather than inventing an independent generation budget that would not be comparable across instance sizes.
5. **Mutation probabilities (`p_mutation_sequence`, `p_mutation_assignment`) and tournament/elitism sizes.** Standard GA defaults (§5.4–5.5), not derived from B7; expose all as configuration, not constants, so Chapter 5/6's experimental matrix can treat them as nuisance parameters to calibrate once rather than hidden constants.

None of these five choices contradicts anything explicitly stated in Part B6/B7 — they fill genuine gaps, flagged here precisely so that filling them isn't mistaken, during the thesis defense, for having been dictated by the original framework document.
