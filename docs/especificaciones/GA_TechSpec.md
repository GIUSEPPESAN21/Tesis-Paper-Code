# Genetic Algorithm Hyperparameter Finalization — Technical Spec v1.0
### Concrete starting configuration and cheap tuning protocol for `baselines/ga.py`

**Audience:** a coding agent implementing `baselines/ga.py` next week, per `../docs/especificaciones/Baselines_TechSpec.md` §5 (encoding, operators, decoding — already specified) and §0 (explicitly deferred last sprint, per `BASELINES_RESULTS_WEEK9.md`). That spec fixed the operator *families* (POX/uniform crossover, swap/reassignment mutation, tournament+elitism) and the *principle* that population size and generation count should follow a wall-clock budget rather than a fixed schedule (§5.6), but left the actual starting numbers as "a reasonable configuration... calibrated empirically... not fixed a priori." **This document finalizes those numbers.**

**Source-of-truth cross-reference:** `../docs/especificaciones/Baselines_TechSpec.md` §5 (encoding/operators, unchanged by this document), `../../archive/capitulos/Section4.6.md` §4.6.2 (thesis prose already citing "50–100 individuals... a few hundred generations" as a provisional range), `../../archive/capitulos/Section5.1_5.2.md` §5.1.2 (the empirically validated wall-clock tiers this spec's recommendations are pinned to).

---

## 1. Hardware/compute context (why this isn't a generic GA config)

This thesis's compute budget is fixed: one student, one GPU, nine months (`../docs/planificacion/Step 4 Thesis.md` §1.5). The GA itself is **CPU-bound, not GPU-bound** — chromosome decoding (§`../docs/especificaciones/Baselines_TechSpec.md` §5.2) is a sequential list-scheduling scan with no matrix math worth putting on a GPU — which is a genuine practical advantage: the GA can run on CPU cores in parallel with (or in the gaps around) GPU-bound PPO training, rather than competing for the same resource. This spec's recommendations assume a single modern multi-core CPU (4–8 cores, no GPU acceleration for the GA itself) and are deliberately conservative about total evaluation count, since each "evaluation" is one full chromosome decode plus a feasibility check.

---

## 2. Recommended starting configuration

**Table 2.1.** Default `GAConfig`, to be used as-is for the first full run on the validated instance tiers (`../../archive/capitulos/Section5.1_5.2.md` Table 5.1.1: `ft06`, `la01`–`la10`, up to 75 operations) before any tuning.

| Parameter | Default | Rationale |
|---|---:|---|
| `population_size` | 80 | Matches the upper end of `../../archive/capitulos/Section4.6.md`'s provisional range; small enough that 80 full decodes complete in well under a second on `ft06`-scale instances (Baseline PDR wall-clock times, `BASELINES_RESULTS_WEEK9.md`, are ~5–27 ms for a *single* schedule — 80 decodes is a fraction of a second), leaving the wall-clock budget (§3) almost entirely to generations, not per-generation overhead. |
| `tournament_size` | 3 | Standard GA default (`../docs/especificaciones/Baselines_TechSpec.md` §5.5); low enough to preserve diversity, high enough to give consistent selection pressure. |
| `elite_count` | 2 | ~2.5% of population — enough to guarantee the monotonicity property tested in `test_ga.py::test_elitism_makespan_monotone_non_increasing`, small enough not to collapse diversity early. |
| `p_crossover` | 0.90 | POX applied to 90% of the mating pool; the remaining 10% are copied forward unmodified (standard permutation-GA practice — crossover is the primary search operator here, mutation is a secondary diversity-preserving one). |
| `p_mutation_sequence` (swap) | 0.15 | Per `../docs/especificaciones/Baselines_TechSpec.md` §5.4's "0.1–0.2" range; 0.15 as the midpoint default. Applied **per individual**, not per gene: with probability 0.15, one individual undergoes exactly one swap mutation. |
| `p_mutation_assignment` (reassignment, FJSP only) | 0.15 | Same rate as sequence mutation, applied independently; a no-op when `assignment is None` (classic JSSP, this sprint's actual evaluated data). |
| `immigrant_fraction` | 0.05 | **New in this spec, not in `../docs/especificaciones/Baselines_TechSpec.md` §5** — see §5 below (stagnation handling). |

### 2.1 Population-size scaling rule across instance-size tiers

`../docs/especificaciones/Baselines_TechSpec.md` §5.6 already directs population size to shrink for larger instances "toward `population_size=40`" without fixing where the transition happens. This spec fixes it as a simple, auditable rule tied to `../../archive/capitulos/Section5.1_5.2.md`'s existing size bands (Table 5.1.1), rather than a separate ad hoc threshold:

```python
def default_population_size(num_operations: int) -> int:
    """population_size, scaled down for larger instances.

    Bands match ../../archive/capitulos/Section5.1_5.2.md Table 5.1.1 exactly, so a change to the thesis's
    instance-size tiers and a change to this function are made in the same place
    conceptually, even though they live in different files.
    """
    if num_operations <= 100:   # "pequeña"/"moderada" bands, validated tiers
        return 80
    if num_operations <= 450:   # "intermedia" band (la11-la40), not yet executed
        return 60
    return 40                    # "grande" band (ta, up to 2000 ops), not yet executed
```

This is a **starting rule, not a tuned law** — exactly like the CP-SAT wall-clock tiers in `../../archive/capitulos/Section5.1_5.2.md` Table 5.1.2, the two untested bands are a documented extrapolation to be revisited once the intermediate band is actually run (same "measure before committing further budget" principle already established for CP-SAT).

---

## 3. Wall-clock budget (primary stopping criterion, per `../docs/especificaciones/Baselines_TechSpec.md` §5.6)

The GA's `time_limit_s` is **not** a GA-specific choice — it is set equal to CP-SAT's budget for the same instance and tier (`../../archive/capitulos/Section5.1_5.2.md` Table 5.1.2), for the fairness reason already established: 60s for the validated tiers (≤100 operations), 300s for the intermediate tier, 600–900s for the large tier. Do not introduce a separate, GA-only time budget; `GABaseline.solve(instance, time_limit_s=...)` receives the same value `CPSATBaseline.solve` would receive on the same instance.

**Expected generations reached at the recommended population size** (planning estimate, to confirm empirically on the first real run, not to hard-code as a target): a single generation for `population_size=80` on a 100-operation instance costs roughly $80\times(\text{one decode})$; PDR wall-clock times (§2, Table 2.1's rationale) put one decode at low tens of milliseconds for instances this size, so one generation costs on the order of 1–2 seconds, meaning a 60s budget should reach on the order of **30–60 generations** on the moderate tier — enough for POX+elitism to show clear improvement over the initial random population, but this is an estimate to log and verify (`BaselineResult.extra["generations_run"]`, already specified in `../docs/especificaciones/Baselines_TechSpec.md` §5.6), not a number to assume without checking.

---

## 4. Cheap tuning protocol (single-GPU/CPU student budget)

A full grid search over population size × crossover rate × mutation rate × tournament size, each cell requiring multiple seeds at the full 60s budget, is not affordable inside this thesis's compute budget and is not necessary — GA performance on JSSP is not razor-sensitive to these choices within the standard ranges (§2), and the marginal value of exhaustive tuning is low relative to the time it costs a single researcher. This spec recommends a **cheap proxy-and-confirm** protocol instead of a full sweep:

### 4.1 Step 1 — proxy search on the cheapest instance

Run the search on `ft06` (36 operations, the cheapest instance in the validated tier) with a **short** time budget (`time_limit_s=10`, not the full 60s) — ten seconds is enough for dozens of generations on an instance this small, and the known optimum (55, `VALIDATION_WEEK1.md`/`BASELINES_RESULTS_WEEK9.md`) gives an immediate, exact ground truth for scoring a configuration, with no need to wait for the full evaluation suite.

**Search grid** (small, 3×3, not a full factorial over every parameter):

```python
population_sizes = [40, 80, 120]
mutation_rates = [0.05, 0.15, 0.30]   # p_mutation_sequence; p_mutation_assignment mirrors it
# crossover rate, tournament size, elite count: held fixed at the Table 2.1 defaults —
# these two axes are the ones GA practice consistently reports as most impactful for
# permutation-based encodings; do not expand the grid further without a specific reason.
```

For each of the 9 cells, run **5 seeds**, 10s each (45 runs total, ≈7.5 minutes of wall-clock on a single core, trivially parallelizable across cores if available) and record the median final RPD against `ft06`'s known optimum.

### 4.2 Step 2 — confirm, don't re-search, on one larger instance

Take the single best-performing cell from Step 1 and run it **once more**, 5 seeds, on `la05` (a mid-sized instance in the validated tier, chosen because it is the one instance where a single-pass PDR — FIFO — already matched the CP-SAT optimum exactly, `../docs/planificacion/BaselineAnalysis.md` §2, making it a useful sanity check: the GA should comfortably reach 0% RPD here too, and failing to do so is a signal of a decoder or operator bug, not a tuning issue) at the full `time_limit_s=60`. If the confirmation run's median RPD is reasonable (informally: at or below FIFO's RPD on the same instance, which is 0% here — a low bar deliberately chosen so this step is a sanity check, not a second optimization pass), lock the configuration and move on. **Do not iterate Step 1 based on Step 2's result** — that would silently turn a cheap proxy search into an expensive multi-instance grid search through the back door, defeating the point of a *cheap* protocol.

### 4.3 What this protocol deliberately does not do

It does not tune `p_crossover`, `tournament_size`, or `elite_count` (held at the Table 2.1 defaults throughout) — these are lower-impact axes for a permutation-GA in the GA literature's general experience, and expanding the grid to cover them multiplies the proxy search's cost for a return this thesis's budget does not justify. If Chapter 6 results later suggest the GA is a materially weak baseline (e.g., losing to FIFO on instances where CP-SAT shows a GA-favorable structure), revisit this decision explicitly rather than silently expanding the search after the fact.

---

## 5. Stagnation handling: random immigrants (new, not in `../docs/especificaciones/Baselines_TechSpec.md` §5)

`../docs/especificaciones/Baselines_TechSpec.md` §5 specifies selection, crossover, mutation, and elitism, but nothing that prevents premature convergence (the whole population collapsing to near-identical chromosomes before the wall-clock budget is exhausted, wasting the remaining generations). This spec adds one small, standard mechanism: at each generation, replace the worst `immigrant_fraction` (default 0.05, i.e. ~4 individuals at `population_size=80`) of the non-elite population with **freshly random** chromosomes (a random permutation-with-repetition sequence, and — FJSP only — a random eligible-machine assignment), rather than descendants of the current population. This is a one-line addition to the generation loop (`../docs/especificaciones/Baselines_TechSpec.md` §5.6's pseudocode), costs one extra random-chromosome generation and decode per generation, and is a standard, low-risk mitigation — flag it in `../../archive/capitulos/Section4.6.md`/Chapter 4 prose as a documented addition to the operator set specified in Part B7, not something B7 itself calls for.

```python
def one_generation(population, rng, config):
    elites = top_k(population, config.elite_count)
    mating_pool = tournament_select(population, config.tournament_size, rng, n=len(population) - config.elite_count)
    offspring = crossover_and_mutate(mating_pool, config, rng)
    n_immigrants = max(1, int(config.immigrant_fraction * len(population)))
    worst_idx = bottom_k_indices(offspring, n_immigrants)
    for i in worst_idx:
        offspring[i] = random_chromosome(instance, rng)   # replaces, does not append
    return elites + offspring
```

---

## 6. Required test additions (extends `../docs/especificaciones/Baselines_TechSpec.md` §7's `tests/test_ga.py` list)

- `test_default_population_size_matches_tier_bands` — `default_population_size(36) == 80`, `default_population_size(100) == 80`, `default_population_size(101) == 60`, `default_population_size(450) == 60`, `default_population_size(451) == 40`, pinning the band boundaries exactly as specified in §2.1 (a tripwire against silent drift from `../../archive/capitulos/Section5.1_5.2.md`'s bands).
- `test_random_immigrants_replace_worst_not_elite` — after one generation, the elite individuals from the previous generation are unchanged (already covered by the existing monotonicity test), and the replaced individuals are exactly the `immigrant_fraction`-worst of the *offspring*, not of the elites.
- `test_ga_reaches_zero_rpd_on_la05` — the confirmation check from §4.2, promoted to a permanent regression test (not just a one-off tuning step): median RPD over 5 seeds at `time_limit_s=60` on `la05` is `0.0`. If this test starts failing after an unrelated change, it is a decoder/operator regression, not a tuning drift, and should be treated with the same urgency as `../docs/especificaciones/Baselines_TechSpec.md` §7's `test_existing_rules_byte_identical`.

---

## 7. Open design decisions this spec resolves (flagged, same culture as prior specs)

1. **The exact starting numbers in Table 2.1** are this spec's own defaults, not derived from Part B7 (which specifies operator families, not rates) — standard permutation-GA practice, not a cited empirical source; the cheap tuning protocol (§4) is the intended mechanism for revising them with actual evidence rather than leaving them as an unexamined guess.
2. **The population-size-vs-instance-size scaling rule (§2.1)** ties directly to `../../archive/capitulos/Section5.1_5.2.md`'s size tiers by construction, so the two documents cannot silently drift apart, but the two untested bands' population sizes (60, 40) are extrapolated, not measured — same status as those bands' CP-SAT time limits.
3. **Random immigrants (§5)** is an addition beyond what `../docs/especificaciones/Baselines_TechSpec.md` §5 specifies; included here because premature convergence is a well-known, cheap-to-prevent failure mode for permutation GAs, and leaving it out would risk an unfair GA-vs-CP-SAT/GA-vs-DRL comparison in Chapter 6 for a reason unrelated to the actual research question.
4. **The proxy-and-confirm tuning protocol (§4) trades tuning thoroughness for compute-budget realism.** A committee member could reasonably ask for a fuller sweep; the documented justification is the same "single student, single GPU, nine months" constraint already invoked for CP-SAT/GA time limits in `../../archive/capitulos/Section4.6.md` §4.6.2 and `../docs/planificacion/Step 4 Thesis.md` §1.5 — this is a consistent application of an already-established constraint, not a new one invented for the GA alone.
