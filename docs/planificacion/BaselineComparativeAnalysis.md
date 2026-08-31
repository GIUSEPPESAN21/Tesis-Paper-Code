# Comparative Analysis of Week 12 Baseline Results (PDR + GA + CP-SAT)

**Source:** `BASELINES_RESULTS_WEEK12.md` and its underlying data, `experiments/results/baselines_week12.csv` (143 rows, 13 instances × 11 methods). Every figure below was recomputed independently from the raw CSV, not copied from the report's own summary tables, and matches exactly.

**Scope correction, stated up front:** the brief for this analysis referred to results "across ft/la/MK instances." The actual evaluated data covers **only `ft06`/`ft10`/`ft20` and `la01`–`la10`** — 13 JSSP instances, no FJSP data. `BASELINES_RESULTS_WEEK12.md` is explicit that MK01–MK10 (the FJSP benchmark family) is deliberately deferred: no FJSP instance loader exists yet, and the project has an established policy (`scripts/fetch_instances.py`) against hand-transcribing benchmark data from memory rather than a verified source. This analysis covers what was actually run; MK/FJSP comparison is not possible from this data and should not be implied in the thesis text until that instance loader lands.

---

## 1. Which method dominates, and on which axis — quality and speed are not the same ranking

Solution quality alone gives one ranking; wall-clock cost gives a completely different one. Collapsing them into a single "best method" statement would hide the most useful finding in this data.

**Table 1.1.** Mean RPD (%) vs. proven optimum, and wall-clock cost, across all 13 instances.

| method | mean RPD % | min | max | wall-clock cost |
|---|---:|---:|---:|---|
| cpsat | 0.00 | 0.00 | 0.00 | 15 ms – 3.53 s (proves optimality every time) |
| ga | 2.61 | 0.00 | 10.64 | fixed 60.00 s (always consumes its full budget) |
| fifo | 17.80 | 0.00 | 44.29 | 5 – 51 ms |
| mor | 35.33 | 4.07 | 68.50 | 6 – 57 ms |
| mwkr | 38.05 | 12.31 | 71.25 | 5 – 46 ms |
| spt / lpt | 148–161 | 76 | 215 | 5 – 47 ms |
| edd / lwkr / lor / cr | 196–236 | 140 | 296 | 5 – 53 ms |

**On solution quality alone**, the ranking is unambiguous and holds on every single instance, not just on average: CP-SAT (optimal by construction) > GA (2.61% mean, exact optimum on 8/13) > FIFO (17.80%, the best PDR) > MOR/MWKR (35–38%) > the remaining six PDRs (148–236%). GA beats every PDR on every one of the 13 instances, typically by one to two orders of magnitude — this is a much stronger, more consistent result than the PDR-only picture from Week 9 (`../docs/planificacion/BaselineAnalysis.md`), where FIFO's win over MOR/MWKR was already notable at a 2x margin; GA's margin over FIFO is roughly 7x on average and reaches values like `ft20` (FIFO 44.29% vs. GA 8.76%, a 5x gap) or `la09` (FIFO 3.37% vs. GA 0.00%).

**On wall-clock cost, the ranking inverts almost completely.** The nine PDRs are three to four orders of magnitude faster than GA (milliseconds vs. a fixed 60 seconds — GA does not adapt its runtime to instance difficulty, it always consumes the full budget by design, per `../docs/especificaciones/GA_TechSpec.md` §3). CP-SAT sits, perhaps surprisingly, **closer to the PDRs than to the GA** in wall-clock terms on this instance set: its worst case (`ft10`, 3.53s) is still roughly 17x under its own 60s budget, and every other instance solves in under 150ms.

### 1.1 The finding worth stating plainly: on this instance set, CP-SAT Pareto-dominates the GA

For every one of the 13 evaluated instances, CP-SAT is **both** more accurate (0% RPD vs. GA's 0–10.64%) **and** faster (worst case 3.53s vs. GA's fixed 60s) than the GA. There is no instance in this data where running the GA instead of CP-SAT would have been the better choice on any axis. This is worth flagging explicitly rather than treating the GA's "beats every PDR" result as the headline on its own: beating a weak baseline is a low bar; the GA's real test is against CP-SAT, and on this instance set it loses on both dimensions that matter. This does not mean the GA is poorly implemented — `BASELINES_RESULTS_WEEK12.md` documents a real bug found and fixed during this sprint (an under-reported makespan, caught precisely because it produced an impossible below-optimal value), and the GA's behavior post-fix is exactly what unit tests (`test_ga.py`) confirm it should be. It means that **for the instance sizes tested so far, CP-SAT's exact search is simply cheap enough that there is no practical reason to prefer an approximate metaheuristic.** The GA's value proposition is not demonstrated by this data — it is a hypothesis about larger instances that has not yet been tested (§3).

---

## 2. Dominance by instance size

Within the 13 evaluated instances (36 to 100 operations), no method's relative ranking changes with size in a way that contradicts §1 — CP-SAT proves optimal everywhere, GA beats every PDR everywhere, FIFO remains the best PDR on 11/13 instances (the two exceptions, `la03` and `la10`, go to MWKR and MOR respectively — unchanged from the Week 9 PDR-only analysis, `../docs/planificacion/BaselineAnalysis.md` §1). What **does** change with size is the *margin*, in two directions worth separating:

- **GA's gap to CP-SAT widens on the two largest, most "square" instances** (`ft10`, 10×10, 100 operations: GA 10.64% RPD; `ft20`, 20×5, 100 operations: GA 8.76% RPD) — the two worst GA results in the entire table, both at the 100-operation ceiling of what's been tested. This is consistent with a generic expectation (more operations, larger search space, same fixed population/time budget reaches proportionally less of it), but the sample is small (2 instances at this size) and should not be treated as a confirmed size trend yet — the intermediate tier (`la11`–`la40`, up to ~450 operations, `../../archive/capitulos/Section5.1_5.2.md` Table 5.1.1) is the actual test of whether this is a trend or a coincidence of these two specific instances.
- **GA's gap does *not* scale cleanly with size within the `la01`–`la05` block, all identically sized (10×5, 50 operations).** `la01` and `la05` reach the exact optimum (0.00%); `la02`, `la03`, `la04` do not (6.26%, 5.36%, 2.88% respectively), despite being the same size as `la01`/`la05` and run under the identical configuration (population 80, ~60s, same operators). This means instance *structure*, not just instance *size*, determines whether this GA configuration finds the optimum within budget — a more interesting and more actionable finding than a pure size trend, because it means "bigger is harder" is not the whole story even within this dataset. Chapter 6 should not present GA's optimality gap as a monotonic function of instance size without qualifying it against this counter-example.

**A caveat the Week 12 report itself flags and this analysis inherits directly:** every GA number above is a **single seed** (`seed=0`). The one instance with dedicated multi-seed evidence, `la05` (5 seeds, `tests/test_ga.py::test_ga_reaches_zero_rpd_on_la05`), confirms median RPD is exactly 0.0% — but that is reassurance for one instance, not for the `la02`/`la03`/`la04` gaps above, which could plausibly narrow, widen, or stay put under a proper multi-seed sweep. Treat the non-zero GA entries in Table 1.1/§2 as provisional until `--ga-seeds 5` is actually run across all 13 instances (already flagged as a "next step, not this sprint" in `BASELINES_RESULTS_WEEK12.md`, and the explicit prerequisite for Section 5.3's Wilcoxon/Friedman procedure, §4 below).

---

## 3. Where CP-SAT's time limit becomes binding

**On this data: nowhere.** The worst observed CP-SAT wall-clock time across all 13 instances is 3.53s (`ft10`), against a 60s budget — roughly a 17x margin, tightening the already-generous 16x margin observed at Week 9 (`../docs/planificacion/BaselineAnalysis.md` §3) very slightly (Week 12's `ft10` run: 3.528s vs. Week 9's 3.787s for the same instance — the small difference is normal run-to-run solver variance, not a regression). The 60-second budget currently used has never once been the reason a CP-SAT result is reported as `FEASIBLE` instead of `OPTIMAL`; every single one of the 13 `status` values in the CSV is `OPTIMAL`.

**This means the question, as posed, doesn't have an answer yet — and that itself is the answer worth recording.** `../../archive/capitulos/Section5.1_5.2.md` Table 5.1.2 already anticipated this and pre-committed to a size-tiered budget precisely because 60 seconds was expected to stop being generous once instances grow past the ~100-operation range tested so far: 300s for the intermediate tier (`la11`–`la40`, up to ~450 operations), 600–900s for the large tier (`ta`, up to 2000 operations), both explicitly marked as **unvalidated extrapolations** in that document. Nothing in the Week 12 data changes that plan — if anything, the fact that even `ft10`/`ft20` (100 operations, the largest instances tested so far) solve in under 4 seconds suggests the 300s intermediate-tier budget may itself turn out to be generous, but this is exactly the kind of claim `../../archive/capitulos/Section5.1_5.2.md` §5.1.2 already flagged as "a prerequisite to measure, not a scaling law to assume" — CP-SAT's runtime on JSSP/FJSP does not scale linearly with operation count, and 2000-operation `ta` instances are a large enough jump that extrapolating from a 100-operation data point would be exactly the mistake that document warned against.

**Practical recommendation:** the `la11`–`la40` block should be run next, specifically because it is the cheapest way to get a real (not extrapolated) data point on whether CP-SAT's binding point is at 200 operations, 400 operations, or still not reached — before committing the larger time budget to the `ta` block.

---

## 4. The gap the DRL model needs to close — a two-axis target, not one number

Section 1's quality/speed split makes a single "beat baseline X" framing misleading for the DRL policy. Three distinct, useful comparisons follow from this data, replacing the single-tier floor/ceiling framing from Week 9 (`../docs/planificacion/BaselineAnalysis.md` §3, which only had FIFO and CP-SAT to work with):

- **Quality floor to clear decisively: FIFO (17.80% mean RPD).** Unchanged from Week 9 — a DRL policy that cannot beat FIFO has not learned anything a five-line heuristic couldn't already do. This remains the minimum bar, not the target.
- **Quality target to approach or beat: the GA (2.61% mean RPD, exact optimum on 8/13 instances).** This is now the real, non-trivial quality bar — an order of magnitude harder to clear than FIFO. A DRL policy landing between FIFO and GA (say, 5–10% mean RPD) would be a solid, defensible result relative to the PDR floor, but should not be presented as "matching the state of the art among these baselines" — the GA already occupies that ground on quality alone.
- **Speed target the GA itself cannot claim: PDR-level decision latency.** This is the DRL policy's actual structural advantage and the comparison this thesis's own reward/architecture design (a single forward pass per decision epoch, Section 4.3/4.4) is built to exploit: a trained policy's inference cost is milliseconds per full episode, in the same regime as the PDRs (5–60ms observed here), not the GA's fixed 60-second search. **The single most informative comparison Chapter 6 can make is not "DRL RPD vs. GA RPD" in isolation, but a quality-vs-time plot** — RPD on one axis, wall-clock cost on the other, all methods (9 PDRs, GA, CP-SAT, DRL) plotted together — because it is entirely possible for the DRL policy to land at, say, 4–6% mean RPD (worse than GA's 2.61%) while still being the practically preferable method, if it gets there in milliseconds rather than 60 seconds. Presenting only a single RPD ranking would bury exactly the result (fast, good-enough scheduling) this thesis's architecture is designed to produce.
- **The oracle reference stays CP-SAT (0% RPD), not a peer.** Unchanged from every prior document (`../../archive/capitulos/Section4.6.md` §4.6.4, `../../archive/capitulos/Section5.1_5.2.md` §5.2.3): CP-SAT defines the RPD denominator; it is not a method the DRL policy is expected to beat under a comparable budget, and on this instance set it dominates the GA on both axes anyway (§1.1), so "beating CP-SAT" was never a meaningful target to begin with at this instance scale.

**One caution for Chapter 6's narrative:** given §1.1's finding that CP-SAT currently dominates the GA outright, a DRL policy that only manages to land between FIFO and GA in quality, without a decisive speed advantage over the GA to justify the comparison, would face the same "why not just run CP-SAT" question the GA currently cannot answer on this instance set. The speed argument (above) is therefore not a nice-to-have framing choice — it is the argument that makes the DRL policy's eventual result meaningful at all on instances this small, and it should be built into the Chapter 6 results presentation from the start, not added after the fact if the RPD numbers alone look unconvincing.
