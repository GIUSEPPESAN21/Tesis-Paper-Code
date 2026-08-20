# Analysis of Week 9 Baseline Results

**Source:** `BASELINES_RESULTS_WEEK9.md` — 9 PDRs + CP-SAT, evaluated on `ft06`, `ft10`, `ft20`, `la01`–`la10` (13 instances, 130 runs). All figures below were independently recomputed from the report's full results table (not just copied from its summary) and match exactly.

## 1. Which PDR performs best on average

**FIFO is the strongest single PDR on this instance set by a wide margin**, not by a narrow one:

| rank | method | mean RPD (%) | best-on (of 13) |
|---|---|---:|---:|
| 1 | fifo | 17.80 | 11 |
| 2 | mor | 35.33 | 1 (la10) |
| 3 | mwkr | 38.05 | 1 (la03) |
| 4 | spt | 148.15 | 0 |
| 5 | lpt | 160.63 | 0 |
| 6 | edd | 195.70 | 0 |
| 7 | lwkr | 197.88 | 0 |
| 8 | lor | 212.17 | 0 |
| 9 | cr | 236.24 | 0 |

FIFO's mean RPD (17.80%) is roughly **half** of MOR's (35.33%) and MWKR's (38.05%), and it wins 11 of 13 instances outright, losing only `la03` (to MWKR) and `la10` (to MOR) — and only by a few points there (MWKR 26.88% vs. FIFO's presumable near-tie on la03 isn't in the "best" column but the gap is not large). This is worth stating plainly rather than smoothing over: it runs against the general expectation in the scheduling literature (and against the framework document's own B6 note that "SPT y MWKR tienden a producir buenos makespans empíricos, mientras que FIFO es 'justa' pero débil"). The Week 9 report frames FIFO's win as "consistent with the literature's finding that MWKR-family rules are strong... with MOR/MWKR close behind" — but a 2x gap in mean RPD is not "close behind," and this framing understates a genuinely interesting empirical result specific to this environment and instance family.

**A plausible explanation, not yet confirmed:** the environment's list-scheduling transition (`start = max(job_ready_time[j], machine_free_time[m])`, Section 4.3.2) inserts every dispatched operation at the earliest feasible time with no backward insertion into idle gaps. Under this transition, a rule that avoids starving any one job of machine time (FIFO, which serves whichever job has been waiting longest) may keep the whole shop's machines more evenly loaded than a rule that aggressively prioritizes high-workload jobs (MWKR/MOR), which can create long queues behind the prioritized jobs on shared machines. Whether this holds up as a structural property of the `ft`/`la` families (moderate job-to-machine ratios, 5–10 machines) or is closer to a coincidence of this particular instance set is worth a short dedicated note in Chapter 6, and possibly a rerun on the `ta` family (larger, more machines) once those instances are added (Section 5.1) to see if the ordering holds.

**Recommendation for the thesis text:** report FIFO, not MWKR, as "the best PDR" wherever Chapter 4/5/6 prose needs a single reference rule — and flag explicitly that this is an empirical finding for this instance set, not an a priori literature assumption. This has a direct, concrete consequence for the pre-registration (D7, H1): H1 is stated as "the policy beats the best PDR in mean RPD," and the best PDR turns out to be FIFO here, not MWKR as the framework document's B6 discussion might lead one to assume. Section 5.1/5.2 (below) makes this substitution explicit rather than leaving it implicit.

## 2. How far CP-SAT's proven optima are from the PDRs

CP-SAT proved optimality on **13/13 instances** (matching the published optima exactly: ft06=55, ft10=930, ft20=1165, and the `la01`–`la10` values used as `C*`), so its RPD is 0.00% everywhere by construction — it is not a competing "method" in the same sense as the PDRs, it is the **reference** the PDRs (and later the GA and the DRL policy) are measured against.

The gap is large and highly instance-dependent:

- **Smallest gap:** `la05`, where FIFO reaches 593 = the proven optimum exactly (RPD 0.00%) — the one instance where a single-pass PDR matches CP-SAT.
- **Typical gap (FIFO, the best PDR):** 8–24% RPD on most `la` instances, 21.8% on `ft06`, up to 44.3% on `ft20`.
- **Largest gap (weakest PDRs — LWKR/EDD/LOR/CR):** 150–296% RPD, i.e. the resulting makespan is up to roughly **4x** the proven optimum (`cr` on `ft10`: makespan 3682 vs. optimum 930).

This spread — 0% at the best case down to a ~4x blow-up at the worst — is itself informative: it shows the instance set is discriminating (it doesn't compress every method into a narrow band), which is what makes RPD a meaningful axis for the Friedman/Nemenyi comparison planned in Section D6/5.2.

## 3. Floor and ceiling for the DRL model

This gives a concrete two-sided bar, not a single number:

- **Ceiling (oracle, not a same-class opponent): CP-SAT's proven optimum, RPD = 0.** The DRL policy's RPD is always measured against this value, per the D6 formula, but CP-SAT should not be treated as a peer method in the Friedman/Nemenyi ranking (Section 4.6.4 already flags this: CP-SAT's decision-time and search paradigm are not comparable to a single constructive pass). It is the fixed reference point that makes RPD meaningful, not a target the policy is expected to beat under a comparable time budget.
- **Floor the model must clear decisively: FIFO (mean RPD 17.80%).** This is the actual empirical best PDR on this instance set, not MWKR — the DRL policy failing to beat FIFO's ~18% mean RPD would be a strong negative result (H1's rejection criterion). A policy that only beats the weak rules (CR, LOR, LWKR, EDD, all >195% mean RPD) has not learned anything a five-line heuristic couldn't already do.
- **Secondary, tougher bar: MOR/MWKR (mean RPD 35–38%).** These are the theoretically-motivated "strong PDR" candidates the framework document expects to be hardest to beat; empirically they trail FIFO here but remain a meaningful second checkpoint, particularly if the ordering changes on larger/different instance families (`ta`).
- **The four weak rules (SPT, LPT, LWKR, LOR, EDD, CR) are not meaningful opponents** — they exist to complete the required 9-rule suite (Part B6) and to give the Friedman/Nemenyi ranking (Section D6/5.3) a full ordinal spread, not because beating them is informative on its own.

**One practical implication for the CP-SAT time budget going forward:** the worst observed CP-SAT wall-clock time across all 13 instances was 3.79s (`ft10`), against a 60s budget — a ~16x margin. This means 60s is comfortably sufficient for every instance up to 10x10/15x5 (100/75 operations) tested so far, but says nothing about the much larger `ta` instances (up to 100x20 = 2000 operations) planned for the size-generalization experiment (Part D4/E3); Section 5.1 below sets a separate, larger time-limit tier for that band rather than assuming the same 60s budget scales.
