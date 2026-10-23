# Final Experiment Matrix — E0–E8 Execution Plan

**Status:** Reviewed and finalized against the actual codebase state as of **2026-07-30**.
**Author of this review:** Cowork (doctoral thesis auditor role), at Joseph Sanchez's request.
**Provenance:** operationalizes `../docs/planificacion/Step 4 Thesis.md` Part D4 (original E0–E8 matrix) against `../../thesis/PreRegistration.md` (frozen 2026-07-29T05:22:26Z UTC, H1–H5 decision rules), `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md` (dated 2026-07-30, trained checkpoints and compute accounting), `../docs/planificacion/AblationPlan.md`, `REFACTOR_NOTES_WEEK15.md`, and a direct read of `code/` (`envs/`, `baselines/`, `data/`) as it stands today.
**Companion documents this plan assumes exist and are frozen:** `../../thesis/PreRegistration.md` (H1–H5 decision rules — this document does not alter them), `../docs/planificacion/AblationPlan.md` (Axis 3 = H4, not double-commissioned).

---

## 0. Two findings that change the scope of D4's original matrix

Before confirming instance sets and replications per experiment, two things surfaced during this review that D4 (drafted earlier in the project) and even `../../thesis/PreRegistration.md` (frozen one day ago) could not have accounted for. Both need a decision now, in writing, before E1–E8 execution starts — exactly the discipline the pre-registration protocol exists to enforce.

### Finding A — A real Brandimarte (`MK*`) FJSP loader now exists, undocumented in the freeze

`../../thesis/PreRegistration.md` §1 (frozen 2026-07-29) states as fact that no real FJSP benchmark loader is bundled, and restricts E2/H1's FJSP arm to the synthetic `generate_fjsp_instance()` variant of the `ft`/`la` set. That was accurate **at freeze time**. It is no longer accurate: `envs/fjsp_loader.py`, `scripts/fetch_fjsp_instances.py`, `data/fjsp_instances/{mk01..mk07}.txt`, and `data/fjsp_known_optima.csv` all exist in the repository today, file-timestamped **2026-07-30** — the same day as `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`, i.e. built *after* the pre-registration freeze. `run_evaluation.py` already imports `load_fjsp_by_name`/`available_fjsp_instances` and prefers the real instance whenever `--fjsp <name>` matches one of the seven bundled names, falling back to the synthetic generator otherwise.

What is actually bundled: **`mk01`–`mk07`** only (not the full `mk01`–`mk10`, and no Hurink instances D4 also named), with reference makespans of mixed provenance — `mk01`, `mk03`, `mk04` are `status=proven`; `mk02`, `mk05`, `mk06`, `mk07` are `status=best_known` (per `data/fjsp_known_optima.csv`).

Two things are **not** yet true, even though the loader exists:
- **No test coverage.** The loader's own module docstring promises `tests/test_fjsp_loader.py` pins the rectangular-padding neutrality claim ("CP-SAT reaches the literature optimum on the padded instance"). That file does not exist in `code/tests/`. This is new, unvalidated code being proposed for a confirmatory thesis result.
- **No baselines measured.** `BASELINES_RESULTS_WEEK12.md` explicitly deferred `MK01–MK10`/FJSP evaluation for lack of a loader. That deferral is now stale but nothing has filled it — there is no CP-SAT or GA result on file for `mk01`–`mk07`.

**Per `../../thesis/PreRegistration.md` §1's own contingency clause** ("If a real FJSP loader is built before H1 is executed, it should be added as a second, separately-reported FJSP arm — an addition, not a substitution"), this is exactly the situation anticipated. Since **zero FJSP results of any kind exist yet** (the freeze's purpose — preventing HARKing — is not at risk), amending now is legitimate and should be done before E2 executes, not silently. §5 of this document gives the amendment text to append to `../../thesis/PreRegistration.md` §7.

### Finding B — H3/H4's blocker has no dedicated engineering slot in the roadmap

`../../thesis/PreRegistration.md` §0 is explicit: H3 (E4–E6) and H4 (E7) are **blocked**, not merely unscheduled — `TrainedRLBaseline` (`baselines/interface.py`) constructs a static `JSSPEnv` rollout only; it is not wired against `DynamicJSSPEnv`'s growable-observation, disruption-aware contract. `../docs/planificacion/AblationPlan.md` §2.3 and `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`'s "carried forward" list both confirm this is still true today, one day after the freeze.

The master project plan (internal planning document) schedules Semana 24 as "execute E1–E6" using "the trained model checkpoint from Week 21" — but three of those six experiments (E4, E5, E6) cannot run at all until this wiring exists, and **no week in the roadmap is explicitly allocated to building it.** This is the single highest-risk item in the whole two-sprint plan, and it is a bounded software task (not open research — `DynamicJSSPEnv` already runs PDRs correctly under all three disruption axes per `VALIDATION_WEEK6.md`'s calibration checks and `TestConstraintSafetyUnderBreakdown`; only the trained policy's observation-contract compatibility is missing), so the fix belongs in the schedule explicitly rather than being discovered mid-sprint. §4 and §6 below treat this as the first, most urgent task of the two-sprint window — ahead of E2's extra scope, ahead of everything except E1/E3/E8 which have zero dependencies at all.

---

## 1. Confirmed matrix: E0–E8

All replication counts below are the **already-frozen D4/`../../thesis/PreRegistration.md` values**, kept as specified. The compute accounting in §3 shows why: for every experiment except E7 (which needs two new training runs), the actual bottleneck is not GPU/CPU time — inference rollouts are sub-second — it is baseline-measurement prep work and, for E4–E7, the wiring blocker in Finding B. Reducing replicate counts would trade away statistical power for savings the budget doesn't actually require, so none are recommended here.

| Exp. | Hypothesis | Confirmed instance set | Confirmed factor levels | Replications | Primary metric | Tier | Blocker |
|---|---|---|---|---|---|---|---|
| **E0** Validación DES | (supports Ch.6.1) | Existing evidence: `VALIDATION_WEEK1.md` (env mechanics, CP-SAT/GA hit known optima on `ft06`/`ft10`), `VALIDATION_WEEK6.md` (disruption-generator calibration vs. target `A_k`/`σ`/`ρ`), 593 passing tests | — | already run | qualitative pass/fail + calibration error | **1 — near-free** | None. No M/M/1 analytic-queue check was found in the repo; add one cheap unit test if the committee will expect it (see §2.0), but do not schedule new compute for this. |
| **E1** Estático JSSP | H1 (JSSP arm) | 43 bundled instances: `ft06`, `ft10`, `ft20`, `la01`–`la40`, `--disruption none` | single condition | 1 deterministic + 4 stochastic (seeds 1–4, descriptive) per instance = 5/instance, 215 total | RPD vs. best-PDR (Wilcoxon, paired, N=43) | **1** | None |
| **E2** Estático FJSP | H1 (FJSP arm) | **Arm A (frozen, keep):** synthetic FJSP variant of the same 43 instances, `seed=0`. **Arm B (new, requires §5 amendment):** real Brandimarte `mk01`–`mk07` | single condition, two arms reported separately | 5/instance each arm (same 1+4 convention) | Arm A: raw makespan vs. GA/CP-SAT. Arm B: RPD vs. `known_optimum` (proven or best-known, flagged by `status`) | **2** | Arm B needs: `test_fjsp_loader.py` written and passing, CP-SAT+GA baselines run fresh on `mk01`–`mk07` (none exist), §5 amendment logged |
| **E3** Generalización tamaño | H2 | Full tiers per `../../thesis/PreRegistration.md` §3: `la21`–`la25`,`la26`–`la30` (1.5–2×), `la36`–`la40` (2.25×), `la31`–`la35` (3×) — 20 instances | 4 tiers | zero-shot, 1 rollout/instance (deterministic weight transfer, no retraining) | Friedman + Nemenyi per tier | **1** | None. `la21/la25/la30/la35/la40` already have a real RL number from Week 21's generalization tracking (19.9–35.6% RPD) using exactly the checkpoint recommended below — the remaining 15 instances just extend the same run. Fresh PDR baselines needed for all 20 (Week 12 stops at `la10`). |
| **E4** Robustez averías | H3 | `ft06`, `la01` (already-validated disruption pair) | $A_k \in \{0.95, 0.90, 0.80\}$ | 30 seeds/level | RI, Wilcoxon per regime | **1, at-risk** | Finding B (wiring) |
| **E5** Tiempos estocásticos | H3 | `ft06`, `la01` | $\sigma \in \{0.1, 0.25, 0.5\}$ | 30 | RI, Wilcoxon per regime | **1, at-risk** | Finding B |
| **E6** Llegadas dinámicas | H3 | `ft06`, `la01` | $\rho \in \{0.7, 0.85, 0.95\}$ | 30 | RI, E[ΣT_j], Wilcoxon per regime | **1, at-risk** | Finding B |
| **E7** Recompensa consciente | H4 | `ft06`, `la01`, same 9 regimes as E4–E6 | $\kappa \in \{0, 0.5, 1.0\}$ ($\kappa=0$ = reused E4–E6 policy, not retrained) | 30/regime; 2 new training runs (κ=0.5, κ=1.0) | ΔRI (Wilcoxon), ΔRPD_nominal (margin δ=2pp) | **3 — supplementary this window** | Finding B **and** $\Delta_{\text{inest}}$ reward term (currently only `info["planned_start_deviation"]` plumbing exists) |
| **E8** Tiempo de decisión | H5 | Same 43 instances as E1 (piggy-backs on E1's rollouts — see §2.8) + a 4-instance CPU-only subset (`ft06`,`la05`,`la20`,`la35`) | GPU (training hardware) primary; CPU-only secondary | 5/instance (shared with E1) | ms/decision vs. CP-SAT (ratio, no significance test) | **1 — near-free** | None |

**Tier legend:** 1 = load-bearing, zero blockers, must complete. 2 = load-bearing, needs one day of prep work first. 1-at-risk = load-bearing *and* central to the thesis's own title ("bajo Perturbaciones Estocásticas") — not demoted for importance, flagged because it carries real schedule risk from Finding B. 3 = genuinely supplementary for this two-sprint window; already scheduled for Sprint 2 by the master plan, contingent on Tier-1-at-risk clearing first.

---

## 2. Notes and deviations per experiment

### 2.0 E0 — do not schedule new compute

Chapter 6.1 needs a validation narrative, not a new experiment. Consolidate `VALIDATION_WEEK1.md` + `VALIDATION_WEEK6.md` + the 593-test suite pass into a one-page E0 summary during Week 24 (an hour of writing, not a code task). The one honest gap: D4 named an "M/M/1 analytic case" check and I could not find one in the test suite. If the committee is likely to ask for it specifically, it is a single cheap unit test (compare simulated mean queue length/wait time against the closed-form M/M/1 formula over a long run) — flagging it as optional, not blocking.

### 2.1 E1 — no open questions, but pin the checkpoint explicitly

Neither D4 nor `../../thesis/PreRegistration.md` name *which* saved checkpoint `TrainedRLBaseline` should load for the 43-instance sweep, and this matters: the Week 18 `ppo_hetero_stageA_ft06_final.zip` checkpoint is trained on `ft06` only (0.00% RPD there, but zero-shot everywhere else), while `curriculum_full/checkpoints/curriculum_stage19_la20_final.zip` is the size-agnostic policy exposed to `la01`–`la20` during the 25-stage curriculum (5.43% RPD on `la20`, and the best available general JSSP policy per Week 21's own appendix). **Use `curriculum_stage19_la20_final.zip` for E1, E3, and E8** — it is the only checkpoint that makes "43-instance benchmark competitiveness" and "generalization to 2–3× larger instances" a coherent pair of claims about the same policy. Report in Chapter 6.2 explicitly that `ft06` is at the edge of this checkpoint's curriculum exposure while `la21`+ is genuinely zero-shot, so a reader can tell competitiveness from generalization apart.

### 2.2 E2 — sequence Arm A first, Arm B only if the test-validation step is clean

Arm A (synthetic, 43-instance) has zero new dependencies — it reuses E1's exact infrastructure with `--fjsp`. Run it same-day as E1. Arm B (real `mk01`–`mk07`) is a genuine upgrade toward D4's original MK-benchmark intent, but only if done properly: write `test_fjsp_loader.py` first (the padding-neutrality check the module docstring already promises), confirm it passes, *then* run CP-SAT/GA baselines fresh, *then* evaluate RL. Report `mk01`–`mk05` (seen during curriculum training, stages 20–24) and `mk06`/`mk07` (genuinely zero-shot per Week 21's own generalization table, 122–193% RPD) as two visibly distinct rows — conflating in-sample and zero-shot FJSP numbers under one "H1 FJSP arm" line would be a reporting error, not just an imprecision.

### 2.3 E3 — extend, don't repeat, Week 21's own generalization run

Five of the twenty frozen-tier instances already have a number, for free, from `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`'s zero-shot table (`la21`=35.6%, `la25`=27.4%, `la30`=19.9%, `la35`=26.1%, `la40`=21.4%, all after the `la20` curriculum stage — i.e., exactly the checkpoint §2.1 recommends). The Week 24 task is: run the same zero-shot evaluation on the remaining 15 instances (`la22`–`la24`, `la26`–`la29`, `la31`–`la34`, `la36`–`la39`), and run a fresh 9-PDR baseline sweep across all 20 (Week 12 only covers up to `la10`). Both are cheap — PDRs are near-instant, and RL rollouts at this scale are sub-second per Week 21's own timing numbers.

### 2.4–2.6 E4/E5/E6 — the grid itself is cheap; the wiring is not

Once `TrainedRLBaseline` can roll out against `DynamicJSSPEnv`, the entire 9-regime × 30-replicate × 2-instance grid (RL + up to 9 PDRs per regime, 6 arrival-safe PDRs for E6 per `REFACTOR_NOTES_WEEK15.md`'s `ARRIVAL_UNSAFE_PDR_NAMES` guard) is on the order of a few thousand short rollouts — comfortably a few hours of wall-clock, not days. **Per `../../thesis/PreRegistration.md` §4, GA and CP-SAT are explicitly *not* part of H3's comparator** (only the PDR subset is); do not spend budget re-running GA under all 9×30×2 conditions; at ~60s/GA-call that would be roughly 9 hours for zero contribution to the frozen decision rule. If GA/CP-SAT-under-disruption is wanted for descriptive context, it is optional and should be scoped down (e.g., one regime, one instance) rather than run at full replication.

### 2.7 E7 — genuinely defer to Sprint 2, as the master plan already intends

The master plan already schedules E7 for the second coding sprint (Semana 27) rather than the first — this review confirms that sequencing is correct and should not be compressed forward, since it depends on E4–E6's wiring clearing first. The two additional training runs (κ=0.5, κ=1.0) are cheap by the ablation study's own precedent (~10–11 min/seed at 400k steps on the RTX 5070 for a `ft06`-scale run; if trained at `la01` scale via `DynamicJSSPEnv`, budget somewhat more given the added event-driven overhead, but still on the order of tens of minutes, not hours).

### 2.8 E8 — merge with E1's rollouts instead of running it separately

`../../thesis/PreRegistration.md` §6 asks for exactly 5 timed replicates per instance on training hardware — the same 1 deterministic + 4 stochastic rollouts E1 already runs. Instrument that one rollout pass to record wall-clock per decision epoch, and E1 and E8 are satisfied by a single execution pass, not two. Only the CPU-only secondary timing pass (recommended on a 4-instance subset spanning the scale range, not all 43, since it is a descriptive ratio check, not a second confirmatory test) needs a separate run.

---

## 3. Compute budget: is replication count actually the constraint?

Grounding this in `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`'s own accounting (single RTX 5070, 12GB VRAM; CPU 16-thread fallback):

| Task | Basis | Estimated cost |
|---|---|---|
| E1 (215 rollouts, ≤300 ops each) | ft06 greedy rollout = 0.18s | minutes, not hours |
| E2 Arm A (215 rollouts) + Arm B (35 rollouts) + CP-SAT/GA on 7 new instances | GA ≈60s/run at `ft06` scale; `mk03` (120 ops) may run longer | GA: ~10–15 min for 7 instances × few seeds; CP-SAT: seconds–minutes each |
| E3 (20 instances × 1 zero-shot rollout + 9-PDR sweep) | PDRs near-instant; RL rollout sub-second at this scale | well under an hour total |
| E4+E5+E6 (once unblocked): ~4,860 short rollouts (9 regimes × 30 reps × 2 instances × up to 10 methods) | ft06-scale rollout ≈0.2–2s | **~2–3 hours**, one-time |
| E7: 2 new training runs | 400k-step ablation run ≈10–11 min/seed (GPU) | well under an hour of training compute |
| E8 CPU-only pass (4 instances) | — | minutes |

**Conclusion: raw compute is not the binding constraint anywhere in this matrix.** The binding constraints are (a) the DynamicJSSPEnv/TrainedRLBaseline wiring (Finding B — an engineering task, not a compute cost), (b) the FJSP loader's missing test coverage and missing baselines (Finding A), and (c) calendar days available given upcoming public holidays (§4). None of these are solved by reducing the pre-registered replicate counts, so they are kept as frozen.

---

## 4. Priority order if time runs short

This is deliberately **not** a simple "H1 > H2 > H3" ranking — H3 (robustness under disruption) is the thesis's own titular contribution, not a lesser hypothesis. The ordering below reflects *risk and dependency*, not importance:

1. **Unconditionally achievable, do first, no excuses:** E1, E3, E8 (share one execution pass per §2.8), E0 close-out. These have zero blockers and already-frozen decision rules. If nothing else in this plan completes, this tier alone is a defensible, internally consistent empirical chapter (H1 JSSP arm + H2 + H5).
2. **Highest-leverage single action in the whole plan:** attempt the `DynamicJSSPEnv`↔`TrainedRLBaseline` wiring (Finding B) starting immediately — ideally during Week 23's "light" week, not deferred to Week 24. It is bounded engineering (the dynamic environment already handles all three disruption axes correctly for PDRs), and clearing it early is what makes E4–E6 (and downstream, E7) possible at all inside this calendar. Treat it as time-boxed: if it is not working by the middle of Week 24, escalate to a documented, honestly-reported blocker (per `../../thesis/PreRegistration.md` §4's own instruction) rather than let it silently consume the whole sprint.
3. **Load-bearing, one day of prep, do once Tier 1 is running:** E2 (Arm A same-day as E1; Arm B — the real Brandimarte set — only after `test_fjsp_loader.py` passes and fresh baselines exist).
4. **Execute the moment Tier 2's wiring clears, cheap once unblocked:** E4, E5, E6.
5. **Sprint 2, contingent on Tier 4:** E7, plus the full Wilcoxon/Friedman/Nemenyi/CD-diagram pipeline over everything (E1–E8) and the hypothesis-by-hypothesis verdict table, exactly as the master plan already schedules for Week 27.

If the wiring in Tier 2 does not clear in time: report H3 and H4 explicitly as "pre-registered, scheduled, blocked at time of writing" in Chapter 6.4/7, per `../../thesis/PreRegistration.md` §4's own contingency — this is the honest outcome the freeze protocol was built to allow, not a failure to hide.

---

## 5. Required pre-registration amendment (append verbatim to `../../thesis/PreRegistration.md` §7)

```
### Amendment 1 — 2026-07-30

Reason: envs/fjsp_loader.py, scripts/fetch_fjsp_instances.py, data/fjsp_instances/
{mk01..mk07}.txt, and data/fjsp_known_optima.csv were added to the codebase on
2026-07-30, one day after this document's freeze (2026-07-29T05:22:26Z UTC). §1's
statement that no real FJSP benchmark loader is bundled is therefore stale as of this
amendment. Per §1's own contingency clause, this is logged as an ADDITION to H1's FJSP
arm (E2), not a substitution for the synthetic-instance arm already frozen there. No
FJSP results of any kind existed at the time this amendment was written, so this
addition does not constitute HARKing.

Change: E2/H1's FJSP arm now has two reported sub-arms:
  (a) Arm A (unchanged): synthetic FJSP variant of the 43 base JSSP instances via
      generate_fjsp_instance(seed=0), evaluated on raw makespan vs. GA/CP-SAT, exactly
      as §1/§2 originally specified.
  (b) Arm B (new): the real Brandimarte instances mk01-mk07 (only 7 of the 10 named in
      ../docs/planificacion/Step 4 Thesis.md D4's E2 row are bundled; no Hurink instances are available),
      evaluated via RPD against data/fjsp_known_optima.csv's best_makespan column
      (status=proven for mk01/mk03/mk04; status=best_known for mk02/mk05/mk06/mk07 --
      report this distinction, do not treat both statuses as equivalent optimality
      gaps). mk01-mk05 were exposed to the policy during Week 21's curriculum training
      (stages 20-24) and must be reported as in-sample; mk06/mk07 are zero-shot
      held-out, per ../docs/logs/HPO_ABLATION_REPORT_WEEK21.md's own generalization table.

Statistical test: identical Wilcoxon procedure as Arm A, computed separately, not
pooled with Arm A or with each other's in-sample/zero-shot subsets.

Precondition for this arm to be used in any confirmatory result: tests/test_fjsp_loader.py
must exist and pass (padding-neutrality: CP-SAT reaches the literature optimum on the
padded instance), and CP-SAT/GA baselines must be freshly measured on mk01-mk07 before
any RL-vs-baseline comparison is drawn, since none existed prior to this amendment.
```

---

## 6. Week-by-week execution schedule

The master plan labels these Semana 22–27 with placeholder December/January dates. Actual project execution is running on today's real calendar (`../../thesis/PreRegistration.md` was frozen yesterday, 2026-07-29; this review is dated 2026-07-30), so dates below are the real upcoming weeks, adjusted for actual Colombian public holidays in this window: **Friday, August 7, 2026** (Batalla de Boyacá) and **Monday, August 17, 2026** (Asunción de la Virgen, observed per Ley Emiliani) — both confirmed via current Colombian holiday calendars. The Semana N labels are kept as cross-references, since later planned documents (`../docs/logs/EXPERIMENT_LOG_WEEK24.md`, etc.) already use them.

| Week | Dates (2026) | Working days | Label | Content |
|---|---|---|---|---|
| Semana 22 | Mon Jul 27 – Fri Jul 31 | 5 | Cowork — this document | Finalize E0–E8 matrix (this file), log the §5 amendment, brief the wiring task for immediate start. |
| Semana 23 | Mon Aug 3 – Fri Aug 7 | **4** (Fri holiday) | Cowork, light + urgent engineering | Per the master plan: prepare (don't execute) the statistical-analysis tech spec. **Add:** start the `DynamicJSSPEnv`↔`TrainedRLBaseline` wiring now rather than waiting for Semana 24 — this is the one task whose early start determines whether E4–E7 are possible at all this cycle. |
| **Semana 24** | Mon Aug 10 – Fri Aug 14 | 5 | **🖥️ Code sprint 1** | Day 1: finish/validate wiring if not already done; E0 close-out write-up; start E1 rollouts (shared with E8) and E3's fresh PDR baseline sweep. Day 2: finish E1 + E8 stats; finish E3 zero-shot sweep + Friedman/Nemenyi. Day 3: E2 — write `test_fjsp_loader.py`, run CP-SAT/GA on `mk01`–`mk07`, log the amendment, run both FJSP arms. Day 4: if wiring succeeded, execute E4/E5/E6 (~2–3h compute, so this is mostly buffer/QA time); if not, document the blocker per §4. Day 5: buffer, descriptive stats only (no significance testing yet, per the roadmap's own STEP4), write `../docs/logs/EXPERIMENT_LOG_WEEK24.md`. |
| Semana 25 | Mon Aug 17 – Fri Aug 21 | **4** (Mon holiday) | Cowork — review | Review `../docs/logs/EXPERIMENT_LOG_WEEK24.md` against this priority order; decide rerun/scope-down/accept-as-limitation for anything incomplete; finalize E7 scope (`../../thesis/E7_E8_Plan.md`) including whether the 9-regime grid needs trimming based on what E4–E6 actually delivered. |
| Semana 26 | Mon Aug 24 – Fri Aug 28 | 5 | Cowork — writing | Draft Chapter 6 §§6.1–6.4 from real E1–E6 CSVs (../../archive/capitulos/Chapter6_template.md already exists from Week 20); report H1/H2 findings against `../../thesis/PreRegistration.md`'s frozen rules; report H3 honestly whether complete, partial, or blocked. |
| **Semana 27** | Mon Aug 31 – Fri Sep 4 | 5 | **🖥️ Code sprint 2** | E7 (train κ=0.5/1.0, evaluate under the same grid); E8's CPU-only pass if not already done; full Wilcoxon/Friedman/Nemenyi/CD pipeline over all completed experiments; `../../thesis/RESULTS_REPORT.md` with the hypothesis-by-hypothesis verdict table. |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation already built into this plan |
|---|---|---|---|
| `DynamicJSSPEnv` wiring takes longer than one week | Medium | High (H3/H4 both depend on it) | Started in Semana 23, not 24; time-boxed with an explicit escalation point mid-Semana 24; H3/H4 have a pre-sanctioned honest "blocked" reporting path |
| Real `mk01`–`mk07` loader has an undiscovered bug (padding neutrality, machine-index off-by-one) | Medium (zero test coverage today) | Medium (would invalidate Arm B only, Arm A unaffected) | `test_fjsp_loader.py` gate required before any Arm B result is used confirmatively |
| Two more holiday-shortened weeks reduce available days further | Low (already accounted) | Low | Schedule above already reflects the two known Aug holidays; no hidden slack assumed |
| E7's two new training runs reveal training instability at κ>0 (untested reward term) | Medium | Low (E7 is Tier 3/supplementary this window) | Already sequenced last, in Sprint 2, after everything else |
