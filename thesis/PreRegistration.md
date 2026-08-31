# Pre-Registration — GNN+PPO for Dynamic FJSP under Stochastic Perturbations

**Status: FROZEN.**
**Freeze timestamp: 2026-07-29T05:22:26Z UTC (2026-07-29, 00:22 America/Bogotá).**
**Frozen by: Joseph Sanchez, prior to any GNN+PPO training or evaluation result.**

**Purpose of this freeze.** This document exists to protect the methodological defense of this thesis against **HARKing** (Hypothesizing After Results are Known) — `../docs/planificacion/Step 4 Thesis.md`'s own execution-order recommendation (item 4) states this requirement explicitly: *"Congelar el pre-registro (D7) antes de generar cualquier resultado."* As of this timestamp, **zero GNN+PPO training results exist**: `experiments/train_ft06.py`'s Week 1 run was an explicitly-labeled pipeline smoke test ("deliberately not a convergence run"), and the heterogeneous encoder + size-agnostic policy specified in `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` had not yet been implemented as of the Week 17 training plan this document builds on. Every threshold, statistical test, and comparator instance set below is fixed **now**, in writing, before that changes.

**Amendment protocol.** Any change to a hypothesis, threshold, test, or instance set below, made after this freeze timestamp, must be recorded as a dated addendum in §7, with an explicit justification distinguishing it from a post-hoc rationalization of an observed result. Silent edits to §§1–6 after this timestamp defeat the entire purpose of this document and are not permitted by the protocol it establishes.

**Provenance.** This document operationalizes `../docs/planificacion/Step 4 Thesis.md` Part D7 ("Pre-registro") against the codebase as it actually stands after the Week 15 refactor (`REFACTOR_NOTES_WEEK15.md`) and the Week 17 architecture/training plan (`../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md`, `../docs/logs/TrainingPlan_Week17.md`). D7's five hypotheses (H1–H5) and their qualitative decision rules are reproduced and made quantitatively explicit below; nothing in D7's substance is changed, only made executable against real instance names, real CSV columns, and real code that now exists.

---

## 0. What is actually testable today, and what is blocked — read this section first

Freezing this document does not mean every hypothesis below can be tested immediately. Being honest about which hypotheses are currently blocked, and on exactly what, is itself part of pre-registering in good faith — a blocked hypothesis whose blocker is discovered *after* trying to test it invites exactly the kind of after-the-fact rationalization this document exists to prevent.

| Hypothesis | Status as of this freeze | Blocking dependency |
|---|---|---|
| **H1** (static competitiveness) | **Testable once Week 17/18's trained policy checkpoint exists.** No architectural blocker. | Trained `state_dict` from `../docs/logs/TrainingPlan_Week17.md` Stage A/B, wrapped as `TrainedRLBaseline` per `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §8. |
| **H2** (size generalization) | **Testable once H1's checkpoint exists**, via the same weight-transfer mechanism (`../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §3.4). No additional blocker beyond H1's. | Same as H1. |
| **H3** (robustness under disruption) | **Blocked.** `TrainedRLBaseline` (§8.1 of the architecture spec) is explicitly scoped to `--disruption none` only (§8.3): the trained policy targets `JSSPEnv`'s static observation contract, and wiring it against `DynamicJSSPEnv` (the machine-`broken` state, the growable-array observation contract) is deferred past this sprint — `VALIDATION_WEEK6.md`'s "Next steps" and `REFACTOR_NOTES_WEEK15.md`'s scope boundaries both independently confirm this is still undone. | `DynamicJSSPEnv` integration (a future sprint, not yet scheduled as of this freeze). |
| **H4** (disruption-aware reward) | **Blocked**, on two independent items. (a) Same `DynamicJSSPEnv` wiring blocker as H3. (b) The $\kappa \cdot \Delta_{\text{inest}}$ reward term itself (`../docs/planificacion/Step 4 Thesis.md` Part D3) is **not implemented** — `VALIDATION_WEEK6.md`'s "Next steps" lists it explicitly as unbuilt: *"Implement the `kappa * Delta_inest` robustness-aware reward term (Part D3); this sprint only wired the plumbing (`info["planned_start_deviation"]`)."* | `DynamicJSSPEnv` wiring **and** the reward term itself. Both tracked jointly in `../docs/planificacion/AblationPlan.md` Ablation C, since H4 and that ablation are the same experiment. |
| **H5** (decision latency) | **Testable once H1's checkpoint exists.** Does not require disruption or `DynamicJSSPEnv` — pure inference-time measurement against the static policy. | Same as H1. |

**Consequence for how this document is used:** H1, H2, and H5's decision rules (§§2, 3, 6) can be executed as soon as a trained checkpoint lands. H3 and H4 (§§4, 5) are pre-registered in full now, precisely so that when their blockers clear — in a later sprint, not this one — the decision rule is already fixed and cannot be adjusted in light of whatever the dynamic-disruption results turn out to look like.

---

## 1. Frozen test-instance universe and baseline reference values

**Deviation flagged against `../docs/planificacion/Step 4 Thesis.md` D4/E1/E2:** D4's experimental matrix names `ta` (Taillard) and `MK01–MK10`/Hurink (Brandimarte FJSP) benchmark families for E1/E2. As of this freeze, neither is bundled: `scripts/fetch_instances.py`'s own `WANTED_PREFIXES = ("ft", "la")` only explodes the Fisher-Thompson and Lawrence families from the cached OR-Library file (`data/raw/jobshop1.txt`) into `data/instances/`; Taillard's `ta*` set is a separate published source not currently fetched at all, and `REFACTOR_NOTES_WEEK15.md` explicitly lists "no real FJSP benchmark file loader" (no Brandimarte `MK*`/Hurink parser) as out of scope for that sprint. **Freezing the test set as what is actually reproducible today, rather than pre-registering against benchmark families the codebase cannot yet load:**

- **JSSP static test set (H1, H2 JSSP arm):** all 43 bundled JSSP instances — `ft06`, `ft10`, `ft20`, `la01`–`la40` — with proven optima recorded in `data/known_optima.csv` (verified: every row's `status` column reads `proven`, so RPD against `known_optimum` is always a true optimality gap, never a best-known-bound approximation).
- **FJSP test set (H1 FJSP arm):** the synthetic FJSP variant of the same 43 base instances via `envs.fjsp_instance.generate_fjsp_instance(base, seed=0)` (`run_evaluation.py --fjsp --fjsp-seed 0`, matching the CLI default). Per that module's own docstring, `known_optimum`/`rpd_pct` are undefined for this family (a synthetic instance's true optimum is generically better than the base JSSP instance's, since routing flexibility only helps) — **H1's FJSP arm is therefore evaluated on raw makespan and on relative comparison to the other baselines on the same synthetic instances, not on RPD against a literature value.** If a real FJSP loader (`MK*`/Hurink) is built before H1 is executed, it should be added as a second, separately-reported FJSP arm — an *addition*, not a substitution for this one (§7 amendment, not a silent swap).
- **Extending the family set:** `scripts/fetch_instances.py`'s `WANTED_PREFIXES` tuple already documents itself as extensible ("the remaining families stay available in the cached raw file"). If `abz`/`orb`/`swv`/`yn` are fetched before H1/H2 execution, they may be added to the frozen set as a dated addendum (§7) — they were not available to reason about at freeze time, so they are not silently assumed here.

**Baseline reference values are frozen from already-published, already-tested measurements, not re-run ad hoc.** `BASELINES_RESULTS_WEEK12.md` / `experiments/results/baselines_week12.md` already contain one measured row per `(instance, method, seed=0)` for all nine PDRs (`spt`, `lpt`, `mwkr`, `lwkr`, `mor`, `lor`, `fifo`, `edd`, `cr`), GA, and CP-SAT, across `ft06`/`ft10`/`ft20` and the full `la01`–`la40` set — this is the exact source `REFACTOR_NOTES_WEEK15.md` says was deliberately preserved (`experiments/eval_baselines.py` "left untouched... so the numbers already cited... stay reproducible from the exact script that produced them"). **H1's "best PDR" comparator per instance is the minimum makespan among that instance's nine already-recorded PDR rows in that file — not a value to be re-measured, re-selected, or re-run once RL results are in hand.** Concrete anchor values reused throughout §§2–6: `ft06` optimum 55, best PDR (`fifo`) 67 (21.82% RPD); `ft10` optimum 930, best PDR (`fifo`) 1218 (30.97% RPD); GA/CP-SAT reach the proven optimum on `ft06`/`ft10` (0% RPD in both cases).

**Disruption-grid values are frozen from `../docs/planificacion/Step 4 Thesis.md` D4/D5, not from `run_evaluation.py`'s CLI defaults.** `run_evaluation.py --disruption breakdowns` etc. ship with convenience defaults (`--availability 0.8`, `--proc-sigma 0.25`, `--utilization 0.6`, `--max-dynamic-jobs 20`) that do **not** match D4's registered experimental grid (`A_k ∈ {0.95, 0.90, 0.80}`, `σ ∈ {0.1, 0.25, 0.5}`, utilization `ρ ∈ {0.7, 0.85, 0.95}`). **§§4–5 below use D4's grid, passed explicitly via CLI flags when H3/H4 are eventually executed — the tool's own defaults are for ad hoc exploration, not the frozen experimental design.**

---

## 2. H1 — Static competitiveness

**Claim (verbatim from D7, reproduced):** the policy outperforms the best PDR in mean RPD over the benchmark set.

**Operationalized test set:** the 43-instance JSSP set (§1) at `--disruption none`, plus the FJSP-synthetic arm (§1), per D4's E1/E2 rows.

**Comparator:** for each instance, `RPD_bestPDR` = the minimum makespan's RPD among the nine already-recorded PDR rows in `experiments/results/baselines_week12.md` (JSSP arm) or a freshly-run equivalent under `--fjsp` (FJSP arm, since Week 12's results predate FJSP support). `RPD_RL` = the trained policy's RPD via `TrainedRLBaseline` (`../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §8), one deterministic rollout per instance (`deterministic=True` in `.predict()`) plus, per D4's "5 (política estocástica)" replication note, four additional stochastic rollouts (`deterministic=False`, seeds 1–4) to characterize variance — the deterministic rollout's RPD is the primary value entering the paired test below; the four stochastic rollouts are reported descriptively (mean ± std) and are not part of the accept/reject computation, to keep the primary test's unit of analysis at exactly one paired observation per instance, matching D4's own per-instance pairing for the Wilcoxon procedure in D6.

**Statistical test (fixed in advance):** paired **Wilcoxon signed-rank test** on $(\text{RPD}_{\text{bestPDR},i} - \text{RPD}_{\text{RL},i})$ across all $N=43$ JSSP instances, $\alpha = 0.05$, two-sided, using `scipy.stats.wilcoxon` (matching the tooling already used for `tests/test_disruptions.py`'s calibration checks, so no new statistical dependency is introduced). Report the statistic $W$, the $p$-value, and effect size $r = Z/\sqrt{N}$, per D6's exact formula. The FJSP-synthetic arm is analyzed identically but reported **separately** (not pooled with the JSSP arm into one $N=43+43$ test), since RPD is not even defined the same way for the two arms (§1).

**Decision rule (exact, fixed in advance):**

> **Accept H1** (JSSP arm) if and only if $p < 0.05$ **and** $\text{median}_i(\text{RPD}_{\text{bestPDR},i} - \text{RPD}_{\text{RL},i}) > 0$.
> **Accept H1** (FJSP arm) if and only if the same test, computed on raw makespan difference (best-PDR-family makespan minus RL makespan per instance — restricted to `ga`/`cpsat` as the FJSP-capable baseline comparators per `REFACTOR_NOTES_WEEK15.md`'s STEP1b point 5, since no PDR has an FJSP machine-selection rule to serve as "best PDR" in this arm), satisfies $p<0.05$ and median difference $>0$.
> **Reject H1** otherwise, including the case $p \geq 0.05$ regardless of the sign of the median difference (an improvement that is not statistically distinguishable from noise is not evidence for H1, by construction of this rule).

**Descriptive context, not part of the accept/reject rule:** report $\text{RPD}_{\text{RL}}$ against CP-SAT's RPD (0% wherever CP-SAT reaches proven optimum within its time limit) as context for D7/E2's expectation "por encima de CP-SAT" (worse than the exact-solver ceiling) — this is a documented expectation, not a threshold RL must clear, and its violation (RL beating CP-SAT) would not falsify H1, only be a surprising and separately reportable finding.

---

## 3. H2 — Size generalization

**Claim (verbatim from D7, reproduced):** RPD on instances 2–3× larger than the training size stays below a threshold (e.g., ≤ best-PDR RPD at that size).

**Deviation flagged against D4:** D4's E3 row names test sizes "15×15, 20×15, 30×20, ta grandes." The bundled `la` corpus's largest instances are `la31`–`la35` (30×10) and `la36`–`la40` (15×15); `20×15` and `30×20` do not exist in the bundled set, and `ta`'s large instances are unavailable per §1. **Test sizes are frozen against what is actually bundled**, grouped by ratio to the two training sizes (`ft06`=36 ops, 10×10 tier=100 ops, per `../docs/logs/TrainingPlan_Week17.md` Stage A/B):

| Tier | Instances | `num_operations` | Ratio to 10×10 training size (100 ops) |
|---|---|---|---|
| ~1.5–2× | `la21`–`la25` (15×10), `la26`–`la30` (20×10) | 150, 200 | 1.5×, 2.0× |
| ~2.25× | `la36`–`la40` (15×15) | 225 | 2.25×, structurally denser (more machines per job) than the other tiers at similar op count |
| ~3× | `la31`–`la35` (30×10) | 300 | 3.0× |

`la01`–`la15` (40–100 ops, i.e. at or below the training range) are **not** part of the H2 generalization test — they are the in-distribution / interpolation set, already covered descriptively by H1, and mixing them into H2's decision rule would understate the generalization claim by diluting it with near-training-size instances.

**Comparator, per tier:** the same nine-PDR best-of comparator as H1, computed per instance within the tier from a fresh `run_evaluation.py --instances <tier> --methods pdr` run (Week 12's results only cover `ft06`/`ft10`/`ft20`/`la01`–`la10`, so tiers above `la10` need a first measurement — this is a baseline measurement, run once, before RL results exist, exactly like Week 9/12's runs were).

**RL evaluation mechanism:** zero-shot weight transfer, per `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §3.4 — load the Stage B (10×10-trained) `state_dict` into a freshly constructed `PerNodeMaskableActorCriticPolicy` wrapping each tier's instances, **no additional gradient updates**. This is the literal mechanism the architecture was built to support, not an approximation of it.

**Statistical test (fixed in advance):** per tier, a **Friedman test** across $k=10$ methods (RL + the nine PDRs) ranked over the $N$ instances in that tier, $\alpha=0.05$; if the Friedman null is rejected, a **Nemenyi post-hoc** with critical difference $CD = q_\alpha\sqrt{k(k+1)/6N}$ (D6's exact formula), producing average ranks and pairwise significant-difference flags.

**Decision rule (exact, fixed in advance):**

> **Accept H2** if and only if, for **every** tier in the table above: (a) the Friedman test rejects the equal-performance null ($p<0.05$) — a non-rejection means the methods aren't distinguishable at all in that tier, which is neither an accept nor a reject of H2's specific claim and must be reported as **inconclusive for that tier**, not silently folded into either side; and (b) given rejection, RL's average rank is not significantly worse than the best PDR's average rank per the Nemenyi critical difference (RL's rank may be *better* or *statistically tied* with the best PDR, but not significantly worse).
> **Reject H2** if RL is significantly worse than the best PDR (by Nemenyi CD) in **any** tier.
> Report the **degradation gradient** — $\text{RPD}_{\text{RL}}(\text{tier}) - \text{RPD}_{\text{RL}}(\text{10×10 training tier})$ across the three tiers — descriptively; D7 sets no formal statistical test for monotonicity or magnitude of degradation, so this is reported for Chapter 7's discussion, not used in the accept/reject computation.

---

## 4. H3 — Robustness (blocked — see §0)

**Claim (verbatim from D7, reproduced):** the model's robustness index $RI$ is lower than the PDRs' under breakdowns/stochastic times/arrivals.

**Frozen experimental grid (D4 E4/E5/E6, D5), independent of `run_evaluation.py`'s CLI defaults per §1:**

| Axis | Levels | Replicates (seeds) | `DisruptionConfig` fields |
|---|---|---|---|
| Breakdowns (E4) | $A_k \in \{0.95, 0.90, 0.80\}$ | 30 | `target_availability`, `mttr_multiplier=2.0` (codebase default), `ttr_shape_sigma=0.5` (codebase default) |
| Stochastic times (E5) | $\sigma \in \{0.1, 0.25, 0.5\}$ | 30 | `processing_time_sigma` |
| Arrivals (E6) | utilization $\rho \in \{0.7, 0.85, 0.95\}$ | 30 | `target_utilization`, `max_dynamic_jobs` fixed at a value large enough not to bind before `arrival_cutoff_time` — set explicitly per instance rather than left at the CLI's `max_dynamic_jobs=20` default, since that default was chosen for ad hoc sweeps, not this grid |

9 regimes total (3 axes × 3 levels each, evaluated as **separate single-axis sweeps** per D4's E4/E5/E6 rows — `run_evaluation.py --disruption combined`, which turns on all three axes simultaneously, is an available secondary/exploratory arm, reported descriptively, and is explicitly **not** part of H3's primary accept/reject rule, since D4 never registered a combined-axis grid for this hypothesis).

**Instance set:** `ft06` and `la01` per `VALIDATION_WEEK6.md`'s already-established disruption test protocol (`TestConstraintSafetyUnderBreakdown` already exercises exactly this pair) — reusing an instance pair the disruption machinery is already validated against, rather than introducing a new untested pair for the first RL-under-disruption measurement.

**Comparator:** the arrival-safe PDR subset only when arrivals are enabled (`RULES.keys() - ARRIVAL_UNSAFE_PDR_NAMES`, i.e. excluding `spt`/`lpt`/`mwkr`/`edd`/`cr` per `REFACTOR_NOTES_WEEK15.md`'s `ARRIVAL_UNSAFE_PDR_NAMES` guard) rolled out live via `DynamicJSSPEnv` (`run_pdr_online`); all nine PDRs for the breakdown-only and stochastic-time-only regimes, where no arrival-safety restriction applies.

**RI formula (D6, exact):** $RI = \dfrac{\mathbb{E}[C_{\max}^{\text{pert}}] - C_{\max}^{\text{nom}}}{C_{\max}^{\text{nom}}}$, complemented with standard deviation and $\text{CVaR}_{0.9}$ per D6.

**Statistical test (fixed in advance):** paired **Wilcoxon signed-rank test** on $(RI_{\text{PDR},i} - RI_{\text{RL},i})$ per (instance × seed) pair, computed **separately for each of the 9 regimes**, $\alpha=0.05$.

**Decision rule (exact, fixed in advance):**

> **Accept H3** for a given regime if and only if $p<0.05$ **and** median $(RI_{\text{PDR}} - RI_{\text{RL}}) > 0$ in that regime, using the best-performing PDR (lowest $RI$) as the comparator, consistent with H1/H2's "best PDR" convention.
> **Accept H3 overall** only if the regime-level rule holds in **all 9** regimes; partial support (e.g., robust under breakdowns but not arrivals) is reported explicitly as **partial support for H3**, not rounded up to a full accept — this distinction matters methodologically since D7's original wording ("bajo averías/tiempos/llegadas") reads as a conjunctive claim across all three perturbation types.

**Cannot be executed until:** `DynamicJSSPEnv` wiring for the trained policy lands (§0). This decision rule is frozen now specifically so that, once that wiring exists, no adjustment is made in light of how the robustness numbers turn out.

---

## 5. H4 — Disruption-aware reward (blocked — see §0)

**Claim (verbatim from D7, reproduced):** $\kappa>0$ reduces $RI$ without increasing nominal RPD by more than a margin $\delta$ (D7's own example: 2 percentage points).

**Frozen parameters (D4 E7, D3):** $\kappa \in \{0, 0.5, 1.0\}$, 30 replicates, reward $r_t = (LB_{t-1}-LB_t) - \kappa \cdot \Delta_{\text{inest}}$ where $\Delta_{\text{inest}}$ penalizes deviation from planned start times induced by reprogramming after a disruption event (D3's exact formula). $\kappa=0$ recovers the standard dense reward exactly — this is the **same trained policy already used for H1/H2/H3** in that arm, not a fourth separately-trained model, so the $\kappa=0$ column of this experiment and H1–H3's policy are one and the same run, reused rather than duplicated.

**Instance set and disruption regime:** same as H3 (`ft06`, `la01`; the same 9-regime grid, evaluated per-regime since $\Delta_{\text{inest}}$'s relevance and magnitude differ by perturbation type).

**Two quantities per $\kappa \in \{0.5, 1.0\}$, each compared against the $\kappa=0$ policy:**

1. $\Delta RI = RI_{\kappa>0} - RI_{\kappa=0}$ (robustness change), paired by (instance × seed).
2. $\Delta \text{RPD}_{\text{nominal}} = \text{RPD}_{\kappa>0}^{\text{no-disruption}} - \text{RPD}_{\kappa=0}^{\text{no-disruption}}$ (nominal-performance cost), measured under `--disruption none` — the "does the robustness-aware reward hurt static competitiveness" check.

**Statistical test (fixed in advance):** **Wilcoxon signed-rank test** on $\Delta RI$ (paired by instance × seed, per regime, $\alpha=0.05$), matching H3's test exactly. $\Delta \text{RPD}_{\text{nominal}}$ is evaluated against the fixed **margin** $\delta = 2$ **percentage points**, per D7's own example value — **as a threshold comparison, not a significance test**, since D7 specifies a margin, not a hypothesis test, for this quantity; this asymmetry (significance test for the robustness gain, simple threshold for the nominal-cost side) is deliberate and is stated explicitly here rather than silently applying a significance test D7 never specified.

**Decision rule (exact, fixed in advance), evaluated independently for each of $\kappa \in \{0.5, 1.0\}$:**

> **Accept H4** for a given $\kappa$ if and only if: (a) the Wilcoxon test on $\Delta RI$ gives $p<0.05$ with median $\Delta RI < 0$ (robustness improves) in a **majority** of the 9 regimes (5 or more) — a majority rather than all 9, unlike H3's conjunctive rule, because H4's claim is that the mechanism *works*, not that it works uniformly everywhere perturbation exists; **and** (b) $\Delta \text{RPD}_{\text{nominal}} < 2$ percentage points.
> If (a) holds but (b) does not: report as **"robustness gained at an unacceptable nominal cost"** — a specific, named failure mode, not a blanket rejection with no diagnostic content.
> If (b) holds but (a) does not: report as **"no measurable robustness benefit at this $\kappa$"**.
> **Reject H4** for a $\kappa$ if neither (a) nor (b) holds.

**Cannot be executed until:** both blockers in §0 clear (`DynamicJSSPEnv` wiring, and the $\kappa \cdot \Delta_{\text{inest}}$ term itself — currently only `info["planned_start_deviation"]` plumbing exists per `VALIDATION_WEEK6.md`). This is the same experiment as `../docs/planificacion/AblationPlan.md` Ablation C; the two documents are not independently commissioning duplicate work.

---

## 6. H5 — Decision latency

**Claim (verbatim from D7, reproduced):** the model's per-decision time is far smaller than CP-SAT's for large instances.

**Instance set:** the full size range already established in §§1–3 — `ft06` (36 ops) through `la31`–`la35` (300 ops) — specifically to test D7's "para tamaños grandes" qualifier as a *trend across scale*, not a single-size snapshot.

**Measurement protocol:** wall-clock time per decision epoch, averaged over a full episode, over **5 timed replicates per instance** (matching D4's E1 replicate convention) on the same hardware used for training (`../docs/logs/TrainingPlan_Week17.md` §1's single-GPU/CPU assumption) — CPU-only inference timing should also be recorded separately if training used GPU, since a thesis claim about "orders of magnitude faster than CP-SAT" should hold under the hardware a reader is actually likely to deploy on, not only under the training accelerator.

**Comparator:** CP-SAT's wall-clock time per instance, already recorded in `BASELINES_RESULTS_WEEK12.md`'s `wall_clock_s` column for the JSSP set (e.g. `ft06`: 0.035s total for the whole instance solve, `ft10`: 3.528s) — converted to a per-decision figure by dividing by `num_operations` (the number of decision epochs in one complete schedule), for a fair epoch-for-epoch comparison against the RL policy's per-step inference time.

**Statistical test:** none required by D7's own wording — the threshold below is a ratio comparison on medians, not a hypothesis test. Report the **median** decision time per instance for both methods (median, not mean, to avoid one slow outlier epoch dominating the summary, consistent with D6's general preference for non-parametric summaries elsewhere in this protocol).

**Decision rule (exact, fixed in advance):**

> **Accept H5** if and only if, for **every** instance in the size range above, $\text{median}(\text{ms/decision}_{\text{RL}}) < 0.01 \times \text{median}(\text{ms/decision}_{\text{CP-SAT}})$ (i.e., RL's median decision time is under 1% of CP-SAT's, per D7's exact threshold).
> Additionally report whether the **ratio** $\text{ms/decision}_{\text{CP-SAT}} / \text{ms/decision}_{\text{RL}}$ **widens** with instance size — this is the concrete form of D7/E2's "one-shot inference vs. search" reasoning, and is the figure Chapter 6.5 should lead with, since a constant ratio across sizes would undercut the "for large instances especially" framing of the claim even if the 1% threshold is met everywhere.
> **Reject H5** if the 1% threshold is violated for any instance in the size range.

---

## 7. Amendments (empty at freeze time)

No amendments have been made as of this document's freeze timestamp. Any future change to §§1–6 must be appended below, not edited in place, with its own timestamp and a stated reason distinct from "the result came out differently than expected."

### Amendment 2 (2026-07-31) — §6's GPU-primary/CPU-secondary framing collapses to one measurement

**Reason, distinct from "the result came out differently than expected":** this is a framing correction surfaced by reading the actual evaluation code path (`E7_E8_Plan.md` §4.3), not a result-driven change to any threshold or test. §6 asks for timing "on the same hardware used for training... CPU-only inference timing should also be recorded separately if training used GPU" — a GPU-primary/CPU-secondary framing. `baselines.interface.TrainedRLBaseline._policy_for` constructs the policy raw and never calls `.to(device)`, so this codebase's evaluation path runs on CPU regardless of GPU availability (confirmed directly against the saved `state_dict`/construction code, `../docs/logs/EXPERIMENT_LOG_WEEK24.md` Finding #1). The Week 25 clean re-measurement machine additionally has no CUDA-enabled torch build at all (`torch.cuda.is_available()` is unconditionally `False`), so there is no GPU-inference number to report as "primary" under any reading of §6.

**What this changes:** "training hardware" and "CPU-only" collapse to one measurement in this codebase — there is nothing to separately record. The multi-thread/single-thread CPU contrast already captured in `../docs/logs/EXPERIMENT_LOG_WEEK24.md`'s E8 section (and reproduced in the Week 25 clean pass) is retained as the closest honest analogue to §6's "does the picture change under different hardware" intent. **H5's decision rule itself (the 1% threshold, the per-instance median comparison) is unchanged** — this amendment only clarifies what "hardware" means for the measurement, not the test.
