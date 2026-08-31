# Ablation Study Plan — Month 7 Coding Sprint

**Drafted: 2026-07-29.** Companion document to `../../thesis/PreRegistration.md` (frozen the same day) and `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md`. This plan is **not itself a frozen pre-registration** — ablations are design-justification experiments (which architectural/reward choice was actually necessary, not just theoretically motivated), not confirmatory tests of this thesis's core hypotheses. The one exception is Ablation C, which **is** H4 (`../../thesis/PreRegistration.md` §5) under a different name — the two documents commission it once, not twice; see the cross-reference in §2.3.

**Scope:** four ablation axes, executed one-axis-at-a-time against a fixed reference configuration (not a full factorial sweep — §1 explains why), scheduled for Month 7 once Week 17–18's primary training run (`../docs/logs/TrainingPlan_Week17.md`) has produced a working reference checkpoint to ablate against. **This plan cannot start before that checkpoint exists** — an ablation study needs a baseline to subtract from.

---

## 1. Why one-axis-at-a-time, not full factorial

Four axes (encoder: 3 variants; reward density: 2; reward disruption-awareness: 3 including the reference; objective: up to 4, per §2.4) would be $3 \times 2 \times 3 \times 4 = 72$ configurations if crossed fully. `../docs/planificacion/Step 4 Thesis.md` §1.5's compute budget (single GPU, one student, 9 months, already substantially committed to the primary training curriculum in `../docs/logs/TrainingPlan_Week17.md`) does not support that. Standard ablation methodology — vary one factor, hold the rest at the reference configuration's values — is used instead: each axis's table (§2) is evaluated independently against the same reference config, not against each other's variants. This means the plan cannot answer *interaction* questions (e.g., "does the GAT-hybrid encoder change how much the disruption-aware reward helps?") — flagged explicitly as an accepted limitation of a single-GPU-budget ablation plan, not an oversight, and a candidate for a future sprint if a specific interaction becomes theoretically interesting after the main-axis results are in.

**Reference configuration (every ablation's control arm):** Heterogeneous GIN encoder (`../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §2), dense standard reward ($\kappa=0$), single-objective ($C_{\max}$ only) — i.e., exactly the policy `../docs/logs/TrainingPlan_Week17.md` trains and `../../thesis/PreRegistration.md` §§2–3 evaluates for H1/H2. No separate reference run is commissioned by this plan; it is the same checkpoint, reused.

---

## 2. The four axes

### 2.1 Axis 1 — Encoder architecture

| Variant | Description | Code status |
|---|---|---|
| **Homogeneous GIN** (control-of-controls) | `GINEncoder` / `JSSPGraphFeaturesExtractor` — the Sprint-1 architecture, no machine nodes, no eligibility relation. Already implemented and validated (`tests/test_policy.py`, `VALIDATION_WEEK1.md`). | **Exists**, zero new code. |
| **Heterogeneous GIN sum+MLP** (reference/default) | `HeteroGINEncoder` — operation + machine nodes, four relations, uniform sum-then-MLP aggregation including the eligibility relation. `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §2. | Built in Week 17 (blocking dependency for this whole plan). |
| **Heterogeneous GAT-hybrid** | Song et al. (2023)-style: swap `HeteroRelationalGINLayer`'s machine-update term (`mach_msg`) for a single-head `GATedge`-style attention aggregator over the eligibility relation, keeping every operation-side relation (precedence, machine-sharing, eligibility-into-operation) on sum+MLP unchanged — exactly the "documented future ablation" code comment already present in `models/gin.py` per the architecture spec §2.1. | **New code needed**: one new class, `HeteroRelationalGATHybridLayer`, additive alongside `HeteroRelationalGINLayer`; single attention head (`num_heads=1`, matching Song et al.'s actual configuration, not their parameterized capability) so the comparison isolates "attention vs. sum aggregation," not "one head vs. many." |

**Rationale for testing this at all, given `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §2.1 already argues for sum+MLP:** that argument is theoretical (WL-expressiveness, Xu et al. 2019 Theorem 3) and precedent-based (neither L2D nor Song et al. uses attention uniformly). It has not been tested empirically on this thesis's own instances. An ablation that confirms sum+MLP was the right default is a stronger methodological position for Chapter 4/7 than an architecture choice justified by citation alone — and if the GAT-hybrid variant *wins*, that is a real, reportable finding, not a failure of the plan.

**Metric:** RPD vs. optimum, static (`--disruption none`), on the `ft06` + 10×10 training tier (matching `../docs/logs/TrainingPlan_Week17.md`'s Stage A/B instances — this axis asks "does the encoder choice matter at the size it's trained on," not a generalization question, so the larger `la31`–`la40` tiers are out of scope for this specific ablation; a size-crossed encoder comparison is a candidate future extension, not this plan).

**Statistical procedure:** 3 variants ⇒ **Friedman test** across the 3 encoder variants, ranked per instance, over the `ft06` + five 10×10 instances (6 instances total); if rejected, **Nemenyi post-hoc** with the same $CD$ formula as `../../thesis/PreRegistration.md` §3. This is a design-justification comparison, not a hypothesis this thesis stakes a claim on in advance — no pre-registered accept/reject threshold, results are reported as-is in Chapter 7's discussion of design choices.

### 2.2 Axis 2 — Reward density

| Variant | Description | Code status |
|---|---|---|
| **Dense (reference)** | $r_t = (LB_{t-1}-LB_t)\cdot\text{reward\_scale}$, current `JSSPEnv`/`DynamicJSSPEnv` behavior, unchanged. | Exists. |
| **Sparse** | $r_t = 0$ for all $t <$ terminal, $r_T = -C_{\max} \cdot \text{reward\_scale}$ at the terminal step only. Same telescoping total return as the dense case (both sum to $-C_{\max} \cdot \text{reward\_scale}$ up to the $H_0$ constant), differing only in *when* the signal arrives — the direct empirical test of Part B3.2's "dispersa... alta varianza, difícil asignación de crédito" (sparse reward, high variance, difficult credit assignment) claim, and of `../../archive/capitulos/Section4.4.md`'s own justification for the dense choice. | **New code needed**: a `reward_mode: Literal["dense", "sparse"] = "dense"` constructor argument on `JSSPEnv` (and `DynamicJSSPEnv` if this axis is later extended past static — not this sprint, see §4), additive alongside `include_machine_edges`, following that argument's own precedent for how an ablation-only behavior switch is exposed without touching the default path. |

**Metric:** **sample efficiency**, not final RPD alone — the credit-assignment argument predicts sparse reward needs *more* environment interactions to reach the same makespan quality, not that it necessarily converges to a worse final answer given unlimited budget. Report episode-return and makespan learning curves (same `EpisodeStatsCallback`/TensorBoard machinery as `../docs/logs/TrainingPlan_Week17.md` §4) for both variants trained under an **identical fixed timestep budget** (Stage A's 300,000–500,000 timesteps on `ft06`), and compare: (a) final RPD at budget exhaustion, (b) timesteps-to-reach-best-PDR-parity (the same "crosses below `fifo`'s 67" milestone `../docs/logs/TrainingPlan_Week17.md` §3.1 defines for the reference run) — the sparse variant not reaching that milestone at all within budget is itself the expected, reportable outcome if the credit-assignment argument holds.

**Statistical procedure:** 2 variants, one instance (`ft06`) across multiple seeds (5, matching D4's standard replicate count) ⇒ **Wilcoxon signed-rank** on final-RPD paired by seed, plus the descriptive milestone-crossing comparison above (not everything here needs to be a formal test — the learning-curve shape difference is the primary finding, best shown graphically per `../docs/logs/TrainingPlan_Week17.md` §3's own emphasis on curve shape over single-number summaries).

### 2.3 Axis 3 — Reward disruption-awareness (= H4, not a separate commission)

| Variant | Description | Code status |
|---|---|---|
| $\kappa=0$ (reference) | Standard dense reward, no instability penalty. | Exists (same run as Axis 1/2's reference). |
| $\kappa=0.5$ | $r_t = (LB_{t-1}-LB_t) - 0.5\cdot\Delta_{\text{inest}}$ | **Blocked**, same as `../../thesis/PreRegistration.md` H4: `DynamicJSSPEnv` wiring for the trained policy, plus the $\Delta_{\text{inest}}$ term itself (`VALIDATION_WEEK6.md` "Next steps": only `info["planned_start_deviation"]` plumbing exists today). |
| $\kappa=1.0$ | $r_t = (LB_{t-1}-LB_t) - 1.0\cdot\Delta_{\text{inest}}$ | Same blocker. |

**This row is included in this table for completeness of "the four ablation axes," but is not separately scheduled — it is executed exactly once, as H4, with H4's pre-registered decision rule (`../../thesis/PreRegistration.md` §5) as the analysis, and reported in both documents by cross-reference.** Do not re-run this as an "ablation" with a different, looser analysis after H4's confirmatory result is known — that would be a second, unregistered look at the same data, which is precisely what freezing H4 in `../../thesis/PreRegistration.md` was meant to prevent.

**Dated methodological note (2026-07-31), written before E7 training starts, per `../../thesis/E7_E8_Plan.md` §2.2/§5's own instruction that this operational detail — left open by `../docs/planificacion/Step 4 Thesis.md` D3, `../../thesis/PreRegistration.md` §5, and this table — be decided and recorded prospectively, not reconstructed after results exist. Not a `../../thesis/PreRegistration.md` amendment: no frozen instance set, replicate count, or decision-rule threshold changes here, only what the policy is trained against.**

1. **Training disruption regime: domain-randomized, not fixed** (`../../thesis/E7_E8_Plan.md` §2.2 "Option B"). Every training episode samples one of the same 9 regimes `../../thesis/PreRegistration.md` §4 / E4–E6 evaluate on (breakdowns `A_k∈{0.95,0.90,0.80}`, stochastic times `σ∈{0.1,0.25,0.5}`, arrivals `ρ∈{0.7,0.85,0.95}`), uniformly at random each episode — implemented as `experiments/train_dynamic.py::DomainRandomizedRegimeWrapper`. Rationale: H4's decision rule requires a *majority* (≥5/9) of regimes to show significant `ΔRI` improvement (§5 below), which only tests general disruption-awareness, not regime-specific overfitting, if the training distribution already spans all 9 regimes.
2. **Two-stage curriculum (`ft06` then `la01`), not domain-randomized instance size.** SB3's `DummyVecEnv` requires a fixed observation/action space across all parallel envs, and `DynamicJSSPEnv`'s padded space size depends on the base instance — so unlike the disruption regime, the instance cannot be randomized within one `model.learn()` call. Stage D1 fine-tunes on `ft06` (warm-started from `curriculum_stage19_la20_final`, the same κ=0/H1–H3 checkpoint); stage D2 continues fine-tuning D1's checkpoint on `la01`. This reuses `experiments/train_curriculum.py::transfer_weights`'s own `strict=True` weight-transfer mechanism (already validated for size transfer by `tests/test_hetero_policy.py::test_weight_transfer_across_instance_sizes`), not a new mechanism. The final (post-D2) checkpoint is evaluated zero-shot on both `ft06` and `la01`, exactly as `curriculum_stage19_la20_final` (last fine-tuned on `la20`) already is for H1–H3.
3. **`max_dynamic_jobs=20` for every regime**, matching the CLI default Week 24's actual E4–E6 execution used (`run_e4e5e6.sh` / `run_e4e5e6_rl.sh` never override `--max-dynamic-jobs`) rather than `../../thesis/PreRegistration.md` §4's "set explicitly per instance" language, which was never actually implemented in the runs that already produced the accepted-as-final E4–E6 results. Chosen for exact train/eval regime consistency with those results, not as a new choice being introduced at training time.

### 2.4 Axis 4 — Single- vs. multi-objective reward

| Variant | Description | Code status |
|---|---|---|
| **Single-objective (reference)** | $r_t$ as in Axis 2/3's dense reward, makespan only. | Exists. |
| **Multi-objective, equal-weighted** | $r_t = -w_1\Delta C_{\max} - w_2\Delta\bar T - w_3\Delta_{\text{inest}}$ (D3's scalarized alternative), $w=(1/3, 1/3, 1/3)$. | **New code needed**, see below. |
| **Multi-objective, makespan-weighted** | Same formula, $w=(0.6, 0.2, 0.2)$ — makespan-dominant, tardiness/instability as secondary tie-breakers. | Same new code, different weight config — a one-line change once the mechanism exists. |
| **Multi-objective, tardiness-weighted** | Same formula, $w=(0.34, 0.33, 0.33)$ *(near-equal, included as a second equal-ish point to check sensitivity to exact weight values, not just the equal-vs-skewed distinction)* — **flagged as this plan's own choice of grid, not literature-sourced**; if Chapter 7's discussion needs a denser weight sweep (e.g., a proper Pareto front), that is additional scope beyond what a single-GPU budget affords in Month 7 and should be flagged as future work rather than silently expanded here. | Same new code. |

**Code status detail:** $\Delta C_{\max}$ is already available (the existing dense reward term). $\Delta \bar T$ (mean tardiness change) needs per-job due dates — **`baselines/due_dates.py::synthesize_due_dates` already exists** (TWK rule, $d_j = r_j + k \cdot \text{job\_totals}[j]$, default tightness $k=1.5$, already used for the `edd`/`cr` PDRs) and can be reused directly: call it once per episode (release times all-zero for the static case) to get $d_j$, then compute $T_j = \max(0, C_j - d_j)$ incrementally as operations complete. $\Delta_{\text{inest}}$ needs the same term Axis 3/H4 needs (`info["planned_start_deviation"]`) — **so the multi-objective variants in this axis inherit Axis 3's blocker for the $w_3$ term specifically**; a two-objective version ($w_3=0$, makespan + tardiness only) is **not blocked** and can run against `JSSPEnv` alone (no `DynamicJSSPEnv` needed, since tardiness is well-defined without any disruption machinery) — recommended as the Month 7 fallback if the disruption-aware blocker has not cleared by then (§4).

**Metric:** report as a small **Pareto-style table** (not a single scalar) — $C_{\max}$ RPD, $\bar T$, and (if unblocked) mean $\Delta_{\text{inest}}$ per weight configuration, so a reader can see the actual trade-off surface rather than a single "best" configuration, consistent with D6's own framing of $RI$/tardiness/makespan as complementary, not substitutable, metrics.

**Statistical procedure:** 4 variants (or 2, in the fallback case) ⇒ **Friedman + Nemenyi** on $C_{\max}$ RPD (the metric shared across all variants, including single-objective) across the `ft06` + 10×10 tier, same as Axis 1's procedure — plus the descriptive Pareto table above, which is not itself a hypothesis test.

---

## 3. Table of planned runs

One row per run; "Blocked on" is blank for anything executable as soon as the Week 17/18 reference checkpoint exists.

| # | Axis | Variant | Instance(s) | Disruption | Seeds | Est. relative compute (vs. reference run) | Blocked on |
|---|---|---|---|---|---|---|---|
| 0 | — (reference, shared) | Heterogeneous GIN, dense, $\kappa=0$, single-obj | `ft06` + `ft10`/`la16`–`la20` | none | 1 (train) | 1× (baseline unit) | `../docs/logs/TrainingPlan_Week17.md` Stage A/B |
| 1a | Encoder | Homogeneous GIN | `ft06` + `ft10`/`la16`–`la20` | none | 1 | ~0.7× (smaller network, no eligibility relation) | Run 0's timestep budget as a fixed comparison point |
| 1b | Encoder | Heterogeneous GAT-hybrid | `ft06` + `ft10`/`la16`–`la20` | none | 1 | ~1.3× (attention overhead) | `HeteroRelationalGATHybridLayer` implementation |
| 2 | Reward density | Sparse | `ft06` | none | 5 | 1× per seed (same architecture, same budget, by design — §2.2) | `reward_mode` flag |
| 3a | Reward disruption-awareness (= H4) | $\kappa=0.5$ | `ft06`, `la01` | breakdowns / stochastic\_times / arrivals (9 regimes, §`../../thesis/PreRegistration.md` §4) | 30 | ~1× (same architecture, disruption rollout overhead only) | `DynamicJSSPEnv` wiring + $\Delta_{\text{inest}}$ term |
| 3b | Reward disruption-awareness (= H4) | $\kappa=1.0$ | `ft06`, `la01` | same 9 regimes | 30 | ~1× | Same as 3a |
| 4a | Objective | Multi-obj, equal-weighted | `ft06` + 10×10 tier | none (or with $\Delta_{\text{inest}}$ if 3a/3b unblocked by Month 7) | 5 | ~1.1× (due-date bookkeeping is cheap) | `due_dates` reuse (unblocked) + $\Delta_{\text{inest}}$ term (blocked, fallback below) |
| 4b | Objective | Multi-obj, makespan-weighted | same | same | 5 | ~1.1× | Same as 4a |
| 4c | Objective | Multi-obj, tardiness-weighted | same | same | 5 | ~1.1× | Same as 4a |
| 4-fallback | Objective (2-objective fallback) | Multi-obj, makespan+tardiness only ($w_3=0$) | `ft06` + 10×10 tier | none | 5 | ~1.1× | **Not blocked** — use if 3a/3b's blocker hasn't cleared by Month 7 |

**Total new runs this plan commissions beyond the shared reference (Run 0):** 4 unblocked (1a, 1b, 2, and either the 4a–4c triple or the 4-fallback single run depending on blocker status at execution time) plus 2 shared with H4 (3a, 3b — already commissioned by `../../thesis/PreRegistration.md`, not double-counted here).

---

## 4. Sequencing and Month 7 decision point

1. **Confirm Run 0 (reference) is complete and passes `../docs/logs/TrainingPlan_Week17.md` §3's success criteria** before starting any ablation — an ablation compared against an under-trained or broken reference is not informative, and would risk misattributing a reference-training problem to the wrong axis.
2. **Run 1a, 1b, 2 first** (all unblocked, all decision-independent of the disruption-machinery blocker). These can start on Month 7 day 1.
3. **Check the `DynamicJSSPEnv`-wiring and $\Delta_{\text{inest}}$-implementation blocker status.** If cleared: run 3a/3b (= H4, per its pre-registered protocol) and the full 4a–4c triple. If not cleared: run only the 4-fallback (2-objective) row, and record 3a/3b/4a–4c as **carried forward, not abandoned**, to whichever sprint completes the blocker — do not quietly drop H4 from the thesis's empirical chapter; report it as scheduled-but-not-yet-executed if Month 7 ends before the blocker clears.
4. **Write up Axis 1/2/4 results in Chapter 4's design-justification framing** (why this encoder, why this reward shape) regardless of whether Axis 3/H4 is complete — these ablations support Chapter 4's methodology narrative independently of H4's confirmatory result.

---

## 5. Explicit non-goals for Month 7

- **No full factorial sweep** (§1) — each axis is evaluated in isolation against the shared reference.
- **No size-crossed encoder ablation** (Axis 1 stays at the training-size tier; a "does the encoder choice change generalization behavior" question is a natural Month 8+ extension building on H2's already-scheduled generalization measurements, not this plan).
- **No continuous weight sweep for the multi-objective axis** — three (or two, fallback) discrete weight points, not a Pareto-front search; flagged explicitly as future work if the committee wants a denser trade-off curve.
- **No GAT-hybrid variant of the *disruption-aware* reward** (i.e., no crossing of Axis 1 × Axis 3) — consistent with §1's stated interaction-question limitation.
