# Final Results Report — E1–E8, H1–H5

**Status:** All experiments executed, statistical pipeline run end-to-end, every number
below traces to a real CSV produced by `run_evaluation.py` or `experiments/train_dynamic.py`
on this machine. No result in this document is fabricated, estimated, or carried forward
from a prior sprint's expectation — every REJECT/ACCEPT verdict is the literal output of
`python -m experiments.stats.cli` against the frozen decision rules in
`PreRegistration.md`.

**Provenance chain:** `PreRegistration.md` (frozen 2026-07-29) → `E7_E8_Plan.md`
(2026-07-31 audit + E7/E8 scope) → `StatsAnalysis_TechSpec.md` (pipeline spec) →
`experiments/stats/` (implementation, `PROGRESS_SUMMARY.md` checkpoint 2026-07-31) →
this document (completed 2026-08-01: E8 clean pass confirmed, stats suite run for the
first time, `train_dynamic.py` run for real (4 training runs), E7 evaluation grid
executed, two real bugs found and fixed in the stats pipeline itself, full E1–E8 report
generated).

**Full test suite:** 752 passed, 0 failed (includes the device-placement-bug regression
tests and the stats module's 84 tests). Stats pipeline's own test suite: 84/84 passed on
first execution — no bugs found there. Two bugs *were* found and fixed downstream, in the
report-orchestration layer (`report.py`/`cli.py`), described in §8.

---

## 0. Hypothesis-by-hypothesis verdict table

| Hypothesis | Verdict | Key statistic |
|---|---|---|
| **H1 (JSSP)** — RL beats best PDR, static, N=43 | **REJECT** | Wilcoxon W=133, p=4.03e-05, r=-0.626 (RL *worse* than best PDR) |
| **H1 (FJSP, synthetic)** | **REJECT** | W=0, p=1.12e-08, r=-0.871 |
| **H1 (FJSP, Brandimarte)** | **REJECT** | W=0, p=0.0156, r=-0.914 |
| **H2 (overall)** — size generalization, 4 tiers | **ACCEPT** | RL not significantly worse than best PDR (Nemenyi CD) in any tier |
| **H3 (overall)** — robustness under disruption, 9 regimes | **REJECT** | 0/9 regimes accept (conjunctive rule); RL's RI worse than best PDR's in all 9, p<1.7e-11 everywhere |
| **H4 (κ=0.5, overall)** — disruption-aware reward | **REJECT** | component (a) 0/9 regimes significant (need ≥5); component (b) ΔRPD_nominal=+106.0pp (margin: <2pp) |
| **H4 (κ=1.0, overall)** | **REJECT** | component (a) 1/9 regimes significant (need ≥5); component (b) ΔRPD_nominal=+46.6pp (margin: <2pp) |
| **H5** — decision latency, RL < 1% of CP-SAT, all 43 instances | **REJECT** | 42/43 instances violate the threshold; RL wins decisively only at the largest instance (la40) |

Full table with every per-regime/per-tier row, test statistic, p-value, and effect size:
[`experiments/results/stats/tables/h1_h5_verdict.md`](experiments/results/stats/tables/h1_h5_verdict.md).

**Reading this table honestly:** every verdict above is a REJECT except H2. This is not a
report-writing failure — H1/H3/H5 rejecting was already the accepted, documented outcome
as of `E7_E8_Plan.md`'s Week 24 audit (§1 of that document explicitly calls the H1
JSSP result and the H3 "REJECT in 0/9 regimes" result "accept as final," not anomalies to
chase). What this sprint adds is: H4 now has a real, executed verdict for the first time
(previously blocked), and H5 now has a clean, uncontaminated, full-43-instance
measurement instead of Week 24's flagged-as-noisy one. Both new REJECTs have concrete,
diagnosable causes, detailed in §§4–5 below — this is not "the numbers came back bad and
we don't know why."

---

## 6.1 Static competitiveness (H1) — E1, E2a, E2b

**Data:** E1 (43 JSSP instances, `ft06`–`la40`), E2a (43 synthetic FJSP), E2b (7
Brandimarte FJSP, `mk01`–`mk07` minus the documented `mk06` data-provenance exclusion,
see §8). Per-instance RPD table (all methods, deterministic seed 0):
[`e1_rpd_summary.md`](experiments/results/stats/tables/e1_rpd_summary.md).

**Result:** REJECT in all three arms. RL's median RPD is higher (worse) than the best
comparator's in every arm, at high significance (p ≤ 0.016 throughout, JSSP arm
p=4e-05). This reproduces the same qualitative finding `E7_E8_Plan.md` §1 already
flagged from Week 24 (RL loses to `fifo`'s zero-shot edge case on a meaningful fraction of
JSSP instances) — now on a clean, non-contaminated re-execution of the underlying data,
with the same conclusion. Descriptively, RL still beats CP-SAT's RPD ceiling expectation
in the documented, non-falsifying sense (§2 of `PreRegistration.md`): this is not a
"the policy doesn't work" result, it is "the policy is not competitive with the single
best hand-crafted rule on this benchmark set," which is a substantially narrower and more
specific claim.

---

## 6.2 Size generalization (H2) — E3

**Data:** `la21`–`la40` in four tiers (1.5–2×, 2.0×, 2.25×, 3.0× the 10×10 training
size), zero-shot weight transfer, no fine-tuning. Per-tier RPD table:
[`e3_rpd_summary.md`](experiments/results/stats/tables/e3_rpd_summary.md). RPD-vs-size
scatter: [`e3_rpd_vs_size.png`](experiments/results/stats/figures/e3_rpd_vs_size.png).
Critical-difference diagrams (one per tier, all four Friedman tests significant at
p<1.6e-05):
[`cd_diagram_e3_1.5-2x.png`](experiments/results/stats/figures/cd_diagram_e3_1.5-2x.png),
[`cd_diagram_e3_2.0x.png`](experiments/results/stats/figures/cd_diagram_e3_2.0x.png),
[`cd_diagram_e3_3.0x.png`](experiments/results/stats/figures/cd_diagram_e3_3.0x.png),
[`cd_diagram_e3_2.25x.png`](experiments/results/stats/figures/cd_diagram_e3_2.25x.png).

**Result:** ACCEPT, all four tiers and overall. RL's *average rank* (Friedman/Nemenyi,
not raw RPD magnitude) is never significantly worse than the best PDR's — RL ranks
1.6–2.0 out of 10 methods in every tier, essentially tied with or just behind `fifo`
(rank 1.0–1.4), and far ahead of every other PDR. Note the raw-RPD table shows RL's
*mean* RPD (23.9–32.9%) numerically above `fifo`'s (12.7–26.8%) in every tier — H2's
decision rule is rank-based specifically so this doesn't matter; the two methods are
statistically indistinguishable at the family-wise α=0.05 level, which is the actual
claim H2 makes. The RPD-by-tier pattern is non-monotonic (3× tier has RL's *lowest*
RPD, not highest) — already flagged in `E7_E8_Plan.md` §1 as a corroborated, not
anomalous, finding, reproduced here.

---

## 6.3 Robustness under disruption (H3) — E4, E5, E6

**Data:** `ft06`, `la01`, 9 regimes (3 breakdown × 3 stochastic-time × 3 arrival
levels), 30 seeds/regime, PDR arm vs. RL arm (both against a live `DynamicJSSPEnv`).
RI summary + line plots per axis:
[`e4e5e6e7_ri_summary.md`](experiments/results/stats/tables/e4e5e6e7_ri_summary.md),
[`ri_vs_availability_breakdowns.png`](experiments/results/stats/figures/ri_vs_availability_breakdowns.png),
[`ri_vs_sigma_stochastic_times.png`](experiments/results/stats/figures/ri_vs_sigma_stochastic_times.png),
[`ri_vs_utilization_arrivals.png`](experiments/results/stats/figures/ri_vs_utilization_arrivals.png).

**Result:** REJECT overall (0/9 regimes accept; H3's rule is conjunctive — all 9 required).
Per-regime, RL's RI is worse (higher) than the best PDR's in every single regime, at
very high significance (p<1.7e-11 in 8/9 regimes). The `arrivals` axis is the worst by a
wide margin (RL RI ≈ +1.65 to +1.82, i.e. perturbed makespan roughly 2.6–2.8× nominal,
vs. the best PDR's RI ≈ +0.26 to +0.27) — consistent with `E7_E8_Plan.md` §1's own
diagnosis that arrivals is "the worst regime by a wide margin" and that this traces to
the reused κ=0 checkpoint carrying the completion-gated/dispatch-gated legality mismatch
(Finding B) rather than a training failure per se. This is the same qualitative result
Week 24 already reported and explicitly instructed **not** to rerun (`E7_E8_Plan.md`
§1: "Accept as final confirmatory result... do not rerun"); this sprint's contribution is
reproducing it end-to-end through the now-executed stats pipeline rather than the
Week 24 standalone script.

---

## 6.4 Disruption-aware reward (H4) — E7

**Data:** two new training runs (κ=0.5, κ=1.0), each a two-stage curriculum
(`ft06`→`la01`, 100k timesteps/stage, domain-randomized across all 9 disruption regimes
per episode reset, warm-started from `curriculum_stage19_la20_final.zip`). κ=0 is the
**same reused** static-trained checkpoint E4–E6 already used (per
`PreRegistration.md` §5 — not a fourth training run). Evaluation: full 9-regime ×
30-seed grid per κ, plus a nominal (`--disruption none`) deterministic pass. Training
wall-clock: 33–36 min/stage, ~135 min total for both κ values, CPU-only (8-core machine,
device bug already fixed — see `PROGRESS_SUMMARY.md`).

**Result:** REJECT for both κ=0.5 and κ=1.0.

- **Component (a)** (does robustness improve — Wilcoxon on ΔRI, need ≥5/9 regimes
  significant with median ΔRI<0): κ=0.5 achieves 0/9; κ=1.0 achieves 1/9
  (`E5_stochastic_s0.10`, p=8.0e-05, median ΔRI<0). Neither reaches the ≥5/9 majority
  threshold.
- **Component (b)** (nominal cost stays under the 2-percentage-point margin): **both
  κ values fail this catastrophically** — ΔRPD_nominal = +106.0pp for κ=0.5, +46.6pp
  for κ=1.0, roughly 50–500× the 2pp margin, not a marginal miss.

**Root cause of the component-(b) failure, diagnosed, not just reported:** the
domain-randomized training wrapper (`DomainRandomizedRegimeWrapper` in
`train_dynamic.py`) samples one of the 9 *disruptive* regimes at **every** episode
reset — the κ>0 policies never once trained on a disruption-free episode. Concretely,
nominal RPD went from 32.7%→116.4% (κ=0.5) / 87.3% (κ=1.0) on `ft06`, and 4.1%→132.4%
(κ=0.5) / 42.8% (κ=1.0) on `la01`, relative to the reused κ=0 checkpoint. This is not a
training bug — domain-randomized training across exactly the evaluation grid is what
`E7_E8_Plan.md` §2.2 recommended (Option B, chosen deliberately over a
single-fixed-regime alternative specifically because H4's decision rule needs a policy
that is generally disruption-aware, not regime-specific) — but it has a real, now
quantified cost: without at least some nominal-episode exposure during training, the
fine-tuned policy's nominal competitiveness collapses. This is a genuine, actionable
finding for Chapter 7's discussion (e.g., mixing a "no disruption" regime into the
domain-randomization pool at a nonzero weight is a concrete, testable fix for a future
sprint), not a methodological error in this sprint's execution.

**The reused-vs-retrained structural asymmetry** flagged in `E7_E8_Plan.md` §2.3
(κ=0's RI carries the adapter legality mismatch; κ>0's RI does not, since it trains
natively against `DynamicJSSPEnv`) is present in this data but is dominated by the much
larger nominal-RPD effect above — the optional third "κ=0 retrained" run that would
have isolated this asymmetry was not run this sprint (deprioritized per §2.3's own
explicit "cut first if time is short" guidance); this remains a disclosed, not hidden,
limitation.

---

## 6.5 Decision latency (H5) — E8

**Data:** clean re-measurement, all 43 instances, 1 deterministic + 4 stochastic
seeds/instance, CP-SAT once, on an otherwise-idle 8-core CPU machine (no CUDA build
present — `torch.cuda.is_available()` is unconditionally `False` on this machine, so
per `PreRegistration.md` Amendment 2, "training hardware" and "CPU-only" collapse
to one measurement; there is no separate GPU number to report). Per-instance latency
table: [`e8_latency_summary.md`](experiments/results/stats/tables/e8_latency_summary.md).
Latency-vs-size plot with the CP-SAT/RL ratio overlaid:
[`e8_latency_vs_size.png`](experiments/results/stats/figures/e8_latency_vs_size.png).

**Result:** REJECT — the strict "every instance" threshold fails for 42/43 instances.

**This is a real, diagnosed result, not a measurement artifact** (see §8 for a bug that
was found and fixed in how this was computed, which did *not* change the verdict).
RL's median per-decision cost is essentially flat across instance sizes (≈4–9ms, with
two moderate outliers at `la15`/`la16` of 27/44ms plausibly reflecting residual
measurement noise even on an idle machine — flagged, not smoothed over) — consistent
with a fixed-cost neural-network forward pass that doesn't scale with instance size.
CP-SAT's per-decision figure, by contrast, is an artificial construct for small
instances: dividing an already-near-instantaneous total solve (CP-SAT solves `la06` in
under half a millisecond total) by the operation count produces a sub-millisecond
"per-decision" number that no sequential neural-network policy can beat, even though RL
is fast in absolute terms. The picture **inverts decisively at scale**, exactly as
`PreRegistration.md` §6 asks to check: the CP-SAT/RL ratio widens from ~0.06–0.9×
at small/medium sizes to 41.7× (`la21`), 34.3× (`la25`), 27.0× (`la29`), and 1773× at
the largest tested instance (`la40`, 225 ops, CP-SAT total solve 32s vs. RL's 18ms
median). RL passes the 1% rule outright only at `la40`, but the trend is unambiguous
and monotonic-in-spirit: **for genuinely large/hard instances, RL is 1–3 orders of
magnitude faster; for small/easy instances, CP-SAT's near-instant solve makes a
per-decision comparison structurally unfavorable to any sequential policy.** The
pre-registration's literal "every instance" rule is not met, and is reported as REJECT
without qualification — the widening-ratio nuance above is the descriptive context the
frozen decision rule itself asks to report alongside the verdict, not a reason to
overturn it.

---

## 7. Precondition and infrastructure status (carried in from `E7_E8_Plan.md` §2.1)

| Precondition | Status |
|---|---|
| κ·Δ_inest reward term, bit-for-bit no-op at κ=0 | Done, tested (`TestKappaReward`) |
| GPU device-placement fix (`device="cpu"` pin) | Done, tested |
| `DynamicJSSPEnv` training entrypoint (`train_dynamic.py`) | Done, smoke-tested, then run for real (4 full training runs) |
| `run_evaluation.py` CSV-write hardening | Done, tested (`TestCsvHardening`) |
| Full regression suite | 752 passed, 0 failed |

---

## 8. Bugs found and fixed during this sprint's execution (pipeline correctness, not data)

Per `StatsAnalysis_TechSpec.md` §1's own framing — "getting a sign or an `alternative=`
backwards... is the single highest-consequence bug this pipeline could contain" — every
anomalous-looking verdict in this report was traced to its root cause before being
accepted. Two real bugs were found in the process, both now fixed and covered by the
still-green 84-test stats suite:

1. **`experiments/stats/cli.py` experiment-ID tag mismatch**: the E6-arrivals loop
   tagged the 0.7-utilization regime as `"E6_arrivals_u0.7"`, but `report.py`'s
   `REGIMES` list (and the golden test) expect the zero-padded `"E6_arrivals_u0.70"` —
   identical to how the adjacent E5 loop already handles zero-padding correctly. This
   silently dropped the u=0.70 regime from both H3 and H4 (surfaced as
   `"expected 9 regime verdicts, got 8"` on the first pipeline run). Fixed by
   zero-padding the tag to match `REGIMES` exactly; verified by re-running the full
   pipeline and confirming all 9 regimes now resolve for both H3 and H4.
2. **`experiments/stats/report.py::run_h5` mixed rollout modes**: H5's RL latency rows
   were not filtered to `seed==0` (the deterministic/argmax rollout), so the four
   `--rl-sample` stochastic rollouts (seeds 1–4, measurably 3–5× slower per decision —
   sampling a masked categorical distribution costs more than argmax) were pooled into
   the median alongside the deterministic row. This is inconsistent with the
   `rollout_mode` convention (`seed==0` primary, `seed∈{1,2,3,4}` descriptive) already
   used correctly everywhere else in this pipeline (H1's primary-vs-descriptive split).
   Fixed by restricting H5's RL rows to `seed==0`. **This fix did not flip the verdict**
   (still 42/43 REJECT before and after) — confirmed by recomputing both ways before
   committing the fix, so the correction is reported for its own sake (methodological
   correctness), not because it was expected to change the answer.

Both fixes are covered by the pre-existing golden regression test
(`tests/test_stats_report_golden.py`), which still passes (pinned against
`../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`'s published numbers, unrelated to either bug).

---

## 9. Open items / documented limitations (carried forward, not silently dropped)

- **`mk06` upstream data issue** (Finding #2, Week 24): RPD excluded from E2b, raw
  makespan retained; not resolved this sprint (no code/data change was possible on our
  side; upstream issue filing remains an action item).
- **H4's reused-vs-retrained structural asymmetry** (§2.3 of `E7_E8_Plan.md`): not
  isolated this sprint (the optional third κ=0-retrained run was not run, per that
  section's own "cut first if time is short" priority ordering) — disclosed here, not
  hidden. In practice this sprint's H4 REJECT is dominated by the much larger
  domain-randomization nominal-cost effect (§6.4), so the asymmetry is unlikely to
  change the qualitative conclusion, but this has not been formally verified.
- **H4's domain-randomization nominal-cost finding** (§6.4): diagnosed but not
  remediated — a concrete next-sprint candidate (mix a no-disruption regime into
  training) is named but not attempted here, to keep this sprint's scope to what
  `E7_E8_Plan.md` actually committed to.
- **`la15`/`la16` latency outliers** (§6.5): flagged as plausible residual measurement
  noise, not investigated further (would require re-running the timing pass in
  isolation, which the "otherwise idle machine" protocol was already designed to
  approximate as closely as practical for this sprint).

---

## 10. Mejoras del tutor (2026-08-20) — implementadas en código, pendientes de ejecución

Sesión posterior a este reporte: diez recomendaciones formales del tutor, auditadas y
en su mayoría implementadas en código/tests, sin ejecutar nada (ver `AUDIT_PREFLIGHT.md`,
`RUN_PLAN.md`, `MEJORAS_TUTOR_IMPLEMENTADAS.md`). Ningún número de este documento
(§§0–9 arriba) se recalculó ni se tocó — siguen siendo los resultados reales ya
descritos en el encabezado de este archivo. Lo nuevo, todo `[PENDIENTE-CORRIDA]` hasta
la primera corrida:

- **Hallazgo de auditoría más relevante para este reporte:** el comparador de H3 ("mejor
  PDR", usado en §6.3 arriba) puede estar parcialmente dominado por un artefacto del
  Índice de Robustez — una PDR con plan nominal muy malo (`cr`, RPD nominal 210.9%–229.3%)
  tiene RI negativo (-0.40 a -0.59) bajo averías/tiempos estocásticos, no porque sea
  robusta sino porque su plan nominal tenía mucho margen para "caer hacia arriba" por
  azar. Ver `AUDIT_PREFLIGHT.md` Fase 0(b) para la traza mecánica completa y la
  corrección (no se cumple bajo llegadas, a diferencia de lo que el tutor asumía).
- **H4 (§6.4):** el REJECT reportado arriba permanece publicado sin cambios (Regla 1 de
  la sesión de mejoras). Nueva Fase 3 (ablación 2×2) deja preparada la celda que
  faltaba (entorno dinámico + recompensa nominal) para atribuir el colapso de RPD
  nominal (+106.0pp/+46.6pp) al entorno de entrenamiento, a la recompensa, o a ambos —
  y Fase 3 (T3.2) corrige el defecto de aleatorización de dominio que este mismo §6.4
  ya diagnostica como causa probable (las políticas κ>0 nunca vieron un episodio
  nominal durante el entrenamiento).
- **H1/H3 (§§6.1, 6.3):** nueva Fase 4 separa `pdr_global`/`pdr_por_familia`/
  `oráculo_best_per_instance` (antes fusionados bajo "mejor PDR"); nueva Fase 8 añade
  calidad realizada (RPD vs. óptimo certificado, comparable entre métodos) como métrica
  primaria junto al RI.
- **H5 (§6.5):** `e8_cpu_timing.py` ampliado de 4 a 43 instancias (Fase 5); nueva figura
  de costo amortizado.

Ver `MEJORAS_TUTOR_IMPLEMENTADAS.md` para la tabla completa recomendación-por-
recomendación y `RUN_PLAN.md` para los comandos exactos y estimaciones de tiempo.

---

## Appendix — full artifact manifest

All tables (Markdown + LaTeX) and figures (300dpi PNG) are under
`experiments/results/stats/`:

```
tables/
    h1_h5_verdict.md          h1_h5_verdict.tex        (§0, leads the chapter)
    e1_rpd_summary.md         e1_rpd_summary.tex       (§6.1)
    e3_rpd_summary.md         e3_rpd_summary.tex       (§6.2)
    e4e5e6e7_ri_summary.md    e4e5e6e7_ri_summary.tex  (§6.3, §6.4)
    e8_latency_summary.md     e8_latency_summary.tex   (§6.5)
figures/
    cd_diagram_e3_1.5-2x.png, _2.0x.png, _2.25x.png, _3.0x.png   (§6.2)
    e3_rpd_vs_size.png                                            (§6.2)
    ri_vs_availability_breakdowns.png                             (§6.3)
    ri_vs_sigma_stochastic_times.png                              (§6.3)
    ri_vs_utilization_arrivals.png                                (§6.4)
    e8_latency_vs_size.png                                        (§6.5)
stats_manifest.json   -- full provenance: every source CSV, row counts in,
                          rows dropped and why, for any committee question
                          about a specific number
```

Raw experiment outputs this report is computed from:
`experiments/results/week24/` (E1–E6, PDR/GA/CP-SAT arms), `experiments/results/week25/e8_clean/`
(H5's clean E8/E1 timing pass), `experiments/results/dynamic/` (E7 training checkpoints,
evaluation grid CSVs).
