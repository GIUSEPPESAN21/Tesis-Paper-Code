# Code appendix summary

**Purpose.** This document is written to be included as a thesis appendix (e.g. "Anexo
I — Resumen del código fuente"). It answers three things a reader of `Thesis.md`
needs and the chapter text itself does not provide: (1) what is in the accompanying
repository, (2) which code module implements which chapter/section, and (3) concrete
evidence that every reported number is reproducible from the archived code, not just an
assertion that it should be.

Companion documents this appendix summarizes rather than duplicates: `README.md` (full
installation/reproduction instructions with exact commands and measured runtimes),
`REPRODUCIBILITY_REPORT.md` (the reproducibility audit and its findings),
`FIGURES_INDEX.md` (figure-by-figure provenance), and `../code/README.md`
(code-level design decisions). This document is the map between them and the thesis
prose; it does not re-derive anything they already establish.

---

## 1. Repository structure

```
Tesis-Paper-Code/                         <- archived repository root (git-tagged, see §3)
├── README.md                               <- reproduction guide (start here to re-run E0-E8)
├── thesis/
│   ├── Thesis.md                          <- the thesis: single integrated document, all 8
│   │                                         chapters (no separate per-chapter files anymore --
│   │                                         cite by section number, see ../archive/README.md)
│   ├── RESULTS_REPORT.md                   <- E1-E8 results, H1-H5 verdicts, full traceability
│   ├── REPRODUCIBILITY_REPORT.md           <- reproducibility audit + smoke-test record
│   ├── FIGURES_INDEX.md                    <- every figure -> thesis section, caption, source
│   ├── PreRegistration.md                  <- frozen (2026-07-29) H1-H5, decision rules
│   ├── StatsAnalysis_TechSpec.md           <- spec for experiments/stats/
│   ├── E7_E8_Plan.md                       <- E7/E8 design rationale (also folded into §5.4)
│   ├── CODE_APPENDIX_SUMMARY.md            <- this document
│   └── defense/                            <- active defense-prep material
├── paper/PaperCIE_LaTeX/                   <- companion Computers & Industrial Engineering submission
├── docs/especificaciones/Baselines_TechSpec.md, docs/especificaciones/DynamicEnv_TechSpec.md,
│   docs/especificaciones/GNN_PPO_Architecture_Spec.md            <- Anexos E-G (§1.7 of the thesis)
├── docs/                                    <- active, non-duplicated support (especificaciones,
│                                                planificacion, auditorias, logs, future_work) --
│                                                see docs/README.md; not cited as primary sources
├── archive/                                 <- superseded/historical material, git mv not deleted --
│                                                includes the former docs/capitulos/ (absorbed into
│                                                Thesis.md, verified by diff) and the paper drafts
│                                                superseded by the RI-artifact metric correction,
│                                                see archive/README.md
├── figures/                                 <- camera-ready figures (Spanish, 300dpi PNG+PDF)
├── results/
│   ├── tablas/                               <- thesis-numbered tables (Tabla_6.X.Y.*)
│   ├── datos_crudos/                          <- raw per-row CSVs, by experiment batch
│   └── README.md                              <- what's here vs. code/
├── CITATION.cff, .zenodo.json               <- archival/citation metadata
└── code/                        <- the installable Python package (below)
```

### `../code/` — the Python package (`thesis_fjsp_drl`, ~9,500 lines across the
modules below, excluding tests)

```
code/
├── envs/            <- Gymnasium environments (JSSP/FJSP, static + dynamic/stochastic);
│                       rescheduling.py (2026-08-20, tutor rec 9): periodic/event-triggered
│                       reoptimization execution policies for the offline (ga/cpsat) methods
├── models/           <- GNN encoder + PPO policy
├── baselines/         <- PDR, GA, CP-SAT reference methods
├── experiments/        <- training entrypoints + experiments/stats/ (the H1-H5 pipeline)
├── scripts/             <- figure/table generation, instance fetching, optimum verification
├── tests/                 <- 752 tests (pytest), one file per module below
├── data/                    <- benchmark instances (Fisher-Thompson/Lawrence/Brandimarte)
├── run_evaluation.py          <- unified evaluation harness (instances x methods x disruption)
├── requirements.txt, pyproject.toml  <- pinned dependencies (see REPRODUCIBILITY_REPORT.md §1)
└── Dockerfile, .dockerignore    <- CPU-only reproduction environment
```

---

## 2. Chapter/section -> code module map

This table is the authoritative cross-reference between `Thesis.md`'s prose and
the code that produced it. All paths are relative to `../code/` unless noted.

### Chapter 4 — Marco metodológico y arquitectura de la solución

| Thesis section | What it describes | Code module(s) |
|---|---|---|
| §4.2 (4.2.1–4.2.4) | Discrete-event simulation core, Gymnasium contract, static env as a special case | `envs/base_env.py`, `envs/events.py`, `envs/jssp_env.py`, `envs/fjsp_env.py` |
| §4.3.1–4.3.3 | MDP formulation: state space, action masking, reward function | `envs/base_env.py` (state/action-mask construction), `envs/jssp_env.py`/`envs/fjsp_env.py` (`step`/`reset`), `envs/disjunctive_graph.py` (the $H(s)$ lower-bound used by the reward) |
| §4.4.1–4.4.3 | Heterogeneous graph representation, relational message passing, size-independent actor/critic heads | `envs/disjunctive_graph.py` (graph construction), `models/gin.py` (`GINEncoder`, the relational message-passing layer) |
| §4.4.4 | MaskablePPO training algorithm | `models/policy.py` (wires `GINEncoder` into an SB3-family actor-critic policy); `experiments/train_hetero.py`, `experiments/train_curriculum.py` (training drivers) |
| §4.5.1–4.5.4 | Disruption models: breakdowns (TTF/TTR), stochastic processing times, dynamic arrivals, preempt-resume policy | `envs/disruptions.py` (independently testable generators), `envs/dynamic_jssp_env.py` (integrates them into the env) |
| §4.5.6 | Statistical validation of the three disruption generators | `tests/test_disruptions.py` (validates generator statistics directly, not just that they run) |
| §4.6.1 | Priority dispatching rules (9 PDR) + EDD/CR due-date synthesis | `baselines/dispatching_rules.py`, `baselines/due_dates.py` |
| §4.6.2 | Genetic algorithm | `baselines/ga.py` |
| §4.6.3 | CP-SAT exact solver | `baselines/cpsat.py` |
| §4.6.4 | Common evaluation harness across all baseline + RL methods | `baselines/interface.py`, `run_evaluation.py` |

### Chapter 5 — Diseño experimental y protocolo de evaluación

| Thesis section | What it describes | Code module(s) |
|---|---|---|
| §5.1 | Factor matrix, instance blocks, wall-clock budgets (E0–E8) | `run_evaluation.py` (CLI flags implementing the budgets), `data/` (instance files), `README.md` (exact commands per experiment) |
| §5.2.1–5.2.2 | Reference bound $C^*$, metric formulas (RPD, RI, latency) | `experiments/stats/metrics.py` |
| §5.2.3, §5.3.1–5.3.6 | Statistical procedure: input schema, Wilcoxon, Friedman+Nemenyi, CD diagrams, inclusion rules | `experiments/stats/io.py` (schema/ingestion), `experiments/stats/pairwise.py` (Wilcoxon + H1/H3/H4 wrappers), `experiments/stats/omnibus.py` (Friedman + Nemenyi/CD), `experiments/stats/cd_diagram.py` (CD diagram rendering) |
| §5.4.1–5.4.5 | E7 κ-grid design, domain-randomization training choice, E8 clean-remeasurement protocol | `experiments/train_dynamic.py` (E7 entrypoint; `DomainRandomizedRegimeWrapper`), `run_evaluation.py` (E8 latency remeasurement) |

### Chapter 6 — Resultados

| Thesis section | Hypothesis | Code that produced the reported numbers |
|---|---|---|
| §6.1 | Environment validation (E0) | `scripts/verify_optima.py` (regenerates `data/known_optima.csv` via CP-SAT proof) |
| §6.2 | H1 (static competitiveness) | `experiments/stats/pairwise.py` (`h1_jssp_test`, `h1_fjsp_test`), `experiments/stats/report.py` (`run_h1`) |
| §6.3 | H2 (size generalization) | `experiments/stats/omnibus.py` (`friedman_test`, `critical_difference`), `experiments/stats/report.py` (`run_h2`) |
| §6.4.1 | H3 (robustness, κ=0) | `experiments/stats/pairwise.py` (`h3_regime_test`), `experiments/stats/report.py` (`run_h3`) |
| §6.4.2 | H4 (κ=0.5, κ=1.0) | `experiments/train_dynamic.py` (produces the κ>0 policies), `experiments/stats/report.py` (`run_h4_full`) |
| §6.5, §6.5.1 | H5 (decision latency), la40 anomaly | `run_evaluation.py` (`ms_per_decision` measurement), `experiments/stats/report.py` (`run_h5`) |
| All of Ch. 6's tables/figures | — | `experiments/stats/cli.py` (orchestrates the full run), `scripts/export_thesis_tables.py`, `scripts/generate_thesis_figures.py` |

### Chapter 7 — Discusión

Chapter 7 is interpretive prose over Chapter 6's already-computed verdicts; it does not
introduce new computation. Every specific figure it cites (e.g. GA's 5.52% mean RPD, the
la40 sensitivity ratio) traces to the same modules as the corresponding Chapter 6 row
above — see `experiments/stats/report.py`'s verdict objects, which both chapters draw
from via `RESULTS_REPORT.md`.

### `experiments/stats/` internals (supports all of Chapter 5 §5.3 and Chapter 6)

| Module | Role |
|---|---|
| `io.py` | Loads/validates the E1–E8 CSVs, computes derived columns (`rpd_pct`, etc.) |
| `metrics.py` | RPD, Robustness Index, decision-latency, summary-statistic formulas |
| `pairwise.py` | Wilcoxon signed-rank + the H1/H3/H4 hypothesis-specific test wrappers |
| `omnibus.py` | Friedman test, Nemenyi post-hoc, critical-difference computation |
| `cd_diagram.py` | Renders Demšar-style critical-difference diagrams |
| `tables.py` | Markdown/LaTeX table formatting |
| `report.py` | Orchestrates the above into the H1–H5 verdict objects Chapter 6/7 cite |
| `cli.py` | Command-line entrypoint (`python -m experiments.stats.cli`) that runs the full pipeline end-to-end |
| `comparators.py` | (2026-08-20, tutor recs 6/7) `pdr_global`/`pdr_por_familia`/`oracle_best_per_instance`, always reported separately; H2's dual reading |
| `seeds.py` | (tutor rec 3) mean/std/95% CI (t)/range aggregation across training seeds |
| `posthoc_multiseed.py` | (tutor rec 3) `Tabla_H4_multisemilla_posthoc` generator — explicitly post-hoc, H4's pre-registered verdict unchanged |
| `ablation_2x2.py` | (tutor rec 5) Friedman + factorial effects (entorno/recompensa/interacción) for the 2×2 ablation |
| `realized_quality.py` | (tutor rec 8, T8.2/T8.4) realized RPD vs. certified optimum, compared against the three T4.1 comparators |
| `rescheduling_policies.py` | (tutor rec 9, T6.3) `right_shift` vs. `periodic`/`event_triggered` comparison, optimality-vs-sequence-commitment diagnosis |

### Figures and tables (supports Chapter 6/7; the live companion paper is `../paper/PaperCIE_LaTeX/`, self-contained)

| Output | Generator | Notes |
|---|---|---|
| `../figures/Fig_6.*.png/pdf` | `scripts/generate_thesis_figures.py` | Spanish labels, thesis figure numbering |
| `../results/tablas/Tabla_6.*.{csv,md,tex}` | `scripts/export_thesis_tables.py` | Format-converts already-tested pipeline tables, no recomputation |
| `../archive/figures_paper/Fig_P*.png/pdf`, `Table_S*.{csv,md}` | `scripts/generate_paper_figures.py`, `scripts/export_paper_supplement_tables.py` | **Historical** — supported the now-archived `ConferencePaper_DRAFT.md` (superseded by the RI-artifact metric correction, see `../archive/README.md`), not the live `../paper/PaperCIE_LaTeX/` submission |

---

## 2.1 Mejoras del tutor (2026-08-20) — code additions, not yet executed

Ten formal advisor recommendations, audited and mostly implemented in code/tests this
session, with nothing executed (Regla Cero). New modules are listed inline in the
tables above, tagged "(2026-08-20, tutor rec N)". Full recommendation-by-recommendation
status: `MEJORAS_TUTOR_IMPLEMENTADAS.md`. Exact commands and time estimates:
`RUN_PLAN.md`. Audit findings (including one correction to the advisor's own
"cr/lwkr, RI negative" claim — true for `cr` under breakdowns/stochastic times, not
for `lwkr` under arrivals): `AUDIT_PREFLIGHT.md`.

---

## 3. Reproducibility confirmation

This section records what was actually run, when, and with what result — not an
assertion that reproduction "should" work.

### 3.1 Test suite

- **752 tests total**, one file per code module (`tests/test_*.py`), covering the
  environments, models, baselines, the full `experiments/stats/` pipeline (84 of the 752
  are stats-pipeline-specific), and `run_evaluation.py`'s CLI surface.
- Full suite (`pytest -q`, no marker filter): **752 passed, 0 failed** — re-confirmed
  during this final packaging pass (see `REPRODUCIBILITY_REPORT.md` §4 for the
  smoke-test-triggered bug find/fix that preceded this run, and this document's own
  re-run below).
- Fast subset (`pytest -m "not slow" -q`, skips CP-SAT-heavy tests): 748 passed, 4
  deselected — used for the quick smoke test since it is not meaningfully faster than
  the full suite on this codebase (measured, not assumed — see `README.md`).

### 3.2 End-to-end smoke test (re-run for this packaging pass)

The exact 4-step sequence in `README.md`'s "Quick smoke test" section, re-run natively
(Docker unavailable in this environment) immediately before this document was written:

1. `pytest -m "not slow" -q` → 748 passed, 4 deselected.
2. `run_evaluation.py`, `{spt, fifo, ga, cpsat, rl}` × `{ft06, la01}`, `--time-limit 5` →
   10/10 rows, all `feasible=True`; CP-SAT reached `known_optimum` on both instances
   (`ft06`: Cmax=55, RPD=0.00%; `la01`: Cmax=666, RPD=0.00%).
3. Disrupted rollout, `fifo`+`rl` under `breakdowns` (`--availability 0.90`) on `ft06` →
   2/2 rows, `feasible=True`.
4. `train_dynamic.py`, 2000 timesteps, κ=0.5, `--output-dir` pointed outside the
   repository → completed cleanly (checkpoint + summary JSON both written), confirming
   the `_display_path()` fix documented in `REPRODUCIBILITY_REPORT.md` §4 still
   holds and no regression was introduced by the subsequent thesis-revision or
   paper-supplement work.

**Result: all four steps passed with no new issues found.** No code changes were needed
during this packaging pass — this run exists to catch late-breaking regressions, and
found none.

### 3.3 Traceability chain

Every number in Chapter 6/7 traces through the same chain, auditable by any reader:

```
PreRegistration.md (frozen hypotheses/decision rules, 2026-07-29)
  -> run_evaluation.py / experiments/train_*.py (raw per-row CSVs, experiments/results/)
  -> experiments/stats/ (84/84 own tests passing; io -> metrics -> pairwise/omnibus -> report)
  -> results/tablas/, figures/ (format-converted or re-rendered, not recomputed)
  -> Thesis.md Chapters 6-7 (cites the tables/figures above by exact filename)
```

No manual number-entry step exists between the stats pipeline's output and the thesis
text — table/figure generation scripts (`scripts/export_thesis_tables.py`,
`scripts/generate_thesis_figures.py`) read the pipeline's own already-tested output and
reformat it, so a reader who re-runs `experiments/stats/cli.py` and regenerates
`../results/`/`../figures/` should get byte-for-byte identical numbers to what the thesis
cites (deterministic given fixed seeds and fixed `experiments/results/` input — see
`README.md` for the seed/hardware documentation this depends on).

### 3.4 Known, disclosed limitations (not blockers to reproduction)

- **Docker image not build-tested** — Docker is unavailable in the environment this
  thesis and its reproducibility tooling were developed in. The `Dockerfile` mirrors the
  exact pinned `requirements.txt`/`pyproject.toml` used for every native run, but has
  not itself been verified to build. Flagged, not silently assumed correct.
- **CP-SAT under `num_search_workers > 1`** retains residual non-determinism for
  `FEASIBLE` (non-optimal) results even with a fixed `random_seed` — documented in
  `baselines/cpsat.py` and `REPRODUCIBILITY_REPORT.md` §1. This affects a small
  number of large/hard instances where CP-SAT does not prove optimality within its time
  budget (see the `la40` discussion in Chapter 6 §6.5.1); it does not affect any
  `OPTIMAL`-status result.
- **Trained checkpoints are excluded from the git repository** (`.gitignore`) but
  **included in the Zenodo archival deposit** (see `REPRODUCIBILITY_REPORT.md`
  §5) — a reader working from the git repository alone can reproduce E1–E8 by retraining
  (commands and measured runtimes in `README.md`), or reproduce them without retraining
  by using the Zenodo deposit's checkpoints directly.

---

## 4. Package identity

- Package name: `thesis-fjsp-drl` (`pyproject.toml`), version `0.1.0`.
- Python: 3.11.9. Dependencies pinned exactly in `requirements.txt`/`pyproject.toml`
  (torch 2.12.1, stable-baselines3 2.9.0, sb3-contrib 2.9.0, ortools 9.15.6755, scipy
  1.15.2, statsmodels 0.14.6, gymnasium 1.3.0 — full list and installed-environment
  cross-check in `REPRODUCIBILITY_REPORT.md` §1).
- CPU-only throughout (`torch.cuda.is_available()` was `False` on the machine that
  produced every reported result — `PreRegistration.md` Amendment 2).
