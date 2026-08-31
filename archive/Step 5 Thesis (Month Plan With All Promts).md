# Plan Semanal de Ejecución — Tesis DRL para FJSP Dinámico
## Estructura: 1 mes = 3 semanas | 1 semana de código al mes | Teoría/análisis en Cowork, código en el asistente de IA

> **Cómo leer este documento:** cada semana tiene un objetivo, entregables concretos, y un **prompt listo para copiar y pegar** — en Cowork (teoría, redacción, análisis, diseño) o en el asistente de IA (implementación). Los prompts están en inglés porque así rinden mejor los modelos técnicos, pero tú y yo seguimos hablando en español en todo lo demás. La numeración de semanas es continua (S1, S2, S3…) para que sea fácil ubicarlas en el calendario; los "meses" son bloques de 3 semanas, no meses naturales.

**Inicio:** lunes 27 de julio de 2026 (semana actual). **Duración total:** 9 meses ≈ 39 semanas ≈ 13 bloques de 3 semanas. **Regla fija:** la 3.ª semana de cada bloque es la **semana de código** (el asistente de IA); las semanas 1 y 2 de cada bloque son teoría/diseño/análisis (Cowork, lectura, redacción).

---

## Vista general por bloques (mes = 3 semanas)

| Bloque | Semanas | Fase (del plan original) | Semana de código (S3) dedicada a |
|---|---|---|---|
| Mes 1 | S1–S3 | Fase 1: Fundamentos y revisión de literatura | Reproducir entorno base + primer intento de L2D sobre ft06/ft10 |
| Mes 2 | S4–S6 | Fase 1→2: cierre de literatura, inicio de entorno DES | Esqueleto del entorno DES + wrapper Gymnasium |
| Mes 3 | S7–S9 | Fase 2: construcción y V&V del entorno | Inyección de eventos dinámicos (llegadas, averías, tiempos estocásticos) |
| Mes 4 | S10–S12 | Fase 2→3: cierre V&V, inicio líneas base | PDRs + arnés CP-SAT |
| Mes 5 | S13–S15 | Fase 3: líneas base | GA para FJSP + consolidación de resultados de líneas base |
| Mes 6 | S16–S18 | Fase 4: desarrollo del modelo (parte 1) | Codificador GNN heterogéneo + política PPO (v1) |
| Mes 7 | S19–S21 | Fase 4: desarrollo del modelo (parte 2) | Entrenamiento, ablaciones de recompensa/codificador, HPO |
| Mes 8 | S22–S24 | Fase 5: experimentos (generalización + robustez) | Ejecución de E1–E6 (matriz experimental) |
| Mes 9 | S25–S27 | Fase 5: cierre experimentos + estadística | E7–E8, análisis estadístico (Wilcoxon/Friedman/Nemenyi) |
| Mes 10 | S28–S30 | Fase 6: redacción (resultados + discusión) | Generación de figuras/tablas finales desde resultados |
| Mes 11 | S31–S33 | Fase 6: redacción (conclusiones, pulido) | Reproducibilidad final: limpieza de repo, Docker, README |
| Mes 12 | S34–S36 | Colchón / artículo de congreso | Extracción de resultados para paper (IEEE CASE/WSC) |
| Mes 13 | S37–S39 | Colchón final, defensa | Empaquetado final del código + reproducción end-to-end |

> Los meses 12–13 son **colchón deliberado** (buffer) frente a los 9 meses "ideales" del plan original, dado que ahora solo 1 de cada 3 semanas es de ejecución técnica. Si el avance es más rápido de lo previsto, se usan para el artículo de congreso y para pulir la defensa.

---

# MES 1 (S1–S3) — Fundamentos y arranque

## Semana 1 (27–31 jul 2026) — Cowork: cimentar el marco teórico y validar la bibliografía

**Objetivo:** convertir el marco teórico ya redactado en un documento verificado y blindar la revisión de literatura con búsquedas reales.

**Entregables:**
- Bibliografía completa re-verificada (Scopus/WoS/IEEE Xplore/doi.org) con correcciones aplicadas.
- Barrido PRISMA ejecutado con las 4 cadenas de búsqueda ya definidas; conteos por etapa.
- Documento de brechas → preguntas de investigación, actualizado con hallazgos 2024–2026.

**Prompt para Cowork:**
```
You are acting as my thesis research assistant in Cowork. I have an existing theoretical
framework document for a master's thesis titled "Generalizable Deep Reinforcement Learning
for Dynamic Flexible Job-Shop Scheduling under Stochastic Disruptions" (attached/referenced).

TASK 1 — Bibliographic verification:
Take the reference list (Part F of the framework document) and verify EVERY entry against
live sources: Scopus, Web of Science, IEEE Xplore, ScienceDirect, doi.org, and arXiv.
For each reference, confirm or correct: author order, title, venue, year, volume, issue,
pages, and DOI. Flag anything that cannot be found. Produce a corrected reference list in
IEEE format, with a change-log noting every correction made versus the original.

TASK 2 — PRISMA systematic search:
Execute these four search strings (adapt syntax per database as needed) across Scopus, Web
of Science, and IEEE Xplore, restricted to 2018–2026:
1. ("reinforcement learning" OR "deep reinforcement learning" OR "graph neural network")
   AND ("job shop" OR "job-shop" OR "flexible job shop" OR "FJSP") AND (scheduling)
2. ("reinforcement learning") AND ("job shop") AND ("dynamic" OR "stochastic" OR
   "machine breakdown" OR "disruption" OR "rescheduling")
3. ("neural combinatorial optimization" OR "graph neural network") AND ("generalization"
   OR "size generalization") AND (scheduling OR "combinatorial optimization")
4. ("large language model" OR "LLM agent") AND (scheduling OR "job shop" OR
   "combinatorial optimization")
Report hit counts per string per database, remove duplicates, and screen titles/abstracts
against these inclusion criteria: peer-reviewed article or arXiv preprint with code/impact;
applies RL or GNN to JSSP/FJSP; reports quantitative benchmarks. Produce a PRISMA-style
flow count (identified → deduplicated → screened → included) and a short list of the most
relevant 2024-2026 papers I had not yet identified, with full citations.

TASK 3 — Gap table update:
Using the newly found papers, update the gap-analysis table (identified gap → evidence →
research question) to confirm my thesis still occupies open territory, or flag where I need
to narrow the novelty claim.

Output as a single structured Markdown document I can review, with clear section headers.
```

---

## Semana 2 (3–7 ago 2026) — Cowork: cerrar Capítulo 1 y afinar Capítulo 3

**Objetivo:** redactar la introducción completa y avanzar el estado del arte con lo verificado en S1.

**Entregables:**
- Capítulo 1 completo (introducción, planteamiento, preguntas, objetivos, alcance, contribuciones).
- Capítulo 3 (estado del arte) actualizado con la tabla comparativa llena y la narrativa 2018–2026 corregida.

**Prompt para Cowork:**
```
Using the verified bibliography and PRISMA results from last week, and the existing thesis
skeleton (Part A) and theoretical framework (Part B), do the following:

1. Write the complete Chapter 1 (Introduction) of my master's thesis in Spanish, following
   the structure already defined: 1.1 Context and motivation, 1.2 Problem statement,
   1.3 Research questions and hypotheses, 1.4 Objectives, 1.5 Scope and delimitations,
   1.6 Expected contributions, 1.7 Document structure. Target ~10 pages. Academic tone,
   third person or passive voice as is standard in Spanish-language engineering theses.
   Cite sources using IEEE numbered style consistent with the verified reference list.

2. Update Chapter 3 (State of the Art): rewrite the narrative section (3.2) chronologically
   from 2018-2026 using the verified papers, fill in the comparative table (3.3) with real
   extracted values (state representation, architecture, RL algorithm, objective, benchmarks,
   baselines, reported results, limitations) pulled directly from the papers' abstracts/
   results tables, and finalize the gap-analysis table (3.6).

Flag any claim you are not fully confident about with [VERIFICAR]. Output two separate
Markdown files: Chapter1.md and Chapter3.md.
```

---

## Semana 3 (10–14 ago 2026) — 🖥️ CÓDIGO (el asistente de IA): entorno base + primer intento de L2D

**Objetivo:** montar el entorno de desarrollo, reproducir el entorno JSSEnv/schlably y correr una primera reproducción de L2D sobre ft06/ft10 como validación mínima.

**Entregables:**
- Repositorio inicializado con estructura de proyecto, entorno virtual, dependencias fijadas.
- Entorno JSSP funcional (Gymnasium) validado contra ft06=55, ft10=930.
- Primer notebook/script de entrenamiento PPO+GIN corriendo (aunque no converja aún del todo).

**Prompt para el asistente de IA:**
```
SKILL CONTEXT: Use good Python engineering practices throughout (type hints, docstrings,
pytest tests, pinned dependencies). This is a research codebase for a master's thesis, so
prioritize clarity and reproducibility over premature optimization.

GOAL: Set up the foundational codebase for a DRL job-shop scheduling research project.

STEP 1 — Project scaffolding:
Create a Python project with this structure:
  thesis_fjsp_drl/
    ├── envs/            # simulation + Gymnasium environments
    ├── models/          # GNN encoders, PPO policy
    ├── baselines/        # dispatching rules, GA, CP-SAT wrapper
    ├── data/instances/   # benchmark instance files (Taillard, Lawrence, Fisher-Thompson)
    ├── experiments/       # experiment configs and scripts
    ├── tests/
    ├── notebooks/
    ├── requirements.txt (or pyproject.toml)
    └── README.md
Use conda or venv; pin: torch, torch-geometric, gymnasium, stable-baselines3 (or write a
minimal PPO from scratch if easier to control — your call, explain the tradeoff), numpy,
pandas, matplotlib, ortools, pytest.

STEP 2 — Benchmark instance loader:
Implement a loader for standard JSSP instance file formats (Taillard format) that parses
ft06, ft10, ft20 (Fisher & Thompson) and la01-la40 (Lawrence) instance files into a common
internal representation (jobs, operations, machine assignment, processing time). Write unit
tests that check known optimal makespans: ft06=55, ft10=930, ft20=1165.

STEP 3 — JSSP Gymnasium environment:
Implement a Gymnasium-compatible environment for the static JSSP based on the disjunctive
graph representation:
  - observation_space: operation node features (processing time, remaining work, ready
    status, earliest start estimate) + adjacency for precedence and machine-sharing edges
  - action_space: Discrete over ready operations, with action masking for infeasibility
  - step(): advances simulated time to next decision epoch, applies dense reward = negative
    increment of a lower-bound makespan estimate (implement a simple critical-path lower
    bound), returns (obs, reward, terminated, truncated, info)
  - reset(seed): loads an instance and resets state
Write unit tests verifying: valid schedules never violate precedence or machine constraints;
a hand-computed optimal action sequence on ft06 reaches makespan 55.

STEP 4 — Minimal GIN + PPO baseline:
Implement a minimal Graph Isomorphism Network (GIN) encoder over the disjunctive graph state,
feeding an actor-critic PPO policy (you may use stable-baselines3's PPO with a custom feature
extractor, or a lightweight custom PPO — prefer stable-baselines3 for reliability unless the
custom action space makes that impractical, in which case implement PPO manually and explain
why). Train on ft06 for a small number of steps as a smoke test — the goal this week is a
RUNNING pipeline, not convergence.

STEP 5 — Validation report:
Run the test suite. Produce a short markdown report (VALIDATION_WEEK1.md) confirming:
(a) instance loader passes known-optimum tests, (b) environment respects all constraints
under random and greedy-SPT policies, (c) the PPO+GIN pipeline runs end-to-end without
crashing for at least 1000 timesteps on ft06, with a plot of the reward curve.

Document any design decisions or deviations from the plan in comments and in the README.
```

---

# MES 2 (S4–S6)

## Semana 4 (17–21 ago) — Cowork: cerrar Capítulo 3, redactar 4.1–4.3

**Objetivo:** terminar el estado del arte y comenzar la metodología (enfoque general + formulación MDP), incorporando lo aprendido en la S3 de código.

**Entregables:** Capítulo 3 cerrado; secciones 4.1 (enfoque de solución) y 4.3 (formulación MDP) redactadas y ajustadas a lo realmente implementado.

**Prompt para Cowork:**
```
I completed a first coding sprint implementing the JSSP Gymnasium environment and a minimal
GIN+PPO pipeline (validation report attached: VALIDATION_WEEK1.md, plus the actual state/
action/reward implementation from the codebase). Using this as ground truth:

1. Finalize Chapter 3 (State of the Art) — incorporate any additional papers found, polish
   the narrative and comparative table, write the closing gap-analysis synthesis paragraph.

2. Write Section 4.1 (Overall approach and solution architecture) — describe the end-to-end
   pipeline I actually built: DES/Gymnasium environment, GNN encoder, PPO policy, baselines,
   evaluation protocol. Include an architecture diagram description (I'll render it separately).

3. Write Section 4.3 (MDP formulation: state, action space, reward, masking) — align this
   EXACTLY with what is implemented in the code (state features, action masking logic, the
   lower-bound reward formula). If the code diverges from the original theoretical spec,
   note the deviation and justify it.

Output as Chapter3_final.md and Chapter4_sections_4.1_4.3.md, academic Spanish, IEEE citations.
```

---

## Semana 5 (24–28 ago) — Cowork: redactar 4.2 (entorno DES) y diseñar la extensión dinámica

**Objetivo:** especificar en detalle la extensión del entorno estático hacia el dinámico/estocástico, lista para implementar en la semana de código.

**Entregables:** Sección 4.2 completa; especificación técnica detallada (pseudocódigo, eventos, parámetros de las distribuciones) lista para pasar a el asistente de IA.

**Prompt para Cowork:**
```
Write Section 4.2 (Discrete-event simulation environment and Gymnasium interface contract)
of my thesis methodology chapter, in academic Spanish, based on the procedural specification
already drafted (Part D1 of the framework document) and the actual static environment code
from Week 1.

Then produce a SEPARATE, highly detailed technical specification (in English, engineering-
spec style, not thesis prose) for extending the static environment to the dynamic/stochastic
case, to be handed directly to a coding agent next week. It must include:
- Exact event types and their routines (job arrival, machine breakdown, repair completion)
- Exact distributions and parameters: TTF ~ Exp(lambda_k) calibrated to target availability
  A_k in {0.95, 0.90, 0.80}; TTR ~ Lognormal; processing time multiplier ~ Lognormal(-s^2/2, s^2)
  for s in {0.1, 0.25, 0.5}; job arrivals ~ Poisson(lambda_a) calibrated to target utilization
  rho in {0.70, 0.85, 0.95}
- How preemption is handled when a machine breaks down mid-operation (preempt-resume vs
  preempt-restart — pick one, justify it)
- Required new unit tests to validate the dynamic environment

Output: Section4.2.md (thesis prose, Spanish) + DynamicEnv_TechSpec.md (English, ready to
paste into el asistente de IA next week).
```

---

## Semana 6 (31 ago–4 sep) — 🖥️ CÓDIGO (el asistente de IA): entorno dinámico/estocástico

**Objetivo:** implementar llegadas dinámicas, averías y tiempos de proceso estocásticos sobre el entorno base, con validación estadística.

**Entregables:** entorno dinámico funcional y testeado; reporte de validación con verificación de las distribuciones empíricas vs. teóricas.

**Prompt para el asistente de IA:**
```
CONTEXT: Attached is DynamicEnv_TechSpec.md, a detailed specification for extending the
static JSSP/FJSP Gymnasium environment (already implemented in envs/) to support dynamic
job arrivals, stochastic processing times, and machine breakdowns.

GOAL: Implement the dynamic/stochastic extension exactly per the spec, on top of the
existing codebase — do not rewrite the static environment, extend it.

STEP 1: Implement the event-list-based DES core (future event list, next-event time
advance) if not already present in a suitable form; add event types for job arrival,
machine breakdown, and repair completion alongside the existing operation-completion event.

STEP 2: Implement the three disruption generators as separate, independently testable
modules: (a) breakdown generator (TTF ~ Exp, TTR ~ Lognormal, calibrated to target
availability), (b) stochastic processing time sampler (Lognormal multiplier), (c) Poisson
arrival process calibrated to target utilization. Each should be a pure function/class that
can be unit tested in isolation with statistical assertions (e.g., sample mean within
tolerance of theoretical mean over N draws).

STEP 3: Wire these into the Gymnasium environment: extend observation_space to include
machine availability/failure state features, ensure action masking correctly excludes
operations on failed machines, and confirm the environment gracefully handles new jobs
arriving mid-episode (extending the disjunctive graph dynamically).

STEP 4: Decide and implement preemption handling as specified (preempt-resume or
preempt-restart) with a clear comment explaining the choice and its consequence for the
reward calculation.

STEP 5: Write a validation report (VALIDATION_WEEK6.md): statistical tests confirming
empirical TTF/TTR/processing-time/arrival distributions match theoretical targets within
tolerance across 1000+ samples; confirm the environment still respects all hard constraints
(precedence, no machine double-booking) under disruptions; confirm episodes still terminate
correctly with dynamically arriving jobs.

Update tests/ accordingly. Do not touch the baseline PPO+GIN training code yet — that comes
in a later sprint.
```

---

# MES 3 (S7–S9)

## Semana 7 (7–11 sep) — Cowork: Capítulo 4 (modelos de perturbación, 4.5) + revisión de V&V

**Objetivo:** documentar formalmente los modelos de perturbación implementados y preparar el protocolo de verificación y validación estadística del entorno.

**Entregables:** Sección 4.5 redactada con los parámetros realmente usados; protocolo de V&V (réplicas necesarias, pruebas de bondad de ajuste) listo para ejecutar.

**Prompt para Cowork:**
```
Using VALIDATION_WEEK6.md and the actual disruption-generator code, write Section 4.5
(Disruption models) of the thesis in academic Spanish, presenting the exact distributions
and parameters used (with justification for the chosen availability/utilization targets),
matching the equations already drafted in the framework's Part D5.

Then design a formal Verification & Validation protocol for the simulation environment,
to be executed empirically (not this week, but specified now): number of replications
needed for a target confidence-interval half-width using the formula R >= (z*S/h)^2,
goodness-of-fit tests to confirm empirical distributions match theoretical ones
(Kolmogorov-Smirnov or similar), and a checklist of Sargent-style validation techniques
(analytical comparison cases, degeneracy tests, sensitivity analysis) applicable to this
environment. Output as Section4.5.md and VandV_Protocol.md.
```

---

## Semana 8 (14–18 sep) — Cowork: Capítulo 4.6 (líneas base) + diseño del arnés de comparación

**Objetivo:** especificar en detalle las líneas base (PDRs, GA, CP-SAT) que se implementarán en la siguiente semana de código.

**Entregables:** Sección 4.6 redactada; especificación técnica del arnés de benchmarking común.

**Prompt para Cowork:**
```
Write Section 4.6 (Baselines: dispatching rules, genetic algorithm, CP-SAT) of the thesis
in academic Spanish, formalizing each baseline exactly as defined in the framework's
Part B6-B7, and describing how they will be integrated into a common evaluation harness.

Then produce a technical specification (English) for a coding agent to implement next week:
- A common `Baseline` interface so PDRs, GA, and CP-SAT all expose a uniform
  `solve(instance) -> schedule, makespan, wall_clock_time` method
- The list of PDRs to implement: SPT, LPT, MWKR, LWKR, MOR, LOR, FIFO, EDD, CR
- GA design: chromosome encoding (operation sequence + machine assignment vectors),
  crossover operators (POX for sequence, uniform for assignment), mutation (swap,
  reassignment), selection (tournament + elitism), population size and generation count
  recommendations for a student compute budget
- CP-SAT wrapper design using OR-Tools: interval variables, NoOverlap/Cumulative
  constraints for JSSP, OptionalInterval for FJSP machine alternatives, with a
  configurable wall-clock time limit
Output: Section4.6.md + Baselines_TechSpec.md.
```

---

## Semana 9 (21–25 sep) — 🖥️ CÓDIGO (el asistente de IA): implementar PDRs y arnés CP-SAT

**Objetivo:** implementar todas las reglas de despacho y el wrapper de CP-SAT, con validación cruzada.

**Entregables:** módulo `baselines/` con PDRs y CP-SAT funcionando y testeado sobre ft06–ft20 y la01–la40.

**Prompt para el asistente de IA:**
```
CONTEXT: Attached is Baselines_TechSpec.md. The environment and instance loader already
exist in envs/ and data/instances/.

GOAL: Implement the baseline solvers per spec.

STEP 1: Implement all 9 priority dispatching rules (SPT, LPT, MWKR, LWKR, MOR, LOR, FIFO,
EDD, CR) as a family of classes/functions sharing a common interface, operating directly
on the Gymnasium environment (i.e., they should be usable as the `policy` in the same loop
that will later host the trained RL agent).

STEP 2: Implement the OR-Tools CP-SAT wrapper for JSSP (interval variables + NoOverlap) and
extend it to FJSP (OptionalInterval + exactly-one-machine constraint). Make the wall-clock
time limit configurable. Return: schedule, makespan, whether optimality was proven, and
solve time.

STEP 3: Write a small evaluation script that runs all PDRs + CP-SAT (with a 60s limit) on
ft06, ft10, ft20, and a sample of la01-la10, and produces a results table (RPD vs known
optimum, wall-clock time) as a CSV and a Markdown table.

STEP 4: Unit tests: each PDR produces a valid, constraint-respecting schedule; CP-SAT
recovers the known optimum on ft06/ft10/ft20 within its time limit.

Output a BASELINES_RESULTS_WEEK9.md summarizing the results table and any issues found.
Do not implement the genetic algorithm yet — that's next month's code sprint.
```

---

# MES 4 (S10–S12)

## Semana 10 (28 sep–2 oct) — Cowork: análisis de resultados de líneas base + Capítulo 5.1–5.2

**Objetivo:** interpretar los resultados de PDRs/CP-SAT obtenidos, y comenzar el capítulo de diseño experimental.

**Entregables:** análisis narrativo de los resultados de líneas base; secciones 5.1 (matriz de experimentos) y 5.2 (protocolo de evaluación) redactadas y ajustadas a la realidad del código.

**Prompt para Cowork:**
```
I have baseline results (BASELINES_RESULTS_WEEK9.md, attached) from priority dispatching
rules and CP-SAT on ft06/ft10/ft20/la01-la10.

1. Write a short analytical section interpreting these results: which PDR performs best on
   average, how far CP-SAT's proven optima are from PDR results, and what this implies as
   the "floor" and "ceiling" my DRL model must be measured against.

2. Write Section 5.1 (Experiment factor and design matrix) and Section 5.2 (Evaluation
   protocol and metrics) of the thesis in academic Spanish, using the framework's Part D4
   and D6 as the base, but adjusted with concrete instance names/sizes now that I know
   what runs in reasonable time on my hardware (reference the wall-clock times observed).

Output: BaselineAnalysis.md + Section5.1_5.2.md.
```

---

## Semana 11 (5–9 oct) — Cowork: diseño del GA + Sección 4.4 (arquitectura GNN+PPO)

**Objetivo:** especificar en profundidad el algoritmo genético y la arquitectura del codificador GNN + política PPO antes de codificarlos.

**Entregables:** Sección 4.4 redactada; especificación técnica completa del GA y de la arquitectura GNN+PPO para la semana de código.

**Prompt para Cowork:**
```
Write Section 4.4 (Heterogeneous GNN encoder and PPO policy) of the thesis in academic
Spanish, formalizing the architecture per the framework's Part B5 and B2.2, but written to
match what will actually be implemented: a heterogeneous graph with operation and machine
nodes, relation-specific message passing, and a PPO actor-critic head.

Then produce two technical specs for next week's coding sprint (English):
1. GA_TechSpec.md: finalize genetic algorithm hyperparameters (population size, generations,
   crossover/mutation rates) appropriate for a single-GPU/CPU student budget, with a
   recommended starting configuration and a note on how to tune it cheaply.
2. GNN_PPO_Architecture_Spec.md: exact layer-by-layer architecture — node feature
   dimensions, number of GNN layers, hidden dimensions, aggregation type (GIN vs GAT — 
   recommend one with justification), actor and critic head structure, action masking
   integration, and the PPO hyperparameters to start from (learning rate, clip epsilon,
   GAE lambda, entropy coefficient, batch size, rollout length) citing standard starting
   points from the PPO and L2D/Song et al. literature.
```

---

## Semana 12 (12–16 oct) — 🖥️ CÓDIGO (el asistente de IA): implementar GA

**Objetivo:** implementar y validar el algoritmo genético para FJSP, integrado al mismo arnés de evaluación que las PDRs y CP-SAT.

**Entregables:** módulo GA funcional, comparado contra PDRs y CP-SAT en las mismas instancias.

**Prompt para el asistente de IA:**
```
CONTEXT: Attached is GA_TechSpec.md. The baseline harness (PDRs, CP-SAT) and evaluation
script from Week 9 already exist.

GOAL: Implement the genetic algorithm baseline for JSSP/FJSP per spec, integrated into the
same `Baseline` interface as the PDRs and CP-SAT wrapper.

STEP 1: Implement chromosome encoding (operation sequence permutation + machine assignment
vector for FJSP), decoding to an active/semi-active schedule.

STEP 2: Implement POX crossover for the sequence part, uniform crossover for the assignment
part, swap mutation and machine-reassignment mutation, tournament selection with elitism.

STEP 3: Implement the main GA loop with configurable population size, generation count,
crossover/mutation rates, and a convergence/early-stopping criterion.

STEP 4: Run the GA on the same instance set as Week 9 (ft06, ft10, ft20, la01-la10, plus
now MK01-MK10 if the FJSP instance loader is ready — extend the loader if needed for the
Brandimarte MK format) and extend the results table/CSV from Week 9 with GA results
(makespan, RPD vs optimum, wall-clock time).

STEP 5: Unit tests for crossover/mutation operators producing valid chromosomes; test that
GA makespan on ft06 converges close to the known optimum (55) given enough generations.

Output BASELINES_RESULTS_WEEK12.md with the updated comparison table across all four
baselines (PDRs, GA, CP-SAT) — this becomes the reference table for the whole thesis.
```

---

# MES 5 (S13–S15)

## Semana 13 (19–23 oct) — Cowork: consolidar líneas base + Capítulo 5.3 (protocolo estadístico)

**Objetivo:** análisis comparativo completo de las tres familias de líneas base, y especificación del protocolo estadístico.

**Entregables:** análisis comparativo (PDRs vs GA vs CP-SAT) listo para el Capítulo 6; Sección 5.3 redactada.

**Prompt para Cowork:**
```
Using BASELINES_RESULTS_WEEK12.md (PDRs, GA, CP-SAT compared across ft/la/MK instances),
write a thorough comparative analysis: which baseline dominates at which instance size,
where CP-SAT's time limit becomes binding, and what performance gap the future DRL model
needs to close or beat.

Then write Section 5.3 (Statistical testing procedure) of the thesis in academic Spanish,
formalizing the Wilcoxon signed-rank test, the Friedman test with Nemenyi post-hoc, and
critical-difference diagrams exactly as in framework Part D6, but now written against the
ACTUAL baseline result data structure I have (describe the CSV schema so the statistical
code next relevant coding sprint can consume it directly).

Output: BaselineComparativeAnalysis.md + Section5.3.md.
```

---

## Semana 14 (26–30 oct) — Cowork: revisión integral de Capítulos 1–5 (borrador intermedio)

**Objetivo:** primera revisión de coherencia de todo lo redactado hasta ahora, antes de empezar el modelo principal.

**Entregables:** borrador consolidado de Capítulos 1–5 con retroalimentación de consistencia, y lista de vacíos pendientes.

**Prompt para Cowork:**
```
Act as a doctoral-level thesis auditor. I'm attaching my consolidated draft of Chapters
1 through 5 (introduction, theoretical framework, state of the art, methodology, experimental
design). Review it for:
- Logical coherence: does each chapter build correctly on the previous one?
- Consistency: do the state, action, and reward definitions match EXACTLY across the
  theoretical framework, the methodology chapter, and (implicitly) the code I've described
  in weekly validation reports?
- Terminology consistency in Spanish (are technical terms translated consistently?)
- Citation completeness: any claim lacking a citation?
- Gaps: what's missing before I can credibly move into model development?
Produce a structured critique with a prioritized action list, in Spanish, organized by
chapter and severity (critical / moderate / minor).
```

---

## Semana 15 (2–6 nov) — 🖥️ CÓDIGO (el asistente de IA): consolidación y refactor del entorno + arnés unificado

**Objetivo:** semana de "salud del código" — refactorizar, consolidar el arnés de evaluación completo (PDRs+GA+CP-SAT+entorno dinámico) antes de introducir el modelo GNN+PPO, e incorporar el soporte FJSP completo.

**Entregables:** codebase refactorizado, arnés de evaluación unificado, soporte completo para JSSP y FJSP estático y dinámico.

**Prompt para el asistente de IA:**
```
GOAL: Consolidation and refactor sprint before starting the main GNN+PPO model. No new
research features — this is a code-health sprint.

STEP 1: Review the full codebase (environments, baselines, instance loaders) built over
the past four sprints. Refactor for consistency: ensure the static and dynamic environments
share a clean common base class, ensure PDRs/GA/CP-SAT all truly share the `Baseline`
interface, ensure FJSP support (machine eligibility sets, OptionalInterval in CP-SAT,
machine-assignment in GA) is complete and tested, not just JSSP.

STEP 2: Build a single unified `run_evaluation.py` script/CLI that takes an instance set,
a list of methods (PDRs, GA, CP-SAT, and a placeholder for the future RL policy), and a
disruption configuration (none / breakdowns / stochastic times / arrivals / combined), and
outputs a standardized results CSV with columns matching what the statistical analysis
(Wilcoxon/Friedman/Nemenyi) will need.

STEP 3: Add integration tests that run this harness end-to-end on a small instance set
and check the output CSV schema and validity.

STEP 4: Update the README with clear setup and usage instructions, and write
REFACTOR_NOTES_WEEK15.md documenting what changed and why.

This sprint sets the foundation the RL model will plug into next month — prioritize a
clean, well-documented interface over new functionality.
```

---

# MES 6 (S16–S18) — Desarrollo del modelo (parte 1)

## Semana 16 (9–13 nov) — Cowork: revisar arquitectura GNN+PPO a la luz del refactor

**Objetivo:** ajustar la especificación de arquitectura tras el refactor, y preparar el plan de entrenamiento e hiperparámetros.

**Entregables:** especificación de arquitectura actualizada; plan de entrenamiento (curriculum, hiperparámetros iniciales, criterios de éxito) para la semana de código.

**Prompt para Cowork:**
```
Given REFACTOR_NOTES_WEEK15.md and the unified evaluation harness now in place, refine
GNN_PPO_Architecture_Spec.md (from Week 11) to match the actual environment interface.

Produce a detailed training plan for next week's coding sprint:
- Curriculum: train on small instances first (which sizes exactly, given my hardware),
  with a clear plan for scaling up
- Hyperparameter starting point table (learning rate, PPO clip epsilon, GAE lambda,
  entropy coefficient, batch size, rollout length, number of GNN layers, hidden dim)
  with justification from L2D/Song et al. and standard PPO practice
- Success criteria for this sprint: what loss/reward curve behavior indicates the pipeline
  is learning correctly vs. broken (e.g., reward should trend toward 0 as it approaches
  -C_max optimum; entropy should decay gradually, not collapse immediately)
- A monitoring/logging plan (what metrics to track with e.g. TensorBoard or Weights&Biases)
Output: GNN_PPO_Architecture_Spec_v2.md + TrainingPlan_Week17.md.
```

---

## Semana 17 (16–20 nov) — Cowork: preparar protocolo de ablaciones y pre-registro

**Objetivo:** finalizar el documento de pre-registro de hipótesis (antes de tener cualquier resultado del modelo) y el plan de ablaciones.

**Entregables:** documento de pre-registro congelado (H1–H5 con reglas de decisión); plan de ablaciones (codificador, recompensa) para más adelante.

**Prompt para Cowork:**
```
This is a critical methodological step: freeze the pre-registration document BEFORE any
GNN+PPO training results exist, to protect against hypothesizing after results are known
(HARKing).

Using framework Part D7 as the base, and the training plan from last week, write the final
pre-registration document: hypotheses H1 (static competitiveness), H2 (size generalization),
H3 (robustness), H4 (disruption-aware reward), H5 (decision latency), each with an explicit,
falsifiable decision rule (statistical test + threshold) fixed in advance. Timestamp this
document clearly.

Also draft the ablation study plan (encoder: GIN vs GAT; reward: dense vs sparse, standard
vs disruption-aware; single- vs multi-objective) as a table of planned runs, to be executed
in Month 7's coding sprint.

Output: PreRegistration_FINAL.md (date it) + AblationPlan.md.
```

---

## Semana 18 (23–27 nov) — 🖥️ CÓDIGO (el asistente de IA): implementar GNN heterogéneo + PPO (v1)

**Objetivo:** implementar la arquitectura completa GNN heterogéneo + PPO y lograr una primera corrida de entrenamiento estable en instancias pequeñas.

**Entregables:** modelo entrenable end-to-end; primeras curvas de entrenamiento en ft06/ft10 y una instancia FJSP pequeña.

**Prompt para el asistente de IA:**
```
CONTEXT: GNN_PPO_Architecture_Spec_v2.md and TrainingPlan_Week17.md are attached. The
unified evaluation harness (Week 15) and dynamic environment (Week 6) already exist.

GOAL: Implement the full heterogeneous GNN encoder + PPO policy per spec, and get a first
stable training run.

STEP 1: Implement the heterogeneous graph construction from the environment state:
operation nodes, machine nodes, precedence edges, eligibility edges, machine-sharing edges,
with the feature vectors specified in the architecture doc.

STEP 2: Implement the heterogeneous GNN encoder (relation-specific message passing per
edge type) using PyTorch Geometric's heterogeneous graph support (HeteroConv or similar).

STEP 3: Implement the actor-critic heads on top of the encoder: actor produces logits over
the (operation, machine) action space with masking; critic produces a state-value estimate.

STEP 4: Wire this into a PPO training loop (stable-baselines3 with a custom policy, or a
custom PPO implementation if the heterogeneous graph batching makes SB3 impractical --
your call, but document the reasoning) using the hyperparameters from the training plan.

STEP 5: Train on ft06 and one small FJSP instance (e.g., MK01) for a substantial number of
timesteps, log reward, loss, and entropy curves, and compare the trained policy's makespan
against the PDR/GA/CP-SAT baselines already in the results CSV.

STEP 6: Produce TRAINING_REPORT_WEEK18.md: training curves (as saved plots), final makespan
comparison table, and an honest assessment against the success criteria from the training
plan -- if it's not learning well, diagnose likely causes (reward scale, action masking bug,
graph construction bug, hyperparameters) rather than just reporting failure.

This is a hard sprint -- prioritize getting a correct, even if not yet optimal, training
loop over premature tuning.
```

---

# MES 7 (S19–S21) — Desarrollo del modelo (parte 2)

## Semana 19 (30 nov–4 dic) — Cowork: analizar resultados de entrenamiento v1 y ajustar

**Objetivo:** interpretar TRAINING_REPORT_WEEK18.md, diagnosticar problemas y preparar el plan de HPO/ablaciones para la próxima semana de código.

**Entregables:** diagnóstico y plan de acción; especificación de HPO (Optuna) lista para implementar.

**Prompt para Cowork:**
```
Attached: TRAINING_REPORT_WEEK18.md (first GNN+PPO training run, curves and results).

1. Analyze the training curves and results as a research methodologist would: is the reward
   trending correctly? Is entropy collapsing prematurely (insufficient exploration) or not
   decaying at all (policy not committing)? How does the learned policy's makespan compare
   to PDR/GA/CP-SAT? Diagnose likely causes for any issues (reward shaping, graph
   construction, hyperparameters, insufficient training steps).

2. Produce a prioritized action plan for next week's coding sprint: what to fix first, what
   hyperparameter search to run (design an Optuna search space over learning rate, GNN
   layers/width, PPO clip epsilon, entropy coefficient, discount factor), and what the
   ablation runs (GIN vs GAT, dense vs sparse reward) from AblationPlan.md should validate.

Output: TrainingDiagnosis.md + HPO_TechSpec.md.
```

---

## Semana 20 (7–11 dic) — Cowork: redactar avance de Capítulo 6 (metodología de resultados) + revisar Anexo B

**Objetivo:** preparar la estructura de reporte de resultados y el anexo de hiperparámetros/arquitectura, para que estén listos en cuanto haya resultados finales.

**Entregables:** plantilla de Capítulo 6 con tablas y figuras vacías pero especificadas; Anexo B (hiperparámetros y arquitectura) actualizado con la arquitectura real implementada.

**Prompt para Cowork:**
```
Even though final experimental results don't exist yet, prepare the reporting scaffolding
now so results can be dropped in as they're produced.

1. Draft Chapter 6 (Results) as a template in Spanish: for each subsection (6.1 Environment
   validation, 6.2 Static performance, 6.3 Size generalization, 6.4 Robustness under
   disruption, 6.5 Decision latency), write the framing prose, specify exactly which table(s)
   and figure(s) will go there (with captions pre-written and placeholder markers like
   [TABLA: resultados RPD por instancia]), and note which hypothesis (H1-H5) each subsection
   addresses.

2. Update Appendix B (hyperparameters and architecture) with the ACTUAL architecture
   implemented (from Week 18's code and this month's HPO spec) — layer counts, dimensions,
   final hyperparameter values once chosen.

Output: Chapter6_template.md + AppendixB.md.
```

---

## Semana 21 (14–18 dic) — 🖥️ CÓDIGO (el asistente de IA): HPO + ablaciones + entrenamiento en instancias medianas

**Objetivo:** ejecutar la búsqueda de hiperparámetros, correr las ablaciones definidas, y escalar el entrenamiento a instancias medianas (la, MK medianas).

**Entregables:** mejor configuración de hiperparámetros identificada; resultados de ablación (GIN vs GAT, recompensa densa vs dispersa); modelo entrenado en instancias medianas.

**Prompt para el asistente de IA:**
```
CONTEXT: HPO_TechSpec.md and AblationPlan.md are attached. The GNN+PPO training pipeline
from Week 18 exists and produced a working (if imperfect) baseline run.

GOAL: Hyperparameter optimization and ablation sprint.

STEP 1: Implement an Optuna study over the specified search space (learning rate, GNN
layers/width, PPO clip epsilon, entropy coefficient, discount factor), using a reduced
training budget per trial (short runs) and selecting on validation-instance makespan, not
training-instance makespan. Report the best configuration found and the study's parameter
importance plot.

STEP 2: With the best hyperparameters, run the ablation study from AblationPlan.md:
GIN vs GAT encoder, dense vs sparse reward. Train each variant under matched compute budget
and compare final makespan and training stability.

STEP 3: Retrain the best configuration with a full training budget on a curriculum
including la01-la20 and MK01-MK05 (small-to-medium), tracking generalization behavior
even at this stage (train small, periodically eval on slightly larger held-out instances).

STEP 4: Produce HPO_ABLATION_REPORT_WEEK21.md: best hyperparameters with justification,
ablation comparison table, and updated training curves for the final configuration. Save
the best model checkpoint.

Flag honestly if compute/time constraints prevented completing the full Optuna study or
ablation matrix, and note what was prioritized and why.
```

---

# MES 8 (S22–S24) — Experimentos: generalización y robustez

## Semana 22 (21–24 dic, semana corta por fiestas) — Cowork: preparar matriz experimental final

**Objetivo:** revisar y congelar la matriz experimental (E0–E8) con los recursos e infraestructura ya disponibles, ajustando alcance si es necesario por el calendario navideño.

**Entregables:** matriz experimental final confirmada; plan detallado de ejecución para las próximas dos semanas de código.

**Prompt para Cowork:**
```
Given everything built so far (environment, baselines, trained GNN+PPO model with best
hyperparameters from HPO_ABLATION_REPORT_WEEK21.md), review and finalize the experimental
design matrix (E0-E8 from framework Part D4) for actual execution.

For each experiment, confirm: exact instance sets to use, exact factor levels (disruption
intensities, sizes), number of replications feasible given my compute budget and the
holiday-shortened weeks ahead, and which pre-registered hypothesis (H1-H5) it tests.

Adjust scope realistically if needed — better to run fewer experiments rigorously (with
proper replication and statistics) than many experiments superficially. Recommend a
priority order in case time runs short: which experiments are load-bearing for the thesis
argument and which are supplementary.

Output: FinalExperimentMatrix.md with a week-by-week execution schedule for the next two
coding sprints.
```

---

## Semana 23 (28 dic–1 ene, semana corta) — Cowork: ligero — preparar scripts de análisis estadístico

**Objetivo:** semana de bajo volumen por fiestas; preparar (sin ejecutar) los scripts de análisis estadístico que se aplicarán a los resultados una vez generados.

**Entregables:** especificación completa de los scripts de análisis estadístico (Wilcoxon, Friedman, Nemenyi, diagramas de diferencia crítica) lista para implementar.

**Prompt para Cowork:**
```
Light week -- prepare, don't execute. Write a complete technical specification (English)
for the statistical analysis pipeline that will process experimental results once
E1-E8 have been run:
- Input: the standardized results CSV schema (from the Week 15 unified harness)
- Wilcoxon signed-rank test for pairwise method comparisons, with exact statistic, p-value,
  and effect size (r = Z/sqrt(N)) reporting
- Friedman test across all methods per experiment, with Nemenyi post-hoc when significant,
  computing the critical difference CD = q_alpha * sqrt(k(k+1)/(6N)) and producing a
  critical-difference diagram
- A function to compute RPD, robustness index (RI), and decision latency summary statistics
  per experiment condition
- Output format: publication-ready tables (Markdown/LaTeX) and figures (matplotlib)
Output: StatsAnalysis_TechSpec.md, ready to hand to el asistente de IA.
```

---

## Semana 24 (4–8 ene 2027) — 🖥️ CÓDIGO (el asistente de IA): ejecutar E1–E6

**Objetivo:** correr los experimentos estáticos (E1, E2), de generalización (E3), y de robustez (E4, E5, E6) según la matriz final.

**Entregables:** resultados completos de E1–E6 en CSV, listos para análisis estadístico.

**Prompt para el asistente de IA:**
```
CONTEXT: FinalExperimentMatrix.md defines the exact instances, factor levels, and
replication counts for E1-E6. The trained model checkpoint from Week 21 and the unified
evaluation harness are ready.

GOAL: Execute experiments E1 through E6 per the matrix.

STEP 1 (E1 - static JSSP, E2 - static FJSP): Evaluate the trained policy, all PDRs, GA, and
CP-SAT on the specified static benchmark instances. Multiple stochastic policy rollouts per
instance as specified (the policy is stochastic; PDRs/CP-SAT are deterministic so a single
run suffices for those).

STEP 2 (E3 - size generalization): Evaluate the SAME trained policy (no retraining) on
progressively larger held-out instances as specified in the matrix, without fine-tuning.
Record RPD vs instance size to characterize the generalization curve.

STEP 3 (E4, E5, E6 - robustness): Using the dynamic environment, evaluate the policy and
baselines under each disruption regime (breakdown intensities, processing-time variance
levels, arrival-rate/utilization levels) with the full replication count (30 per condition
as specified). PDRs and GA need re-evaluation under stochastic conditions too (not just
CP-SAT/deterministic comparisons) -- confirm how each baseline handles dynamic replanning
(e.g., PDRs naturally reschedule reactively; GA/CP-SAT may need periodic re-optimization --
implement a sensible re-optimization trigger for them and document it).

STEP 4: Save all results in the standardized CSV schema, organized by experiment ID. Do NOT
run any statistical significance testing yet -- that's next month, with the dedicated script.

STEP 5: Produce EXPERIMENT_LOG_WEEK24.md documenting any anomalies, instances that failed
to run, timeouts, or unexpected behavior, plus basic descriptive statistics (means, std
devs) per condition as a sanity check before formal analysis.

This is a long, compute-heavy sprint -- if time runs short, follow the priority order from
FinalExperimentMatrix.md and clearly document what was and wasn't completed.
```

---

# MES 9 (S25–S27) — Cierre de experimentos y análisis estadístico

## Semana 25 (11–15 ene) — Cowork: revisar EXPERIMENT_LOG_WEEK24 y ajustar plan de E7–E8

**Objetivo:** evaluar qué tan completa quedó la ejecución de E1–E6, y ajustar el alcance de E7 (recompensa consciente de perturbación) y E8 (tiempo de decisión) en consecuencia.

**Entregables:** plan ajustado y confirmado para la última semana de código de experimentos.

**Prompt para Cowork:**
```
Attached: EXPERIMENT_LOG_WEEK24.md with results and any issues from E1-E6.

Review completeness against FinalExperimentMatrix.md's priority order. For any incomplete
or anomalous experiments, decide: rerun, scope down, or accept as a documented limitation.

Finalize the exact scope for E7 (disruption-aware reward comparison, kappa in {0, 0.5, 1.0})
and E8 (decision latency benchmarking across instance sizes) for next week's final
experimental coding sprint, adjusting replication counts if compute time is a constraint.

Output: E7_E8_FinalPlan.md.
```

---

## Semana 26 (18–22 ene) — Cowork: redactar Capítulo 6 con resultados reales de E1–E6

**Objetivo:** empezar a llenar el Capítulo 6 con los resultados ya obtenidos (E1–E6), dejando E7–E8 pendientes hasta la semana de código.

**Entregables:** secciones 6.1–6.4 redactadas con datos reales; tablas y figuras generadas a partir de los CSV.

**Prompt para Cowork:**
```
Using the real results CSVs from E1-E6 (attached) and the Chapter6_template.md prepared
in Week 20, write Sections 6.1 (environment validation), 6.2 (static performance), 6.3
(size generalization), and 6.4 (robustness under disruption) with REAL data. Generate
publication-quality tables (Markdown, convertible to LaTeX) and describe what figures are
needed (I will generate the actual plots from the CSVs separately, or we can do this in a
short follow-up coding task).

Write in academic Spanish, present findings objectively -- if the model underperforms CP-SAT
or a baseline in some regime, report that honestly and note it for the discussion chapter
rather than obscuring it. Explicitly reference which hypothesis (H1-H3) each section's
findings support or refute, per the frozen pre-registration decision rules.

Output: Chapter6_sections_6.1-6.4_DRAFT.md.
```

---

## Semana 27 (25–29 ene) — 🖥️ CÓDIGO (el asistente de IA): ejecutar E7–E8 + análisis estadístico completo

**Objetivo:** correr los últimos experimentos y ejecutar todo el pipeline de análisis estadístico (Wilcoxon, Friedman, Nemenyi) sobre el conjunto completo de resultados.

**Entregables:** resultados de E7–E8; análisis estadístico completo con tablas de significancia y diagramas de diferencia crítica.

**Prompt para el asistente de IA:**
```
CONTEXT: E7_E8_FinalPlan.md and StatsAnalysis_TechSpec.md are attached. All E1-E6 results
CSVs exist from Week 24.

STEP 1 (E7): Train or fine-tune policy variants with the disruption-aware reward at
kappa in {0, 0.5, 1.0} per the plan, evaluate each under the disruption regimes from E4-E6,
and compare robustness index (RI) vs nominal RPD tradeoffs across kappa values.

STEP 2 (E8): Benchmark decision latency (milliseconds per decision epoch, and total
inference time per instance) for the trained policy vs CP-SAT vs GA across all tested
instance sizes. Use consistent hardware/timing methodology (document CPU/GPU used, whether
timing includes graph construction overhead, etc.).

STEP 3: Implement the full statistical analysis pipeline per StatsAnalysis_TechSpec.md and
run it over ALL experimental results (E1-E8): Wilcoxon signed-rank for pairwise comparisons,
Friedman + Nemenyi for multi-method comparisons, critical-difference diagrams, and RPD/RI/
latency summary tables. Test each pre-registered hypothesis (H1-H5) against its frozen
decision rule from PreRegistration_FINAL.md and report accept/reject explicitly.

STEP 4: Produce FINAL_RESULTS_REPORT.md: every table and figure needed for Chapter 6,
organized by section (6.1-6.5), plus a clear hypothesis-by-hypothesis verdict table.

This is the last major experimental sprint -- be thorough with the statistics, since this
is what the thesis defense will be tested on most rigorously.
```

---

# MES 10 (S28–S30) — Redacción: resultados y discusión

## Semana 28 (1–5 feb) — Cowork: completar Capítulo 6 (6.5) y comenzar Capítulo 7 (discusión)

**Prompt para Cowork:**
```
Using FINAL_RESULTS_REPORT.md (complete statistical results for E1-E8, all hypotheses
tested), finish Section 6.5 (decision latency) and integrate the final statistical
significance results into Sections 6.2-6.4 (add significance annotations to tables,
reference critical-difference diagrams).

Then draft Chapter 7 (Discussion): interpret the results in light of the literature from
Chapter 3 -- where do my findings agree or disagree with L2D, Song et al., and other prior
work? Discuss the theoretical implications for generalization (Part B8) and robust
scheduling theory. Be honest about limitations (Part M10 threats to validity from the
methodology doc) and discuss them explicitly: internal, external, construct, and conclusion
validity threats specific to what actually happened in the experiments.

Output: Chapter6_FINAL.md + Chapter7_DRAFT.md, academic Spanish.
```

## Semana 29 (8–12 feb) — Cowork: pulir Capítulo 7 + redactar Capítulo 8 (conclusiones)

**Prompt para Cowork:**
```
Review and strengthen Chapter7_DRAFT.md as a doctoral-level auditor would: are the
interpretations well-supported by the statistics? Is anything overclaimed? Add a
comparison table positioning my results against the published L2D/Song et al. gaps
(from Part E1 reference figures, now compared against my own verified results).

Then write Chapter 8 (Conclusions and future work): summarize contributions against the
original objectives (Chapter 1.4/1.6), state clearly which hypotheses were supported/
refuted, discuss practical implications for manufacturing/quality-maintenance contexts,
and propose concrete future work directions grounded in what was learned (not generic
boilerplate).

Output: Chapter7_FINAL.md + Chapter8.md.
```

## Semana 30 (15–19 feb) — 🖥️ CÓDIGO (el asistente de IA): generación de figuras finales + limpieza de resultados

**Prompt para el asistente de IA:**
```
GOAL: Generate all final publication-quality figures for the thesis from the experimental
result CSVs (E1-E8) and the statistical analysis outputs.

STEP 1: Produce a complete set of matplotlib figures matching every [FIGURA: ...] 
placeholder in Chapter6_FINAL.md and Chapter7_FINAL.md: RPD comparison bar charts per
method, generalization curves (RPD vs instance size), robustness comparison plots across
disruption intensities, critical-difference diagrams for each Friedman test, and decision
latency comparison plots (log-scale where appropriate).

STEP 2: Ensure consistent, thesis-appropriate styling (readable fonts, colorblind-safe
palette, consistent axis labeling in Spanish to match the thesis text), export as
high-resolution PDF/PNG suitable for LaTeX/Word inclusion.

STEP 3: Clean up the results CSVs and code into a final, well-organized `results/` directory
with a clear naming convention matching the thesis figure/table numbering.

Output all figures to figures/ with descriptive filenames, and a FIGURES_INDEX.md mapping
each figure file to its thesis section and caption.
```

---

# MES 11 (S31–S33) — Redacción final y reproducibilidad

## Semana 31 (22–26 feb) — Cowork: integrar figuras y compilar borrador completo

**Prompt para Cowork:**
```
Using FIGURES_INDEX.md and all finalized chapter files (Chapter1 through Chapter8), compile
a single, integrated thesis draft in Spanish: insert figure references and captions at the
correct points, ensure consistent numbering (sections, tables, figures, equations),
consistent terminology throughout, and a consistent citation style. Generate the front
matter (abstract in Spanish and English -- now with real final numbers from the results,
table of contents, list of figures/tables, nomenclature).

Do a full-document consistency pass flagging: any [VERIFICAR] tags still unresolved, any
broken cross-references, any terminology drift between chapters.

Output: Thesis_FullDraft_v1.md + ConsistencyCheckReport.md.
```

## Semana 32 (1–5 mar) — Cowork: revisión de auditor doctoral + pulido de redacción

**Prompt para Cowork:**
```
Act as my embedded doctoral thesis advisor performing a full pre-defense audit of
Thesis_FullDraft_v1.md. Evaluate: logical coherence across all chapters, methodological
rigor (does the executed methodology match what was specified? are deviations justified?),
feasibility and honesty of claims, alignment with academic standards for a master's thesis
in industrial engineering. Flag unsupported claims, structural problems, or any remaining
inconsistencies. Provide constructive, prioritized feedback (critical/moderate/minor) with
specific section references and suggested fixes.

Output: PreDefenseAudit.md.
```

## Semana 33 (8–12 mar) — 🖥️ CÓDIGO (el asistente de IA): reproducibilidad final del repositorio

**Prompt para el asistente de IA:**
```
GOAL: Final reproducibility sprint. A reader should be able to clone this repository and
reproduce the thesis results end-to-end.

STEP 1: Audit the full codebase for reproducibility: confirm all random seeds are
controlled and documented, all dependencies pinned (requirements.txt or environment.yml
with exact versions), and hardware/software requirements documented.

STEP 2: Write a comprehensive README.md: installation instructions, directory structure
explanation, how to reproduce each experiment (E0-E8) with exact commands, expected runtime
and hardware for each, and how to regenerate all figures/tables from raw results.

STEP 3: Add a Dockerfile (or conda environment.yml, whichever is more appropriate) that
fully captures the environment for long-term reproducibility.

STEP 4: Do a final end-to-end smoke test: fresh environment, run the full pipeline on a
reduced/fast configuration (few instances, few seeds) to confirm nothing is broken, and
document the smoke-test procedure in the README as a quick sanity check for examiners.

STEP 5: Prepare the repository for archival: clean commit history or a clear release tag,
and draft the text needed to archive it on Zenodo with a DOI (I will do the actual upload).

Output FINAL_REPRODUCIBILITY_REPORT.md confirming everything above is complete.
```

---

# MES 12 (S34–S36) — Colchón / artículo de congreso

## Semana 34 (15–19 mar) — Cowork: resolver hallazgos del PreDefenseAudit

**Prompt para Cowork:**
```
Using PreDefenseAudit.md, address every critical and moderate finding: revise the
corresponding thesis sections, strengthen weak arguments, add missing citations, resolve
inconsistencies. Produce Thesis_FullDraft_v2.md with a change-log summarizing what was
revised in response to each audit finding.
```

## Semana 35 (22–26 mar) — Cowork: redactar artículo de congreso (IEEE CASE / WSC)

**Prompt para Cowork:**
```
Using Thesis_FullDraft_v2.md as source material, draft a conference paper (target: IEEE
CASE or the Winter Simulation Conference, ~6-8 pages, standard IEEE two-column format
content -- I'll handle final LaTeX formatting) condensing the thesis into a standalone
paper: abstract, introduction, related work (condensed from Chapter 3), method (condensed
from Chapter 4), experiments and results (the strongest subset from Chapter 6), discussion,
and conclusion. Write in professional academic English suitable for peer review. Identify
the single strongest, most defensible claim to lead with.

Output: ConferencePaper_DRAFT.md.
```

## Semana 36 (29 mar–2 abr) — 🖥️ CÓDIGO (el asistente de IA): extraer resultados/figuras específicos del paper

**Prompt para el asistente de IA:**
```
GOAL: Support the conference paper with any additional analysis or figure variants needed
that weren't required for the thesis itself (e.g., more compact combined figures for page
budget, additional robustness checks a reviewer might request, or a supplementary results
table). Work from ConferencePaper_DRAFT.md's needs and the existing results/ directory --
do not rerun full experiments unless a genuine gap is identified. Output any new figures to
figures/paper/ and document what was added and why in PAPER_SUPPLEMENT_NOTES.md.
```

---

# MES 13 (S37–S39) — Cierre final y defensa

## Semana 37 (5–9 abr) — Cowork: pulido final del documento de tesis

**Prompt para Cowork:**
```
Final polish pass on Thesis_FullDraft_v2.md: proofread for academic Spanish quality, verify
every citation against the final reference list, verify every cross-reference and figure/
table number, ensure the abstract accurately reflects final results, and produce a final
PDF-ready Markdown (or note any LaTeX/Word conversion needs). Output: Thesis_FINAL.md +
a final proofreading changelog.
```

## Semana 38 (12–16 abr) — Cowork: preparar presentación de defensa

**Prompt para Cowork:**
```
Using Thesis_FINAL.md, draft a defense presentation outline (15-20 slides): problem
statement, research questions, theoretical framework highlights, methodology summary,
key results (prioritizing the strongest, most defensible findings), discussion, conclusions
and future work. For each slide, write speaker notes in Spanish anticipating likely
committee questions, especially around methodological choices and any limitations flagged
in the PreDefenseAudit. Also draft a Q&A prep document: likely tough questions (e.g., "why
does your model underperform CP-SAT here?", "how do you know this generalizes beyond your
benchmarks?") with strong, honest prepared answers.

Output: DefensePresentation_Outline.md + QA_Prep.md.
```

## Semana 39 (19–23 abr) — 🖥️ CÓDIGO (el asistente de IA): empaquetado final end-to-end

**Prompt para el asistente de IA:**
```
GOAL: Final packaging sprint before defense/submission.

STEP 1: Run the full smoke test from FINAL_REPRODUCIBILITY_REPORT.md one more time on a
clean environment to catch any late-breaking issues.

STEP 2: Tag a final release version of the repository, ensure the README, Dockerfile/
environment file, and all documentation are current and consistent with Thesis_FINAL.md.

STEP 3: Produce a final CODE_APPENDIX_SUMMARY.md suitable for inclusion as a thesis
appendix: repository structure, how each thesis chapter/section maps to specific code
modules, and confirmation that all reported results are reproducible from the archived code.

This closes the technical work. Congratulations on reaching submission.
```

---

# Notas de uso

- **Cowork vs. el asistente de IA:** las semanas de Cowork están pensadas para trabajo de investigación, redacción y análisis que se beneficia de búsqueda web, manejo de documentos largos y razonamiento extendido sin necesidad de un entorno de ejecución de código persistente. Las semanas de el asistente de IA son para todo lo que implica escribir, ejecutar y depurar código real de forma sostenida.
- **Adjuntar contexto:** cada prompt asume que le adjuntas los documentos generados en semanas anteriores (reportes de validación, especificaciones técnicas, CSVs de resultados). Es importante mantener esos artefactos organizados semana a semana.
- **Flexibilidad:** si una semana de código no alcanza para completar todo lo planeado, el patrón recomendado es: documentar honestamente qué quedó pendiente (como se pide explícitamente en varios prompts) y absorberlo en la semana de Cowork siguiente antes de avanzar.
- **Los meses 12–13 son colchón:** si el proyecto va según lo previsto, se usan para el artículo de congreso y la preparación de la defensa; si hay atrasos, absorben el desfase sin comprometer la fecha final.
