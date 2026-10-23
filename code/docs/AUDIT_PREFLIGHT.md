# AUDIT_PREFLIGHT.md — Auditoría de partida (Fase 0)

**Alcance:** solo lectura de código y CSV ya existentes. No se ejecutó ningún script del
pipeline del repositorio (`run_evaluation.py`, `experiments/stats/`,
`scripts/generate_thesis_figures.py`, `pytest`, ni ningún entrenamiento). Los valores
citados abajo son lecturas directas o agregaciones aritméticas simples (media, RPD,
Spearman cuando así se indica) sobre archivos CSV que ya existían en el repositorio antes
de esta sesión — no son resultados de una corrida nueva. Donde el cálculo requerido
excede una simple lectura/agregación de datos ya existentes (Fase 0a), se deja como
`[PENDIENTE-CORRIDA: experiments/audit/ri_artifact_check.py]`, según regla del prompt del
tutor.

**Fuentes leídas:** `code/experiments/results/week24/e1_seed0_baselines.csv`,
`code/experiments/results/week24/e4e5e6/*_with_rl.csv`,
`code/experiments/results/week24/baselines_week12.csv`,
`code/envs/dynamic_jssp_env.py`, `code/envs/disjunctive_graph.py`,
`code/baselines/dispatching_rules.py`, `code/baselines/due_dates.py`,
`code/experiments/stats/*.py`, `results/tablas/*.md`,
`../../thesis/PreRegistration.md`, `../../thesis/StatsAnalysis_TechSpec.md`, `../../thesis/RESULTS_REPORT.md`,
`../../thesis/E7_E8_Plan.md`.

---

## (a) El Índice de Robustez premia mecánicamente los planes nominales malos

**Script escrito, no ejecutado:** [`experiments/audit/ri_artifact_check.py`](experiments/audit/ri_artifact_check.py).
Calcula la correlación de Spearman entre RPD nominal (E1, `disruption=="none"`, seed 0) y
RI medio (30 semillas) por PDR, sobre las 9 reglas disponibles en cada uno de los seis
regímenes de averías (E4: A∈{0.95,0.90,0.80}) y tiempos estocásticos (E5:
σ∈{0.10,0.25,0.50}), más un reporte descriptivo separado para los tres regímenes de
llegadas (E6), donde solo hay 4 PDR (ver (c)).

**Valor a reproducir:** rho de Spearman entre -0.93 y -0.98, p < 3e-4, en los seis
regímenes de averías/tiempos estocásticos.

**Estado: EJECUTADO el 2026-08-21** (corrida corta, autorizada explícitamente por el
usuario para producir resultados preliminares — ver `EXECUTION_LOG_2026-08-21.md`).
Resultado real, sobre las 10 series disponibles por régimen (9 PDR + RL):

| Régimen | rho | p | ¿Dentro del rango del tutor (-0.93 a -0.98, p<3e-4)? |
|---|---|---|---|
| E4 averías A=0.95 | -0.9774 | 1.34e-13 | **Sí** |
| E4 averías A=0.90 | -0.9586 | 2.93e-11 | **Sí** |
| E4 averías A=0.80 | -0.9586 | 2.93e-11 | **Sí** |
| E5 tiempos σ=0.10 | -0.9752 | 3.13e-13 | **Sí** |
| E5 tiempos σ=0.25 | -0.9744 | 4.09e-13 | **Sí** |
| E5 tiempos σ=0.50 | -0.9714 | 1.10e-12 | **Sí** |
| E6 llegadas ρ=0.70 (descriptivo, n=5) | -0.9483 | 2.93e-05 | fuera del rango numérico exacto, pero misma dirección/fuerza |
| E6 llegadas ρ=0.85 (descriptivo, n=5) | -0.9240 | 1.33e-04 | fuera del rango numérico exacto, pero misma dirección/fuerza |
| E6 llegadas ρ=0.95 (descriptivo, n=5) | -0.9240 | 1.33e-04 | fuera del rango numérico exacto, pero misma dirección/fuerza |

**Los seis regímenes de averías/tiempos estocásticos caen EXACTAMENTE dentro del rango
que predijo el tutor, con significancia muy superior al umbral pedido** (p entre 1e-11 y
1e-13, muy por debajo de 3e-4). Esto es una confirmación fuerte, medida, del artefacto
mecánico: el RI de una PDR está casi perfectamente anti-correlacionado con qué tan malo
es su plan nominal, en los seis regímenes donde las 9 PDR compiten en igualdad de
condiciones. Bajo llegadas (solo 4 PDR + RL, n=5), la correlación sigue siendo fuerte y
en la misma dirección, pero se reporta aparte (rango numérico del tutor no calibrado
para n=5) — consistente con (c) más abajo.

Tabla completa: [`../results/tablas/Tabla_auditoria_ri_artefacto.csv`](../results/tablas/Tabla_auditoria_ri_artefacto.csv).

Lo que **también** se verificó por lectura directa, como evidencia consistente con la
hipótesis del tutor (ver (b)): la PDR con el RPD nominal más alto del conjunto (`cr`,
210.9%–229.3%) es también la que tiene el RI más negativo en los seis regímenes
(-0.40 a -0.59), mientras que la PDR con mejor RPD nominal del conjunto (`fifo`, la
mejor de las 9, ver (d)) tiene RI positivo en todos los regímenes medidos. Esta dirección
del efecto es la misma que predice el rho negativo esperado; el número exacto queda
pendiente de la primera corrida del script.

---

## (b) El comparador de H3 es un artefacto — traza mecánica de código

**Afirmación del tutor:** la PDR de menor RI es `cr` (averías, tiempos estocásticos) o
`lwkr` (llegadas), con RI negativo entre -0.40 y -0.59.

**Verificado por lectura de CSV existentes (no por ejecución de pipeline):**

| Régimen | `cr` ft06 | `cr` la01 | `lwkr` ft06 | `lwkr` la01 |
|---|---|---|---|---|
| E4 A=0.80 | -0.3966 | -0.4119 | -0.3296 | -0.3434 |
| E4 A=0.90 | -0.4495 | -0.5280 | -0.3928 | -0.4670 |
| E4 A=0.95 | -0.5287 | -0.5526 | -0.4588 | -0.4812 |
| E5 σ=0.10 | -0.5916 | -0.5922 | -0.5161 | -0.5252 |
| E5 σ=0.25 | -0.5839 | -0.5807 | -0.5011 | -0.5162 |
| E5 σ=0.50 | -0.5521 | -0.5679 | -0.4554 | -0.5066 |

**Confirmado para `cr` bajo averías/tiempos estocásticos** — RI en el rango afirmado
(-0.40 a -0.59, con un valor levemente fuera del extremo alto en E4 A=0.80).

**Refutado para `lwkr` bajo llegadas.** Bajo E6 (arrivals), `lwkr` es la mejor de las 4
PDR evaluadas (ver (c)) pero su RI es **positivo**, no negativo: ft06 u=0.70/0.85/0.95 →
+0.3906/+0.4055/+0.3744; la01 → +0.1757/+0.1620/+0.1437. Ninguna PDR bajo llegadas cae en
el rango -0.40/-0.59 (la más cercana a cero es `lor` en la01/u=0.95, con -0.0207). El
mecanismo de RI negativo descrito abajo es específico de `cr` bajo averías/tiempos
estocásticos, no un patrón general de "la peor regla nominal siempre tiene RI negativo
bajo cualquier eje de disrupción" — bajo llegadas el efecto no aparece con la magnitud
afirmada. **Esta es una corrección al enunciado del tutor, no solo una confirmación.**

**RPD nominal real** (E1, seed 0): `cr` ft06 = 210.909% (tutor: ~220%), `cr` la01 =
229.279%; `lwkr` ft06 = 180.0% (tutor: ~187%), `lwkr` la01 = 194.895%. Los valores reales
son ligeramente inferiores a los citados por el tutor pero confirman la misma conclusión
cualitativa: `cr` y `lwkr` son, respectivamente, la peor y la tercera peor regla nominal
del conjunto de 9.

**Makespan de `cr`, ft06, A=0.90 — confirmado exactamente:** nominal = 171.0
(`e1_seed0_baselines.csv`), perturbado (n=30, `e4_breakdowns_avail0.90_with_rl.csv`):
media = 94.141, mediana = 87.0, rango [59, 182.95]. Coincide con la cifra "171→94" citada
por el tutor.

### Traza mecánica exacta (lectura de código, sin ejecución)

1. **Las fechas de entrega (`due_dates`) se sintetizan una sola vez, a partir de
   duraciones nominales, y nunca se recalculan bajo disrupción.**
   `baselines/due_dates.py::ensure_due_dates` cachea `env.due_dates = release_times +
   tightness * job_totals`, donde `job_totals` viene de las duraciones base de la
   instancia (no de las duraciones efectivas post-avería/post-muestreo estocástico). Esta
   cota queda fija durante todo el episodio, sin importar cuántas averías ocurran.

2. **`cr_rule` (`baselines/dispatching_rules.py`) recalcula el ratio crítico en cada
   época de decisión con `t_now = max(job_ready_time[job], machine_free_time[machine])`.**
   Bajo un breakdown, `machine_free_time`/el estado de la máquina se ve desplazado por
   `machine_repair_complete_time` (fijado en
   `DynamicJSSPEnv._on_breakdown`, `envs/dynamic_jssp_env.py`, cuando la máquina pasa a
   estado `BROKEN` y se programa el evento `REPAIR_COMPLETE`). El ratio
   `(due_date - t_now) / remaining_work` se reevalúa por época con `t_now` desplazado por
   la avería, pero `due_date` y `remaining_work` (basado en duraciones nominales vía
   `remaining_work_per_op`) permanecen fijos.

3. **Consecuencia mecánica:** el desplazamiento de `t_now` reordena qué operación parece
   "más urgente" en cada época, de forma **desacoplada de la calidad real del despacho**
   — no es una corrección hacia el óptimo, es ruido de reordenación inducido por el
   evento. Como el despacho nominal de `cr` ya es muy malo (RPD nominal 210.9%–229.3%,
   la peor o casi peor regla del conjunto), hay mucho más margen para que una
   reordenación aleatoria mejore accidentalmente el makespan que para que empeore un
   despacho ya casi óptimo (caso de `fifo`, con RPD nominal bajo). Este es un efecto de
   **regresión a la media / techo de mejora** genérico de la fórmula
   `RI = (E[C_max^pert] - C_max^nom]) / C_max^nom` tal como está definida hoy — no
   normaliza por qué tan lejos del óptimo está el plan nominal, así que un RI negativo no
   es evidencia de robustez, es un artefacto de que el nominal tenía mucho espacio para
   "caer hacia arriba". `lwkr_rule` bajo `arrivals` es mecánicamente análogo (usa
   `remaining_work_per_op()`, que se recalcula incluyendo jobs recién llegados en cada
   época, reordenando el ranking LWKR de forma no correlacionada con la calidad del
   makespan final) pero, medido, **no** produce RI negativo en este banco de datos — ver
   la refutación arriba.

4. Esta traza está implementada como corrección de diseño en **T8.3** (Fase 8): una
   advertencia explícita cuando `RI < 0`, en vez de dejar que el número se reporte sin
   contexto.

---

## (c) Alcance real de E4/E5/E6

**Confirmado exactamente por lectura de los CSV crudos**
(`experiments/results/week24/e4e5e6/*_with_rl.csv`):

- Instancias: exactamente `{"ft06", "la01"}` en los 9 archivos de régimen.
- Semillas: exactamente 30 por instancia (seed 0–29) en cada régimen.
- E4 (breakdowns) y E5 (stochastic_times): 600 filas por archivo, 9 métodos PDR
  (`cr, edd, fifo, lor, lpt, lwkr, mor, mwkr, spt`) + `rl`.
- E6 (arrivals): 300 filas por archivo, **solo 4 métodos PDR**
  (`fifo, lor, lwkr, mor`) + `rl` — confirma exactamente la restricción a las PDR
  arrival-safe (`ARRIVAL_UNSAFE_PDR_NAMES = {spt, lpt, mwkr, edd, cr}` en
  `run_evaluation.py`).
- Pares por régimen: 2 instancias × 30 semillas = 60 — confirmado en los 9 regímenes.

---

## (d) Procedencia del Estudio 1

**Búsqueda exhaustiva realizada** sobre todo `.md` y `.csv` del repositorio (dentro y
fuera de `code/`) de los términos "Estudio 1", "estudio1", "study1", "47520",
y de datasets con la forma 16 instancias × 9 regímenes × 11 métodos × 30 escenarios.

**Resultado: los datos crudos del "Estudio 1" (16×9×11×30 = 47,520 corridas) NO existen
en este repositorio, bajo ningún nombre reconocible.** La única coincidencia textual de
"Estudio 1"/"47520" en todo el repositorio es el propio archivo que contiene el documento
de recomendaciones del tutor — cero
menciones en `../../thesis/Thesis.md`, `../../thesis/Thesis.md`, `../../thesis/Thesis.md`, o en cualquier
CSV bajo `results/` o `code/experiments/results/`.

Los dos datasets existentes más cercanos por forma, y por qué **no** son el Estudio 1:

- `e1_seed0_baselines.csv`: 43 instancias × 9 PDR + GA + CP-SAT, **1 sola semilla**
  (seed 0), **sin** dimensión de régimen de disrupción (todo bajo `disruption="none"`).
  Cubre la dimensión de instancias/métodos pero no la de régimen ni la de réplicas.
- `baselines_week12.csv` (legado, `experiments/eval_baselines.py`): 13 instancias × 11
  métodos = 143 filas, **CP-SAT alcanza `OPTIMAL` en 13/13 instancias** (no en "7 de 16"
  como afirma el tutor en T7.1, y ese dato tampoco es reconstruible desde ningún archivo
  existente) — este dataset tampoco tiene dimensión de régimen ni de 30 escenarios.

**Consecuencia para T7.1:** la cifra "el óptimo certificado se alcanza en 7 de 16
instancias del Estudio 1, con desviación nominal media de 3.08%" **no se puede verificar
contra ningún dato real del repositorio** y queda marcada como
`[PENDIENTE-CORRIDA: no hay fuente de datos identificada]` en toda la documentación
nueva que la referencie. No se inventa ningún valor sustituto.

**Propuesta para cerrar la brecha (dos caminos, no excluyentes):**

1. **Regenerar el Estudio 1 con este código.** El repositorio ya tiene todo lo necesario
   para producir una malla de esa forma: `run_evaluation.py --instances <16 nombres>
   --methods pdr ga cpsat --disruption <9 regímenes vía combined con severidades, ver
   T7.2> --seeds 0..29`. El comando exacto, con estimación de tiempo, se deja preparado
   en `RUN_PLAN.md` §8 (T8.1 ya amplía la malla de disrupción a 8+ instancias con las 9
   PDR completas en los 9 regímenes — el Estudio 1 sería un superconjunto natural de esa
   ampliación, agregando GA/CP-SAT/RL para llegar a 11 métodos y 16 instancias).
2. **Corregir la Declaración de disponibilidad de datos del manuscrito** si no se va a
   regenerar: declarar explícitamente que el Estudio 1 tal como está descrito en el
   manuscrito no tiene datos crudos depositados en este repositorio, y que las cifras
   "7/16" y "3.08%" deben eliminarse o sustituirse por las del Estudio E1 real (43
   instancias, 1 semilla, sin régimen) con su alcance correctamente acotado.

**Esta decisión requiere al usuario** — no es una decisión que el código pueda tomar por
sí mismo (Regla Cero / instrucción del tutor "propone cómo cerrar la brecha"). Se deja
registrada como pregunta abierta en `RUN_PLAN.md` §"Supuestos a verificar en la primera
corrida".

---

## Tabla de afirmaciones del manuscrito afectadas

| Hallazgo | Afirmación concreta del manuscrito afectada | Acción |
|---|---|---|
| (a) RI artefacto | Cualquier pasaje que interprete un RI negativo de una PDR como "esa regla es robusta bajo disrupción" | T8.3 añade advertencia explícita cuando RI<0; Fase 1 (holgura) da un mecanismo alternativo medible en vez de solo interpretativo |
| (b) `cr`/`lwkr` RI negativo | El comparador de H3 (RI de la mejor PDR) puede estar dominado por una regla con plan nominal pésimo, no por una regla genuinamente robusta — el tutor tenía razón para `cr`/averías-tiempos, no para `lwkr`/llegadas | Fase 4 (T4.1) separa `pdr_global` (fifo, la única desplegable a priori) de `oracle_best_per_instance`, para que H3 no se lea como "la mejor PDR ad hoc" sin decir cuál ni por qué |
| (c) Alcance E4/E5/E6 | Cualquier pasaje que describa la malla de disrupción como cubriendo "las 9 PDR" bajo llegadas, o más de 2 instancias | T8.1 amplía la malla a 8+ instancias y 9 PDR completas en los 9 regímenes; hasta esa corrida, el texto actual debe decir explícitamente "2 instancias, 4 PDR bajo llegadas" |
| (d) Estudio 1 sin datos crudos | "el óptimo certificado se alcanza en 7 de 16 instancias del Estudio 1, con desviación nominal media de 3.08%" (T7.1) y cualquier cifra derivada del Estudio 1 en el manuscrito | Requiere decisión del usuario (regenerar vs. corregir Declaración de disponibilidad de datos); no se sustituye por ningún valor inventado |
