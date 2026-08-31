# EXECUTION_LOG_2026-08-21.md — Primera ejecución real de las mejoras del tutor

**Contexto:** todo el código de `RUN_PLAN.md`/`MEJORAS_TUTOR_IMPLEMENTADAS.md` se escribió
(2026-08-20) sin ejecutar nada (Regla Cero de esa sesión). El usuario autorizó
explícitamente, en esta sesión, correr algo corto para obtener resultados preliminares
citables en el paper y mostrables al tutor, antes de comprometerse a las corridas largas
de `RUN_PLAN.md`. Esto es exactamente ese primer paso, documentado en detalle.

**Máquina:** la del usuario, Windows, `.venv` ya existente con las dependencias fijadas
(`torch==2.12.1+cpu`, `numpy==2.4.6`, `pandas==2.2.2`, `scipy==1.15.2`,
`pytest==9.1.1`, `ortools==9.15.6755`, `statsmodels==0.14.6` — confirmado por
importación directa antes de correr nada).

---

## 1. Validación del código nuevo (nunca antes ejecutado)

Antes de generar ningún resultado citable, se corrió la batería de tests de TODOS los
módulos nuevos de la sesión de mejoras del tutor (Fases 0–8), sin tocar entrenamiento
real ni las 43 instancias completas — subconjuntos rápidos que ya existían como tests
unitarios/de integración.

**Resultado final: 518 tests pasando, 0 fallando**, tras encontrar y corregir:

### Bugs reales de producción (no de test)

1. **`envs/rescheduling.py` — desempate faltante en la selección de siguiente operación.**
   `periodic_reopt`/`event_triggered_reopt` elegían, entre las operaciones actualmente
   legales, la de menor tiempo planeado (`min(candidates, key=lambda op:
   planned_starts[op])`), sin desempate. Cuando dos operaciones compartían el mismo
   tiempo planeado (frecuente en t=0), Python resolvía el empate por orden de iteración
   de un `set` (esencialmente arbitrario), lo que podía permutar el orden de máquina
   respecto al plan real del solver residual. Corregido añadiendo el mismo desempate
   `(start_time, job, position)` que `baselines.cpsat.dispatch_order_from_starts` ya usa
   en el resto del repositorio. **No cambió el resultado cualitativo** (ver hallazgo
   metodológico abajo), pero es la corrección correcta y ahora documentada.

2. **`tests/test_run_evaluation.py::TestCsvHardening` (pre-existente, no de esta sesión)
   — 3 stubs de `evaluate_one` no aceptaban los nuevos parámetros de T6.2**
   (`execution_policy`, `reopt_period`, `reopt_time_limit_s`). Corregido añadiendo
   `**kwargs` a los stubs (`flaky_evaluate_one`, `always_fails`) y reenviándolos a la
   función real.

### Bugs de test (no de producción)

3. `tests/test_stats_metrics_realized_quality.py` — regex de `pytest.warns` con
   minúscula (`"not interpretable"`) no coincidía con el mensaje real
   (`"NOT interpretable"`, mayúsculas deliberadas en el código de producción). Corregido
   el regex del test, no el mensaje.
4. `tests/test_stats_report_comparators_integration.py` — el fixture sintético de dos
   tests no incluía la columna `ms_per_decision` que `report.py`'s sección E8/H5 ya
   asume presente (correctamente, cuando los datos vienen del pipeline real vía
   `io.load_results`/`add_derived_columns`). Corregido añadiendo la columna al fixture.

### Hallazgo metodológico real, encontrado por ejecución (no un bug)

Al validar `envs/rescheduling.py` con CP-SAT sobre `ft06` sin ninguna disrupción, un
único plan residual óptimo (55, makespan óptimo probado) realiza **60** al reproducirse
contra `DynamicJSSPEnv` (una brecha del ~9%), y solo baja a 59 con reoptimización
frecuente — nunca cierra del todo. **Esto no es un defecto del código nuevo**: es la
misma discrepancia de legalidad "completion-gated vs. dispatch-gated" que
`../../thesis/E7_E8_Plan.md` §2.3 ya documentaba para el adaptador de la política RL ("a few
percent of makespan on ft06/la01 with disruptions off") — `DynamicJSSPEnv.action_masks()`
exige que la máquina de una operación esté libre EN ESE INSTANTE, no solo que su
predecesora de precedencia haya terminado. Documentado en detalle en el docstring de
`envs/rescheduling.py` y en los tests corregidos (`tests/test_rescheduling.py`). Este
hallazgo es evidencia adicional, independiente, de un fenómeno que el propio repositorio
ya había cuantificado por otra vía — refuerza, no contradice, el resto del análisis.

---

## 2. Resultado preliminar citable: confirmación del artefacto del RI (Fase 0a)

**Comando ejecutado:**
```
python -m experiments.audit.ri_artifact_check \
    --e1-csv experiments/results/week24/e1_seed0_baselines.csv \
    --e4e5e6-dir experiments/results/week24/e4e5e6 \
    --out results/tablas/Tabla_auditoria_ri_artefacto.csv \
    --instances ft06 la01
```
**Tiempo de ejecución:** menos de 2 segundos (solo lee CSV ya existentes de `week24`,
sin entrenar ni resolver nada nuevo).

**Resultado:** correlación de Spearman entre RPD nominal y RI medio, por régimen, sobre
las 9 PDR + RL:

| Régimen | rho | p |
|---|---|---|
| E4 averías A=0.95 | -0.9774 | 1.34e-13 |
| E4 averías A=0.90 | -0.9586 | 2.93e-11 |
| E4 averías A=0.80 | -0.9586 | 2.93e-11 |
| E5 tiempos σ=0.10 | -0.9752 | 3.13e-13 |
| E5 tiempos σ=0.25 | -0.9744 | 4.09e-13 |
| E5 tiempos σ=0.50 | -0.9714 | 1.10e-12 |

**Los seis regímenes caen exactamente dentro del rango que predijo el tutor (-0.93 a
-0.98), con p entre 4 y 9 órdenes de magnitud por debajo del umbral pedido (p<3e-4).**
Esto confirma, con datos reales y no solo por lectura de código, que el Índice de
Robustez está casi perfectamente anti-correlacionado con qué tan malo es el plan nominal
de cada regla — el artefacto que el tutor sospechaba es real y medible, no una
posibilidad teórica. Ver `AUDIT_PREFLIGHT.md` Fase 0(a) para el detalle completo y
`results/tablas/Tabla_auditoria_ri_artefacto.csv` para la tabla cruda.

**Cómo usarlo en el paper / para el tutor:** esta es la pieza más fácil de defender de
todo el trabajo de esta sesión, porque no depende de ningún entrenamiento nuevo —
usa exactamente los mismos CSV que ya sustentan `../../thesis/RESULTS_REPORT.md`, solo aplica
una prueba estadística adicional que faltaba. Es evidencia de que el mecanismo de RI
necesita, como mínimo, reportarse junto con la calidad realizada absoluta (Fase 8,
`Tabla_calidad_realizada_disrupcion`) — que sigue pendiente de correr, ver §3.

---

## 3. Qué queda pendiente para la corrida completa

Este fue un chequeo deliberadamente acotado (segundos de cómputo, cero entrenamiento).
`RUN_PLAN.md` sigue siendo la referencia para la corrida completa por fases; nada de lo
de hoy reemplaza esos comandos. Próximos pasos cortos razonables, si se quiere seguir
construyendo evidencia preliminar antes de comprometerse a las corridas largas:

- **Fase 1 (holgura), reducida:** `experiments/run_slack_study.py --instances ft06 la01
  --methods pdr --out results/datos_crudos/slack/slack_metrics.csv` — segundos, sin
  CP-SAT/GA, ya validado por los tests de esta sesión.
- **Fase 4/8 (comparadores, calidad realizada):** ya se ejecutan solas dentro de
  `python -m experiments.stats.cli` una vez que decida correrse el pipeline completo de
  stats sobre los CSV `week24` existentes — no requieren entrenamiento nuevo tampoco.

Las corridas que sí requieren tiempo real (semillas múltiples de entrenamiento, ablación
2×2, malla ampliada de disrupción, políticas de reprogramación en escala) siguen en
`RUN_PLAN.md`, sin tocar hoy.
