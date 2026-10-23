# RUN_PLAN.md — Plan de ejecución de las mejoras del tutor

**Nada en este documento se ejecutó.** Todo lo que sigue es código escrito y listo
para correr (Regla Cero del documento de recomendaciones del tutor). Cada comando indica
qué script lo produce, qué archivo(s) genera, sus dependencias, y una estimación de
tiempo de reloj basada en la referencia medida del usuario: 400k pasos en ft06 ≈ 2213 s
(DE 143 s, 5 semillas); 1M pasos FJSP en mk01 ≈ 8713 s.

Ejecutar en el orden de fases: 0, 7, 4, 5, 1, 8, 2, 3, 6. La Fase 6 es la de mayor
riesgo de regresión (toca ejecución en línea) y debe correrse — y revisarse — de
último, con el mayor escrutinio.

---

## Fase 0 — Auditoría (solo lectura, ya completada en esta sesión)

`AUDIT_PREFLIGHT.md` ya está escrito con los hallazgos de lectura directa de CSV
existentes. Lo único pendiente de ejecución real:

```
python -m experiments.audit.ri_artifact_check \
    --e1-csv "code/experiments/results/week24/e1_seed0_baselines.csv" \
    --e4e5e6-dir "code/experiments/results/week24/e4e5e6" \
    --out results/tablas/Tabla_auditoria_ri_artefacto.csv
```
**Produce:** `results/tablas/Tabla_auditoria_ri_artefacto.csv`.
**Tiempo estimado:** < 5 s (solo lee CSV existentes y calcula Spearman).
**Dependencias:** ninguna, los CSV ya existen.

---

## Fase 7 — Semántica correcta (T7.1, T7.2)

Sin comandos nuevos que correr: T7.1 (mip_gap/certified_optimal) y T7.2
(multiple_disruption_classes, `--combined-severity`) son cambios de código en
`baselines/cpsat.py`, `baselines/interface.py`, `run_evaluation.py`,
`experiments/stats/io.py`. Se activan automáticamente en cualquier corrida nueva de
`run_evaluation.py` (columna `mip_gap` nueva en el CSV) y en `experiments/stats/cli.py`
(columna derivada `certified_optimal`). Verificar con:

```
python -m pytest tests/test_cpsat.py::TestComputeMipGap tests/test_run_evaluation.py::TestMipGapColumn tests/test_stats_io.py -v
```
**Tiempo estimado:** < 10 s (son tests unitarios puros, sin CP-SAT real salvo los ya existentes).

---

## Fase 4 — Comparadores PDR (T4.1, T4.2)

Se ejecutan automáticamente dentro de `python -m experiments.stats.cli` (ya wireado en
`report.py::generate_chapter6_artifacts`) una vez exista `experiments/results/week24/e1_seed0_baselines.csv`
(ya existe) y, para T4.2, `experiments/results/week24/e3_generalization_la21_la40.csv`
(ya existe). No requiere un comando separado — correr el pipeline de stats de la Fase
cierre ya las produce (`comparadores_pdr_e1.md`, `e3_h2_dual_reading.md`).

**Verificación aislada (opcional, sin correr todo el pipeline):**
```
python -m pytest tests/test_stats_comparators.py tests/test_stats_report_comparators_integration.py -v
```
**Tiempo estimado:** < 10 s.

---

## Fase 5 — Costo computacional (T5.1–T5.3)

### T5.3 — ampliar e8_cpu_timing.csv a las 43 instancias
```
python "code/experiments/results/week24/e8_cpu_timing.py"
```
**Produce:** `experiments/results/week24/e8_cpu_timing.csv` (43 filas, no 4).
**Tiempo estimado:** 43 instancias × (5 réplicas RL multihilo + 5 monohilo + 1 CP-SAT
30s máx) ≈ 43 × (10×~0.05s + 30s peor caso) — dominado por CP-SAT en las instancias
grandes (la31–la40): estimar **~15–25 min** en total (la mayoría de instancias
resuelven CP-SAT en <1s; solo las la31+ pueden acercarse al límite de 30s).

### T5.1 — Tabla_costo_computacional
```
python -m experiments.cost_accounting \
    --e1-csv "code/experiments/results/week24/e1_seed0_baselines.csv" \
    --e8-timing-csv "code/experiments/results/week24/e8_cpu_timing.csv" \
    --summary-glob "code/experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json" \
    --out "code/experiments/results/stats/tables/costo_computacional.md"
python "code/scripts/export_thesis_tables.py"
```
**Depende de:** T5.3 (e8_cpu_timing.csv completo) y de que existan
`summary_dynamic_kappa0.0_D1_ft06_seed*.json` (Fase 2, T2.1 — si aún no hay corridas
multi-semilla de kappa=0, el `--summary-glob` no encuentra archivos y la fila de
entrenamiento queda `[PENDIENTE-CORRIDA]`, no falla).
**Tiempo estimado:** < 5 s (solo agrega datos ya existentes).

### T5.2 — Fig_costo_amortizado
```
python "code/scripts/generate_thesis_figures.py" \
    --e8-timing-csv "code/experiments/results/week24/e8_cpu_timing.csv" \
    --training-summary-glob "code/experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json"
```
**Depende de:** lo mismo que T5.1. **Tiempo estimado:** < 5 s.

---

## Fase 1 — Holgura vs. robustez (T1.1–T1.4)

### T1.2 — generar programas nominales + métricas de holgura
```
python -m experiments.run_slack_study \
    --instances ft06 ft10 ft20 $(python -c "print(' '.join(f'la{n:02d}' for n in range(1,41)))") \
    --methods cpsat ga pdr \
    --seed 0 \
    --out results/datos_crudos/slack/slack_metrics.csv
```
**Produce:** `results/datos_crudos/slack/slack_metrics.csv` (43 instancias × 11 métodos
= hasta 473 filas, menos las que fallen).
**Tiempo estimado:** dominado por CP-SAT (hasta 60s/instancia en las 43 instancias,
peor caso ~43 min) + GA (presupuesto fijo por instancia, típicamente unos segundos
cada una, ~5–10 min total) + 9 PDR (segundos en total). **Estimar 45–60 min en el peor
caso**, mucho menos si la mayoría de instancias resuelven CP-SAT rápido (como ya
ocurrió en `baselines_week12.csv`).

### T1.3 — correlación holgura vs. RI/RPD realizada
```
python -m experiments.slack_vs_robustness \
    --slack-csv results/datos_crudos/slack/slack_metrics.csv \
    --ri-csv <CSV_long_format_con_columnas_instance,method,regime,ri> \
    --out results/tablas/Tabla_holgura_vs_robustez.csv
```
**Supuesto a verificar en la primera corrida:** este script espera un CSV *ya
agregado* a formato largo `instance,method,regime,ri` — hoy ese CSV no existe como
archivo independiente (el RI vive calculado inline dentro de `report.py`). Antes de
correr esto, exportar esa vista con un script corto de una línea desde
`report.py::add_row_ri` sobre los CSV de E4/E5/E6, o extender `report.py` para
volcarla — decisión del usuario, no tomada aquí.
**Tiempo estimado:** < 5 s una vez el CSV de entrada exista.

### T1.4 — Fig_holgura_vs_RI
Se genera automáticamente dentro de `generate_thesis_figures.py --slack-csv
results/datos_crudos/slack/slack_metrics.csv` (ya wireado). **Tiempo estimado:** < 5 s.

---

## Fase 8 — Ampliación de alcance (T8.1–T8.4)

### T8.1 — malla de disrupción ampliada (8+ instancias, 9 PDR completas)
```
python "code/run_evaluation.py" \
    --instances ft06 ft10 ft20 la01 la11 la21 la31 la36 \
    --methods pdr \
    --disruption breakdowns --availability 0.95 --seeds $(python -c "print(' '.join(str(i) for i in range(30)))") \
    --out "code/experiments/results/week24_wide/e4_breakdowns_A0.95_wide.csv"
# repetir con --availability 0.90/0.80, --disruption stochastic_times --proc-sigma 0.10/0.25/0.50,
# y --disruption arrivals --utilization 0.70/0.85/0.95 (ahora con las 9 PDR completas
# gracias a T8.1's arrival-safety fix, no solo 4)
```
**Supuesto a verificar en la primera corrida:** las 8 instancias elegidas
(ft06/ft10/ft20/la01/la11/la21/la31/la36) cubren el rango de tamaños (36 a 300+ ops);
el usuario puede sustituir por su propia selección — no es una elección única correcta.
**Tiempo estimado por régimen:** 8 instancias × 9 PDR × 30 semillas = 2160 filas,
cada una un rollout PDR (rápido, milisegundos–segundos) → **~10–20 min por régimen**,
× 9 regímenes ≈ **1.5–3 horas en total** si se corre todo secuencialmente. Paralelizar
por régimen con `--num-workers` no aplica aquí (es específico de CP-SAT); considerar
lanzar los 9 regímenes en procesos separados si el usuario quiere acelerar.

### T8.2/T8.4 — calidad realizada + comparación contra los tres comparadores
```
python -m experiments.stats.realized_quality \
    --results-dir "code/experiments/results" \
    --out results/tablas/Tabla_calidad_realizada_disrupcion
```
**Depende de:** E1 (ya existe) + E4/E5/E6 (ya existen para ft06/la01; T8.1 los amplía).
**Tiempo estimado:** < 10 s.

### T8.3 — advertencia RI negativo
Se activa automáticamente al correr `python -m experiments.stats.cli` (wireado en
`report.py`). No requiere comando separado. Revisar
`experiments/results/stats/tables/ri_negative_warnings.json` si se genera.

---

## Fase 2 — Semillas múltiples de entrenamiento (T2.1–T2.4)

**Orden de prioridad si el presupuesto no alcanza (instrucción explícita del
usuario): kappa=0 primero, luego kappa=1.0, luego kappa=0.5.**

```
# kappa=0 (prioridad 1)
python "code/experiments/train_dynamic.py" \
    --instance ft06 --kappa 0.0 --stage D1 \
    --init-checkpoint "code/experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip" \
    --timesteps 100000 --seeds 0 1 2 3 4 --nominal-episode-fraction 0.2

# kappa=1.0 (prioridad 2)
python "code/experiments/train_dynamic.py" \
    --instance ft06 --kappa 1.0 --stage D1 \
    --init-checkpoint "code/experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip" \
    --timesteps 100000 --seeds 0 1 2 3 4 --nominal-episode-fraction 0.2

# kappa=0.5 (prioridad 3)
python "code/experiments/train_dynamic.py" \
    --instance ft06 --kappa 0.5 --stage D1 \
    --init-checkpoint "code/experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip" \
    --timesteps 100000 --seeds 0 1 2 3 4 --nominal-episode-fraction 0.2
```
**Produce:** 5 × `summary_dynamic_kappa{k}_D1_ft06_seed{0..4}.json` por valor de kappa.
**Tiempo estimado por valor de kappa (5 semillas, secuencial, `--n-workers 1`):**
usando la referencia del usuario (400k pasos ft06 ≈ 2213s) escalado a 100k pasos
(~1/4): **~550s/semilla × 5 = ~46 min por valor de kappa**, **~2.3 horas para los 3
valores** si se corre secuencial. Con `--n-workers 5` (una subprocess por semilla, si
el hardware tiene núcleos suficientes): **~9–10 min por valor de kappa** (limitado por
contención de CPU real, no división exacta).

**Supuesto a verificar en la primera corrida:** `--nominal-episode-fraction 0.2` (T3.2)
cambia el comportamiento de entrenamiento respecto a las corridas E7 ya publicadas
(que usaban 0% episodios nominales) — estas NUEVAS corridas de kappa=0.5/1.0 NO
reproducirán bit-a-bit los resultados de `../../thesis/RESULTS_REPORT.md` §6.4. Si se quiere
reproducir exactamente el E7 ya publicado, añadir `--nominal-episode-fraction 0.0`.

### T2.4 — Tabla_H4_multisemilla_posthoc
```
python -m experiments.stats.posthoc_multiseed \
    --kappa-glob 0.0 "code/experiments/results/dynamic/summary_dynamic_kappa0.0_D1_ft06_seed*.json" \
    --kappa-glob 0.5 "code/experiments/results/dynamic/summary_dynamic_kappa0.5_D1_ft06_seed*.json" \
    --kappa-glob 1.0 "code/experiments/results/dynamic/summary_dynamic_kappa1.0_D1_ft06_seed*.json" \
    --out results/tablas/Tabla_H4_multisemilla_posthoc
```
**Depende de:** las corridas de arriba. **Tiempo estimado:** < 5 s.

---

## Fase 3 — Ablación 2×2 (T3.1–T3.3)

```
python -m experiments.run_2x2_ablation \
    --instance ft06 \
    --static-checkpoint "code/experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip" \
    --kappa-robust 0.5 \
    --timesteps 100000 --seeds 0 1 2 3 4 --nominal-episode-fraction 0.2 \
    --execute
```
**Produce:** checkpoints + `summary_*.json` para las celdas B y C (A y D reutilizan
checkpoints existentes — si `--kappa-d-checkpoint` no se pasa, D también se reentrena).
**Tiempo estimado:** 2 celdas nuevas (B, C) × 5 semillas × ~550s/semilla (100k pasos)
≈ **~1.5 horas** si D se reutiliza; **~2.3 horas** si D también se reentrena.

```
python -m experiments.stats.ablation_2x2 \
    --cell-csv A "code/experiments/results/ablation_2x2/eval_cellA_none.csv" \
    --cell-csv B "code/experiments/results/ablation_2x2/eval_cellB_none.csv" \
    --cell-csv C "code/experiments/results/ablation_2x2/eval_cellC_none.csv" \
    --cell-csv D "code/experiments/results/ablation_2x2/eval_cellD_none.csv" \
    --out results/tablas/Tabla_ablacion_2x2
```
**Supuesto a verificar en la primera corrida:** `run_2x2_ablation.py --execute` NO
genera automáticamente los CSV `eval_cell{X}_none.csv` — solo entrena. Falta correr
`run_evaluation.py --rl-policy <checkpoint de cada celda>` por separado para cada
celda antes de este paso (el comando exacto está en
`evaluation_command()` de `run_2x2_ablation.py`, no incluido aquí por brevedad).
**Tiempo estimado:** < 10 s una vez los 4 CSV existan.

---

## Fase 6 — Reprogramación más fuerte (T6.1–T6.3)

**Máximo riesgo de regresión — correr primero con instancias pequeñas y revisar
`tests/test_rescheduling.py` completo (incluyendo `-m slow`) antes de confiar en
cualquier número que produzca.**

```
python -m pytest "code/tests/test_rescheduling.py" -v
python -m pytest "code/tests/test_rescheduling.py" -m slow -v
```
**Tiempo estimado:** los tests no-`slow` (backend `ga`/priority-search): ~10–30 s
en total. El test `slow` (CP-SAT, ft06, sin disrupción): ~5–30 s (ft06 resuelve
óptimo rápido).

```
python "code/run_evaluation.py" \
    --instances ft06 la01 --methods ga cpsat \
    --disruption breakdowns --availability 0.90 --seeds 0 1 2 \
    --execution-policy periodic --reopt-period 20 --reopt-time-limit 5 \
    --out "code/experiments/results/reprogramacion/e4_periodic.csv"

python "code/run_evaluation.py" \
    --instances ft06 la01 --methods ga cpsat \
    --disruption breakdowns --availability 0.90 --seeds 0 1 2 \
    --execution-policy event_triggered --reopt-time-limit 5 \
    --out "code/experiments/results/reprogramacion/e4_event_triggered.csv"

python "code/run_evaluation.py" \
    --instances ft06 la01 --methods ga cpsat \
    --disruption breakdowns --availability 0.90 --seeds 0 1 2 \
    --execution-policy right_shift \
    --out "code/experiments/results/reprogramacion/e4_right_shift.csv"
```
**Tiempo estimado:** cada reoptimización cuesta hasta `--reopt-time-limit` segundos
(5s aquí); con ~10-20 reoptimizaciones típicas por episodio bajo A=0.90, 2
instancias × 2 métodos × 3 semillas × ~15 reopt × 5s ≈ **~30 min por política**,
**~1 hora para las tres políticas**. Bajar `--reopt-time-limit` a 1-2s si el
presupuesto es corto (a costa de peor calidad por reoptimización).

Luego:
```python
import pandas as pd
from experiments.stats.rescheduling_policies import compare_execution_policies, diagnose_fragility_source, tabla_politicas_reprogramacion_markdown

df = pd.concat([pd.read_csv(p) for p in [".../e4_right_shift.csv", ".../e4_periodic.csv", ".../e4_event_triggered.csv"]])
comparison_ga = compare_execution_policies(df, method="ga")
comparison_cpsat = compare_execution_policies(df, method="cpsat")
diag_ga = diagnose_fragility_source(comparison_ga)
diag_cpsat = diagnose_fragility_source(comparison_cpsat)
md = tabla_politicas_reprogramacion_markdown(
    {"ga": comparison_ga, "cpsat": comparison_cpsat}, {"ga": diag_ga, "cpsat": diag_cpsat}
)
# escribir md a results/tablas/Tabla_politicas_reprogramacion.md manualmente
# (no se escribió un CLI dedicado para este paso final, dado el tiempo de esta sesión)
```

---

## Supuestos a verificar en la primera corrida (consolidado)

1. **Fase 0(d) — Estudio 1 sin datos crudos.** Decisión pendiente del usuario:
   ¿regenerar con este código (T8.1 lo deja preparado) o corregir la Declaración de
   disponibilidad de datos del manuscrito? Ver `AUDIT_PREFLIGHT.md` §(d).
2. ~~**Fase 0(a) — rho de Spearman del artefacto de RI.**~~ **RESUELTO 2026-08-21**:
   corrido, ver `EXECUTION_LOG_2026-08-21.md` y `AUDIT_PREFLIGHT.md` §(a). Los seis
   regímenes de averías/tiempos estocásticos caen dentro del rango predicho
   (-0.93 a -0.98), p entre 1e-11 y 1e-13.
3. **Fase 1(T1.3) — formato del CSV de entrada.** `slack_vs_robustness.py` espera un
   CSV largo `instance,method,regime,ri` que hoy no existe como archivo independiente
   (el RI vive calculado inline en `report.py`); hay que exportarlo primero.
4. **Fase 2/3 — cambio de comportamiento por defecto.** `--nominal-episode-fraction 0.2`
   (default nuevo, pedido explícitamente por el tutor en T3.2) hace que cualquier
   corrida nueva de `train_dynamic.py` NO reproduzca bit-a-bit las corridas E7 ya
   publicadas salvo que se pase `--nominal-episode-fraction 0.0` explícitamente.
5. **Fase 6 — riesgo de implementación no verificado por ejecución.** `envs/rescheduling.py`
   (T6.1) es la pieza de código más compleja de esta sesión y NO pudo probarse
   ejecutándola (Regla Cero). Antes de confiar en cualquier tabla de la Fase 6, correr
   `tests/test_rescheduling.py` completo primero y tratar cualquier fallo como trabajo
   de depuración esperado, no como señal de que el enfoque general está mal.
6. **Fase 6 — el GA residual es una búsqueda de prioridades aleatorias, no un GA
   poblacional completo.** `solve_residual_priority_search` es un sustituto
   simplificado, documentado como tal en `envs/rescheduling.py` y en
   `MEJORAS_TUTOR_IMPLEMENTADAS.md`. Si el comité pide un GA genético completo con
   cruce/mutación para el subproblema residual, eso es trabajo adicional no cubierto
   aquí.
7. **T7.1 — "7 de 16 instancias, 3.08% RPD nominal" del Estudio 1.** No verificable
   (ver punto 1). Cualquier tabla que cite esta cifra debe marcarla
   `[PENDIENTE-CORRIDA: no hay fuente de datos identificada]` hasta que el Estudio 1
   se regenere o el manuscrito se corrija.
