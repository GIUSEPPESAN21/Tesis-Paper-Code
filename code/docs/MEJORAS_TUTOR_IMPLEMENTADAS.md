# MEJORAS_TUTOR_IMPLEMENTADAS.md

Estado de las 10 recomendaciones del tutor tras esta sesión. **Nada se ejecutó**
(Regla Cero) — todo lo listado como "implementado" es código/tests/documentación
escritos y listos para correr, no resultados. Ver `RUN_PLAN.md` para los comandos
exactos y `AUDIT_PREFLIGHT.md` para los hallazgos de la auditoría de partida.

| # | Recomendación | Qué se implementó | Archivos tocados | Comando para reproducir | Salida esperada | Qué afirmación del manuscrito cambiará y cómo |
|---|---|---|---|---|---|---|
| 1/10 | Semántica correcta CP-SAT (solver/status/certificación) y regímenes de disrupción | `mip_gap`/`certified_optimal` distintos de `is_optimal`; renombrado a "nueve regímenes de tres clases de disrupción"; `--combined-severity {low,medium,high}` para el régimen `combined` | `baselines/cpsat.py`, `baselines/interface.py`, `run_evaluation.py`, `experiments/stats/io.py` | `python run_evaluation.py --methods cpsat --disruption combined --combined-severity high` | Columna `mip_gap` nueva en el CSV; columna derivada `certified_optimal` en el pipeline de stats | Ninguna tabla/figura puede decir "exacto-óptimo" para una fila con `status=FEASIBLE`; la cifra "7/16 instancias, 3.08% RPD" (Estudio 1) queda `[PENDIENTE-CORRIDA]` — no verificable, ver Fase 0(d) |
| 2 | Medir la holgura explícitamente (CPM) | `schedule_slack_metrics` (ES/LS, 6 métricas), `run_slack_study.py` (CP-SAT/GA/9 PDR sobre 43 instancias), `slack_vs_robustness.py` (Spearman/Pearson vs. RI y RPD realizada), `Fig_holgura_vs_RI` | `experiments/slack_analysis.py`, `experiments/run_slack_study.py`, `experiments/slack_vs_robustness.py`, `scripts/generate_thesis_figures.py` | Ver RUN_PLAN.md Fase 1 | `results/datos_crudos/slack/slack_metrics.csv`, `Tabla_holgura_vs_robustez.{csv,md,tex}`, `Fig_holgura_vs_RI.{png,pdf}` | El mecanismo "más compacto = más frágil" pasa de interpretación a medición cuantitativa (signo y magnitud reales, sin forzar el resultado esperado) |
| 3 | Semillas múltiples de entrenamiento | `--seeds`/`--n-workers` en `train_dynamic.py`/`train_hetero.py` (byte-idéntico en el caso de una sola semilla); `experiments/stats/seeds.py` (media/DE/IC95-t/rango); `Tabla_H4_multisemilla_posthoc` (post-hoc, etiquetada explícitamente) | `experiments/train_dynamic.py`, `experiments/train_hetero.py`, `experiments/stats/seeds.py`, `experiments/stats/posthoc_multiseed.py` | Ver RUN_PLAN.md Fase 2 | 5 `summary_..._seed{N}.json` por condición; `Tabla_H4_multisemilla_posthoc.{md,tex}` | El veredicto pre-registrado de H4 (REJECT, `../../thesis/RESULTS_REPORT.md` §6.4) **no cambia**; la tabla nueva solo evalúa su estabilidad entre semillas, marcada post-hoc en su propio encabezado |
| 4 | (numeración del tutor salta de 3 a 5) | — | — | — | — | — |
| 5 | Ablación 2×2 entorno × recompensa (celda C, la que falta) | `envs/dynamic_jssp_env.py` no se modificó (kappa=0 ya era no-op exacto, verificado por lectura + `TestKappaReward` ya existente); `--training-regime {domain_randomized,none}` y `--nominal-episode-fraction` (T3.2, corrige el defecto de aleatorización diagnosticado por el propio manuscrito) en `train_dynamic.py`; `run_2x2_ablation.py`; `experiments/stats/ablation_2x2.py` (Friedman + efectos factoriales); plantilla `EXPERIMENT_LOG_ABLACION_2x2.md` | `experiments/train_dynamic.py`, `experiments/run_2x2_ablation.py`, `experiments/stats/ablation_2x2.py`, `EXPERIMENT_LOG_ABLACION_2x2.md` | Ver RUN_PLAN.md Fase 3 | Celdas B y C entrenadas; `Tabla_ablacion_2x2.{csv,md,tex}` con efecto de entorno, efecto de recompensa, e interacción | Permite atribuir el colapso de RPD nominal (+106.0pp/+46.6pp, H4) al entorno de entrenamiento, a la recompensa, o a su interacción — hoy confundidos en una sola cifra |
| 6/7 | Comparadores PDR bien separados + doble lectura de H2 | `pdr_global`/`pdr_por_familia`/`oraculo_best_per_instance` siempre separados, oráculo siempre etiquetado explícitamente; H2 con lectura de rango + lectura competitiva absoluta en la misma tabla | `experiments/stats/comparators.py`, `experiments/stats/report.py` | Automático dentro de `python -m experiments.stats.cli` | `comparadores_pdr_e1.md` (`Tabla_6.2.2`), `e3_h2_dual_reading.md` (`Tabla_6.3.3`) | H1 rechazado y H2 aceptado ya no se leen contra un "mejor PDR" ambiguo — el manuscrito debe citar explícitamente cuál de los tres comparadores usa en cada afirmación |
| 8 | Costo computacional homogéneo | `experiments/cost_accounting.py` (entrenamiento, inferencia, amortizado, CP-SAT, GA, PDR, siempre filas separadas); `Fig_costo_amortizado`; `e8_cpu_timing.py` ampliado de 4 a 43 instancias, documentado qué columna alimenta qué afirmación | `experiments/cost_accounting.py`, `scripts/generate_thesis_figures.py`, `experiments/results/week24/e8_cpu_timing.py`, `scripts/export_thesis_tables.py` | Ver RUN_PLAN.md Fase 5 | `Tabla_6.5.2_costo_computacional`, `Fig_6.5.2_costo_amortizado` | Las cifras de latencia/costo de H5/Cap. 6.5 quedan respaldadas por las 43 instancias, no un subconjunto de 4 |
| 9 | Estrategia de reprogramación más fuerte | `envs/rescheduling.py` (capa nueva, NO modifica `dynamic_jssp_env.py`): `periodic_reopt`, `event_triggered_reopt`, subproblema residual vía CP-SAT propio o búsqueda de prioridades (sustituto simplificado de GA, documentado); `--execution-policy {right_shift,periodic,event_triggered}` en `run_evaluation.py`, con costo de reoptimización registrado; `Tabla_politicas_reprogramacion` con diagnóstico explícito optimalidad-vs-secuencia | `envs/rescheduling.py`, `run_evaluation.py`, `experiments/stats/rescheduling_policies.py` | Ver RUN_PLAN.md Fase 6 | Nuevas columnas `execution_policy`/`n_reoptimizations`/`reoptimization_wall_clock_s` en el CSV; tabla de diagnóstico | Permite separar si la pérdida de CP-SAT/GA bajo disrupción viene de la optimalidad del solver o del compromiso con una secuencia fija — hoy indistinguibles |
| 10 | (fusionada con 1, ver fila 1/10) | — | — | — | — | — |

**Ampliación de alcance (T8.1, fuera de la numeración 1–10 pero pedida explícitamente):**
las PDR `spt`/`lpt`/`mwkr` pasan de arrival-unsafe a arrival-safe (`env.duration_per_op()`/
`env.machine_per_op()`/`env.remaining_work_per_op()`, nuevos métodos en
`envs/jssp_env.py`/`envs/dynamic_jssp_env.py`, bit-idénticos al comportamiento previo
en el caso estático) — de 4 a 7 PDR seguras bajo llegadas (`edd`/`cr` siguen bloqueadas,
la síntesis de fechas de entrega para trabajos llegados dinámicamente sigue sin
implementarse, gap disclosed, no resuelto). `results/tablas/Tabla_calidad_realizada_disrupcion`
(T8.2/T8.4) y la advertencia de RI negativo con registro en el manifiesto (T8.3)
también están implementadas.

---

## Resumen honesto (T9.4)

**Completamente implementadas en código** (con tests escritos, sin ejecutar): Fase 0
(auditoría por lectura directa + script de correlación pendiente de correr), Fase 7
(mip_gap/certified_optimal, multiple_disruption_classes, combined-severity), Fase 4
(los tres comparadores PDR separados + doble lectura de H2), Fase 5 (costo
computacional homogéneo + figura de amortización + ampliación de e8_cpu_timing a 43
instancias), Fase 1 (métricas de holgura CPM + script de generación + script de
correlación + figura), Fase 8 T8.2/T8.3/T8.4 (calidad realizada, advertencia RI
negativo, tabla comparativa) y la ampliación de PDR arrival-safe de spt/lpt/mwkr, Fase
2 (semillas múltiples con `--seeds`/`--n-workers`, agregación estadística, tabla
post-hoc), y Fase 3 (ablación 2×2 completa: la celda C que faltaba, la corrección de
aleatorización de dominio T3.2, el análisis factorial).

**Parcialmente implementadas, y por qué:** Fase 8 T8.1 (ampliación de la malla a 8+
instancias) — el código y el comando ya soportan esto sin cambios adicionales, pero
bajo llegadas solo 7 de 9 PDR quedaron arrival-safe (edd/cr siguen bloqueadas: arreglar
eso exige rastrear el tiempo real de llegada de cada job y extender
`ensure_due_dates` para recalcular fechas de entrega dinámicamente — un cambio de
estado nuevo en `DynamicJSSPEnv`, deliberadamente no intentado junto con el resto
porque es una pieza de riesgo comparable a la Fase 6 y esta sesión ya tenía una pieza
de máximo riesgo que cubrir). Fase 6 (T6.1–T6.3) — está completamente escrita
(`envs/rescheduling.py`, la extensión de `run_evaluation.py`, la tabla de
diagnóstico) pero es la pieza de código más compleja de toda la sesión y NO pudo
verificarse por ejecución (Regla Cero); además, el "GA" del subproblema residual es
una búsqueda de prioridades aleatorias con reinicios, no un GA poblacional con
cruce/mutación — un sustituto simplificado, documentado como tal, suficiente para
ejercitar la capa de políticas de ejecución pero no equivalente en calidad a
`baselines/ga.py`'s GA completo.

**No se pudieron implementar sin decisiones del usuario:** el Estudio 1 (16
instancias × 9 regímenes × 11 métodos × 30 escenarios, T7.1) no tiene datos crudos en
este repositorio bajo ningún nombre reconocible (`AUDIT_PREFLIGHT.md` §(d)) — regenerarlo
o corregir la Declaración de disponibilidad de datos del manuscrito es una decisión
que solo el usuario puede tomar, y no se sustituyó por ningún valor inventado en
ningún lugar de esta sesión.
