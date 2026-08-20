# Capítulo 6 — Resultados (§§6.1–6.4, BORRADOR con datos reales)

**Estado:** Borrador con datos reales de E0–E6 (corte Semana 24, 2026-07-31). Sustituye los marcadores `[TABLA: …]`/`[FIGURA: …]` de `../docs/capitulos/Chapter6_template.md` (Semana 20) por las cifras efectivamente producidas por `run_evaluation.py` sobre los CSV en `experiments/results/week24/`. §§6.5 (tiempos de decisión, H5) y la síntesis H1–H5 completa no se abordan en este borrador — dependen de la remedición limpia de E8 y de H4, ambas en curso per `E7_E8_FinalPlan.md`.

**Nota metodológica sobre las pruebas estadísticas de este borrador.** `../docs/logs/EXPERIMENT_LOG_WEEK24.md` señala explícitamente que la batería de pruebas de significancia (Wilcoxon para H1, Friedman+Nemenyi para H2) fue diferida a "el script de análisis dedicado del próximo mes" (`StatsAnalysis_TechSpec.md`), con la excepción de H3, cuyo Wilcoxon **sí** se ejecutó como parte de la propia regla de decisión congelada (`experiments/results/week24/h3_verdict.csv`, calculado con `scipy` en el entorno real del proyecto). Para que este capítulo pueda citar un veredicto de H1/H2 en vez de dejarlo `[PENDIENTE]`, las prueba de Wilcoxon (H1) y de Friedman+Nemenyi (H2) de este borrador **fueron calculadas aquí mismo**, sobre los CSV reales, con una implementación en Python puro (sin `scipy`, no disponible en este entorno de redacción) del procedimiento exacto fijado en `PreRegistration_FINAL.md` §§2–3: rangos con signo de Wilcoxon con aproximación normal y corrección de continuidad, y estadístico χ² de Friedman con función de supervivencia calculada por expansión de la función gamma incompleta regularizada; los valores críticos de Nemenyi usan la tabla estándar de Demšar (2006). Esta implementación fue verificada contra valores de referencia conocidos (p. ej. χ²(9, α=0.05)=16.919) antes de aplicarse a los datos reales, y los anclajes de `PreRegistration_FINAL.md` §1 (`ft06` mejor PDR=fifo, 21.82 % RPD; `ft10`=30.97 % RPD) se reproducen exactamente a partir del CSV de la Semana 24, confirmando que el barrido de PDR recién ejecutado coincide con el de `baselines_week12.md` tal como exige la regla congelada. **Aun así, estos números deben tratarse como preliminares y re-verificarse con el pipeline `scipy`/`scikit-posthocs` de `StatsAnalysis_TechSpec.md` antes de considerarse definitivos para la defensa** — se reportan aquí con total trazabilidad de método, no como sustituto silencioso de ese pipeline.

---

## 6.1 Validación del entorno de simulación

**Hipótesis que atiende:** ninguna de H1–H5 directamente — corresponde a **E0**, que condiciona la confiabilidad de todo lo que sigue (`../docs/capitulos/Chapter6_template.md` §6.1).

**Estado de ejecución:** completo, consolidado en `../docs/logs/E0_CLOSEOUT_WEEK24.md`. No se programó cómputo nuevo para E0 en la Semana 24 (per `../docs/planificacion/FinalExperimentMatrix.md` §2.0): se trata de una consolidación narrativa de evidencia ya producida en `VALIDATION_WEEK1.md`/`VALIDATION_WEEK6.md` y de la nueva suite de pruebas re-ejecutada esta semana.

**Tabla 6.1.1.** Óptimos publicados frente a los reproducidos por CP-SAT.

| Instancia | Familia | Óptimo publicado | Óptimo reproducido por CP-SAT | Error relativo | Estado |
|---|---|---|---|---|---|
| `ft06` | Fisher–Thompson | 55 | 55 | 0.00 % | proven |
| `ft10` | Fisher–Thompson | 930 | 930 | 0.00 % | proven |
| `ft20` | Fisher–Thompson | 1165 | 1165 | 0.00 % | proven |
| `mk01` (FJSP) | Brandimarte | 40 | 40 | 0.00 % | proven |

CP-SAT reproduce el óptimo publicado exactamente en las cuatro instancias de referencia, sobre la instancia JSSP directa y sobre la instancia FJSP rectangularizada (verificando simultáneamente el parser, la indexación de máquinas y la neutralidad del relleno de operaciones *ragged*, `tests/test_fjsp_loader.py`).

**Tabla 6.1.2.** Calibración de los tres generadores de perturbación (`VALIDATION_WEEK6.md`, $N=5000$ muestras/aserción, `ft06`).

*Averías* (`mttr_multiplier=2.0`, `ttr_shape_sigma=0.5`):

| $A_k$ objetivo | MTBF objetivo | TTF medio empírico | Error | MTTR objetivo | TTR medio empírico | Error | $p$-valor KS (TTF vs. Exp) | Disponibilidad empírica |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 207.944 | 207.006 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9500 |
| 0.90 | 98.500 | 98.055 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9001 |
| 0.80 | 43.778 | 43.580 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.8001 |

*Tiempos de proceso estocásticos* (multiplicador Lognormal de media 1):

| $\sigma$ | Media empírica | Media objetivo | Varianza empírica | Varianza teórica $e^{\sigma^2}-1$ | Error |
|---|---|---|---|---|---|
| 0.10 | 0.9980 | 1.0 | 0.00991 | 0.01005 | −1.38 % |
| 0.25 | 0.9949 | 1.0 | 0.06339 | 0.06449 | −1.71 % |
| 0.50 | 0.9894 | 1.0 | 0.27667 | 0.28403 | −2.59 % |

*Llegadas dinámicas* (Poisson, calibrado a $\rho$ objetivo):

| $\rho$ | $\lambda_a$ objetivo | $\lambda_a$ empírico | Error |
|---|---|---|---|
| 0.70 | 0.12792 | 0.12691 | −0.79 % |
| 0.85 | 0.15533 | 0.15410 | −0.79 % |
| 0.95 | 0.17360 | 0.17223 | −0.79 % |

Los tres generadores calibran dentro del 2.6 % de su valor teórico en todos los niveles probados; la prueba KS no rechaza la hipótesis exponencial para el tiempo entre fallos ($p=0.929 \gg 0.01$).

**Tabla 6.1.3.** Estado de la suite de pruebas (evidencia de correctitud del motor DES, estática y dinámica).

| Bloque | Pruebas | Resultado |
|---|---|---|
| Óptimos reproducidos (`test_loader.py`) | 104 | 104/104 ✓ |
| Restricciones estáticas (`test_env.py::TestConstraintSafety`, 6 reglas × 2 instancias × 3 semillas) | 51 | 51/51 ✓ |
| Calibración de perturbaciones (`test_disruptions.py`) | 25 | 25/25 ✓ |
| Restricciones dinámicas bajo avería (`test_dynamic_env.py::TestConstraintSafetyUnderBreakdown`) | 88 | 88/88 ✓ |
| Suite completa (mañana del 2026-07-31, antes del arreglo de cableado E4–E6) | 592 | 589/592 (3 fallos, ver nota) |
| Suite completa (misma tarde, tras el arreglo de cableado `DynamicJSSPEnv`↔política) | 635 | 632/635 (mismos 3 fallos, cero regresiones nuevas) |

**Nota sobre los 3 fallos persistentes:** confinados a `tests/test_hetero_policy.py`, causados por un problema de ubicación de dispositivo (`device="auto"` de `build_hetero_maskable_ppo` interactúa mal con un *driver* de GPU que reporta una incompatibilidad de versión en esta máquina) en una ruta de construcción exclusiva del **entrenamiento**, confirmado — por lectura directa del código y por una prueba de humo dedicada — que **no** afecta la ruta de evaluación (`baselines.interface.TrainedRLBaseline`, que fija `device="cpu"` explícitamente). No invalida ningún resultado de E1–E8. Queda como precondición activa antes de que la Sección 2.7 del `E7_E8_FinalPlan.md` inicie los dos entrenamientos nuevos de E7.

**Brecha honesta:** el caso analítico M/M/1 que D4/E0 exige como cuarto punto de referencia no se ejecutó — `../docs/planificacion/VandV_Protocol.md` §4.4 ya especifica el protocolo exacto (contrastar $\hat W$/$\hat L_q$ simulados contra las fórmulas cerradas $W=1/(\mu-\lambda_a)$, $L_q=\rho^2/(1-\rho)$, tolerancia relativa 5 %), pero no se ha ejecutado en ningún documento revisado. Es una prueba barata (una única corrida de cola de un servidor), marcada como opcional y no bloqueante para este capítulo, per `../docs/planificacion/FinalExperimentMatrix.md` §2.0.

**Figura 6.1.1 (pendiente de generar).** Histogramas empíricos de TTF, TTR, multiplicador de tiempo de proceso y tiempo entre llegadas, superpuestos a su densidad teórica (Exponencial/Lognormal/Poisson respectivamente) — un panel por generador, 3×1 o 2×2. Fuente: los mismos 5000 muestreos de `VALIDATION_WEEK6.md`/`test_disruptions.py` (si no se guardaron las muestras crudas, re-muestrear con la misma semilla es suficiente, ya que la calibración es determinística dado el generador de números aleatorios).

---

## 6.2 Desempeño estático (JSSP/FJSP)

**Hipótesis que atiende: H1** (`PreRegistration_FINAL.md` §2).

**Estado de ejecución:** completo sobre el conjunto de 43 instancias congelado, más el brazo FJSP-sintético (Arm A) y, como adición documentada per la Enmienda 1 (`PreRegistration_FINAL.md` §7, Finding A de `../docs/planificacion/FinalExperimentMatrix.md`), el brazo FJSP real de Brandimarte (Arm B, `mk01`–`mk07`).

### 6.2.1 Brazo JSSP (43 instancias)

**Tabla 6.2.1.** RPD por instancia, mejor PDR y método (N=43).

| Instancia | Familia | n×m | Mejor PDR | RPD mejor PDR (%) | RPD RL det. (%) | RPD CP-SAT (%) | RPD GA (%) |
|---|---|---|---|---|---|---|---|
| ft06 | fisher-thompson | 6×6 | fifo | 21.82 | 32.73 | 0.00 | 0.00 |
| ft10 | fisher-thompson | 10×10 | fifo | 30.97 | 41.72 | 0.00 | 10.64 |
| ft20 | fisher-thompson | 20×5 | fifo | 44.29 | 50.30 | 0.00 | 5.15 |
| la01 | lawrence | 10×5 | fifo | 20.42 | 4.05 | 0.00 | 0.00 |
| la02 | lawrence | 10×5 | fifo | 24.58 | 32.82 | 0.00 | 6.26 |
| la03 | lawrence | 10×5 | mwkr | 33.50 | 48.91 | 0.00 | 5.36 |
| la04 | lawrence | 10×5 | fifo | 11.86 | 34.41 | 0.00 | 2.88 |
| la05 | lawrence | 10×5 | fifo | 0.00 | 7.76 | 0.00 | 0.00 |
| la06 | lawrence | 15×5 | fifo | 9.50 | 12.85 | 0.00 | 0.00 |
| la07 | lawrence | 15×5 | fifo | 9.66 | 14.16 | 0.00 | 0.00 |
| la08 | lawrence | 15×5 | fifo | 8.92 | 16.57 | 0.00 | 0.00 |
| la09 | lawrence | 15×5 | fifo | 3.37 | 14.09 | 0.00 | 0.00 |
| la10 | lawrence | 15×5 | mor | 4.07 | 6.16 | 0.00 | 0.00 |
| la11 | lawrence | 20×5 | fifo | 1.80 | 10.64 | 0.00 | 0.00 |
| la12 | lawrence | 20×5 | fifo | 5.20 | 6.54 | 0.00 | 0.00 |
| la13 | lawrence | 20×5 | fifo | 0.00 | 4.17 | 0.00 | 0.00 |
| la14 | lawrence | 20×5 | fifo | 0.00 | 2.86 | 0.00 | 0.00 |
| la15 | lawrence | 20×5 | fifo | 21.46 | 17.81 | 0.00 | 0.00 |
| la16 | lawrence | 10×10 | fifo | 28.15 | 30.79 | 0.00 | 3.92 |
| la17 | lawrence | 10×10 | mor | 23.60 | 18.88 | 0.00 | 5.74 |
| la18 | lawrence | 10×10 | fifo | 10.26 | 22.76 | 0.00 | 1.53 |
| la19 | lawrence | 10×10 | fifo | 17.22 | 21.85 | 0.00 | 8.31 |
| la20 | lawrence | 10×10 | fifo | 28.27 | 5.43 | 0.00 | 5.99 |
| la21 | lawrence | 15×10 | fifo | 22.75 | 35.56 | 0.00 | 13.67 |
| la22 | lawrence | 15×10 | fifo | 29.45 | 41.75 | 0.00 | 14.35 |
| la23 | lawrence | 15×10 | fifo | 18.99 | 25.48 | 0.00 | 3.88 |
| la24 | lawrence | 15×10 | fifo | 33.26 | 32.51 | 0.00 | 10.59 |
| la25 | lawrence | 15×10 | fifo | 27.23 | 27.43 | 0.00 | 12.38 |
| la26 | lawrence | 20×10 | fifo | 16.42 | 28.49 | 0.00 | 11.41 |
| la27 | lawrence | 20×10 | fifo | 22.75 | 32.55 | 0.00 | 13.28 |
| la28 | lawrence | 20×10 | fifo | 28.37 | 46.71 | 0.00 | 6.09 |
| la29 | lawrence | 20×10 | fifo | 37.07 | 36.98 | 0.87 | 11.89 |
| la30 | lawrence | 20×10 | fifo | 22.14 | 19.85 | 0.00 | 7.97 |
| la31 | lawrence | 30×10 | fifo | 11.77 | 24.27 | 0.00 | 0.73 |
| la32 | lawrence | 30×10 | fifo | 13.30 | 26.86 | 0.00 | 3.40 |
| la33 | lawrence | 30×10 | fifo | 15.07 | 16.70 | 0.00 | 0.00 |
| la34 | lawrence | 30×10 | fifo | 16.91 | 25.33 | 0.00 | 5.69 |
| la35 | lawrence | 30×10 | fifo | 6.57 | 26.11 | 0.00 | 1.91 |
| la36 | lawrence | 15×15 | fifo | 28.55 | 26.58 | 0.00 | 8.91 |
| la37 | lawrence | 15×15 | fifo | 28.06 | 38.58 | 0.00 | 12.10 |
| la38 | lawrence | 15×15 | fifo | 27.09 | 37.21 | 0.00 | 15.55 |
| la39 | lawrence | 15×15 | fifo | 29.36 | 38.60 | 0.00 | 12.73 |
| la40 | lawrence | 15×15 | fifo | 21.11 | 21.36 | 0.00 | 14.89 |

100 % de las filas son factibles, cero *timeouts*, cero fallos. `fifo` es la PDR mínima en 40/43 instancias (`mwkr` en `la03`, `mor` en `la10`/`la17`).

**Tabla 6.2.1b.** Resumen agregado por método (todas las instancias, n=43 excepto RL n=215 [1 determinista + 4 estocásticas/instancia]).

| Método | RPD medio (%) | Desv. estándar (%) | Tiempo de pared medio (s) |
|---|---|---|---|
| cpsat | 0.02 | 0.13 | 4.667 |
| ga | 5.52 | 5.31 | 60.012 |
| fifo | 19.16 | 11.10 | 0.023 |
| **RL (política)** | **29.08** | **14.73** | **0.398** |
| mor | 39.83 | 16.91 | 0.024 |
| mwkr | 42.59 | 16.19 | 0.023 |
| spt | 268.23 | 114.01 | 0.027 |
| lpt | 281.88 | 116.91 | 0.023 |
| edd | 387.34 | 177.00 | 0.022 |
| lwkr | 391.91 | 177.61 | 0.023 |
| lor | 401.15 | 174.56 | 0.023 |
| cr | 430.31 | 183.62 | 0.024 |

**Tabla 6.2.2.** Prueba de Wilcoxon de rangos con signo, política RL vs. mejor PDR por instancia (criterio de H1, `PreRegistration_FINAL.md` §2).

| N | W | z | p-valor | Tamaño de efecto $r=Z/\sqrt N$ | Mediana(RPD_mejorPDR − RPD_RL) | Decisión |
|---|---|---|---|---|---|---|
| 43 | 133.0 | −4.10 | 4.14×10⁻⁵ | −0.625 | **−7.648 pp** | **RECHAZA H1** |

**Lectura honesta, sin atenuar:** la prueba es estadísticamente significativa ($p \ll 0.05$), pero **en la dirección contraria a la reclamada por H1**: la mediana de la diferencia (RPD del mejor PDR menos RPD de RL) es negativa, es decir, **RL tiene un RPD significativamente mayor (peor) que el mejor PDR por instancia**, no menor. RL supera a la mejor PDR en solo 8 de 43 instancias (`la01`, `la15`, `la17`, `la20`, `la24`, `la29`, `la30`, `la36`). El criterio congelado exige $p<0.05$ **y** mediana $>0$ para aceptar H1; aquí se cumple la significancia pero se viola el signo, así que la regla de decisión da **rechazo**, no un resultado ambiguo. Esto es consistente con lo ya señalado descriptivamente en `../docs/logs/EXPERIMENT_LOG_WEEK24.md` §E1 (RL supera a 7 de 9 PDR individuales pero pierde frente a `fifo` específicamente, la PDR de mejor desempeño) y con la explicación ya documentada: el *checkpoint* usado (`curriculum_stage19_la20_final`) es una política *size-agnostic* cuya exposición de currículo llega hasta `la20` — `ft06` está en el borde de esa exposición y todo lo posterior a `la20` es *zero-shot* genuino (`../docs/planificacion/FinalExperimentMatrix.md` §2.1) — por lo que una brecha real frente a la PDR más fuerte en un subconjunto de instancias es un resultado esperable, no una anomalía de medición. Descriptivamente, CP-SAT y GA sí alcanzan el óptimo probado con brecha mínima (0.02 % y 5.52 % medios), confirmando que la brecha de RL es del método, no del arnés de evaluación.

### 6.2.2 Brazo FJSP-sintético (Arm A, 43 instancias, sin RPD publicable)

**Tabla 6.2.3.** Comparación de makespan bruto (sin óptimo de literatura — instancias sintéticas, `generate_fjsp_instance(seed=0)`; ver `PreRegistration_FINAL.md` §1).

| Método | n | Makespan medio | Desv. estándar | Tiempo de pared medio (s) | Máx. tiempo de pared (s) |
|---|---|---|---|---|---|
| CP-SAT (FJSP) | 43 | 893.19 | 320.78 | 39.21 | 60.03 |
| GA | 43 | 980.56 | 355.17 | 60.02 | 60.05 |
| RL (todas las tiradas) | 215 | 1295.78 | 464.54 | 0.92 | 38.52 |

CP-SAT alcanza el óptimo probado (`status=OPTIMAL`) en 17/43 instancias; en las 26 restantes reporta el mejor factible al límite de 60 s (`status=FEASIBLE`, no probado). Razón RL/CP-SAT (determinista, por instancia): media 1.403 (σ=0.205); usando el promedio de las 5 tiradas de RL por instancia (comparable con la cifra descriptiva de `../docs/logs/EXPERIMENT_LOG_WEEK24.md`): media **1.456** (σ=0.214). Razón GA/CP-SAT: media 1.094 (σ=0.063).

**Tabla 6.2.2b.** Prueba de Wilcoxon, brazo FJSP-sintético (mejor de {GA, CP-SAT} vs. RL determinista, por makespan, N=43).

| N | W | z | p-valor | $r$ | Mediana(mejor{GA,CPSAT} − RL) | Decisión |
|---|---|---|---|---|---|---|
| 43 | 0.0 | −5.71 | 1.16×10⁻⁸ | −0.870 | **−352.0** | **RECHAZA H1 (brazo FJSP)** |

RL no supera al mejor de {GA, CP-SAT} en **ninguna** de las 43 instancias sintéticas ($W=0$) — el rechazo es aún más decisivo que en el brazo JSSP. Esto es coherente con lo ya señalado en `../docs/logs/EXPERIMENT_LOG_WEEK24.md`: la cabeza de acción por-par (operación, máquina) de FJSP escala con `num_ops × num_machines`, y el *checkpoint* generalista usado no fue optimizado específicamente para esta familia sintética.

### 6.2.3 Brazo FJSP real — Brandimarte `mk01`–`mk07` (Arm B, adición per Enmienda 1)

**Tabla 6.2.4.** Resultados sobre las siete instancias reales de Brandimarte (una tirada determinista, semilla 0).

| Instancia | GA makespan | GA RPD | CP-SAT makespan | Estado CP-SAT | RL (det.) makespan | RL RPD | Óptimo/mejor conocido | Condición |
|---|---|---|---|---|---|---|---|---|
| mk01 | 42 | 5.00 % | 40 | OPTIMAL (probado) | 60 | 50.00 % | 40 | probado, in-sample |
| mk02 | 32 | 23.08 % | 26 | OPTIMAL (probado) | 63 | 142.31 % | 26 | best_known, in-sample |
| mk03 | 204 | 0.00 % | 204 | OPTIMAL (probado) | 374 | 83.33 % | 204 | probado, in-sample |
| mk04 | 76 | 26.67 % | 60 | OPTIMAL (probado) | 78 | 30.00 % | 60 | probado, in-sample |
| mk05 | 181 | 5.23 % | 173 | FEASIBLE (60 s, no probado) | 201 | 16.86 % | 172 | best_known, in-sample |
| mk06† | 81 | 39.66 %† | 59 | FEASIBLE (60 s)† | 170 | 193.10 %† | 58† | best_known, **zero-shot, RPD inválido** |
| mk07 | 164 | 17.99 % | 142 | FEASIBLE (60 s, no probado) | 309 | 122.30 % | 139 | best_known, zero-shot |

† `mk06` excluido de cualquier comparación basada en RPD: el índice de referencia del *mirror* `SchedulingLab/fjsp-instances` declara 15 máquinas, pero su propio archivo de datos usa solo 10 (índices 0–9), confirmado no transitorio (`tests/test_fjsp_loader.py::test_mk06_known_upstream_machine_count_mismatch_is_still_present`). El makespan bruto (81/59/170) es un resultado válido para la instancia de 10 máquinas realmente cargada, pero no comparable al óptimo de 58 de la `mk06` canónica de 15 máquinas. **Acción recomendada, no ejecutada en este borrador:** reportar el problema en `SchedulingLab/fjsp-instances`.

RL es notablemente más débil en este brazo real que en el sintético — incluso en las instancias *in-sample* (`mk01`–`mk05`, vistas durante el currículo, etapas 20–24), donde el RPD de RL (17–50 %) es sustancialmente peor que el 5.0 % que el mismo tipo de *checkpoint*, pero entrenado de forma monoinstancia (Semana 18, 1M pasos), alcanzó sobre `mk01` en el informe de la Semana 21 — la diferencia es exactamente la esperada entre una política generalista (`stage24_mk05`, usada aquí por consistencia con el brazo A y con E3) y una especializada; ambos números deben reportarse por separado en la discusión, no promediarse. En `mk06`/`mk07` (*zero-shot* genuino) el RPD de RL sube a 122–193 %, la brecha más severa de todo §6.2 — un patrón sensible a la forma (más máquinas que trabajos) que merece una nota explícita en el Capítulo 7.

**Veredicto H1 (síntesis).** Con los tres subconjuntos evaluados — JSSP ($N=43$), FJSP-sintético ($N=43$) y FJSP-Brandimarte ($N=7$, descriptivo, no forma parte de la prueba confirmatoria per el diseño de `PreRegistration_FINAL.md` §1 al no tener el tamaño muestral registrado — **H1 se rechaza en los dos brazos con prueba confirmatoria (JSSP y FJSP-sintético)**, ambos con significancia estadística clara mostrando que el mejor PDR (o el mejor de {GA, CP-SAT} en FJSP) supera a la política entrenada, no al revés. Este es un resultado negativo pero informativo, coherente con la propia advertencia del *checkpoint* utilizado (§2.1 de `../docs/planificacion/FinalExperimentMatrix.md`): la política es competitiva en un subconjunto de instancias (8/43 en JSSP) pero no domina el conjunto de referencia en promedio.

**Figuras requeridas para §6.2 (a generar por separado a partir de los CSV):**

- **Figura 6.2.1** — Diagrama de diferencia crítica (Friedman + Nemenyi post-hoc) de rango medio sobre las 9 PDR + GA + RL en las 43 instancias JSSP (CP-SAT excluido del ranking conjunto, fija $C^*$). Requiere ejecutar el mismo procedimiento de §6.3 (Friedman/Nemenyi) sobre el conjunto de 43 en vez de por *tier* — no calculado en este borrador porque no es parte de la regla de decisión de H1 (esa regla usa Wilcoxon pareado, no Friedman), pero es una figura de contexto útil para el Capítulo 6.2 igualmente.
- **Figura 6.2.2** — Curvas de entrenamiento (6 paneles: retorno, makespan vs. óptimo, entropía, pérdida de valor, varianza explicada, tamaño de paso PPO) del *checkpoint* `curriculum_stage19_la20_final` usado en esta sección — reutilizar el formato ya validado en `TRAINING_REPORT_WEEK18.md` §3, sobre los CSV de `experiments/results/train_hetero_stage*` si ese *checkpoint* específico los generó, o regenerar desde los *logs* de `train_curriculum.py`.
- **Figura 6.2.3 (nueva, recomendada para el brazo FJSP-B)** — Diagrama de dispersión RPD *in-sample* (mk01–mk05) vs. RPD *zero-shot* (mk06–mk07) por método, para visualizar de un vistazo la brecha de generalización de la Tabla 6.2.4.

---

## 6.3 Generalización en tamaño

**Hipótesis que atiende: H2** (`PreRegistration_FINAL.md` §3).

**Estado de ejecución:** completo. **Nota de conciliación:** la definición congelada de *tiers* en `PreRegistration_FINAL.md` §3 agrupa `la21`–`la25` y `la26`–`la30` en un **único** *tier* "~1.5–2×" ($N=10$ instancias), mientras que el desglose descriptivo de `../docs/logs/EXPERIMENT_LOG_WEEK24.md` §E3 los reporta como dos grupos de 5 por separado (siguiendo el propio agrupamiento de `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md`). **Este borrador usa la definición de tres *tiers* congelada en el pre-registro para la prueba confirmatoria de H2** (Tabla 6.3.3), no el desglose de cuatro grupos del log — ambas lecturas de los datos coexisten sin conflicto (los 20 valores por instancia son los mismos), pero solo la agrupación de tres *tiers* es la que la regla de decisión de H2 usa.

**Tabla 6.3.1.** *Tiers* de generalización (frozen, `PreRegistration_FINAL.md` §3).

| Tier | Instancias | `num_operations` | Razón vs. tamaño de entrenamiento (100 ops) |
|---|---|---|---|
| ~1.5–2× | `la21`–`la30` (N=10) | 150, 200 | 1.5×, 2.0× |
| ~2.25× | `la36`–`la40` (N=5) | 225 | 2.25× (más denso: más máquinas/trabajo) |
| ~3× | `la31`–`la35` (N=5) | 300 | 3.0× |

**Tabla 6.3.2.** RPD por instancia (transferencia de pesos *zero-shot*, `deterministic=True`, un rollout).

| Tier | Instancia | n_ops | RPD RL *zero-shot* (%) | Mejor PDR | RPD mejor PDR (%) | RPD CP-SAT (%) | RPD GA (%) |
|---|---|---|---|---|---|---|---|
| ~1.5–2× | la21 | 150 | 35.56 | fifo | 22.75 | 0.00 | 13.67 |
| ~1.5–2× | la22 | 150 | 41.75 | fifo | 29.45 | 0.00 | 14.35 |
| ~1.5–2× | la23 | 150 | 25.48 | fifo | 18.99 | 0.00 | 3.88 |
| ~1.5–2× | la24 | 150 | 32.51 | fifo | 33.26 | 0.00 | 10.59 |
| ~1.5–2× | la25 | 150 | 27.43 | fifo | 27.23 | 0.00 | 12.38 |
| ~1.5–2× | la26 | 200 | 28.49 | fifo | 16.42 | 0.00 | 11.41 |
| ~1.5–2× | la27 | 200 | 32.55 | fifo | 22.75 | 0.00 | 13.28 |
| ~1.5–2× | la28 | 200 | 46.71 | fifo | 28.37 | 0.00 | 6.09 |
| ~1.5–2× | la29 | 200 | 36.98 | fifo | 37.07 | 0.87 | 11.89 |
| ~1.5–2× | la30 | 200 | 19.85 | fifo | 22.14 | 0.00 | 7.97 |
| ~3× | la31 | 300 | 24.27 | fifo | 11.77 | 0.00 | 0.73 |
| ~3× | la32 | 300 | 26.86 | fifo | 13.30 | 0.00 | 3.40 |
| ~3× | la33 | 300 | 16.70 | fifo | 15.07 | 0.00 | 0.00 |
| ~3× | la34 | 300 | 25.33 | fifo | 16.91 | 0.00 | 5.69 |
| ~3× | la35 | 300 | 26.11 | fifo | 6.57 | 0.00 | 1.91 |
| ~2.25× | la36 | 225 | 26.58 | fifo | 28.55 | 0.00 | 8.91 |
| ~2.25× | la37 | 225 | 38.58 | fifo | 28.06 | 0.00 | 12.10 |
| ~2.25× | la38 | 225 | 37.21 | fifo | 27.09 | 0.00 | 15.55 |
| ~2.25× | la39 | 225 | 38.60 | fifo | 29.36 | 0.00 | 12.73 |
| ~2.25× | la40 | 225 | 21.36 | fifo | 21.11 | 0.00 | 14.89 |

**Validación cruzada del pipeline (control de calidad, no parte de la regla de H2):** las 5 instancias que ya tenían una cifra de RL independiente en `../docs/logs/HPO_ABLATION_REPORT_WEEK21.md` (`la21`, `la25`, `la30`, `la35`, `la40`) coinciden con las de esta tabla dentro de 0.05 pp en los cinco casos — evidencia de que el arnés de evaluación, la carga del *checkpoint* y el cálculo de RPD de la Semana 24 son consistentes con una medición independiente anterior.

**Tabla 6.3.3.** Prueba de Friedman + Nemenyi por *tier* ($k=10$ métodos: RL + 9 PDR; criterio de H2, `PreRegistration_FINAL.md` §3).

| Tier | N | χ²_F | p-valor | CD (Nemenyi, α=0.05) | Rango medio RL | Mejor PDR (rango) | ¿RL significativamente peor? |
|---|---|---|---|---|---|---|---|
| ~1.5–2× (la21–la30) | 10 | 84.53 | 2.03×10⁻¹⁴ | 4.284 | 1.70 | fifo (1.30) | **No** (Δ=0.40 < CD) |
| ~3× (la31–la35) | 5 | 43.72 | 1.59×10⁻⁶ | 6.059 | 2.00 | fifo (1.00) | **No** (Δ=1.00 < CD) |
| ~2.25× (la36–la40) | 5 | 42.02 | 3.26×10⁻⁶ | 6.059 | 1.80 | fifo (1.20) | **No** (Δ=0.60 < CD) |

**Veredicto H2:** en los tres *tiers*, Friedman rechaza decisivamente la hipótesis nula de igual desempeño ($p \ll 0.05$, condición (a) del criterio), y en los tres, la diferencia de rango medio entre RL y la mejor PDR (`fifo`, consistentemente) queda **por debajo** de la diferencia crítica de Nemenyi (condición (b)) — es decir, **RL no es significativamente peor que la mejor PDR en ningún tier**. Con las dos condiciones satisfechas en los tres *tiers*, **se ACEPTA H2**.

**Matiz honesto que la prueba de significancia no captura por sí sola:** el RPD medio de RL sí es numéricamente mayor que el de `fifo` en los tres *tiers* (32.7 % vs. 25.8 % en 1.5–2×; 23.9 % vs. 12.7 % en 3×; 32.5 % vs. 26.8 % en 2.25×) — la aceptación de H2 refleja que, con $N=5$ o $N=10$ instancias por *tier*, la diferencia de rango entre RL y `fifo` no alcanza el umbral (conservador) de Nemenyi para declararse estadísticamente distinguible, no que ambos métodos rindan igual en la práctica. Ambas lecturas — "H2 se acepta por la regla congelada" y "la brecha numérica frente a la mejor PDR persiste y es consistente entre *tiers*" — deben aparecer juntas en el Capítulo 7, no solo la primera.

**Gradiente de degradación (descriptivo, no forma parte de la regla de decisión):** el patrón **no es monótono** en el tamaño de instancia — el *tier* ~3× (`la31`–`la35`) muestra el RPD medio *más bajo* de los tres (23.9 %), por debajo incluso del *tier* más pequeño (~1.5–2×, 32.6 %). Esto ya es visible en la propia tabla de 5 instancias de la Semana 21 (`la35`=26.1 %, `la40`=21.4 %, ambos menores que `la21`=35.6 %) y se confirma con las 15 instancias adicionales de este corte — un hallazgo genuino, no un artefacto de muestreo, que el Capítulo 7 debe describir como tal en vez de asumir una curva de generalización estrictamente decreciente.

**Figuras requeridas para §6.3:**

- **Figura 6.3.1** — RPD de RL en función de `num_operations` (eje x: 36 a 300; eje y: RPD %), una serie para RL y una para la mejor PDR de cada *tier*, línea vertical en el tamaño de entrenamiento (100 ops). Los datos de la Tabla 6.3.2 alimentan directamente esta figura; el punto clave a visualizar es la no-monotonicidad del *tier* ~3×.
- **Figura 6.3.2** — Tres diagramas de diferencia crítica (uno por *tier*, mismo formato que 6.2.1), usando los rangos medios y el CD ya calculados en la Tabla 6.3.3.

---

## 6.4 Robustez bajo perturbaciones

**Hipótesis que atiende: H3** (`PreRegistration_FINAL.md` §4). H4 (recompensa consciente de la perturbación, κ) permanece **bloqueada** — el término $\kappa\cdot\Delta_{\text{inest}}$ no está implementado más allá de la instrumentación (`info["planned_start_deviation"]`), confirmado por lectura directa de `envs/dynamic_jssp_env.py::step()` — y su ejecución está planificada para el esprint de la semana del 3–7 de agosto de 2026 per `E7_E8_FinalPlan.md` §2, fuera del alcance de este borrador. La Tabla 6.4.3/Figura 6.4.2 de κ permanecen `[BLOQUEADAS]` tal como ya las marcaba `../docs/capitulos/Chapter6_template.md`.

**Estado de ejecución (H3):** completo, incluyendo el brazo RL — el bloqueador de cableado `DynamicJSSPEnv`↔política se resolvió el mismo día de `../docs/logs/EXPERIMENT_LOG_WEEK24.md` (ver Finding #4 de ese documento), con una verificación explícita de equivalencia de reproducción numérica (24/24 + 3/3 pruebas nuevas) antes de ejecutar la rejilla completa.

### 6.4.1 Índice de robustez ($RI$) por régimen y método

**Tabla 6.4.1a.** RI medio (desv. estándar), E4 averías, por $A_k$ (n=60 pares instancia×semilla por celda: `ft06`+`la01` × 30 semillas).

| Método | $A_k$=0.95 | $A_k$=0.90 | $A_k$=0.80 |
|---|---|---|---|
| spt | −0.339 (0.114) | −0.280 (0.177) | −0.149 (0.175) |
| lpt | −0.469 (0.092) | −0.433 (0.099) | −0.305 (0.119) |
| mwkr | −0.054 (0.117) | +0.031 (0.177) | +0.274 (0.239) |
| lwkr | −0.470 (0.084) | −0.430 (0.139) | −0.337 (0.115) |
| mor | −0.040 (0.124) | +0.048 (0.169) | +0.260 (0.199) |
| lor | −0.478 (0.114) | −0.435 (0.181) | −0.316 (0.162) |
| fifo | +0.064 (0.126) | +0.154 (0.188) | +0.418 (0.229) |
| edd | −0.470 (0.070) | −0.426 (0.137) | −0.336 (0.114) |
| cr | **−0.541 (0.069)** | **−0.489 (0.124)** | **−0.404 (0.108)** |
| **RL (política)** | **+0.065 (0.158)** | **+0.154 (0.209)** | **+0.410 (0.286)** |

**Tabla 6.4.1b.** RI medio (desv. estándar), E5 tiempos estocásticos, por $\sigma$.

| Método | σ=0.10 | σ=0.25 | σ=0.50 |
|---|---|---|---|
| spt | −0.377 (0.099) | −0.371 (0.103) | −0.355 (0.128) |
| lpt | −0.526 (0.069) | −0.503 (0.070) | −0.450 (0.109) |
| mwkr | −0.141 (0.034) | −0.130 (0.066) | −0.044 (0.141) |
| lwkr | −0.521 (0.037) | −0.509 (0.053) | −0.481 (0.091) |
| mor | −0.136 (0.043) | −0.110 (0.063) | −0.035 (0.137) |
| lor | −0.529 (0.097) | −0.509 (0.110) | −0.478 (0.126) |
| fifo | −0.033 (0.070) | −0.016 (0.087) | +0.067 (0.171) |
| edd | −0.518 (0.036) | −0.508 (0.053) | −0.488 (0.083) |
| cr | **−0.592 (0.041)** | **−0.582 (0.044)** | **−0.560 (0.078)** |
| **RL (política)** | **−0.021 (0.086)** | **+0.000 (0.112)** | **+0.091 (0.188)** |

**Tabla 6.4.1c.** RI medio (desv. estándar), E6 llegadas dinámicas, por $\rho$ (solo PDR *arrival-safe*: `lwkr`, `mor`, `lor`, `fifo`).

| Método | ρ=0.70 | ρ=0.85 | ρ=0.95 |
|---|---|---|---|
| lwkr | **+0.283 (0.137)** | **+0.284 (0.146)** | **+0.259 (0.136)** |
| lor | +0.336 (0.315) | +0.321 (0.330) | +0.305 (0.338) |
| mor | +1.522 (0.269) | +1.430 (0.227) | +1.372 (0.201) |
| fifo | +1.833 (0.268) | +1.748 (0.275) | +1.706 (0.259) |
| **RL (política)** | **+1.816 (0.352)** | **+1.715 (0.283)** | **+1.654 (0.271)** |

**Lectura honesta de la Tabla 6.4.1, sin atenuar:** en E4/E5, `cr` aparece como la PDR de menor (mejor) RI — pero esto es un artefacto reconocido, no una fortaleza real de `cr`: su RPD nominal (sin perturbación) es el peor de las nueve PDR (430 % en la Tabla 6.2.1b), así que casi cualquier resultado perturbado parece una "mejora" frente a esa base ya mala — un artefacto inherente al diseño de RI (línea base específica por método), no un error de cómputo. La PDR de mejor desempeño **absoluto** (`mwkr`, per RPD nominal) tiene un RI considerablemente peor que `cr` pero mejor que RL en E4/E5. En E6, **RL tiene el segundo peor RI de los cinco métodos comparados, prácticamente empatado con `fifo`** (la PDR de peor RI) y muy por encima de `lwkr`/`lor` — sin ambigüedad, este es el régimen donde RL rinde peor frente a la mejor alternativa disponible.

### 6.4.2 Prueba de Wilcoxon y veredicto de H3

**Tabla 6.4.2.** Wilcoxon pareado, $RI_{\text{RL}}$ vs. $RI_{\text{mejorPDR}}$ (mediana), por (instancia × semilla), calculado con `scipy` en `experiments/results/week24/compute_h3_verdict.py` (fuente: `h3_verdict.csv`).

| Régimen | Mejor PDR (por RI) | RI mediana (mejor PDR) | RI mediana (RL) | Mediana(RI_PDR − RI_RL) | $p$-valor | ¿Acepta H3? |
|---|---|---|---|---|---|---|
| E4 averías, $A_k$=0.95 | cr | −0.547 | +0.010 | −0.609 | 1.62×10⁻¹¹ | No |
| E4 averías, $A_k$=0.90 | cr | −0.504 | +0.115 | −0.630 | 1.63×10⁻¹¹ | No |
| E4 averías, $A_k$=0.80 | cr | −0.402 | +0.405 | −0.818 | 1.63×10⁻¹¹ | No |
| E5 estocástico, σ=0.10 | cr | −0.598 | −0.031 | −0.560 | 1.63×10⁻¹¹ | No |
| E5 estocástico, σ=0.25 | cr | −0.583 | −0.017 | −0.572 | 1.63×10⁻¹¹ | No |
| E5 estocástico, σ=0.50 | cr | −0.577 | +0.067 | −0.635 | 1.63×10⁻¹¹ | No |
| E6 llegadas, ρ=0.70 | lwkr | +0.271 | +1.727 | −1.501 | 1.63×10⁻¹¹ | No |
| E6 llegadas, ρ=0.85 | lwkr | +0.277 | +1.706 | −1.424 | 1.63×10⁻¹¹ | No |
| E6 llegadas, ρ=0.95 | lwkr | +0.257 | +1.631 | −1.414 | 1.63×10⁻¹¹ | No |

**Regímenes que aceptan H3: 0/9. Veredicto: SE RECHAZA H3.** El criterio congelado exige aceptación en los 9 regímenes para un accept global (`PreRegistration_FINAL.md` §4, regla conjuntiva); aquí ningún régimen individual la satisface, y en todos los casos el efecto es significativo **en la dirección contraria** a la reclamada (RL es menos robusto, no más).

**Verificación de sensibilidad (ya ejecutada en `../docs/logs/EXPERIMENT_LOG_WEEK24.md`, reportada aquí por completitud):** dado que `cr` es mecánicamente seleccionada como "mejor PDR" en E4/E5 por un artefacto de su línea base nominal deficiente (no por ser la PDR de mejor desempeño absoluto), se repitió la prueba usando `mwkr` (la PDR de mejor RPD nominal, excluida de E6 por no ser *arrival-safe*) como comparador alternativo: **el resultado cualitativo no cambia** — H3 se rechaza en los 6 regímenes de E4/E5 probados con este comparador alternativo ($p$ entre 7.6×10⁻⁶ y 2.3×10⁻¹¹, mediana de la diferencia negativa en los 6 casos). El rechazo de H3 **no depende de qué PDR concreta selecciona la regla congelada como comparador**.

**Interpretación, para el Capítulo 7:** el RI de RL es peor, por un margen amplio, específicamente bajo **llegadas dinámicas** ($RI\approx+1.6$ a $+1.8$, es decir, el makespan perturbado promedia entre 2.6 y 2.8 veces el nominal) — sustancialmente peor que bajo averías o tiempos estocásticos ($RI$ entre −0.03 y +0.42). Esto es consistente con, y da una magnitud concreta a, la limitación ya documentada en la verificación del cableado `DynamicJSSPEnv` (`../docs/logs/EXPERIMENT_LOG_WEEK24.md`, sección "E4/E5/E6 RL arm"): la política nunca fue entrenada bajo la semántica de legalidad *completion-gated* del entorno dinámico, y las llegadas presentan un escenario genuinamente nuevo (trabajos que entran a mitad de episodio) frente a averías/tiempos estocásticos, que solo perturban la temporización de operaciones que la política ya conoce. **El rechazo de H3 es, por tanto, un resultado negativo pero informativo**, no una falla de medición: cuantifica exactamente el costo de desplegar, sin reentrenamiento, una política estática sobre un entorno dinámico — el mismo hallazgo que motiva el diseño de entrenamiento de E7 en `E7_E8_FinalPlan.md` §2.2–2.3.

**Figuras requeridas para §6.4:**

- **Figura 6.4.1** — Diagramas de caja de $RI$ por régimen y método, un panel por eje de perturbación (3 paneles: averías/tiempos/llegadas), cajas agrupadas por método dentro de cada panel — los datos ya están en `experiments/results/week24/e4e5e6/*_with_rl.csv` (para esto se necesitan los valores de RI por semilla, no solo el resumen de la Tabla 6.4.1; el script de análisis de este borrador ya los calculó en memoria y puede exportarlos a CSV en una pasada adicional si se desea generar la figura directamente).
- **Figura 6.4.3 (nueva, recomendada)** — Barras agrupadas de RI medio por método, un panel por régimen (9 paneles o una cuadrícula 3×3), para visualizar de un vistazo el patrón "RL empata con la peor PDR, no con la mejor" que la Tabla 6.4.1 muestra numéricamente.

---

## Síntesis provisional: H1–H3

| Hipótesis | Sección | Prueba | Resultado | Decisión |
|---|---|---|---|---|
| H1 (competitividad estática) — JSSP | 6.2.1 | Wilcoxon pareado, N=43 | p=4.14×10⁻⁵, mediana=−7.65pp | **RECHAZADA** |
| H1 (competitividad estática) — FJSP sintético | 6.2.2 | Wilcoxon pareado, N=43 | p=1.16×10⁻⁸, mediana=−352 | **RECHAZADA** |
| H2 (generalización en tamaño) | 6.3 | Friedman + Nemenyi, 3 tiers | p≪0.05 en los 3, RL no sig. peor que la mejor PDR en ninguno | **ACEPTADA** |
| H3 (robustez bajo perturbación) | 6.4 | Wilcoxon pareado, 9 regímenes | 0/9 regímenes aceptan; efecto sig. en dirección contraria en los 9 | **RECHAZADA** (0/9) |

Tres de las cuatro pruebas confirmatorias que este borrador puede calcular con los datos de la Semana 24 dan un resultado negativo para la política entrenada frente a la mejor alternativa clásica (H1 en ambos brazos, H3 en los 9 regímenes); solo H2 (generalización) se acepta, y con el matiz de que la brecha numérica frente a `fifo` persiste aunque no sea estadísticamente distinguible al nivel de Nemenyi. Ninguno de estos resultados debe presentarse en el Capítulo 6 sin este matiz — y, per la instrucción explícita de honestidad de este encargo, ninguno debe atenuarse en el Capítulo 7: son exactamente el tipo de resultado negativo-pero-informativo que un pre-registro está diseñado para proteger de la sobre-interpretación retrospectiva, no un fracaso del proyecto.

**Pendiente antes de que este capítulo se considere final (no parte de este borrador):** (a) re-verificación de las pruebas de Wilcoxon/Friedman/Nemenyi de este documento con el pipeline `scipy`/`scikit-posthocs` de `StatsAnalysis_TechSpec.md`; (b) §6.5 (H5, tiempos de decisión) tras la remedición limpia de E8 (`E7_E8_FinalPlan.md` §4); (c) §6.4's Tabla 6.4.3/Figura 6.4.2 (H4) tras el esprint de E7 (`E7_E8_FinalPlan.md` §2); (d) generación efectiva de las figuras listadas en cada sección a partir de los CSV ya disponibles.
