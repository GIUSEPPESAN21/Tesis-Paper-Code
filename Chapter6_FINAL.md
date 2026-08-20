# Capítulo 6 — Resultados (VERSIÓN FINAL, §§6.1–6.5)

**Estado:** Final, con los resultados estadísticos completos de E1–E8 (H1–H5), producidos por el pipeline dedicado (`experiments/stats/`, `StatsAnalysis_TechSpec.md`) y consolidados en `FINAL_RESULTS_REPORT.md` (completado 2026-08-01). Sustituye a `docs/capitulos/Chapter6_sections_6.1-6.4_DRAFT.md` (borrador del 2026-07-31, que calculó Wilcoxon/Friedman/Nemenyi con una implementación Python preliminar sin `scipy`, explícitamente marcada como pendiente de re-verificación). **Esa re-verificación ya ocurrió**: los valores del borrador coinciden, dentro de un margen de redondeo trivial, con los del pipeline oficial (p. ej. H1-JSSP: borrador $p=4.14\times10^{-5}$ vs. oficial $p=4.03\times10^{-5}$; misma decisión en ambos casos) — la única diferencia estructural es que §6.3 usa aquí la partición en **cuatro** *tiers* de 5 instancias que el pipeline oficial ejecutó realmente, no la partición en tres *tiers* (con `la21`–`la30` agrupado) que este capítulo asumió preliminarmente a partir de una lectura literal de la tabla de `PreRegistration_FINAL.md` §3 — ver la nota de conciliación en §6.3.

**Cadena de procedencia:** `PreRegistration_FINAL.md` (congelado 2026-07-29) → `E7_E8_FinalPlan.md` (auditoría + alcance de E7/E8, 2026-07-31) → `StatsAnalysis_TechSpec.md` (especificación del pipeline) → `experiments/stats/` (implementación; 84/84 pruebas propias superadas) → `FINAL_RESULTS_REPORT.md` (2026-08-01: remedición limpia de E8 confirmada, pipeline estadístico ejecutado por primera vez, `train_dynamic.py` ejecutado para las dos políticas $\kappa$, rejilla de evaluación E7 completa, dos errores reales encontrados y corregidos en el pipeline mismo — ver §6.4/§8 de ese documento —, reporte final generado) → este capítulo.

**Suite de pruebas completa a la fecha de este capítulo:** 752 superadas, 0 fallidas (incluye las pruebas de regresión del arreglo de ubicación de dispositivo y las 84 pruebas propias del módulo de estadística).

---

## Tabla 6.0. Veredicto por hipótesis (H1–H5), resumen ejecutivo del capítulo

| Hipótesis | Veredicto | Estadístico clave |
|---|---|---|
| H1 (JSSP) — RL supera a la mejor PDR, estático, N=43 | **RECHAZADA** | Wilcoxon (exacto) W=133, p=4.03×10⁻⁵, r=−0.626 (RL es *peor* que la mejor PDR) |
| H1 (FJSP sintético) | **RECHAZADA** | W=0, p=1.12×10⁻⁸, r=−0.871 |
| H1 (FJSP Brandimarte) | **RECHAZADA** | W=0, p=0.0156, r=−0.914 |
| H2 (global) — generalización en tamaño, 4 *tiers* | **ACEPTADA** | RL no significativamente peor que la mejor PDR (CD de Nemenyi) en ningún *tier* |
| H3 (global) — robustez bajo perturbación, 9 regímenes | **RECHAZADA** | 0/9 regímenes aceptan (regla conjuntiva); RI de RL peor que el de la mejor PDR en los 9, p<1.7×10⁻¹¹ en todos |
| H4 ($\kappa$=0.5, global) — recompensa consciente de la perturbación | **RECHAZADA** | componente (a): 0/9 regímenes significativos (se exigen ≥5); componente (b): $\Delta\text{RPD}_{\text{nominal}}$=+106.0 pp (margen: <2 pp) |
| H4 ($\kappa$=1.0, global) | **RECHAZADA** | componente (a): 1/9 regímenes significativos (se exigen ≥5); componente (b): $\Delta\text{RPD}_{\text{nominal}}$=+46.6 pp (margen: <2 pp) |
| H5 — tiempo de decisión, RL < 1 % del de CP-SAT, 43 instancias | **RECHAZADA** | 42/43 instancias violan el umbral; RL solo gana de forma decisiva en la instancia más grande (`la40`) — ver §6.5 sobre una anomalía de medición detectada en ese mismo punto |

**Lectura honesta de esta tabla, antes de entrar en el detalle:** todos los veredictos son RECHAZO salvo H2. Esto no es un fallo de redacción ni de ejecución — H1/H3 ya se habían identificado como RECHAZO en el corte de la Semana 24 (`E7_E8_FinalPlan.md` §1 los marca explícitamente como "aceptar como resultado confirmatorio final," no como anomalías a perseguir) y H5 confirma, con datos limpios, el patrón que Semana 24's medición contaminada por concurrencia ya insinuaba. Lo que este esprint añade es: H4 tiene ahora un veredicto real y ejecutado por primera vez (antes bloqueada), y H5 tiene ahora una medición limpia y de cobertura completa (43/43 instancias) en vez de la de la Semana 24, señalada como ruidosa. Cada rechazo tiene una causa concreta, diagnosticada, no solo reportada — ver §§6.2–6.5.

---

## 6.1 Validación del entorno de simulación

*(Sin cambios respecto al borrador del 2026-07-31 — E0 no depende de ninguna pieza del esprint de estadística/E7/E8.)*

**Hipótesis que atiende:** ninguna de H1–H5 directamente — corresponde a **E0**, que condiciona la confiabilidad de todo lo que sigue.

**Tabla 6.1.1.** Óptimos publicados frente a los reproducidos por CP-SAT.

| Instancia | Familia | Óptimo publicado | Óptimo reproducido por CP-SAT | Error relativo | Estado |
|---|---|---|---|---|---|
| `ft06` | Fisher–Thompson | 55 | 55 | 0.00 % | proven |
| `ft10` | Fisher–Thompson | 930 | 930 | 0.00 % | proven |
| `ft20` | Fisher–Thompson | 1165 | 1165 | 0.00 % | proven |
| `mk01` (FJSP) | Brandimarte | 40 | 40 | 0.00 % | proven |

**Tabla 6.1.2.** Calibración de los tres generadores de perturbación ($N=5000$ muestras/aserción, `ft06`).

*Averías:*

| $A_k$ objetivo | MTBF objetivo | TTF medio empírico | Error | MTTR objetivo | TTR medio empírico | Error | $p$-valor KS (TTF vs. Exp) | Disponibilidad empírica |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 207.944 | 207.006 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9500 |
| 0.90 | 98.500 | 98.055 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9001 |
| 0.80 | 43.778 | 43.580 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.8001 |

*Tiempos de proceso estocásticos:*

| $\sigma$ | Media empírica | Varianza empírica | Varianza teórica $e^{\sigma^2}-1$ | Error |
|---|---|---|---|---|
| 0.10 | 0.9980 | 0.00991 | 0.01005 | −1.38 % |
| 0.25 | 0.9949 | 0.06339 | 0.06449 | −1.71 % |
| 0.50 | 0.9894 | 0.27667 | 0.28403 | −2.59 % |

*Llegadas dinámicas:*

| $\rho$ | $\lambda_a$ objetivo | $\lambda_a$ empírico | Error |
|---|---|---|---|
| 0.70 | 0.12792 | 0.12691 | −0.79 % |
| 0.85 | 0.15533 | 0.15410 | −0.79 % |
| 0.95 | 0.17360 | 0.17223 | −0.79 % |

**Tabla 6.1.3.** Estado de la suite de pruebas.

| Bloque | Pruebas | Resultado |
|---|---|---|
| Óptimos reproducidos | 104 | 104/104 ✓ |
| Restricciones estáticas | 51 | 51/51 ✓ |
| Calibración de perturbaciones | 25 | 25/25 ✓ |
| Restricciones dinámicas bajo avería | 88 | 88/88 ✓ |
| Suite completa, al cierre de este capítulo | 752 | 752/752 ✓ (0 fallos — el arreglo de ubicación de dispositivo de la Semana 24/25 eliminó los 3 fallos previamente confinados a `test_hetero_policy.py`) |

**Brecha honesta que persiste:** el caso analítico M/M/1 sigue sin ejecutarse — opcional per §2.0 de `docs/planificacion/FinalExperimentMatrix.md`, no bloqueante.

---

## 6.2 Desempeño estático (JSSP/FJSP) — H1

**Hipótesis que atiende: H1** (`PreRegistration_FINAL.md` §2).

**Nota metodológica sobre el cálculo del $p$-valor (añadida en v2, resolviendo el Hallazgo moderado 2.5 de `docs/auditorias/PreDefenseAudit.md`).** Todas las pruebas de Wilcoxon de este capítulo se calcularon con `scipy.stats.wilcoxon` usando su selección automática de método; verificado directamente contra `experiments/results/stats/tables/h1_h5_verdict.md` (columna `method`), **el método efectivamente usado en la totalidad de las pruebas reportadas —incluida H1-JSSP con $N=43$— fue el exacto** (basado en permutación), no la aproximación normal que la Sección 5.3.2 permitía como alternativa aceptable para $N\gtrsim20$. Esto excede, no solo cumple, el mínimo de rigor que la propia especificación estadística del Capítulo 5 exigía.

### 6.2.1 Brazo JSSP (43 instancias)

**Tabla 6.2.1.** RPD (%) por instancia y método (fuente: `e1_rpd_summary.md`, pipeline oficial). Columna **★** marca la mejor PDR de cada instancia.

| Instancia | cpsat | ga | fifo★ | rl | mor | mwkr | spt | lpt | edd | lwkr | lor | cr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ft06 | 0.00 | 0.00 | 21.82 | 32.73 | 32.73 | 34.55 | 121.82 | 140.00 | 180.00 | 180.00 | 140.00 | 210.91 |
| ft10 | 0.00 | 10.64 | 30.97 | 41.72 | 66.99 | 38.60 | 215.27 | 206.56 | 185.70 | 185.70 | 230.00 | 295.91 |
| ft20 | 0.00 | 5.15 | 44.29 | 50.30 | 68.50 | 71.25 | 161.54 | 119.91 | 160.09 | 155.88 | 174.08 | 190.47 |
| la01 | 0.00 | 0.00 | 20.42 | 4.05 | 30.18 | 26.88 | 119.52 | 191.89 | 194.90 | 194.90 | 238.44 | 229.28 |
| la05 | 0.00 | 0.00 | 0.00 | 7.76 | 7.76 | 12.31 | 156.49 | 154.81 | 154.30 | 176.05 | 194.10 | 214.67 |
| la10 | 0.00 | 0.00 | 7.93 | 6.16 | 4.07 | 27.77 | 124.32 | 136.95 | 233.30 | 233.30 | 211.38 | 219.21 |
| la17 | 0.00 | 5.74 | 23.72 | 18.88 | 23.60 | 44.77 | 327.55 | 286.86 | 381.63 | 381.63 | 397.58 | 402.93 |
| la20 | 0.00 | 5.99 | 28.27 | 5.43 | 39.02 | 55.65 | 235.81 | 247.67 | 330.27 | 330.27 | 346.45 | 343.46 |
| la29 | 0.87 | 11.89 | 37.07 | 36.98 | 45.75 | 45.31 | 347.92 | 370.14 | 622.05 | 622.05 | 518.23 | 662.93 |
| la40 | 0.00 | 14.89 | 21.11 | 21.36 | 51.80 | 59.82 | 365.88 | 480.85 | 641.49 | 641.49 | 680.93 | 670.54 |

*(Tabla completa de las 43 instancias × 12 métodos: `experiments/results/stats/tables/e1_rpd_summary.md`/`.tex`; el subconjunto anterior ilustra los extremos y los casos donde RL supera a `fifo`. `fifo` es la PDR de menor RPD en 40/43 instancias.)*

**Tabla 6.2.2.** Prueba de Wilcoxon de rangos con signo (exacta), RL vs. mejor PDR por instancia (criterio de H1, `PreRegistration_FINAL.md` §2).

| Brazo | N | W | p-valor | Efecto ($r$) | Decisión |
|---|---|---|---|---|---|
| JSSP | 43 | 133 | **4.03×10⁻⁵** | −0.626 | **RECHAZA H1** |
| FJSP sintético (Arm A) | 43 | 0 | **1.12×10⁻⁸** | −0.871 | **RECHAZA H1** |
| FJSP Brandimarte (Arm B) | 7 | 0 | **0.0156** | −0.914 | **RECHAZA H1** |

**Lectura honesta, sin atenuar:** en los tres brazos, el efecto es significativo y **en la dirección contraria** a la reclamada por H1: la mediana de la diferencia (RPD/makespan del mejor comparador menos el de RL) es negativa en los tres casos, es decir, **RL es peor, no mejor, que la mejor alternativa clásica disponible**, con tamaños de efecto grandes ($|r|>0.6$ en todos los brazos, $|r|>0.87$ en los dos FJSP). El brazo FJSP-sintético rechaza con el margen más amplio posible ($W=0$: RL pierde en las 43 instancias sin excepción). El brazo de Brandimarte, aunque estadísticamente significativo también ($p=0.0156$), se apoya en solo $N=7$ pares, no $N=8$ — la nota de conclusión-validez correspondiente se retoma en el Capítulo 7.

**Por qué $N=7$ y no $N=8$ (añadido en v2, resolviendo el Hallazgo crítico #1 de `docs/auditorias/PreDefenseAudit.md` — explicación autocontenida, sin depender de un documento externo no anexado):** de las ocho instancias de Brandimarte con datos de referencia disponibles (`mk01`–`mk05`, `mk07` in-sample/zero-shot, más `mk06`), `mk06` se excluyó de toda comparación basada en RPD por un desajuste de datos aguas arriba detectado durante la carga de instancias: el índice de referencia del espejo `SchedulingLab/fjsp-instances` registra `mk06` con 15 máquinas (la cifra canónica de Brandimarte 1993, la que usa el óptimo publicado de 58), pero el archivo de datos real del espejo para esa instancia (`brandimarte/mk06.txt`) usa únicamente los índices de máquina 0–9 (10 máquinas distintas) y su propia línea de cabecera declara `10 10` — una inconsistencia genuina entre los metadatos y el archivo de datos del espejo, no un error transitorio de descarga (una recarga forzada reproduce el mismo archivo). En consecuencia, la instancia efectivamente cargada y evaluada **no es** la `mk06` canónica de 15 máquinas frente a la cual el óptimo de 58 es válido, por lo que su RPD calculado contra ese óptimo carece de sentido y se excluye — el makespan y la factibilidad crudos de `mk06` sí se registraron (todos los métodos corrieron exitosamente sobre ella), pero su columna de RPD queda marcada inválida, fijada por una prueba de regresión dedicada que detectará automáticamente si el espejo corrige el desajuste en el futuro. `mk01`–`mk05` y `mk07` no están afectados.

**Nota nueva en esta versión:** el brazo Brandimarte (Arm B) **ahora tiene una prueba confirmatoria formal**, no solo descriptiva como en el borrador anterior — el pipeline oficial la ejecuta junto con los otros dos brazos, restringida a {GA, CP-SAT} como comparador (sin PDR de referencia para selección de máquina en FJSP, per `PreRegistration_FINAL.md` §2). El resultado confirma, con datos reales y una prueba formal, la brecha ya descrita cualitativamente en el borrador (RL con RPD 17–193 % frente a GA/CP-SAT sobre las 7 instancias de Brandimarte).

**Tabla 6.2.3.** RPD medio de RL frente al mejor de {GA, CP-SAT}, brazos FJSP (nueva en v2, resolviendo el Hallazgo moderado 2.2 de `docs/auditorias/PreDefenseAudit.md` — traza dentro del propio Capítulo 6 las cifras que la Tabla 7.1 del Capítulo 7 ya citaba sin una tabla de origen local).

| Brazo | N | RPD medio de RL vs. mejor{GA,CP-SAT} | Instancias donde RL gana |
|---|---|---|---|
| FJSP sintético (Arm A) | 43 | 48.76 % | 0/43 |
| FJSP Brandimarte (Arm B) | 7 | 89.65 % | 0/7 |

*(Cifras recalculadas de forma independiente en la auditoría del Capítulo 7 directamente contra `e2_armA_merged.csv`/`e2_armB_mk_seed0.csv`; coinciden con el patrón cualitativo ya descrito en el texto — RL pierde en la totalidad de las instancias de ambos brazos, consistente con $W=0$ en ambas pruebas de Wilcoxon, Tabla 6.2.2.)*

**El algoritmo genético supera a RL por un margen aún mayor que el de la propia PDR comparadora (añadido en v2, resolviendo el Hallazgo moderado 2.1 de `docs/auditorias/PreDefenseAudit.md`).** La Tabla 6.2.1 incluye una columna `ga` cuyo RPD medio sobre las 43 instancias JSSP es 5.52 % — muy por debajo tanto del 24.82 % de RL como del 18.96 % de la mejor PDR por instancia (recalculado directamente contra `e1_rpd_summary.md`). El criterio pre-registrado de H1 compara RL específicamente contra la mejor PDR, no contra el GA, por una razón metodológica válida: la PDR es el piso de comparación de menor costo computacional, la pregunta más informativa sobre si la política aprendió algo útil sobre la estructura del problema. Pero la brecha frente al GA es, en términos absolutos, aún más pronunciada, y merece nombrarse explícitamente en vez de quedar visible solo en la tabla: bajo un presupuesto de tiempo de pared equiparado a CP-SAT (60 s), una búsqueda poblacional bien establecida sigue siendo, en esta comparación, sustancialmente más fuerte que una política aprendida entrenada bajo el presupuesto de cómputo de un único investigador con una GPU de consumo — una asimetría de recursos de entrenamiento-vs-búsqueda-en-tiempo-de-evaluación que el Capítulo 7 (§7.2) ya discute como explicación parcial del rechazo de H1, y que se retoma allí explícitamente para el caso del GA (§7.2).

**Veredicto H1 (síntesis).** Rechazada en los tres brazos, con significancia estadística clara en cada uno. Esta es la conclusión final del capítulo — no queda ninguna prueba de H1 pendiente.

---

## 6.3 Generalización en tamaño — H2

**Hipótesis que atiende: H2** (`PreRegistration_FINAL.md` §3).

**Nota de conciliación (reemplaza la del borrador anterior):** el pipeline oficial ejecutó H2 sobre **cuatro** *tiers* de 5 instancias cada uno (`la21`–`la25`, `la26`–`la30`, `la31`–`la35`, `la36`–`la40`), no sobre los tres *tiers* (con `la21`–`la30` agrupado en uno de $N=10$) que el borrador del 2026-07-31 infirió de una lectura literal de la fila compacta de `PreRegistration_FINAL.md` §3. **Este capítulo adopta la partición de cuatro *tiers* como la definitiva**, por ser la que el pipeline dedicado (la implementación de referencia de `StatsAnalysis_TechSpec.md`) ejecutó realmente — la conclusión cualitativa (ACEPTA H2 en todos los *tiers*) es idéntica bajo ambas particiones, así que este ajuste no cambia el veredicto, solo la granularidad con la que se reporta.

**Tabla 6.3.1.** *Tiers* de generalización ejecutados.

| Tier | Instancias | N | `num_operations` | Razón vs. tamaño de entrenamiento (100 ops) |
|---|---|---|---|---|
| 1.5× | `la21`–`la25` | 5 | 150 | 1.5× |
| 2.0× | `la26`–`la30` | 5 | 200 | 2.0× |
| 2.25× | `la36`–`la40` | 5 | 225 | 2.25× (más denso: más máquinas/trabajo) |
| 3.0× | `la31`–`la35` | 5 | 300 | 3.0× |

**Tabla 6.3.2.** RPD medio ± desviación estándar por *tier* y método (fuente: `e3_rpd_summary.md`).

| Tier | fifo | rl | mor | mwkr | spt | lpt | edd | lwkr | lor | cr |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5× (la21–la25) | 26.34±5.60 | **32.55±6.52** | 49.06±9.20 | 56.66±13.77 | 351.86±48.17 | 403.80±38.96 | 506.63±57.92 | 516.58±53.40 | 518.68±48.65 | 543.15±32.85 |
| 2.0× (la26–la30) | 25.35±7.80 | **32.92±9.97** | 52.00±10.51 | 51.36±11.39 | 389.65±73.01 | 354.21±40.24 | 558.90±67.85 | 560.58±64.26 | 545.29±44.66 | 629.85±60.72 |
| 3.0× (la31–la35) | 12.72±3.94 | **23.86±4.12** | 38.29±9.19 | 39.45±7.59 | 338.15±34.51 | 405.77±31.40 | 542.01±32.09 | 556.56±33.65 | 582.62±36.28 | 612.42±40.65 |
| 2.25× (la36–la40) | 26.83±3.30 | **32.47±7.99** | 54.54±9.09 | 54.44±7.55 | 402.24±27.75 | 414.00±40.19 | 633.36±22.12 | 633.36±22.12 | 636.47±49.65 | 661.74±33.72 |

**Nota de inclusión (añadida en v2, resolviendo el Hallazgo moderado 2.3 de `docs/auditorias/PreDefenseAudit.md`).** El GA y CP-SAT se excluyen de la clasificación conjunta de Friedman/Nemenyi de H2 — a diferencia de la Tabla 6.2.1 (H1), donde el GA sí aparece como columna informativa. La razón es específica de lo que H2 mide: la generalización de tamaño es, por construcción, una propiedad de una política **entrenada una vez y transferida sin reentrenamiento** a instancias más grandes; el GA no se entrena — se ejecuta desde cero en cada instancia con un presupuesto de tiempo de pared igualado a CP-SAT —, por lo que no tiene una noción de "tamaño de entrenamiento" que generalizar, y su desempeño en instancias grandes no es comparable, en ese sentido específico, al de una política transferida. CP-SAT permanece excluido por la misma razón ya fijada en la Sección 4.6.4 para toda comparación conjunta: no es un método par de despacho de una sola pasada ni de política aprendida, sino la referencia que fija $C^*$.

**Tabla 6.3.3.** Prueba de Friedman + Nemenyi por *tier* ($k=10$ métodos: RL + 9 PDR).

| Tier | N | p-valor (Friedman) | CD (Nemenyi, α=0.05) | Rango medio RL | Mejor PDR (rango) | ¿RL sig. peor? | Decisión |
|---|---|---|---|---|---|---|---|
| 1.5× (la21–la25) | 5 | 3.348×10⁻⁶ | 6.058 | 1.80 | fifo (1.20) | No (Δ=0.60<CD) | **ACEPTA** |
| 2.0× (la26–la30) | 5 | 1.837×10⁻⁶ | 6.058 | 1.60 | fifo (1.40) | No (Δ=0.20<CD) | **ACEPTA** |
| 3.0× (la31–la35) | 5 | 1.552×10⁻⁶ | 6.058 | 2.00 | fifo (1.00) | No (Δ=1.00<CD) | **ACEPTA** |
| 2.25× (la36–la40) | 5 | 2.923×10⁻⁶ | 6.058 | 1.80 | fifo (1.20) | No (Δ=0.60<CD) | **ACEPTA** |

**Figura 6.3.1.** Diagramas de diferencia crítica, uno por *tier* — `figures/Fig_6.3.1a_dc_tier_1.5x.png`, `Fig_6.3.1b_dc_tier_2.0x.png`, `Fig_6.3.1c_dc_tier_2.25x.png`, `Fig_6.3.1d_dc_tier_3.0x.png` (versión camera-ready en español; también disponibles en PDF vectorial junto a los PNG). En los cuatro diagramas, `fifo` y `rl` caen dentro del mismo grupo estadísticamente indistinguible (la barra horizontal de agrupamiento conecta `fifo` con `rl`, `mwkr`, `mor`, `spt` y `lpt`), separados con significancia del segundo grupo (`edd`, `lwkr`, `lor`, `cr`).

![Diagrama de diferencia crítica, tier 1.5x](figures/Fig_6.3.1a_dc_tier_1.5x.png)
![Diagrama de diferencia crítica, tier 2.0x](figures/Fig_6.3.1b_dc_tier_2.0x.png)
![Diagrama de diferencia crítica, tier 2.25x](figures/Fig_6.3.1c_dc_tier_2.25x.png)
![Diagrama de diferencia crítica, tier 3.0x](figures/Fig_6.3.1d_dc_tier_3.0x.png)


**Figura 6.3.2.** RPD de RL en función de `num_operations` — `figures/Fig_6.3.2_rpd_vs_tamano.png` (también en PDF vectorial).

![RPD de RL en función de num_operations](figures/Fig_6.3.2_rpd_vs_tamano.png)


**Veredicto H2:** en los cuatro *tiers*, Friedman rechaza decisivamente la hipótesis nula de igual desempeño, y en los cuatro, RL no es significativamente peor que la mejor PDR por la diferencia crítica de Nemenyi. **Se ACEPTA H2, global y por *tier*.**

**Matiz honesto:** el RPD medio de RL es numéricamente mayor que el de `fifo` en los cuatro *tiers* (23.9–32.9 % vs. 12.7–26.8 %) — la aceptación de H2 es una afirmación sobre *rango estadístico*, no sobre paridad práctica de desempeño. El patrón por tamaño **no es monótono**: el *tier* 3.0× tiene el RPD medio más bajo de los cuatro (23.9 %), por debajo incluso del *tier* más pequeño (1.5×, 32.6 %) — un hallazgo genuino, corroborado también por `docs/logs/HPO_ABLATION_REPORT_WEEK21.md`, no un artefacto de muestreo.

---

## 6.4 Robustez bajo perturbaciones — H3 y H4

**Hipótesis que atiende: H3** (`PreRegistration_FINAL.md` §4) **y H4** (§5, ahora ejecutada por primera vez).

### 6.4.1 H3 — robustez de la política base ($\kappa=0$)

**Tabla 6.4.1.** Índice de robustez ($RI$, media ± desviación estándar, $n=60$ pares instancia×semilla/celda), por régimen y método, incluyendo los tres valores de $\kappa$ (fuente: recálculo verificado bit-a-bit contra `e4e5e6e7_ri_summary.md` — media coincide exactamente en las 27 celdas $\kappa\in\{0.5,1.0\}$; desviación estándar calculada en este capítulo sobre las mismas filas crudas, con el mismo Cmax_nom compartido, ver nota metodológica).

**Nota metodológica importante:** el $Cmax_{\text{nom}}$ usado en el denominador de $RI$ es, para los **tres** valores de $\kappa$, el mismo makespan nominal de $\kappa=0$ registrado en E1 (no el nominal propio de cada política $\kappa>0$) — esto es intencional, no un descuido: usar un denominador compartido y fijo evita que una política cuyo propio desempeño nominal colapsó (ver §6.4.2) aparezca artificialmente "menos afectada" por la perturbación en términos relativos. Verificado por reproducción exacta de las 27 medias oficiales de `e4e5e6e7_ri_summary.md` a partir de los CSV crudos de `experiments/results/dynamic/`.

| Régimen | Nivel | Mejor PDR | RI mejor PDR | RI $\kappa=0$ | RI $\kappa=0.5$ | RI $\kappa=1.0$ |
|---|---|---|---|---|---|---|
| Averías | $A_k$=0.95 | cr | −0.547 | +0.065 (0.158) | **+0.165 (0.178)** | +0.063 (0.183) |
| Averías | $A_k$=0.90 | cr | −0.504 | +0.154 (0.209) | **+0.253 (0.248)** | +0.169 (0.223) |
| Averías | $A_k$=0.80 | cr | −0.402 | +0.410 (0.286) | **+0.519 (0.327)** | +0.417 (0.291) |
| Tiempos estocásticos | σ=0.10 | cr | −0.598 | −0.022 (0.086) | **+0.068 (0.117)** | −0.063 (0.098) |
| Tiempos estocásticos | σ=0.25 | cr | −0.583 | +0.000 (0.112) | **+0.085 (0.138)** | −0.017 (0.119) |
| Tiempos estocásticos | σ=0.50 | cr | −0.577 | +0.091 (0.188) | **+0.166 (0.200)** | +0.084 (0.184) |
| Llegadas | ρ=0.70 | lwkr | +0.271 | +1.816 (0.352) | **+2.003 (0.303)** | +1.900 (0.314) |
| Llegadas | ρ=0.85 | lwkr | +0.277 | +1.715 (0.283) | **+1.936 (0.226)** | +1.777 (0.262) |
| Llegadas | ρ=0.95 | lwkr | +0.257 | +1.654 (0.271) | **+1.924 (0.257)** | +1.748 (0.244) |

**Lectura de la Tabla 6.4.1:** en las **9 de 9** celdas, $\kappa=0.5$ tiene el RI más alto (peor) de las tres variantes de RL — la recompensa consciente de la perturbación, con este peso, **empeora** la robustez medida, de forma consistente, no solo en algunos regímenes. $\kappa=1.0$ es, en la mayoría de las celdas, ligeramente mejor que $\kappa=0$ (7/9 celdas), pero la diferencia es pequeña frente al tamaño del efecto que separa a cualquier variante de RL de la mejor PDR (`cr`/`lwkr`). Ninguna variante de RL se acerca al RI de la mejor PDR en ningún régimen.

**Figura 6.4.1.** RI vs. nivel de perturbación, un panel por eje, las cuatro series (mejor PDR, $\kappa\in\{0,0.5,1.0\}$) superpuestas — `figures/Fig_6.4.1a_ri_averias.png`, `Fig_6.4.1b_ri_tiempos_estocasticos.png`, `Fig_6.4.1c_ri_llegadas.png` (también en PDF vectorial). En los tres paneles, la serie $\kappa=0.5$ (verde) queda visiblemente por encima de $\kappa=0$ y $\kappa=1.0$, que casi se superponen; la mejor PDR (azul) permanece muy por debajo de las tres variantes de RL en todo el rango.

![RI vs. disponibilidad (averías)](figures/Fig_6.4.1a_ri_averias.png)
![RI vs. sigma (tiempos estocásticos)](figures/Fig_6.4.1b_ri_tiempos_estocasticos.png)
![RI vs. utilización (llegadas)](figures/Fig_6.4.1c_ri_llegadas.png)


**Tabla 6.4.2.** Prueba de Wilcoxon, $RI_{\kappa=0}$ vs. $RI_{\text{mejorPDR}}$, por régimen (criterio de H3).

| Régimen | Mejor PDR | Mediana(RI_PDR − RI_RL) | $p$-valor | ¿Acepta H3? |
|---|---|---|---|---|
| E4, $A_k$=0.95 | cr | −0.609 | 1.62×10⁻¹¹ | No |
| E4, $A_k$=0.90 | cr | −0.630 | 1.63×10⁻¹¹ | No |
| E4, $A_k$=0.80 | cr | −0.818 | 1.63×10⁻¹¹ | No |
| E5, σ=0.10 | cr | −0.560 | 1.63×10⁻¹¹ | No |
| E5, σ=0.25 | cr | −0.572 | 1.63×10⁻¹¹ | No |
| E5, σ=0.50 | cr | −0.635 | 1.63×10⁻¹¹ | No |
| E6, ρ=0.70 | lwkr | −1.501 | 1.63×10⁻¹¹ | No |
| E6, ρ=0.85 | lwkr | −1.424 | 1.63×10⁻¹¹ | No |
| E6, ρ=0.95 | lwkr | −1.414 | 1.63×10⁻¹¹ | No |

**Veredicto H3: 0/9 regímenes aceptan → SE RECHAZA H3** (regla conjuntiva, exige los 9). Efecto significativo en la dirección contraria en los 9 regímenes.

### 6.4.2 H4 — recompensa consciente de la perturbación (κ), ejecutada por primera vez

**Datos:** dos entrenamientos nuevos ($\kappa=0.5$, $\kappa=1.0$), cada uno un currículo de dos etapas (`ft06`→`la01`, 100k pasos/etapa, con muestreo aleatorizado del régimen de perturbación en cada reinicio de episodio sobre los 9 regímenes, arrancado en caliente desde `curriculum_stage19_la20_final.zip`). $\kappa=0$ es la **misma** política reutilizada de H3 (no un cuarto entrenamiento, per `PreRegistration_FINAL.md` §5). Evaluación: rejilla completa de 9 regímenes × 30 semillas por $\kappa$, más una pasada nominal (`--disruption none`) determinista. Cómputo: 33–36 min/etapa, ~135 min en total para ambos $\kappa$, CPU (arreglo de ubicación de dispositivo ya corregido).

**Tabla 6.4.3.** Componente (a) de H4 — Wilcoxon sobre $\Delta RI = RI_{\kappa>0} - RI_{\kappa=0}$, por régimen.

| Régimen | $\kappa$=0.5: W | $\kappa$=0.5: p | $\kappa$=0.5: $r$ | $\kappa$=0.5: ¿sig. y mejora? | $\kappa$=1.0: W | $\kappa$=1.0: p | $\kappa$=1.0: $r$ | $\kappa$=1.0: ¿sig. y mejora? |
|---|---|---|---|---|---|---|---|---|
| E4, $A_k$=0.95 | 46 | 1.885×10⁻⁹ | +0.818 | No (peor, sig.) | 648 | 0.7087 | 0.000 | No |
| E4, $A_k$=0.90 | 139 | 2.032×10⁻⁷ | +0.707 | No (peor, sig.) | 639.5 | 0.2742 | +0.147 | No |
| E4, $A_k$=0.80 | 143.5 | 6.77×10⁻⁷ | +0.689 | No (peor, sig.) | 637.5 | 0.4898 | +0.095 | No |
| E5, σ=0.10 | 71 | 5.192×10⁻¹⁰ | +0.802 | No (peor, sig.) | 379 | 7.95×10⁻⁵ | **−0.509** | **Sí** |
| E5, σ=0.25 | 349 | 3.091×10⁻⁵ | +0.538 | No (peor, sig.) | 723 | 0.1575 | −0.183 | No |
| E5, σ=0.50 | 450 | 6.19×10⁻⁴ | +0.442 | No (peor, sig.) | 862 | 0.6964 | −0.050 | No |
| E6, ρ=0.70 | 20 | 9.883×10⁻¹¹ | +0.849 | No (peor, sig.) | 117.5 | 1.196×10⁻⁷ | +0.727 | No (peor, sig.) |
| E6, ρ=0.85 | 19 | 9.388×10⁻¹¹ | +0.850 | No (peor, sig.) | 252 | 4.998×10⁻⁶ | +0.605 | No (peor, sig.) |
| E6, ρ=0.95 | 6 | 3.25×10⁻¹¹ | +0.864 | No (peor, sig.) | 132 | 2.122×10⁻⁸ | +0.736 | No (peor, sig.) |
| **Total significativos con mejora** | | | | **0/9** | | | | **1/9** |

**Lectura de la Tabla 6.4.3, sin atenuar:** para $\kappa=0.5$, el efecto es significativo en **9 de 9** regímenes, pero en la dirección contraria en los 9 — la robustez no solo no mejora, **empeora significativamente en todos los regímenes probados**. Para $\kappa=1.0$, solo 1 de 9 regímenes (tiempos estocásticos, σ=0.10) muestra una mejora significativa; los otros 8 son no significativos o empeoran. El umbral del criterio (≥5/9 regímenes con mejora significativa) no se alcanza en ningún caso.

**Tabla 6.4.4.** Componente (b) de H4 — costo nominal ($\Delta\text{RPD}_{\text{nominal}}$, margen $\delta=2$ pp).

| $\kappa$ | RPD nominal `ft06` ($\kappa=0$→$\kappa$) | RPD nominal `la01` ($\kappa=0$→$\kappa$) | $\Delta\text{RPD}_{\text{nominal}}$ | ¿Cumple margen (<2pp)? |
|---|---|---|---|---|
| 0.5 | 32.7 % → 116.4 % | 4.1 % → 132.4 % | **+106.0 pp** | **No, por ~53×** |
| 1.0 | 32.7 % → 87.3 % | 4.1 % → 42.8 % | **+46.6 pp** | **No, por ~23×** |

**Veredicto H4: RECHAZADA para $\kappa=0.5$ y para $\kappa=1.0$**, en ambos casos por **ambos** componentes simultáneamente (no un fallo marginal de uno solo) — el criterio de `PreRegistration_FINAL.md` §5 distingue explícitamente entre "sin beneficio medible" (falla (a), cumple (b)) y "robustez ganada a costo inaceptable" (cumple (a), falla (b)); aquí ninguno de los dos matices aplica — **ambos componentes fallan**, por márgenes grandes.

**Causa raíz del fallo del componente (b), diagnosticada, no solo reportada:** el envoltorio de aleatorización de dominio (`DomainRandomizedRegimeWrapper`, `train_dynamic.py`) muestrea uno de los 9 regímenes de perturbación **en cada** reinicio de episodio — las políticas $\kappa>0$ **nunca vieron un episodio sin perturbación durante el entrenamiento**. Esto no es un error de programación — el entrenamiento con aleatorización de dominio sobre exactamente la misma rejilla que se evalúa es lo que `E7_E8_FinalPlan.md` §2.2 recomendó explícitamente (Opción B, elegida deliberadamente sobre una alternativa de régimen único fijo, precisamente porque la regla de decisión de H4 exige una política generalmente consciente de la perturbación, no específica de un régimen) — pero tiene un costo real, ahora cuantificado: sin exposición a al menos algunos episodios nominales durante el entrenamiento, la competitividad nominal de la política ajustada colapsa. **Hallazgo genuino y accionable para el Capítulo 7**: mezclar un régimen "sin perturbación" en el conjunto de aleatorización de dominio, con un peso no nulo, es una corrección concreta y verificable para un esprint futuro — no intentada en este esprint, para mantener el alcance dentro de lo que `E7_E8_FinalPlan.md` efectivamente comprometió.

**Asimetría estructural reutilizada-vs-reentrenada** (señalada en `E7_E8_FinalPlan.md` §2.3: el RI de $\kappa=0$ carga el desajuste de legalidad del adaptador, mientras que $\kappa>0$ entrena nativamente contra `DynamicJSSPEnv` y no lo hereda) está presente en estos datos pero **queda dominada** por el efecto, mucho mayor, del colapso nominal por aleatorización de dominio — la tercera corrida opcional "$\kappa=0$ reentrenada" que habría aislado esta asimetría no se ejecutó este esprint (despriorizada per la guía explícita de esa misma sección); se declara aquí como limitación no resuelta, no oculta.

---

## 6.5 Tiempo de decisión — H5

**Hipótesis que atiende: H5** (`PreRegistration_FINAL.md` §6).

**Datos:** remedición limpia, las 43 instancias, 1 semilla determinista + 4 estocásticas/instancia, CP-SAT una vez, en una máquina de 8 núcleos CPU sin *build* CUDA (`torch.cuda.is_available()` es `False` incondicionalmente en esta máquina — per la Enmienda 2 de `PreRegistration_FINAL.md`, "hardware de entrenamiento" y "solo CPU" colapsan en una única medición; no existe una cifra GPU separada que reportar).

**Tabla 6.5.1.** Tiempo de decisión (ms/época, mediana) — subconjunto representativo por tamaño (tabla completa de 43 instancias: `experiments/results/stats/tables/e8_latency_summary.md`).

| Instancia | num_operations | ms/decisión (RL) | ms/decisión (CP-SAT) | Razón CP-SAT/RL | ¿Pasa umbral 1 %? |
|---|---|---|---|---|---|
| ft06 | 36 | 4.972 | 0.928 | 0.19× | No |
| la05 | 50 | 3.708 | 0.446 | 0.12× | No |
| la20 | 100 | 4.666 | 4.147 | 0.89× | No |
| ft10 | 100 | 4.532 | 82.848 | 18.28× | No |
| la21 | 150 | 9.594 | 400.453 | 41.74× | No |
| la29 | 200 | 11.121 | 300.407 | 27.01× | No |
| la38 | 225 | 10.606 | 266.913 | 25.17× | No |
| **la40** | **225** | **18.085** | **32074.086** | **1773.49×** | **Sí — único caso** |
| la31 | 300 | 8.952 | 6.392 | 0.71× | No |

**Tabla 6.5.2.** Prueba (umbral, no de hipótesis, per D7): mediana(ms/decisión_RL) < 1 % × mediana(ms/decisión_CP-SAT).

| N total | N que pasan | N que fallan | Decisión |
|---|---|---|---|
| 43 | 1 (`la40`) | 42 | **RECHAZA H5** (se exige que pasen las 43) |

**Figura 6.5.1.** Razón CP-SAT/RL (ms/decisión) vs. tamaño de instancia, escala log — `figures/Fig_6.5.1_latencia_vs_tamano.png` (también en PDF vectorial). El panel superior muestra RL (azul) esencialmente plano entre 4–18 ms/decisión en todo el rango de tamaños, consistente con el costo fijo de un *forward pass* de una red neuronal que no escala con el tamaño de instancia; CP-SAT (naranja) es errático, dominado por si la instancia se resuelve rápido (facilidad estructural) o agota el presupuesto de 60 s (dureza estructural). El panel inferior (razón) muestra un pico aislado y extremo exactamente en `la40`.

![Razón CP-SAT/RL vs. tamaño de instancia](figures/Fig_6.5.1_latencia_vs_tamano.png)


### 6.5.1 Anomalía detectada: la medición de CP-SAT en `la40` no es fiable, y el veredicto no depende de ella

**Esta es la observación más importante de §6.5 y debe leerse antes de citar la cifra de "1773×" en cualquier otro capítulo.** Verificado por lectura directa del CSV crudo (`experiments/results/week25/e8_clean/e1_e8_clean_seed0_baselines.csv`): el tiempo de pared de CP-SAT en `la40` es **7216.67 segundos** (~2 horas), frente a **60.0554 s** en `la38` — una instancia de tamaño idéntico (15×15, 225 operaciones), estado `FEASIBLE` idéntico (ninguna de las dos prueba optimalidad), y presupuesto de tiempo nominal idéntico (60 s, `time_limit_s` por defecto de `baselines/cpsat.py`). De las 43 instancias del corte, `la40` es la única que excede los 100 segundos; las siguientes tres más lentas (`la29`, `la21`, `la38`) están todas en el rango 60.05–60.08 s, exactamente donde se espera que caigan las instancias que agotan el presupuesto de 60 s del solver. **Una desviación de ~120× frente al presupuesto configurado, en una única instancia, sin ningún error o advertencia en el registro de ejecución (`run.log`) ni en `stats_manifest.json`, es estadísticamente y operacionalmente anómala** — no hay evidencia de que `la40` sea intrínsecamente ~120× más difícil que `la29`/`la21`/`la38` (todas del mismo orden de tamaño); es mucho más probable que se trate de un artefacto de medición (contención de CPU, *swap*, u otra interferencia del sistema operativo durante la corrida, pese al protocolo de "máquina ociosa"; alternativamente, un fallo puntual del mecanismo `max_time_in_seconds` de OR-Tools en ese proceso específico) que una propiedad genuina del solver sobre esa instancia.

**Consecuencia para el veredicto de H5 — no cambia, se vuelve más decisivo, no menos:** `la40` es el **único** de los 43 casos que pasa el umbral del 1 %, y ese único caso es precisamente el que muestra la medición anómala. Una lectura de sensibilidad, sustituyendo el tiempo de pared de `la40` por el valor observado en `la38` (misma clase de tamaño/dificultad, 60.0554 s → 266.91 ms/decisión), da una razón CP-SAT/RL de **≈14.76×** para `la40` — muy por debajo del umbral de 100× exigido. **Bajo esta corrección de sensibilidad, el resultado sería 0/43 en vez de 1/43 — H5 se rechaza de todas formas, con o sin este dato, y el rechazo se vuelve más limpio, no está condicionado a una única medición atípica.** Se reporta el número crudo (1773×, 1/43) como el resultado literal del corte de datos actual, exactamente como aparece en el CSV, pero **se recomienda explícitamente no citar la cifra de "1773×" ni la afirmación "RL gana decisivamente en la instancia más grande" en el Capítulo 7 o en cualquier resumen ejecutivo sin esta advertencia adjunta**, y se recomienda una re-medición aislada de `la40` (una única instancia, computacionalmente trivial) antes de la sustentación para resolver la anomalía de forma definitiva en vez de reportarla con una salvedad permanente.

**Veredicto H5, con o sin la corrección de sensibilidad: RECHAZADA.** La razón CP-SAT/RL sí se ensancha con el tamaño de instancia en el sentido cualitativo que D7 pedía verificar (de razones <1× en instancias pequeñas/fáciles a razones de docenas en instancias grandes/difíciles como `la21`/`la29`/`la38`), pero el umbral estricto "para toda instancia" del criterio congelado no se cumple en ninguna lectura honesta de los datos. **Interpretación correcta, sin el artefacto de `la40`:** para instancias pequeñas y fáciles, la resolución casi instantánea de CP-SAT hace que la comparación por-decisión sea estructuralmente desfavorable para cualquier política secuencial (el "costo por decisión" de CP-SAT es un artefacto de dividir un total ya minúsculo, no una medida realista de dificultad por decisión); para instancias grandes y difíciles (`la21`, `la29`, `la38`, y otras que agotan el presupuesto de 60 s), RL es 1–2 órdenes de magnitud más rápido por decisión, un margen genuino y sustancial, aunque no alcance el umbral extremo del 1 % que D7 fijó.

---

## Síntesis del capítulo: estado final de H1–H5

**Tabla 6.6.1.** Síntesis del capítulo: estado final de H1–H5.

| Hipótesis | Sección | Decisión | ¿Depende de datos con reservas? |
|---|---|---|---|
| H1 (JSSP) | 6.2 | RECHAZADA | No |
| H1 (FJSP sintético) | 6.2 | RECHAZADA | No |
| H1 (FJSP Brandimarte) | 6.2 | RECHAZADA | Sí — N=7, y `mk06` excluido por Finding #2 (mismatch de datos aguas arriba, ver `docs/logs/EXPERIMENT_LOG_WEEK24.md`) |
| H2 (generalización) | 6.3 | ACEPTADA | No |
| H3 (robustez, κ=0) | 6.4.1 | RECHAZADA (0/9) | No |
| H4 (κ=0.5) | 6.4.2 | RECHAZADA | Sí — un solo entrenamiento/semilla por κ (ver Capítulo 7, M10) |
| H4 (κ=1.0) | 6.4.2 | RECHAZADA | Sí — ídem |
| H5 (latencia) | 6.5 | RECHAZADA | Sí — un punto de dato anómalo (`la40`); no cambia el veredicto |

**Cuatro de las cinco hipótesis se rechazan; H2 es la única aceptada.** El Capítulo 7 interpreta este patrón a la luz de la literatura del Capítulo 3 y discute explícitamente las amenazas a la validez (interna, externa, de constructo y de conclusión) que condicionan cuánto peso puede sostener cada uno de estos veredictos.
