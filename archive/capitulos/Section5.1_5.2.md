## 5.1 Matriz de factores y diseño experimental

La Parte D4 del marco metodológico fija la estructura de la matriz experimental (factores, niveles y métrica principal por experimento) a un nivel todavía abstracto — tamaños de instancia descritos como "train 6×6/10×10 → test 15×15, 20×15, 30×20, ta grandes", sin comprometerse a nombres de archivo concretos. Esta sección instancia esa matriz con los nombres exactos y los presupuestos de tiempo de pared que los resultados de línea base de la Semana 9 (`BASELINES_RESULTS_WEEK9.md`, analizados en `../docs/planificacion/BaselineAnalysis.md`) permiten fijar con evidencia, en lugar de una estimación a priori.

### 5.1.1 Bloques de instancias por tamaño

**Tabla 5.1.1.** Bloques de instancias, tamaño y estado de validación empírica.

| Bloque | Instancias | Tamaño (trabajos × máquinas) | Operaciones totales | Estado |
|---|---|---|---:|---|
| Entrenamiento (tamaño pequeño) | `ft06`; `la01`–`la05` | 6×6; 10×5 | 36; 50 | **Validado** (Semana 9): CP-SAT prueba el óptimo en ≤0.17 s; las 9 PDR corren en ≤0.012 s. |
| Generalización moderada | `ft10`, `ft20`; `la06`–`la10` | 10×10, 20×5; 15×5 | 100, 100; 75 | **Validado** (Semana 9): CP-SAT prueba el óptimo en ≤3.79 s (caso más costoso, `ft10`); PDR ≤0.027 s. |
| Generalización extendida (planificada) | `la11`–`la40` (Lawrence 1984) | hasta 30×15 | hasta ~450 | **No ejecutado.** Tamaño intermedio entre el bloque anterior y `ta`; se ejecutará antes de comprometer presupuesto de cómputo al bloque siguiente. |
| Generalización de gran escala (planificada, condicionada al presupuesto) | `ta01`+ (Taillard 1993), hasta 100×20 | hasta 100×20 | hasta 2000 | **No ejecutado.** Es el bloque que más presiona el supuesto de una única GPU (Sección 1.5); su inclusión completa (80 instancias) se revisará una vez medido el costo real de CP-SAT y del GA en el bloque intermedio. |

El bloque de "entrenamiento" y el de "generalización moderada" son exactamente las 13 instancias ya evaluadas (Semana 9); los dos bloques restantes son una extrapolación planificada, no ejecutada, y se marcan como tal para no confundir una proyección con un resultado.

### 5.1.2 Presupuesto de tiempo de pared por banda de tamaño (CP-SAT y GA)

Siguiendo el principio ya fijado en la Sección 4.6 ("limitar CP-SAT y GA por tiempo de pared para comparaciones justas", `../docs/planificacion/Step 4 Thesis.md` §5), el límite de tiempo no es un valor único para toda la matriz, sino una función escalonada del tamaño de instancia, calibrada con los datos de la Semana 9 en las dos primeras bandas y estimada por extrapolación conservadora en las dos restantes:

**Tabla 5.1.2.** Presupuesto de tiempo de pared por banda de tamaño.

| Banda | Operaciones | Límite CP-SAT | Margen observado / estimado | Base |
|---|---|---:|---|---|
| Pequeña | ≤50 | 60 s | Peor caso observado: 0.166 s (`la04`) → margen ≈360× | Empírico (Semana 9) |
| Moderada | 51–100 | 60 s | Peor caso observado: 3.79 s (`ft10`) → margen ≈16× | Empírico (Semana 9) |
| Intermedia | 101–450 | 300 s (propuesto) | Sin medir; extrapolación conservadora, **a validar** en el bloque `la11`–`la40` antes de fijarse como definitivo | Estimado |
| Grande | 451–2000 | 600–900 s (propuesto) | Sin medir; en este rango es razonable esperar que CP-SAT **no** pruebe optimalidad dentro del límite en varias instancias — el protocolo debe reportar `best_bound` y `status="FEASIBLE"` explícitamente en lugar de asumir `status="OPTIMAL"` (ya soportado por `CPSATResult`/`BaselineResult`, Sección 4.6.4) | Estimado |

El GA (una vez implementado, Sección 4.6.2) usa el mismo límite de tiempo de pared que CP-SAT en la misma instancia, por el mismo argumento de equidad — no un número de generaciones fijo independiente del tamaño.

**Nota metodológica.** Las bandas "intermedia" y "grande" son, en el estado actual, una hipótesis de planificación, no un dato medido; el margen de 16×–360× observado en las dos bandas ya ejecutadas sugiere que 60 s es conservador para instancias de hasta 100 operaciones, pero **no** debe extrapolarse linealmente al rango de 2000 operaciones de `ta` sin una medición dedicada, dado que el tiempo de resolución de CP-SAT no escala de forma lineal con el número de operaciones. La ejecución del bloque `la11`–`la40` (banda intermedia) es, por tanto, un prerequisito para fijar con evidencia el límite de la banda grande, no un paso que pueda omitirse.

### 5.1.3 Matriz de factores y experimentos (Parte D4, instanciada)

**Tabla 5.1.3.** Matriz experimental con instancias, réplicas y métrica principal ya concretadas para los experimentos que dependen del bloque de instancias validado esta semana.

| Exp. | Hipótesis | Factores | Niveles | Instancias (concretadas) | Réplicas | Métrica principal |
|---|---|---|---|---|---|---|
| E0 Validación DES | modelo válido | casos analíticos + óptimos conocidos | M/M/1; `ft06`=55, `ft10`=930, `ft20`=1165 | `ft06`, `ft10`, `ft20` | 10 | error vs. teórico |
| E1 Estático JSSP | H1 | benchmark | `ft`, `la` (validado); `ta` (pendiente) | `ft06`,`ft10`,`ft20`,`la01`–`la10` (ejecutado); `la11`–`la40`,`ta01`+ (pendiente) | 5 (política estocástica) | RPD vs. CP-SAT; **piso empírico = FIFO (RPD media 17.80 %), no MWKR** (§5.1.4) |
| E2 Estático FJSP | H1 | benchmark | MK01–MK10, Hurink | — (bloqueado: carga de instancias FJSP fuera de alcance de este sprint, `../docs/especificaciones/Baselines_TechSpec.md` §0) | 5 | gap vs. CP-SAT |
| E3 Generalización tamaño | H2 | tamaño train→test | train `ft06`/`la01`–`la05` → test `ft10`/`ft20`/`la06`–`la10` (validado); `la11`–`la40`/`ta` grandes (pendiente) | ídem | 5 | gap por tamaño |
| E4–E6 Robustez | H3 | $A_k$, $s$, $\rho$ | Sección 4.5 | `ft06` (calibración ya validada, `VALIDATION_WEEK6.md`); extensión a `la01`–`la10` pendiente | 30 (sujeto al cálculo de réplicas de `../docs/planificacion/VandV_Protocol.md` §2) | $\mathbb{E}[C_{\max}]$, $RI$ |
| E7 Recompensa consciente | H4 | $\kappa$ | {0, 0.5, 1.0} | bloque de entrenamiento | 30 | robustez vs. optimalidad |
| E8 Tiempo de decisión | H5 | tamaño | todos los bloques ejecutados | todos | — | ms/decisión |

### 5.1.4 Ajuste del pre-registro (D7, H1) a la evidencia empírica

D7 fija H1 como "la política supera a la mejor PDR en RPD medio sobre `ta` y MK", sin nombrar la regla de antemano — una redacción deliberadamente agnóstica, correcta como pre-registro. Lo que esta sección fija, **antes de generar cualquier resultado de la política DRL** (siguiendo el propio principio anti-HARKing de `../docs/planificacion/Step 4 Thesis.md`, "congelar el pre-registro antes de generar cualquier resultado"), es que la "mejor PDR" medida empíricamente en el bloque de instancias ya ejecutado es **FIFO** (RPD media 17.80 %, mejor en 11 de 13 instancias), no MWKR como el desarrollo teórico de la Parte B6 podría sugerir por defecto (`../docs/planificacion/BaselineAnalysis.md`, §1). El criterio de aceptación de H1 se reescribe, por tanto, en términos operacionales concretos: *aceptar H1 si la política DRL alcanza una RPD media inferior a la de FIFO, con significancia de Wilcoxon $p<0.05$, sobre el mismo conjunto de instancias*; superar a MOR/MWKR (RPD media 35–38 %) se registra como un segundo umbral, más exigente en el sentido de la literatura general de JSSP pero empíricamente más fácil de alcanzar que FIFO en este conjunto de instancias.

---

## 5.2 Protocolo de evaluación y métricas

### 5.2.1 Cota de referencia $C^*$

Siguiendo la Parte D6, $C^*$ (el denominador de la brecha de optimalidad) es el óptimo publicado cuando existe, o la solución de CP-SAT con límite de tiempo en su defecto. Esta sección fija cómo se resuelve esa disyuntiva de forma concreta, banda por banda (Tabla 5.1.2):

- **Bandas pequeña y moderada (validadas):** $C^*$ = óptimo **probado** por CP-SAT (`status="OPTIMAL"`), que además coincide exactamente con los óptimos publicados en OR-Library para las tres instancias `ft` (55, 930, 1165) — la coincidencia exacta es en sí misma la confirmación empírica de E0 (validación del parser de instancias y del modelo de intervalos de CP-SAT contra una referencia externa, no solo autoconsistente).
- **Bandas intermedia y grande (pendientes):** $C^*$ = óptimo probado si CP-SAT lo alcanza dentro del límite de tiempo de la banda; en caso contrario, $C^*$ = `best_bound` (la mejor cota inferior probada al agotarse el tiempo), y toda tabla de resultados que dependa de esa banda debe reportar explícitamente si $C^*$ proviene de una solución óptima o de una cota — nunca tratar ambos casos como intercambiables en el cálculo de RPD, dado que un RPD calculado contra una cota no probada subestima sistemáticamente la brecha real.

### 5.2.2 Métricas (fórmulas exactas, Parte D6)

$$\text{RPD}=100\cdot\frac{C_{\max}^{\text{método}}-C^{*}}{C^{*}}.$$

Sobre el bloque ya ejecutado, el rango observado de RPD por método (Tabla 5.1.1/`../docs/planificacion/BaselineAnalysis.md` §2) sitúa el **piso** operativo en FIFO (17.80 % de media) y el **techo** en CP-SAT (0 % por construcción, no un método par — Sección 4.6.4); las cuatro reglas más débiles (LWKR, EDD, LOR, CR, 196–236 % de media) se conservan en la tabla comparativa por completitud del conjunto de nueve reglas exigido por la Parte B6, no porque superarlas sea, por sí solo, un resultado informativo.

**Tardanza** ($T_j=\max(0,C_j-d_j)$, $\bar T$, $T_{\max}$, $\sum U_j$): requiere fechas de entrega $d_j$, que — como ya se documentó en la Sección 4.6.1 — no existen en las instancias `ft`/`la`/`ta` y deben sintetizarse (regla TWK, `../docs/especificaciones/Baselines_TechSpec.md` §4). Las corridas de la Semana 9 no reportan tardanza por esta razón (EDD/CR sí se ejecutaron, usando la fecha sintetizada solo para el índice de prioridad, no para reportar $T_j$); el reporte de tardanza como métrica de resultado se activa a partir de que el Capítulo 6 decida el factor de holgura $k$ a usar de forma consistente en todo el documento (Sección 4.6.1, decisión señalada).

**Índice de robustez** ($RI=\frac{\mathbb{E}[C_{\max}^{\text{pert}}]-C_{\max}^{\text{nom}}}{C_{\max}^{\text{nom}}}$): no aplica al bloque ya ejecutado (perturbaciones desactivadas, evaluación puramente estática); se activa en los experimentos E4–E6 sobre el protocolo de réplicas y bondad de ajuste ya especificado en `../docs/planificacion/VandV_Protocol.md`.

**Tiempo de decisión:** reportado ya en la Semana 9 como tiempo de pared total por método e instancia (columna `wall-clock (s)` de `BASELINES_RESULTS_WEEK9.md`), no como milisegundos por época individual — las PDR corren en 5–27 ms totales (todas las épocas de un episodio incluidas) y CP-SAT en 29 ms–3.79 s. Para E8, este protocolo exige reportar **ambas** granularidades por separado una vez la política DRL esté disponible: milisegundos por decisión individual (comparable entre PDR y política) y segundos de pared totales por instancia (comparable entre GA/CP-SAT y el episodio completo de la política) — nunca promediarlas entre sí, dado que no son la misma unidad de comparación (ya señalado en la Sección 4.6.4).

### 5.2.3 Procedimiento estadístico

Sobre las $N=13$ instancias ya evaluadas (a expandirse conforme se ejecuten los bloques pendientes de la Tabla 5.1.1, lo que fortalece la potencia del procedimiento sin cambiar su forma):

- **Comparación pareada** (p. ej., política DRL vs. FIFO específicamente, para el criterio de aceptación de H1, §5.1.4): prueba de **Wilcoxon de rangos con signo**, $\alpha=0.05$, reportando el estadístico $W$, el $p$-valor y el tamaño de efecto $r=Z/\sqrt{N}$.
- **Comparación conjunta de métodos** (las nueve PDR, el GA, y la política DRL — $k$ métodos sobre $N$ instancias): prueba de **Friedman** sobre los rangos medios; si rechaza, post-hoc de **Nemenyi** con diferencia crítica $CD=q_\alpha\sqrt{k(k+1)/(6N)}$ y diagrama de diferencia crítica (Demšar 2006).
- **CP-SAT queda fuera de esta clasificación conjunta como método par**, exactamente por la razón ya señalada en la Sección 4.6.4: no es comparable en paradigma ni en presupuesto de tiempo a un despacho de una sola pasada (PDR) o a un episodio de política aprendida; su rol es exclusivamente el de fijar $C^*$ (§5.2.1), no el de competir en el ranking de Friedman/Nemenyi. El GA, cuando esté disponible, sí entra en esa clasificación conjunta, porque su presupuesto de tiempo de pared es explícitamente igualado al de CP-SAT (Sección 4.6.2) precisamente para poder compararlo con la política DRL bajo el mismo criterio de equidad — pero se reporta en una fila separada de "métodos con presupuesto de tiempo configurable" para no mezclar su interpretación con la de una PDR de una sola pasada.
- Reportar siempre el **tamaño de efecto** (r de Wilcoxon o delta de Cliff), no solo el $p$-valor, siguiendo el mismo criterio ya fijado en D6.
