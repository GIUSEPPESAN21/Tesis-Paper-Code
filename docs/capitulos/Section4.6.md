## 4.6 Líneas base (reglas de despacho, algoritmo genético y CP-SAT)

Toda afirmación de que la política aprendida "generaliza" o es "robusta" carece de contenido sin un piso y un techo de comparación bien definidos. Esta sección formaliza los tres tipos de línea base especificados en las Partes B6 y B7 del marco metodológico — reglas de despacho por prioridad (PDR), un algoritmo genético (GA) y un solver exacto (CP-SAT) — y describe cómo los tres, junto con la política DRL, se integran en un único arnés de evaluación que produce métricas comparables (Sección D6) bajo el mismo protocolo de reproducibilidad (semillas, presupuesto de tiempo de pared) usado en el resto de la tesis.

### 4.6.1 Reglas de despacho por prioridad (PDR)

Una PDR asigna, en cada época de decisión, un índice de prioridad a cada operación de la frontera (la operación inmediatamente dispachable de cada trabajo no terminado) y despacha la de mejor índice. Es la línea base más barata computacionalmente ($O(1)$–$O(|\text{cola}|)$ por decisión) y, por ello, el piso de comparación obligatorio: una política que no supera a MWKR no ha aprendido nada útil sobre la estructura del problema.

La Tabla 4.6.1 formaliza, con la notación de la Parte B6, las nueve reglas requeridas por el diseño experimental. Para la operación candidata $O_{j,h}$ en el instante $t$, con $p_{j,h}$ su tiempo de proceso, $\sum_{h'\ge h}p_{j,h'}$ el trabajo remanente de su trabajo, y $n_j$ el número total de operaciones del trabajo:

**Tabla 4.6.1.** Reglas de despacho por prioridad y su regla de decisión.

| Regla | Nombre | Índice a minimizar (o maximizar, indicado) |
|---|---|---|
| SPT | Shortest Processing Time | $\min\; p_{j,h}$ |
| LPT | Longest Processing Time | $\max\; p_{j,h}$ |
| MWKR | Most Work Remaining | $\max\; \sum_{h'\ge h} p_{j,h'}$ |
| LWKR | Least Work Remaining | $\min\; \sum_{h'\ge h} p_{j,h'}$ |
| MOR | Most Operations Remaining | $\max\; (n_j-h+1)$ |
| LOR | Least Operations Remaining | $\min\; (n_j-h+1)$ |
| FIFO/FCFS | First In First Out | $\min\;$ tiempo de llegada a la cola de listas |
| EDD | Earliest Due Date | $\min\; d_j$ |
| CR | Critical Ratio | $\min\; \dfrac{d_j-t}{\sum_{h'\ge h}p_{j,h'}}$ |

**Desempate.** Los empates son frecuentes en las familias `ft`/`la`/`ta` (muchas duraciones idénticas) y un desempate determinista (p. ej., `argmin` sobre el primer índice) sesga sistemáticamente el resultado medido hacia los trabajos de índice bajo. El arnés de evaluación rompe todo empate de forma aleatoria y uniforme entre los candidatos empatados, con una semilla propia e independiente de la semilla del episodio, siguiendo la práctica ya establecida en el módulo de reglas existente.

**La brecha entre B6 y los datos de benchmark disponibles.** B6 define EDD y CR en términos de una fecha de entrega $d_j$ por trabajo, parámetro que forma parte de la formulación MILP general (Parte B2.1) pero que **no existe** en las instancias `ft`/`la`/`ta` empleadas en esta tesis: son benchmarks de minimización pura de makespan, sin fechas de entrega publicadas. Aplicar EDD/CR tal cual exige, por tanto, **sintetizar** $d_j$ a partir de los datos disponibles de la instancia — una decisión de modelado no dictada por B6 que se señala aquí explícitamente, siguiendo la misma cultura de transparencia metodológica adoptada en la Sección 4.5 para los generadores de perturbación. La especificación técnica complementaria (`../docs/especificaciones/Baselines_TechSpec.md`) fija el método concreto (regla de Contenido de Trabajo Total, $d_j = k\cdot\sum_h p_{j,h}$ para una holgura $k>1$ configurable) y lo deja señalado como una asunción a justificar en la discusión de resultados, no como un valor derivado del marco original.

**Estado de implementación y seguridad frente a llegadas dinámicas.** El módulo existente (`baselines/dispatching_rules.py`) ya implementa `random`, `spt`, `lpt`, `fifo` y `mwkr`, además de una regla equivalente a MOR bajo el nombre `mopnr`; ese módulo quedó **congelado** durante el sprint de la Sección 4.5 (decisión señalada #2 de `VALIDATION_WEEK6.md`) porque tres de sus reglas (`spt`, `lpt`, `mwkr`) indexan directamente los arreglos de forma fija de la instancia base y no cubren operaciones de trabajos llegados dinámicamente, mientras que `random`, `fifo` y `mopnr` sí son seguras frente a llegadas porque solo usan `action_masks()`, `job_ready_time` y aritmética pura de `job_of`/`position_of`. Las cuatro reglas nuevas requeridas por esta sección (LWKR, LOR, EDD, CR) deben implementarse siguiendo el patrón *arrival-safe* desde el inicio — sin indexar arreglos de forma fija — para que el arnés unificado de la Sección 4.6.4 pueda ejercitarlas indistintamente sobre el entorno estático y el dinámico sin la restricción que hoy limita a `spt`/`lpt`/`mwkr`. El detalle de implementación se especifica en `../docs/especificaciones/Baselines_TechSpec.md`.

### 4.6.2 Algoritmo genético (GA)

El GA es la línea base metaheurística de referencia para el FJSP (Parte B7) y explora, mediante búsqueda poblacional, una región del espacio de cronogramas mucho más amplia que cualquier PDR de una sola pasada, a costa de un presupuesto de cómputo mayor y sin garantía de optimalidad.

**Codificación cromosómica (dos vectores, per B7).** Un individuo se representa mediante:

1. **Vector de secuencia** $\pi$: una permutación con repetición de los índices de trabajo, de longitud $\sum_j n_j$, donde la $r$-ésima aparición del índice de trabajo $j$ en $\pi$ designa la $r$-ésima operación (en orden tecnológico) de ese trabajo. Esta codificación garantiza por construcción que toda permutación decodificada respeta la precedencia intra-trabajo, sin necesidad de un operador de reparación.
2. **Vector de asignación de máquina** $\mu$, de longitud $\sum_j n_j$: para el JSSP clásico (enrutamiento fijo, como en `ft`/`la`) este vector es constante y se omite del cromosoma; se activa únicamente para la extensión FJSP, donde codifica, para cada operación, cuál de sus máquinas elegibles $M_{j,h}$ se selecciona.

**Decodificación.** $\pi$ se recorre de izquierda a derecha; cada operación se despacha en la máquina indicada por $\mu$ (o la única disponible, en JSSP) en el instante más temprano factible dado el estado de la máquina y de su trabajo — la misma regla de *list scheduling* que emplea el entorno Gymnasium (Sección 4.3.2), lo que garantiza que todo cromosoma decodifica a un cronograma semiactivo factible y permite reutilizar `envs.disjunctive_graph.is_schedule_feasible` como verificación independiente del decodificador.

**Operadores de cruce.**
- **POX (Precedence Operation Crossover)** sobre el vector de secuencia $\pi$: se particiona aleatoriamente el conjunto de trabajos en dos subconjuntos $J_1,J_2$; el descendiente hereda de un progenitor, en su posición original, todas las operaciones de trabajos en $J_1$, y completa las posiciones restantes con las operaciones de trabajos en $J_2$ en el orden en que aparecen en el otro progenitor. Preserva automáticamente la factibilidad de precedencia de ambos progenitores.
- **Cruce uniforme** sobre el vector de asignación $\mu$ (solo relevante en FJSP): cada gen del descendiente se hereda de uno u otro progenitor con probabilidad $0.5$, independientemente por operación.

**Operadores de mutación.**
- **Intercambio (swap)** sobre $\pi$: se seleccionan dos posiciones al azar y se intercambian sus valores, perturbando el orden de despacho sin invalidar la codificación.
- **Reasignación de máquina** sobre $\mu$ (FJSP): se selecciona una operación al azar y se reemplaza su máquina asignada por otra máquina elegible distinta, uniformemente.

**Selección.** Selección por torneo (tamaño de torneo $k_t$, configurable, valor por defecto recomendado $k_t=3$) para la mayor parte de la población, combinada con **elitismo** (los $e$ mejores individuos de la generación pasan intactos a la siguiente, sin cruce ni mutación) para garantizar que el mejor makespan encontrado nunca empeora entre generaciones.

**Tamaño de población y número de generaciones bajo presupuesto de estudiante único.** El alcance metodológico de la tesis (Sección 1.5) fija explícitamente una única GPU y un único investigador con un horizonte de nueve meses, y el propio marco experimental (Parte D4) exige limitar CP-SAT y GA por tiempo de pared para que la comparación con la política DRL sea justa — no por número de generaciones o de nodos de búsqueda, que no son comparables entre paradigmas. En consecuencia, esta sección recomienda fijar el **presupuesto por tiempo de pared** como parámetro primario (idéntico al usado para CP-SAT en la misma instancia, Sección 4.6.3) y usar tamaño de población y número de generaciones solo como una configuración razonable dentro de ese presupuesto, no como el criterio de parada: población de 50–100 individuos y hasta unas pocas centenas de generaciones para instancias del tamaño de entrenamiento (6×6, 10×10), reduciendo proporcionalmente para instancias grandes donde cada evaluación de fitness (una decodificación completa) es más costosa. El valor exacto se calibra empíricamente contra el presupuesto de tiempo de pared elegido, no se fija a priori; el detalle de implementación (incluyendo el criterio de convergencia temprana) se especifica en `../docs/especificaciones/Baselines_TechSpec.md`.

**Función de aptitud (fitness).** $\text{fitness}(\pi,\mu)=-C_{\max}(\pi,\mu)$ (minimizar makespan), calculada por decodificación completa del cromosoma; ningún atajo de evaluación parcial se emplea, dado que el tamaño de las instancias objetivo (hasta 100×20 en la matriz de generalización, Parte D4/E3) mantiene la decodificación completa dentro de un costo computacional trivial frente al entrenamiento DRL.

### 4.6.3 Solver exacto: CP-SAT (Google OR-Tools)

CP-SAT es el **techo de calidad** de la tesis: para instancias pequeñas y medianas prueba optimalidad, lo que permite calcular la brecha de optimalidad exacta (RPD, Sección D6) de cualquier heurística o política aprendida, en lugar de compararla contra una tabla de la literatura potencialmente obtenida bajo un protocolo distinto.

**Formulación por variables de intervalo (JSSP, ya implementada).** Cada operación $O_{j,h}$ se modela como una variable de intervalo `IntervalVar(start, duration, end)`, con `duration` fijo e igual a $p_{j,h}$. Dos familias de restricciones bastan para el JSSP clásico:
- **`AddNoOverlap`** sobre el conjunto de intervalos asignados a cada máquina $k$, que impone que las operaciones que comparten máquina no se solapen — la traducción directa de las restricciones disyuntivas de la Sección B2.1 (arcos disyuntivos orientados) a la API de CP-SAT, sin necesidad de las variables binarias $y_{j,h,j',h',k}$ ni de la constante *big-M* $V$ de la formulación MILP explícita.
- **Restricciones de precedencia** por trabajo: $\text{start}(O_{j,h+1})\ge \text{end}(O_{j,h})$, encadenadas para cada trabajo.

El objetivo minimiza $C_{\max}=\max_j\,\text{end}(O_{j,n_j})$, modelado con una variable auxiliar `makespan` ligada por `AddMaxEquality` a los tiempos de finalización de la última operación de cada trabajo.

**Extensión a FJSP: intervalos opcionales.** Cuando una operación tiene más de una máquina elegible ($|M_{j,h}|>1$), cada alternativa $(O_{j,h},k)$ con $k\in M_{j,h}$ se modela como un **`OptionalIntervalVar`**, ligado a una variable booleana de presencia $x_{j,h,k}$, con la restricción $\sum_{k\in M_{j,h}} x_{j,h,k}=1$ (exactamente una alternativa está presente) — la traducción directa de la restricción de asignación única de la Sección B2.1. `AddNoOverlap` se aplica entonces sobre el conjunto de intervalos opcionales de cada máquina (CP-SAT ignora automáticamente los intervalos cuya variable de presencia es falsa), evitando tener que codificar manualmente las variables binarias $y$ de la formulación disyuntiva explícita.

**`AddCumulative` como alternativa para recursos agrupados.** Aunque no es necesaria para el JSSP/FJSP de capacidad unitaria por máquina (una operación a la vez), esta tesis reserva `AddCumulative` (capacidad $=1$ por máquina, demanda $=1$ por intervalo) como una formulación funcionalmente equivalente a `AddNoOverlap` para el caso en que un experimento futuro agrupe máquinas idénticas en un único recurso de capacidad $>1$ (fuera del alcance de esta tesis, pero mencionado en `../docs/especificaciones/Baselines_TechSpec.md` como punto de extensión documentado).

**Presupuesto de tiempo de pared y estatus del solver.** Siguiendo el mismo principio de equidad experimental que rige el GA (Sección 4.6.2), cada solución de CP-SAT se limita a un tiempo de pared configurable (instancias pequeñas de las familias `ft01`–`ft06`/`la01`–`la15` prueban optimalidad en segundos; instancias medianas/grandes pueden requerir el límite completo sin prueba de optimalidad). El resultado siempre reporta, además del makespan, si la optimalidad quedó **probada** (`OPTIMAL`) o solo se alcanzó una solución factible dentro del límite de tiempo (`FEASIBLE`), distinción indispensable para no confundir un valor de RPD calculado contra una cota no probada con uno calculado contra el óptimo real.

**Verificación cruzada, no solo generación.** Toda solución que CP-SAT reporta como factible se re-verifica de forma independiente contra `is_schedule_feasible` (Sección B2.2) antes de aceptarse — una comprobación paranoica ya presente en la implementación de referencia, que blinda el arnés de evaluación frente a un eventual error de traducción entre el modelo de intervalos de CP-SAT y la semántica de factibilidad del entorno.

### 4.6.4 Arnés de evaluación común

Las tres familias de línea base (PDR, GA, CP-SAT) y la política DRL entrenada difieren radicalmente en su mecánica interna — una PDR decide en línea dentro del propio bucle del entorno Gymnasium; el GA evalúa cromosomas completos fuera de ese bucle; CP-SAT construye y resuelve un modelo de programación por restricciones independiente del entorno — pero todas producen el mismo objeto de interés: un cronograma factible y su makespan. El arnés de evaluación unifica los cuatro métodos detrás de una **interfaz común** (`solve(instance) -> cronograma, makespan, tiempo_de_pared`), especificada formalmente en `../docs/especificaciones/Baselines_TechSpec.md`, de modo que:

1. **Ningún método se favorece por conveniencia de implementación.** El mismo conjunto de instancias, semillas y presupuesto de tiempo de pared (donde aplique) se usa para los cuatro métodos, y todo cronograma producido — sin excepción, incluida la política DRL — se re-verifica contra `is_schedule_feasible` (o `is_schedule_feasible_dynamic` bajo perturbaciones, Sección 4.5) antes de calcular cualquier métrica.
2. **Las métricas de la Sección D6 se calculan de forma idéntica para los cuatro.** RPD respecto al óptimo o mejor cota conocida (CP-SAT), tiempo de decisión medio (por época para PDR/política aprendida, tiempo de pared total para GA/CP-SAT, reportados por separado y nunca promediados entre sí, dado que no son la misma unidad de medida), e índice de robustez $RI$ bajo los mismos regímenes de perturbación de la Sección 4.5 cuando el método admite reevaluación bajo el entorno dinámico (las PDR sí, vía el mismo `DynamicJSSPEnv`; GA y CP-SAT se evalúan en su variante determinista/nominal, dado que ninguno de los dos está formulado para reaccionar a eventos estocásticos in situ, una limitación que se discute explícitamente en el Capítulo 7 como parte de la comparación de paradigmas).
3. **El resultado alimenta directamente el procedimiento estadístico de la Sección D6/5.3** (Wilcoxon para comparaciones pareadas, Friedman + Nemenyi con diagrama de diferencia crítica para las nueve PDR más GA, CP-SAT y la política DRL simultáneamente), sin transformación adicional: la interfaz común garantiza que las tablas de resultados del Capítulo 6 son directamente comparables entre sí columna por columna.

La especificación técnica completa de esta interfaz, incluyendo las firmas exactas y los criterios de aceptación para su implementación, se detalla en el documento complementario `../docs/especificaciones/Baselines_TechSpec.md`.
