## 5.3 Procedimiento de prueba estadística

La Parte D6 fija el aparato estadístico de la tesis (Wilcoxon para comparaciones pareadas; Friedman con post-hoc de Nemenyi y diagrama de diferencia crítica para comparaciones conjuntas de tres o más métodos) a nivel de fórmula, sin comprometerse a una estructura de datos concreta. Esta sección formaliza ambos procedimientos exactamente como en D6, y los ancla al esquema real de `experiments/results/baselines_week12.csv` — el archivo que ya produce `experiments/eval_baselines.py` (`../docs/especificaciones/Baselines_TechSpec.md` §1) —, de modo que el código estadístico del próximo sprint pueda consumirlo directamente sin una etapa de traducción intermedia no especificada.

### 5.3.1 Esquema de datos de entrada

**Tabla 5.3.1.** Columnas de `experiments/results/baselines_week12.csv` (143 filas: 13 instancias × 11 métodos, `BASELINES_RESULTS_WEEK12.md`), una fila por corrida método–instancia–semilla.

| Columna | Tipo | Contenido |
|---|---|---|
| `instance` | `str` | Nombre de la instancia (`"ft06"`, `"la01"`, …) |
| `num_jobs`, `num_machines` | `int` | Tamaño de la instancia |
| `method` | `str` | Uno de `{spt, lpt, mwkr, lwkr, mor, lor, fifo, edd, cr, ga, cpsat}` — las nueve PDR (Parte B6), el GA y CP-SAT (Parte B7); la política DRL se añadirá como un valor adicional de esta columna, no como una tabla separada, en cuanto exista (§5.3.5) |
| `seed` | `int` | Semilla de la corrida. `0` para las nueve PDR y CP-SAT en la tabla actual (deterministas o con una única semilla registrada por diseño, Sección 4.6.4); **múltiple** para el GA una vez ejecutado el barrido `--ga-seeds 5` que este documento exige como prerrequisito (§5.3.6) |
| `makespan` | `float` | $C_{\max}$ de la corrida |
| `known_optimum` | `int` | $C^*$ (Sección 5.2.1: óptimo probado por CP-SAT, coincidente con OR-Library para las 13 instancias actuales) |
| `rpd_pct` | `float` | $\text{RPD}=100\cdot(C_{\max}-C^*)/C^*$, ya precalculado — la columna que alimenta directamente ambas pruebas de esta sección |
| `wall_clock_s` | `float` | Tiempo de pared de la corrida (Sección 5.2.2) |
| `feasible` | `bool` | Resultado de la verificación independiente contra `is_schedule_feasible` |
| `status` | `str` | `"OK"` (PDR/GA) o el estado de CP-SAT (`"OPTIMAL"`/`"FEASIBLE"`/…) |
| `is_optimal` | `bool` | `True` solo si `status=="OPTIMAL"` |

Esta tabla es una fila por corrida, no una fila por instancia — el procedimiento estadístico de esta sección necesita, en cambio, matrices *instancia × método*, derivadas de esta tabla mediante las transformaciones de §5.3.5, no una estructura distinta que el código de evaluación deba producir por separado.

### 5.3.2 Prueba de Wilcoxon de rangos con signo (comparación pareada)

Para comparar dos métodos específicos sobre el mismo conjunto de instancias — el caso de uso concreto es el criterio de aceptación de H1 (Sección 5.1.4): política DRL vs. FIFO — se agrega `rpd_pct` a una observación por instancia por método (mediana sobre semillas si `method` tiene más de una fila por instancia, valor único si no la tiene) y se calculan las diferencias pareadas $d_i = \text{RPD}_i^{A}-\text{RPD}_i^{B}$ sobre las $N$ instancias comunes a ambos métodos. El estadístico de Wilcoxon:

$$W=\min(W^+,W^-),\qquad W^+=\sum_{i:\,d_i>0} R_i,\quad W^-=\sum_{i:\,d_i<0} R_i,$$

donde $R_i$ es el rango de $|d_i|$ (los empates $d_i=0$ se excluyen antes de rankear, reduciendo $N$ en consecuencia — debe registrarse el $N$ efectivo, no asumirse igual al número de instancias original). Nivel de significancia $\alpha=0.05$; reportar el estadístico $W$, el $p$-valor (aproximación normal si $N\gtrsim20$, exacto en caso contrario — con $N=13$ instancias actuales, usar la distribución exacta, no la aproximación normal) y el tamaño de efecto $r=Z/\sqrt{N}$.

**Con los datos actuales** (§5.3.1), esta prueba es directamente ejecutable para cualquier par de métodos con una fila por instancia — es decir, todo par que no involucre al GA con más de una semilla. Ejemplo concreto ya disponible: FIFO vs. GA sobre las 13 instancias (Tabla 1.1 de `../docs/planificacion/BaselineComparativeAnalysis.md`), aunque su resultado es previsible dado que el GA domina a FIFO en las 13/13 instancias sin excepción — la prueba tiene sentido pleno una vez exista una tercera columna de método (la política DRL) cuyo resultado no esté ya determinado por inspección directa de la tabla.

### 5.3.3 Prueba de Friedman con post-hoc de Nemenyi

Para la comparación conjunta de $k$ métodos sobre las mismas $N$ instancias, se construye la matriz de rangos $r_{ij}$ (rango del método $j$ dentro de la instancia $i$, rango 1 = mejor `rpd_pct`, empates promediados) y el estadístico de Friedman:

$$\chi^2_F = \frac{12N}{k(k+1)}\left(\sum_{j=1}^k \bar R_j^2 - \frac{k(k+1)^2}{4}\right), \qquad \bar R_j=\frac1N\sum_{i=1}^N r_{ij},$$

con $k-1$ grados de libertad bajo $H_0$ (todos los métodos son equivalentes). Si se rechaza $H_0$ ($\alpha=0.05$), el post-hoc de **Nemenyi** compara cada par de métodos contra la diferencia crítica:

$$CD = q_\alpha\sqrt{\frac{k(k+1)}{6N}},$$

donde $q_\alpha$ es el valor crítico de la distribución del rango estudentizado (tabulado, Demšar 2006, Tabla 5b). Dos métodos se declaran significativamente distintos si $|\bar R_i-\bar R_j|>CD$.

**Composición de $k$ con los datos actuales, y su evolución.** Con la tabla de la Semana 12 (una fila por instancia para ocho de los once métodos: las nueve PDR menos ninguna, más CP-SAT — diez métodos con una única observación por instancia), $k=10$ y $N=13$ son ya calculables directamente. El GA se excluye de este cálculo **hasta** que exista una agregación de sus múltiples semillas por instancia (§5.3.6) — incluirlo con una sola semilla trataría una única muestra aleatoria como si fuera el valor representativo del método, inflando artificialmente su rango en cualquier instancia donde esa semilla particular haya tenido suerte o mala suerte, exactamente el riesgo que `BASELINES_RESULTS_WEEK12.md` ya señala para sus propias cifras del GA.

### 5.3.4 Diagrama de diferencia crítica

Siguiendo a Demšar (2006), el diagrama de diferencia crítica ordena los $k$ métodos en un eje horizontal por su rango medio $\bar R_j$ (Friedman) y une con una barra horizontal a todo subconjunto de métodos cuya diferencia de rango medio no supera $CD$ (§5.3.3) — es decir, los grupos estadísticamente indistinguibles al nivel $\alpha$ elegido. Es el artefacto visual que resume, en una sola figura, tanto el ordenamiento como la significancia de cada diferencia, y es el formato de reporte que el Capítulo 6 debe usar para presentar la comparación conjunta de las nueve PDR, el GA, CP-SAT (solo como referencia marcada aparte, §5.3.6) y la política DRL, en lugar de una tabla de rangos sin indicación visual de qué diferencias son o no significativas.

### 5.3.5 Transformaciones requeridas del esquema de entrada (contrato para el código del próximo sprint)

El código estadístico del próximo sprint debe implementar, como funciones puras sobre el `DataFrame` de la Tabla 5.3.1 (o el CSV leído directamente), las siguientes transformaciones — este es el contrato de entrada/salida que evita que cada prueba reimplemente su propia lógica de agregación:

```python
def aggregate_replicas(df: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
    """Colapsa la columna `seed` (mediana por defecto, más robusta a una semilla atípica
    que la media) dentro de cada (instance, method), devolviendo una fila por
    (instance, method). No-op para los métodos que ya tienen una única fila por instancia
    (todas las PDR y CP-SAT en la tabla actual); imprescindible para el GA en cuanto
    exista el barrido multi-semilla (§5.3.6)."""

def rpd_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla dinámica instancia (fila) x método (columna) de rpd_pct, tras
    aggregate_replicas. Entrada directa de la prueba de Wilcoxon (dos columnas
    cualesquiera) y de la matriz de rangos de Friedman (todas las columnas)."""

def rank_matrix(rpd: pd.DataFrame) -> pd.DataFrame:
    """rpd_matrix, con cada fila reemplazada por sus rangos (1 = mejor rpd_pct de esa
    fila, empates promediados vía scipy.stats.rankdata(method='average'))."""

def wilcoxon_pair(rpd: pd.DataFrame, method_a: str, method_b: str) -> WilcoxonResult:
    """W, p-valor, r=Z/sqrt(N_efectivo), sobre las instancias donde ambos métodos tienen
    una observación y la diferencia no es exactamente cero."""

def friedman_nemenyi(rpd: pd.DataFrame, methods: list[str], alpha: float = 0.05) -> FriedmanResult:
    """chi^2_F, p-valor, y — si se rechaza H0 — la matriz de diferencias críticas por
    par de métodos, restringido a `methods` (para excluir CP-SAT del ranking conjunto,
    §5.3.6, sin tener que borrar su columna de rpd_matrix, que sigue siendo la fuente
    de C* para todo lo demás)."""
```

`WilcoxonResult`/`FriedmanResult` deben incluir el $N$ efectivo usado (no solo el estadístico y el $p$-valor), dado que empates y exclusiones de método pueden reducirlo por debajo del número total de instancias — exactamente el mismo principio de no ocultar el tamaño de muestra real ya aplicado en `../docs/planificacion/VandV_Protocol.md` §2 para el cálculo de réplicas.

### 5.3.6 Reglas de inclusión por prueba (heredadas de secciones anteriores, consolidadas aquí)

- **CP-SAT nunca entra en la clasificación conjunta de Friedman/Nemenyi** como método par — ya establecido en la Sección 4.6.4 y en `../docs/capitulos/Section5.1_5.2.md` §5.2.3: su rol es fijar $C^*$ (el denominador de `rpd_pct` para todos los demás), no competir en un ranking cuyo paradigma de tiempo/búsqueda no es comparable a un despacho de una sola pasada o a un episodio de política aprendida. En la función `friedman_nemenyi` (§5.3.5), esto se implementa pasando `methods` sin `"cpsat"`, no eliminando la columna de `rpd_matrix`.
- **El GA solo entra en la clasificación conjunta una vez agregado sobre múltiples semillas** (`aggregate_replicas`, §5.3.5) — con los datos de la Semana 12 (una sola semilla por instancia), el GA puede compararse informalmente (como en `../docs/planificacion/BaselineComparativeAnalysis.md`) pero no debe presentarse como una entrada estadísticamente válida de Friedman/Nemenyi hasta que exista el barrido `--ga-seeds 5` ya solicitado en `BASELINES_RESULTS_WEEK12.md` ("Next steps") y en `../docs/capitulos/Section5.1_5.2.md` §5.2.3 — este documento fija ese barrido como un prerrequisito formal de esta sección, no una mejora opcional.
- **La política DRL, en cuanto exista, se incorpora como un valor adicional de la columna `method`** en el mismo esquema (Tabla 5.3.1) — múltiples semillas por instancia desde el primer momento (a diferencia del GA, que llegó tarde a esa exigencia), de modo que `aggregate_replicas` la trate exactamente igual que al GA sin ninguna rama de código especial por método.
- **El tamaño de muestra $N$ para Friedman/Nemenyi es el número de instancias con datos completos para todos los métodos incluidos** — actualmente $N=13$ (Tabla 5.1.1 de `../docs/capitulos/Section5.1_5.2.md`, bloques "entrenamiento" y "generalización moderada"); crece automáticamente en cuanto se ejecuten los bloques `la11`–`la40`/`ta` ya planificados, sin requerir cambios a las funciones de §5.3.5, que operan sobre el conjunto de instancias presente en el CSV en el momento de la ejecución, no sobre una lista fija.
