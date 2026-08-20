# Capítulo 4 — Marco Metodológico y Arquitectura de la Solución

> **Nota de trazabilidad:** Estas secciones documentan, sin idealizar, el pipeline realmente implementado en el primer sprint de codificación de la tesis (código en `envs/`, `models/`, `baselines/`, `experiments/`; validación en `VALIDATION_WEEK1.md`, 188/188 pruebas superadas). Toda afirmación técnica de este capítulo está anclada a una línea de código o a un docstring citables; donde el pipeline se aparta de la especificación teórica original del marco de la tesis, la desviación se declara explícitamente y se justifica, en vez de describirse como si ya estuviera resuelta. Esta es una reescritura de la versión previa de estas secciones ("Capítulo 4: Marco Metodológico y Arquitectura de la Solución", borrador inicial): esa versión describía tres características de nodo, enmascaramiento sensible a averías de máquina y un simulador DES general — ninguna de las tres corresponde al código del Sprint 1, y se corrigen aquí.

## 4.1 Enfoque general y arquitectura de la solución

El pipeline adopta un paradigma **constructivo** (no de búsqueda iterativa sobre soluciones completas): el cronograma se construye operación por operación, y cada decisión de despacho es una acción de un agente de Aprendizaje por Refuerzo Profundo (DRL) entrenado con Optimización de Política Próxima (PPO) [13] sobre un codificador de Red Neuronal de Grafos (GNN). El pipeline efectivamente implementado en el Sprint 1 consta de cinco componentes modulares, cada uno con un módulo de código dedicado:

1. **Analizador y modelo de instancia (`envs/instance.py`).** La clase `JSSPInstance` representa una instancia estática y determinista del JSSP clásico: `num_jobs` trabajos, cada uno una cadena fija de `ops_per_job` operaciones, cada operación con una máquina y una duración prescritas. Expone las cantidades derivadas (carga total por trabajo/máquina, tiempos de holgura de cabeza/cola) que alimentan tanto la cota inferior estática como las características de estado. El propio docstring del módulo documenta la extensión planeada hacia el FJSP (una estructura `eligible_machines` no rectangular) como trabajo futuro, no como algo ya soportado.
2. **Entorno de decisión secuencial (`envs/jssp_env.py`, `envs/disjunctive_graph.py`).** Implementado sobre la interfaz estándar de Gymnasium [55], `JSSPEnv` expone el grafo disyuntivo parcialmente orientado de Roy y Sussmann [34] como una observación `Dict`, con una máscara de acciones legales recalculada en cada paso. **Es importante precisar, para evitar la sobre-interpretación del borrador anterior de este capítulo, que este entorno es una simulación de eventos discretos *implícita*, no un simulador DES de propósito general**: no existe un reloj global que avance por sí mismo ni una cola de eventos exógenos; cada `step()` *es* una época de decisión, y el "tiempo" reportado es el instante de inicio asignado a la operación despachada (Sección 4.3.1). El entorno no inyecta todavía perturbaciones estocásticas (averías, llegadas dinámicas): es exclusivamente el JSSP estático clásico, tal como se declara sin ambigüedad en el propio docstring de `JSSPEnv` y se retoma en la Sección 4.3.1.
3. **Codificador de estado (`models/gin.py`).** Una Red de Isomorfismo de Grafos (GIN) [19] adaptada para procesar, en una sola pasada, dos relaciones distintas del grafo disyuntivo —precedencia de trabajo y disputa de máquina— mediante `RelationalGINLayer`: cada relación (predecesor, sucesor, máquina) tiene su propia transformación lineal antes de sumarse en el mensaje agregado, que luego pasa por la actualización GIN estándar con suma de vecinos, la forma de agregación que preserva la capacidad discriminativa equivalente al test de Weisfeiler–Lehman [19]. `GINEncoder` apila 3 de estas capas (por defecto) y produce tanto embeddings por nodo como un embedding de grafo agrupado por promedio.
4. **Agente de política (`models/policy.py`).** `JSSPGraphFeaturesExtractor` envuelve al `GINEncoder` como extractor de características de `sb3_contrib.MaskablePPO` [53], concatenando los embeddings de nodo aplanados con el embedding de grafo agrupado. Sobre esas características, cabezas MLP de actor y crítico (128 unidades ocultas por defecto) producen, respectivamente, los logits de acción y la estimación de valor de estado. La máscara de acciones se aplica dentro de la distribución categórica enmascarable de `sb3_contrib` [52], no como un término de penalización en la recompensa (Sección 4.3.2).
5. **Líneas base y protocolo de evaluación (`baselines/dispatching_rules.py`, `baselines/cpsat.py`, `experiments/train_ft06.py`).** Seis reglas de despacho por prioridad (aleatoria, SPT, LPT, FIFO, MWKR, MOPNR) sirven como referencia conductual mínima, y un modelo exacto de intervalos sobre Google OR-Tools CP-SAT [33] sirve como *ground truth* de optimalidad, reemplazando una tabla de óptimos de literatura por una prueba re-ejecutable. `experiments/train_ft06.py` orquesta el entrenamiento sobre una instancia (`ft06` en el Sprint 1) y escribe un punto de control del modelo, una curva de recompensa/makespan por episodio y un CSV de métricas — el núcleo de un protocolo de evaluación que, en fases posteriores del proyecto, se ampliará con las comparaciones estadísticas (Friedman/Nemenyi) identificadas como brecha metodológica en la Sección 3.6.

### 4.1.1 Descripción del diagrama de arquitectura (Figura 4.1)

La figura siguiente se describe en prosa para que el autor la renderice por separado (p. ej. en draw.io o Mermaid); la descripción corresponde exactamente al flujo de datos de `step()` y a la llamada `forward()` del extractor de características.

**Bloques (de izquierda a derecha, en un ciclo cerrado):**

1. **`JSSPInstance`** (parseada una vez, inmutable) → alimenta la construcción de `JSSPEnv` en `reset()`.
2. **`JSSPEnv` — estado `s_t`**: un rectángulo que se bifurca en cuatro tensores etiquetados exactamente como en el código: `node_features (n_ops × 8)`, `adj_precedence (n_ops × n_ops)`, `adj_machine (n_ops × n_ops)`, `action_mask (n_ops,)`.
3. **`GINEncoder`** (tres capas `RelationalGINLayer` apiladas, dibujadas como tres rectángulos apilados con flechas de `adj_precedence` y `adj_machine` entrando a cada una): produce `node_embeddings (n_ops × hidden_dim)` y, por una flecha de agrupación (pool "mean"), `graph_embedding (hidden_dim,)`.
4. **Concatenación** de `node_embeddings` aplanados con `graph_embedding` → vector `features ((n_ops+1) × hidden_dim,)`.
5. **Dos cabezas MLP en paralelo**: "Actor" → logits sobre `n_ops` acciones; "Crítico" → escalar `V(s_t)`.
6. **Enmascaramiento**: una flecha desde `action_mask` (paso 2) entra directamente a la distribución categórica enmascarable que consume los logits del Actor, **antes** del muestreo y del cálculo de log-probabilidad (no como una suma de `-∞`, sino como una re-normalización de la distribución categórica misma [52]) — mostrar esto como un rombo/compuerta entre "logits" y "distribución categórica", con la etiqueta "MaskableCategorical".
7. **Acción `a_t`** muestreada → flecha de regreso hacia `JSSPEnv.step(a_t)`.
8. **`JSSPEnv.step()`** calcula `s_{t+1}`, la recompensa `r_t` (Sección 4.3.3) y `terminated`; estas tres salidas, junto con `V(s_t)` del Crítico, alimentan el búfer de rollout de PPO (rombo etiquetado "GAE + pérdida recortada de PPO" [13], [14]), que periódicamente actualiza los pesos del Actor, el Crítico y el `GINEncoder` compartido.
9. Un bloque lateral, sin flechas hacia el ciclo de entrenamiento, representa **Líneas base** (`dispatching_rules.py`, `cpsat.py`): consume la misma `JSSPInstance` de forma independiente y produce makespans de referencia para el reporte de evaluación (Capítulo 5), no para el gradiente de PPO.

### 4.1.3 Justificación de la elección arquitectónica frente a la literatura (Sección 3.5)

La capa `RelationalGINLayer` no es una GIN homogénea en el sentido estricto de Zhang et al. [23] —donde toda arista recibe el mismo tratamiento—, pero tampoco es un grafo heterogéneo con tipos de nodo distintos como en Song et al. [25] o Liu et al. [36]: usa un único tipo de nodo (la operación), pero **tres transformaciones lineales separadas** según el rol de la arista (predecesor, sucesor, máquina), justificadas explícitamente en el código porque un predecesor *restringe* cuándo puede iniciar una operación, un sucesor es *lo que la operación desbloquea*, y un competidor de máquina *disputa el mismo recurso* — tres significados semánticamente opuestos que una adyacencia simétrica única perdería. Esta posición intermedia se adopta deliberadamente, no por omisión: es más barata computacionalmente que un transformer de grafo heterogéneo completo (evita `torch-geometric` y sus dependencias frágiles de compilación, según el propio docstring de `models/gin.py`), y su justificación se refuerza retroactivamente por Hoss et al. [50], quienes en 2026 muestran que ni siquiera la heterogeneidad de tipos de nodo es condición necesaria para escalar con complejidad lineal — es decir, la literatura más reciente respalda explorar puntos intermedios del espectro homogéneo–heterogéneo en lugar de asumir que la heterogeneidad plena es obligatoria. La validación de que esta elección no sacrifica la señal de aprendizaje frente a la alternativa homogénea de L2D es, precisamente, el resultado (b) de `VALIDATION_WEEK1.md`: la política entrena y reduce el makespan medio de 98.5 a 67.8 en 20 000 pasos sobre `ft06`.

## 4.3 Formulación del Proceso de Decisión de Markov (MDP)

El problema de programación se formaliza como un MDP `(S, A, P, R, γ)` cuya implementación en código —no una reformulación posterior— es la fuente de verdad de esta sección. `envs/jssp_env.py` documenta esta formulación en su docstring de módulo; lo que sigue es una traducción fiel de ese docstring y del código que lo implementa, con la trazabilidad línea-por-línea que exige el encargo de esta sección.

### 4.3.1 Espacio de estados (S)

El estado en la época de decisión *t* es el grafo disyuntivo parcialmente orientado (Sección 2 / [34]), expuesto como una observación `Dict` con cuatro claves (constante `NUM_NODE_FEATURES = 8` en `envs/jssp_env.py`, línea 77):

| Clave | Forma | Contenido |
|---|---|---|
| `node_features` | `(n_ops, 8)` | ocho características por operación, detalladas abajo |
| `adj_precedence` | `(n_ops, n_ops)` | arcos conjuntivos dirigidos, fijos durante todo el episodio |
| `adj_machine` | `(n_ops, n_ops)` | arcos disyuntivos simétricos, **solo entre operaciones aún no despachadas** que comparten máquina — se reduce a medida que avanza el episodio |
| `action_mask` | `(n_ops,)` | máscara binaria de legalidad, recalculada en cada paso |

Las ocho características de `node_features` (orden exacto de `NODE_FEATURE_NAMES`, `envs/jssp_env.py` líneas 81–90), cada una normalizada a `[0, 1]` respecto al horizonte `H = max(tiempo total de procesamiento de la instancia, 1)` o a la duración máxima según corresponda:

1. **`processing_time`** — duración de la operación, escalada por la duración máxima de la instancia (estática).
2. **`is_scheduled`** — 1 si la operación ya fue despachada.
3. **`is_ready`** — 1 si la operación es legal para despachar en este instante (idéntica al valor de `action_mask` para esa operación).
4. **`is_job_complete`** — 1 si el trabajo dueño de la operación ya completó todas sus operaciones.
5. **`earliest_start_estimate`** — inicio realizado (si ya está programada) o estimación optimista de inicio más temprano (si no lo está), escalada por `H`.
6. **`remaining_job_work`** — suma de duraciones aún no despachadas del propio trabajo, desde esta operación en adelante, escalada por `H`.
7. **`job_progress`** — posición de la operación dentro de la cadena de su trabajo, escalada por el número de operaciones por trabajo (estática).
8. **`machine_remaining_load`** — carga total aún no despachada sobre la máquina que requiere esta operación, escalada por `H`.

**Nota de desviación arquitectónica y de alcance (obligatoria por el encargo de esta sección):** la especificación teórica original de la tesis (marco conceptual, Capítulo 1–2) contempla un taller con averías estocásticas y, eventualmente, flexibilidad de máquina (FJSP); una versión previa de esta sección describía —incorrectamente respecto al código— un vector de estado de solo tres dimensiones y un enmascaramiento sensible a averías. El código del Sprint 1 diverge de ambas ideas de dos maneras concretas y verificables:　

- **Ocho características, no tres.** El vector de estado realmente implementado es más rico que la especificación mínima descrita en el borrador anterior, precisamente porque un extractor GIN sin capas recurrentes necesita que la información temporal relevante (holgura, progreso, carga restante) esté disponible como característica de entrada en cada paso, no inferida implícitamente por la red.
- **Sin averías ni flexibilidad de máquina todavía.** El entorno implementado es el JSSP **estático, clásico y determinista**: no hay máquinas que se averíen, no hay elección de máquina por operación, y `adj_machine` cambia únicamente porque las disyunciones se resuelven al despachar, no porque un recurso deje de estar disponible. Esto es una decisión de alcance explícita y documentada (docstring de `JSSPEnv`: *"the dynamic extension... will replace [this] with a real event queue"*; docstring de `JSSPInstance`: *"The FJSP extension planned for later weeks will subclass or generalise this"*), no un descuido. Se justifica metodológicamente por la misma razón dada en `VALIDATION_WEEK1.md`: verificar primero que el entorno, la recompensa, la máscara y el gradiente de PPO están correctamente sincronizados sobre el caso determinista y de enrutamiento fijo, antes de introducir el enmascaramiento dinámico por avería (RQ2, Capítulo 3, Sección 3.6) y la elección de máquina (RQ1). Esta decisión reordena, pero no abandona, los ejes de novedad identificados en el Capítulo 3: el Sprint 1 resuelve la precondición de que la arquitectura relacional aprenda correctamente sobre el caso base, dejando explícitamente para sprints posteriores la inyección de perturbaciones y la flexibilidad de enrutamiento.

### 4.3.2 Espacio de acciones (A) y enmascaramiento (Action Masking)

El espacio de acciones es `Discrete(n_ops)`: cada acción `a` es el identificador global de una operación a despachar (`op_id = job × ops_per_job + posición`, `envs/instance.py`). Esta formulación por-operación es, en el código, deliberadamente equivalente a un espacio "elegir trabajo": en todo instante cada trabajo inconcluso expone como máximo una operación despachable (su operación *frontera*), de modo que la biyección entre "elegir operación frontera" y "elegir trabajo" se mantiene siempre — la ventaja de formular la acción a nivel de operación es que la cabeza de política puede leer un logit directamente del embedding de cada nodo del GNN, lo que mantiene la red agnóstica al tamaño *en principio* (aunque el extractor de características actual, Sección 4.1, todavía aplana esos embeddings en un vector de tamaño fijo — la limitación documentada en la Sección 3.7).

La legalidad se calcula en `JSSPEnv.action_masks()` (código, no paráfrasis):

```
mask[i] = 1  si y solo si  la operación i no ha sido despachada
              Y  i es la operación frontera de su trabajo
              (es decir, i == job × ops_per_job + job_next_pos[job])
mask[i] = 0  en cualquier otro caso
```

No existe hoy ninguna condición de máscara relacionada con disponibilidad de máquina o avería — esa es, precisamente, la desviación señalada en 4.3.1. El enmascaramiento se aplica mediante la convención de auto-descubrimiento de `sb3_contrib.MaskablePPO`, que llama a `env.action_masks()` y usa una distribución `MaskableCategorical` que pone en cero la masa de probabilidad de las acciones ilegales **antes** del muestreo y **antes** del cálculo de log-verosimilitud usado en la pérdida de PPO [52] — no una penalización de recompensa ni una suma de `-∞` a los logits seguida de softmax convencional, que es como lo describía el borrador anterior de esta sección; la diferencia importa porque penalizar por recompensa corrompe el gradiente on-policy, mientras que enmascarar la distribución misma no lo hace [52]. Intentar dispatchar una acción ilegal no se absorbe silenciosamente: `JSSPEnv.step()` lanza `ValueError` de forma deliberada, para no contaminar los datos on-policy de una actualización de PPO con una transición inválida.

### 4.3.3 Función de recompensa (R)

La recompensa es **densa**, no dispersa: se calcula en cada paso, no solo al final del episodio, para dar señal de aprendizaje en un problema NP-difícil donde una recompensa terminal única sería extremadamente rala. Concretamente (`envs/jssp_env.py::step`, `envs/disjunctive_graph.py::estimated_makespan`):

```
r_t = ( H(s_t) − H(s_{t+1}) ) × reward_scale
```

donde `H(s)` es una **estimación de límite inferior del makespan dependiente del estado**, no la cota estática de la instancia (que es una cantidad distinta, usada solo para normalizar). `H(s)` se calcula recorriendo la cadena de cada trabajo de izquierda a derecha: si la operación ya fue despachada, contribuye su tiempo de finalización realizado; si no, se le asume optimistamente un inicio inmediato tras el fin de su predecesora (ignorando la contención de máquina), de modo que su finalización estimada es `finalización_predecesora + duración`. `H(s)` es el máximo de esas finalizaciones estimadas sobre todas las operaciones. Esta construcción garantiza tres propiedades, todas verificadas por la suite de pruebas (`tests/test_disjunctive_graph.py`, `tests/test_env.py`) y consistentes con la teoría clásica de *reward shaping* basado en potenciales [54]:

1. `H(s)` es una cota inferior válida del makespan de cualquier completación de `s`.
2. `H(s)` es no decreciente a medida que se despachan operaciones, de modo que `r_t ≤ 0` siempre.
3. En la terminación, `H(s_T)` coincide exactamente con el makespan realizado `C_max`, por lo que la suma telescópica de recompensas de todo el episodio es exactamente:

```
Σ_t r_t = ( H(s_0) − C_max ) × reward_scale
```

Con `γ = 1`, maximizar el retorno no descontado es entonces *exactamente* minimizar el makespan, salvo por la constante `H(s_0) × reward_scale` que no depende de la política. `reward_scale` es, por defecto, `1 / static_lower_bound(instancia)` — la cota clásica de trabajo/máquina de la instancia (máximo entre la cota de carga de trabajo y la cota de cabeza+carga+cola por máquina; `envs/disjunctive_graph.py::static_lower_bound`) — de modo que los retornos quedan en una escala comparable entre instancias de tamaños distintos (`ft06` y `la40`, por ejemplo); puede fijarse en `1.0` para obtener recompensas en unidades de tiempo bruto.

**Nota de desviación frente al borrador anterior de esta sección:** la descripción previa de esta función de recompensa era cualitativamente correcta ("recompensa positivamente si la acción no empeora el makespan proyectado") pero omitía tanto la fórmula exacta de `H(s)` —una propagación optimista específicamente sobre las **cadenas de trabajo**, no sobre el grafo completo, e ignorando explícitamente la contención de máquina en la parte no despachada— como el rol normalizador de `reward_scale`. Se corrige aquí porque la Sección 4.3 exige alineación exacta con el código, no una paráfrasis conceptual del principio de *reward shaping*.

## Referencias empleadas en este capítulo

Numeración consistente con el Capítulo 3 para las referencias compartidas.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[14] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2015/2016.

[19] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?," in *Proc. ICLR*, 2019, arXiv:1810.00826.

[23] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, C. Xu, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," in *Proc. NeurIPS*, 2020, arXiv:2010.12367.

[25] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 2, pp. 1600–1610, 2023, doi:10.1109/TII.2022.3189725.

[33] L. Perron, V. Furnon, "OR-Tools," Google, v9.x, 2025. [Online]. Available: https://developers.google.com/optimization/. Solver de referencia: L. Perron, F. Didier, S. Gay, "The CP-SAT-LP Solver," in *Proc. CP 2023*, LIPIcs, vol. 280, art. 3, 2023, doi:10.4230/LIPIcs.CP.2023.3.

[34] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.º 9 bis, SEMA, Montrouge, 1964.

[36] G. Liu, M. Liao, W. Chen, Z. Zhang, H. Wan, Y. Lin, "Multi-Faceted Dynamic Flexible Job Shop Scheduling via Heterogeneous Graph Transformer and Deep Reinforcement Learning," *Expert Systems with Applications*, vol. 303, art. 130532, 2025/2026, doi:10.1016/j.eswa.2025.130532.

[50] J. Hoss, M. Link, N. Klarmann, "Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs," arXiv:2604.23841, 2026.

**Nuevas en este capítulo**

[52] S. Huang, S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms," in *Proc. 35th Int. FLAIRS Conf.*, 2022, arXiv:2006.14171.

[53] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

[54] A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," in *Proc. ICML*, 1999, pp. 278–287.

[55] M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," arXiv:2407.17032, 2024. **[VERIFICAR — identificador arXiv exacto y lista completa de autores no reconfirmados de forma independiente en esta ronda]**

## Acciones pendientes antes de la sustentación

1. Renderizar el diagrama descrito en la Sección 4.1.1 como figura formal (Figura 4.1) en la herramienta de preferencia del autor.
2. Confirmar el identificador arXiv exacto y la lista de autores de [55] contra la fuente primaria antes de la entrega final.
3. Cuando se implemente la extensión dinámica (averías/llegadas, RQ2) y la elección de máquina (FJSP, RQ1) en sprints posteriores, esta sección deberá revisarse de nuevo para que el espacio de estados, de acciones y la función de recompensa documentados sigan coincidiendo exactamente con el código — tal como se hizo en esta revisión frente al borrador anterior.
