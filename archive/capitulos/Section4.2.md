# 4.2 Entorno de simulación de eventos discretos y contrato de interfaz Gymnasium

> **Nota de trazabilidad:** esta sección desarrolla, en prosa de tesis, la especificación procedimental de la Parte D1 del documento de marco ("../docs/planificacion/Step 4 Thesis.md", §D1, con sus fundamentos de simulación de eventos discretos en §B9), y la reconcilia explícitamente con el código del entorno estático ya implementado y validado en el Sprint 1 (`envs/jssp_env.py`, `envs/disjunctive_graph.py`; `VALIDATION_WEEK1.md`). Como en el Capítulo 4.1/4.3, el criterio de esta sección es la alineación exacta con lo que existe hoy, no con una aspiración: la Sección 4.2.4 declara, sin ambigüedad, cuáles elementos del contrato general de D1 **no** están todavía instanciados en el código, precisamente para que la especificación técnica complementaria de esta entrega (documento separado, en inglés, listo para el agente de codificación) tenga un punto de partida honesto.

## 4.2.1 Fundamentos de la simulación de eventos discretos

La formulación dinámica del JSSP que motiva esta tesis (Capítulo 3, brechas RQ2/RQ3) requiere, más allá del formalismo de grafo disyuntivo ya introducido (Roy y Sussmann [34]), el aparato conceptual de la **simulación de eventos discretos** (DES): un reloj de simulación, una **lista de eventos futuros** (FEL, *future event list*) ordenada por tiempo, un conjunto de **entidades** (trabajos, operaciones), **recursos** (máquinas), variables de estado, y contadores estadísticos [30]. El motor de simulación avanza el reloj al instante del próximo evento pendiente (*next-event time advance*, en contraposición al avance por incrementos fijos de tiempo), ejecuta la rutina asociada a ese evento —la cual puede, a su vez, programar nuevos eventos futuros en la FEL— y actualiza el estado y las estadísticas acumuladas.

Dos consideraciones metodológicas de la literatura de simulación son directamente aplicables al protocolo experimental de esta tesis (Capítulo 5): primero, la distinción de Sargent [31] entre **verificación** (¿el modelo se implementó correctamente? — depuración, trazas de ejecución, pruebas de consistencia) y **validación** (¿el modelo representa fielmente el sistema? — validez conceptual, validación operacional, validación de los datos de entrada), que estructura tanto el reporte `VALIDATION_WEEK1.md` ya entregado como el `VALIDATION_WEEK6.md` previsto para la extensión dinámica (documento técnico complementario, §8). Segundo, la distinción entre sistemas **terminantes** (con un evento natural de fin de episodio, como en esta tesis: todas las operaciones de todos los trabajos, incluidos los llegados dinámicamente, completadas) y **no terminantes** (de estado estacionario): dado que el JSSP dinámico de esta tesis es terminante por diseño (horizonte de llegada finito, §4.2.4 y especificación técnica complementaria), el protocolo de réplicas independientes con semillas distintas —y no la eliminación de calentamiento de Welch ni el método de *batch means*, reservados a sistemas no terminantes— es el procedimiento aplicable, con el número de réplicas `R` acotado por el semiancho `h` deseado del intervalo de confianza al nivel `1−α` mediante `R ≥ (z_{1−α/2}·S/h)²`, usando la desviación estándar muestral `S` de una corrida piloto [30].

## 4.2.2 Variables de estado y tipos de evento del contrato general

El contrato de simulación especificado en la Parte D1 del marco procedimental define las siguientes variables de estado: el reloj `t`; para cada máquina `k`, su estado (`libre`, `ocupada`, `averiada`), su tiempo de fin de ocupación, y su próximo tiempo hasta el fallo (TTF); para cada operación, su estado (`no lista`, `lista`, `en proceso`, `terminada`), la máquina asignada y sus tiempos de inicio/fin; el conjunto de operaciones actualmente listas; y el grafo disyuntivo parcialmente orientado que resulta de las decisiones de despacho ya tomadas.

Sobre este estado actúan cinco tipos de evento:

- **(E1) Llegada de trabajo** — instancia un nuevo trabajo (con su propia cadena de operaciones) en el sistema en el instante `t`, extendiendo el grafo disyuntivo con nuevos nodos y aristas de precedencia, y potencialmente nuevas aristas de máquina compartida con operaciones ya presentes.
- **(E2) Fin de operación** — libera la máquina que la procesaba, marca la operación como terminada, y activa su operación sucesora en la cadena del trabajo (si existe), añadiéndola al conjunto de operaciones listas.
- **(E3) Avería de máquina** — interrumpe la operación en curso en esa máquina, si la hay (según la política de preempción especificada en el documento técnico complementario, §5), y reprograma el fin de la reparación.
- **(E4) Fin de reparación** — la máquina vuelve al estado libre y, si había una operación preempcionada, esta queda lista para reanudarse.
- **(E5) Época de decisión** — el agente observa el estado y elige una acción; se dispara cada vez que una máquina queda libre y existe al menos una operación lista para procesarse en ella.

El bucle principal del motor DES, tal como lo especifica la Parte D1, es:

```
inicializar reloj t=0, FEL, estado; programar llegadas y primeras averías
reset() → observación inicial
mientras no fin_episodio:
    evento e ← extraer_min(FEL); t ← tiempo(e)
    ejecutar rutina(e); actualizar estado y estadísticas
    si e habilita una época de decisión (máquina libre ∧ operaciones listas):
        obs  ← construir_observación(estado)
        mask ← acciones_factibles(estado)
        a    ← política(obs, mask)        # en entrenamiento: PPO enmascarable
        aplicar(a): asignar operación→máquina; programar evento E2 en t+p
        r    ← recompensa(estado_antes, estado_después)
    hasta próxima_decisión
retornar C_max y estadísticas
```

## 4.2.3 Contrato de interfaz Gymnasium

El contrato de interfaz que expone este motor DES a un agente de aprendizaje —válido tanto para la variante estática ya implementada como para la extensión dinámica— se especifica, siguiendo la convención estándar de Gymnasium [55] ya adoptada en el Sprint 1, como:

- **`observation_space`**: un espacio `Dict` heterogéneo compuesto por una matriz de características de operación (`N_op × d_op`), una matriz de características de máquina (`m × d_m`) —esta última ausente en la variante estática actual, cuyo vector de estado (Sección 4.3.1) no incluye todavía nodos de máquina explícitos— y los índices de arista por tipo de relación (precedencia, elegibilidad operación–máquina, máquina compartida).
- **`action_space`**: un espacio `Discrete` sobre pares (operación lista, máquina elegible), indexados y acompañados de una **máscara** booleana de factibilidad. Las acciones infactibles se excluyen de la distribución categórica de muestreo antes del cálculo de log-verosimilitud usado por PPO —no mediante una penalización de recompensa— siguiendo la misma convención de enmascaramiento de `sb3_contrib.MaskablePPO` ya validada en el Sprint 1 [52].
- **`reset(seed)` → `(obs, info)`**; **`step(a)` → `(obs, reward, terminated, truncated, info)`**; `terminated` se declara cuando todas las operaciones de todos los trabajos —incluidos los llegados dinámicamente hasta el horizonte de llegada— han finalizado.

## 4.2.4 El entorno estático del Sprint 1 como caso particular del contrato general

Es necesario precisar con exactitud la relación entre este contrato general y el entorno realmente implementado y validado (`envs/jssp_env.py`, 51/51 pruebas superadas en `tests/test_env.py`), para que la especificación técnica de la extensión dinámica (documento complementario de esta entrega) parta de una base honesta y no de una sobre-interpretación.

`JSSPEnv` implementa **únicamente** épocas de decisión (E2 y E5, en la nomenclatura de la Sección 4.2.2), fusionadas en una sola transición por cada llamada a `step()`: al no existir un reloj que avance de forma independiente entre decisiones, el fin de la operación despachada (E2) ocurre lógicamente en el mismo instante en que se resuelve la época de decisión (E5) que la originó, mediante la transición de *list-scheduling* ya documentada en el docstring del módulo (`start = max(job_ready_time[j], machine_free_time[m])`). En términos del bucle de la Sección 4.2.2, esto equivale a una FEL degenerada que nunca contiene más de un evento pendiente relevante a la vez, y en la que ese evento siempre es directamente resuelto por la acción del agente —nunca un evento exógeno que el agente deba esperar. Los eventos **E1 (llegada de trabajo)**, **E3 (avería de máquina)** y **E4 (fin de reparación)** **no están instanciados**: el conjunto de trabajos es fijo desde `reset()`, y no existe ninguna fuente de indisponibilidad de máquina más allá de la ocupación por procesamiento normal. Esta es, precisamente, la razón por la que el docstring de `JSSPEnv` (Sección 4.3.1) describe la formulación como "event-driven" sin ambigüedad, pero reserva explícitamente la introducción de "una cola de eventos real" para la extensión dinámica —la que motiva esta sección y su especificación técnica complementaria.

En cuanto al contrato Gymnasium de la Sección 4.2.3, el entorno estático realiza dos simplificaciones consistentes con la ausencia de eventos E1/E3/E4: el espacio de acción `Discrete` sobre pares (operación, máquina) colapsa a `Discrete(n_operaciones)` porque, en el JSSP clásico de enrutamiento fijo, cada operación tiene exactamente una máquina elegible (`envs/instance.py`), de modo que especificar la operación determina la máquina sin ambigüedad; y `observation_space` omite el bloque de características de máquina `(m × d_m)` de la Sección 4.2.3 porque, sin averías, la única información de máquina relevante para la política —su carga remanente— ya se incorpora como característica de operación (`machine_remaining_load`, Sección 4.3.1) en lugar de como un nodo de máquina independiente. Ambas simplificaciones son válidas para el caso estático y deberán levantarse explícitamente al instanciar E1/E3/E4, lo cual se especifica en detalle, componente por componente, en el documento técnico `../docs/especificaciones/DynamicEnv_TechSpec.md` que acompaña esta entrega y que un agente de codificación puede implementar directamente sobre el paquete `envs/` existente sin reescribir el entorno estático ya validado.

## Referencias empleadas en esta sección

[30] A. M. Law, *Simulation Modeling and Analysis*, 6th ed., McGraw Hill, 2024.

[31] R. G. Sargent, "Verification and Validation of Simulation Models," in *Proc. 2011 Winter Simulation Conf.*, S. Jain, R. R. Creasey, J. Himmelspach, K. P. White, M. Fu, Eds., IEEE, 2011, pp. 183–198, doi:10.1109/WSC.2011.6147750.

[34] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.º 9 bis, SEMA, Montrouge, 1964.

[52] S. Huang, S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms," in *Proc. 35th Int. FLAIRS Conf.*, 2022, arXiv:2006.14171.

[55] M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," arXiv:2407.17032, 2024. **[VERIFICAR — identificador arXiv exacto no reconfirmado de forma independiente]**
