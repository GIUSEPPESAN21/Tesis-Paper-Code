# Capítulo 3 — Estado del Arte (versión final)

> **Nota de trazabilidad:** Esta es la versión de cierre del capítulo, construida sobre la iteración anterior ("Capítulo 3 — Estado del Arte", verificación bibliográfica del 27/07/2026) e incorporando (a) tres referencias adicionales de 2024–2026 localizadas en esta ronda de búsqueda abierta (arXiv, motores de búsqueda web), (b) la promoción de las secciones 3.4 y 3.5 de "versión preliminar" a versión final, y (c) un párrafo de síntesis de cierre (Sección 3.7) que ancla el análisis de brechas al estado real del código del primer sprint de implementación (`VALIDATION_WEEK1.md`). Como en la iteración anterior, todo valor numérico proviene de una fuente citada explícitamente; donde no fue posible confirmarlo se mantiene la marca **[VERIFICAR]** en lugar de estimarlo — esta tesis prefiere una brecha de verificación declarada a un dato inventado. Persiste la limitación metodológica ya señalada: este entorno no tiene acceso de suscriptor a Scopus, Web of Science o IEEE Xplore; la re-ejecución de las cuatro cadenas de búsqueda PRISMA con acceso institucional UNIMINUTO sigue siendo una acción pendiente antes de la sustentación (Sección 3.1).

## 3.1 Protocolo de revisión sistemática PRISMA

**Bases de datos objetivo:** Scopus, Web of Science, IEEE Xplore, ScienceDirect (Elsevier); arXiv/DBLP como complemento de preprints. **Bases efectivamente consultadas:** arXiv, motores de búsqueda web de propósito general, Semantic Scholar, DBLP, páginas de editoriales de acceso público (IEEE Xplore, ScienceDirect, MDPI, Taylor & Francis) y repositorios de código en GitHub — **no** las interfaces de suscriptor de Scopus/WoS/IEEE Xplore.

**Cadenas de búsqueda (literales):**

1. `("reinforcement learning" OR "deep reinforcement learning" OR "graph neural network") AND ("job shop" OR "job-shop" OR "flexible job shop" OR "FJSP") AND (scheduling)`
2. `("reinforcement learning") AND ("job shop") AND ("dynamic" OR "stochastic" OR "machine breakdown" OR "disruption" OR "rescheduling")`
3. `("neural combinatorial optimization" OR "graph neural network") AND ("generalization" OR "size generalization") AND (scheduling OR "combinatorial optimization")`
4. `("large language model" OR "LLM agent") AND (scheduling OR "job shop" OR "combinatorial optimization")`

**Ventana temporal:** 2015–2026. **Criterios de inclusión:** artículo revisado por pares o preprint arXiv con evidencia de impacto (código público, citas, publicación posterior en venue arbitrado); aplica RL/DRL, GNN, o agentes LLM a JSSP/FJSP; reporta métricas cuantitativas y benchmarks. **Exclusión:** sin evaluación cuantitativa; *scheduling* ajeno al taller (cloud/grid sin discusión de transferibilidad); duplicados de la misma preimpresión/publicación.

**Flujo PRISMA aproximado (proxy, no oficial — acción pendiente: re-ejecutar en Scopus/WoS/IEEE Xplore antes de la defensa):**

```
Identificados (proxy, 4 cadenas, 2015–2026)              ≈ 90–115 registros
        │
        ▼
Tras eliminación de duplicados entre cadenas              ≈ 58 registros únicos
        │
        ▼
Cribados por título/resumen                                ≈ 42 pasaron
        │
        ▼
Incluidos (cumplen criterios de inclusión)                  ≈ 30–33 registros
        │
        ▼
Priorizados como incorporaciones 2024–2026 de
mayor relevancia para la narrativa y la tabla               17 registros
```

## 3.2 Evolución 2015–2026 de DRL para JSSP/FJSP (narrativa cronológica)

**2015–2019 — Precursores de la optimización combinatoria neuronal (NCO).** El paradigma de resolver problemas combinatorios mediante políticas aprendidas, en lugar de heurísticas diseñadas a mano, se origina fuera del *scheduling* propiamente dicho. Vinyals, Fortunato y Jaitly introducen las **Pointer Networks** (NeurIPS 2015) [20], una arquitectura secuencia-a-secuencia con un mecanismo de atención que apunta directamente a elementos de la entrada, aplicada originalmente al problema del viajante (TSP). Kool, van Hoof y Welling extienden este paradigma con un **modelo de atención** entrenado con REINFORCE y una línea base de rollout determinista, resolviendo TSP y variantes de ruteo con calidad cercana a la óptima (ICLR 2019) [21]. Kwon et al. proponen **POMO**, que explota las simetrías de solución del problema para reducir la varianza del entrenamiento (NeurIPS 2020) [22]. Estos trabajos no abordan *scheduling* de taller, pero establecen el marco conceptual —codificador de atención/grafo más política entrenada con RL, evaluada por generalización— que la familia GNN+DRL para JSSP adopta directamente, incluido el pipeline implementado en esta tesis (Capítulo 4).

**2020–2021 — GNN+RL aplicado por primera vez al JSSP.** El trabajo fundacional de esta línea es **L2D** (Zhang, Song, Cao, Zhang, Tan y Xu, NeurIPS 2020) [23]: una representación de **grafo disyuntivo** con un codificador **GIN** adaptado a grafos dirigidos y una política **PPO** que decide, en cada época, qué operación despachar. L2D reporta brechas frente a OR-Tools de 17.7 % (6×6) a 29.2 % (30×20) sobre instancias generadas, y de 13.6 % a 26.0 % sobre el benchmark de Taillard [7], superando a las PDR clásicas en el 92 % de las instancias detalladas de Taillard y el 87.5 % de las instancias DMU evaluadas. L2D también reporta generalización *fuera de distribución*: una política entrenada en instancias de 20×20/30×20 alcanza brechas de 22.1 %/9.4 % al evaluarse en instancias de 50×20/100×20, muy por debajo de la brecha de la mejor PDR (33.7–54.9 %) en esos mismos tamaños [23]. Park, Chun, Kim, Kim y Park (*International Journal of Production Research*, 2021) formalizan de manera independiente un enfoque similar y reportan un error relativo promedio de 14.7 % sobre un conjunto de prueba propio, con pruebas de transferencia frente a MARL y MDQN de la literatura previa [24]. Ambos trabajos comparten una limitación explícita: se restringen al JSSP **estático y determinista**, sin flexibilidad de máquina ni eventos dinámicos.

**2022–2023 — Extensión al FJSP mediante grafos heterogéneos.** La generalización natural del enfoque GNN+DRL al FJSP requiere representar dos tipos de entidad —operación y máquina— con relaciones distintas entre ellas. Song, Chen, Li y Cao (*IEEE Transactions on Industrial Informatics*, 2023) proponen un **grafo heterogéneo** que extiende el grafo disyuntivo con nodos de máquina explícitos, procesados mediante capas de atención heterogénea, con una política PPO [25]; el acceso al texto completo estuvo bloqueado en esta revisión (IEEE Xplore), por lo que la cifra numérica exacta de mejora se marca **[VERIFICAR]**. En paralelo, Lei, Guo, Zhao, Wang, Qian, Meng y Tang (*Expert Systems with Applications*, 2022) proponen un esquema **multi-PPO** con dos políticas actor-crítico acopladas sobre un codificador GIN, evaluado sobre Brandimarte [9] y variantes sintéticas [26]; de nuevo, el acceso a la tabla de resultados exacta estuvo bloqueado.

**2023–2024 — Consolidación e infraestructura de código abierto.** Tassel, Gebser y Schekotihin publican **JSSEnv**, un entorno de RL para JSSP compatible con Gymnasium [27]; Waubert de Puiseau et al. publican **schlably**, un framework Python para experimentos de *scheduling* con DRL [28]; y Reijnen, Smit, Zhang, Wu, Bukhsh y Zhang unifican múltiples entornos y conjuntos de instancias en un solo *benchmark* de código abierto [29]. Esta infraestructura compartida es la que hace viable, dentro del alcance de un único estudiante con una GPU de consumo, construir el pipeline del Capítulo 4 y reproducir L2D como línea base. En 2024, Smit, Zhou, Reijnen, Wu, Chen, Zhang, Bukhsh, Zhang y Nuijten publican la primera revisión sistemática específica de GNN para JSSP/FJSP (arXiv:2406.14096) [35], usada aquí como referencia de clasificación taxonómica y como ancla de posicionamiento para la Sección 3.6.

**2024–2026 — Frontera: dinámica múltiple, generalización con incertidumbre, el debate homogéneo/heterogéneo, y agentes LLM.** Los trabajos más recientes identificados por la búsqueda PRISMA se organizan en cinco frentes:

- *Robustez ante múltiples perturbaciones simultáneas, sin GNN.* Lassoued, Lier y Schwung (arXiv:2601.09293, 2026) combinan averías de máquina (tiempo hasta el fallo Weibull) y llegadas de trabajo (proceso Gamma) sobre una representación de **red de Petri coloreada y temporizada**, con una política PPO enmascarable (sin codificador de grafo) [38]. Reportan mejoras de ~1.1–3 % sobre la mejor heurística bajo averías solas, y de hasta 8 % en el régimen combinado — pero **sin** experimento de generalización de tamaño.
- *Generalización de tamaño con un único tipo de incertidumbre.* Smit, Wu, Troubil, Zhang y Nuijten (AAAI 2025, arXiv:2412.14052) extienden la arquitectura **DAN** con un **módulo de procesamiento de escenarios** (SPM) que agrega, mediante atención, múltiples realizaciones muestreadas de tiempos de proceso estocásticos [42]. Entrenado en instancias de 20×10, SPM-DAN mantiene una brecha de −7.10 %/−7.07 % frente a CP-SAT al evaluarse, sin reentrenamiento, en instancias de 30×10/40×10 [42]. En la misma familia, Infantes, Roussel, Pereira, Jacquet y Benazera publican **Wheatley** (CPAIOR 2024, arXiv:2404.01308) [49], un enfoque GNN+DRL de código abierto orientado explícitamente a "mejorar la generalización y la escalabilidad" del DRL aplicado al JSSP con duraciones de tarea inciertas (distribución de probabilidad asociada a cada operación), buscando el cronograma que minimiza el makespan *promedio* en vez del determinista; su cifra exacta de brecha frente a CP-SAT/PDR no fue confirmada en esta ronda y se marca **[VERIFICAR]**, pero la disponibilidad pública del código lo hace un candidato natural de comparación directa. Tanto [42] como [49] comparten la misma limitación: una única fuente de incertidumbre (tiempos de proceso), sin averías ni llegadas dinámicas.
- *Reward shaping consciente de la perturbación, de un solo tipo.* Zhang, Yan, Yang y Hu (*Advanced Engineering Informatics*, 2024) proponen un esquema multiagente heterogéneo con **D3QN** y un mecanismo de *reward shaping* para el FJSP dinámico con tiempos de transporte [40]. Es el trabajo más cercano en el eje específico de "recompensa consciente de la perturbación" a la propuesta de esta tesis (Sección 4.3), pero su mecanismo está diseñado y evaluado para un único tipo de evento dinámico. Un trabajo relacionado de 2025 (arquitectura DCMAD3QN) ocupa el mismo nicho **[VERIFICAR]** [41].
- *El debate homogéneo vs. heterogéneo, revisitado en 2026.* Un supuesto casi universal desde 2022–2023 ([25], [26], [35]) es que el FJSP exige una representación de grafo **heterogénea** (nodos de operación y de máquina con relaciones tipadas) para escalar correctamente. Hoss, Link y Klarmann (arXiv:2604.23841, 2026) cuestionan directamente ese supuesto: proponen una **homogeneización basada en características** que proyecta los distintos roles de nodo a un espacio latente compartido, alcanzando complejidad **lineal** (frente a la complejidad cuadrática o mayor de las arquitecturas heterogéneas/atencionales) sin sacrificar, según los propios autores, la escalabilidad en instancias industriales de gran tamaño [50]. Esto matiza —sin invalidar— la Brecha 1 formulada en la Sección 3.6: la heterogeneidad no es la única vía hacia la generalización de tamaño, y la arquitectura *relacional* del Capítulo 4 (pesos separados por tipo de relación sobre un único tipo de nodo, ni plenamente homogénea ni plenamente heterogénea) ocupa precisamente el punto intermedio que este debate deja abierto. En el otro extremo del espectro de complejidad, "PPO-Graph Explorer" combina una **Graph Isomorphism Attention Network (GIAN)** con un PPO de exploración ajustada por entropía (EAE-PPO) para atacar el mismo problema —el conservadurismo exploratorio de PPO frente a la inestabilidad de SAC en espacios de acción discretos— mediante un mecanismo de atención estructurado sobre la topología disyuntiva (*Machines*, MDPI, vol. 14, n.º 3, art. 310, 2026) [51]; la lista completa de autores no pudo confirmarse de forma independiente en esta ronda y se marca **[VERIFICAR]**.
- *Agentes basados en modelos de lenguaje (LLM) para scheduling.* Área emergente, casi enteramente compuesta por preimpresiones de 2025–2026. Abgaryan, Cazenave y Harutyunyan publican **Starjob** (arXiv:2503.01877) [45]. Cao y Yuan proponen **ReflecSched** (arXiv:2508.01724), un agente LLM *zero-shot* evaluado con llegadas, averías con reparación, cancelaciones y trabajos urgentes simultáneos, con una brecha relativa promedio de 6.09–6.83 % [46]. Otros trabajos incluyen **NS4S** (IJCAI 2025) [47] y un enfoque de LLM como diseñador de función de recompensa para un DRL posterior (*Machines*, MDPI, 2025) [48]. Ninguno combina generalización de tamaño tipo GNN con robustez multi-perturbación mediante recompensa diseñada.

**Síntesis de la narrativa:** la trayectoria 2015–2026 muestra una progresión clara desde políticas end-to-end para JSSP estático [20], [23] hacia representaciones heterogéneas para FJSP estático [25], [26], de ahí hacia dos ramas que permanecen **separadas** —generalización de tamaño bajo un solo tipo de incertidumbre ([42], [49]) y robustez multi-perturbación sin generalización de tamaño ni codificador de grafo ([38])— y, más recientemente (2026), hacia un cuestionamiento activo de si la heterogeneidad arquitectónica es siquiera necesaria para escalar [50]. La propuesta de esta tesis se sitúa en la intersección de las dos primeras ramas, con un mecanismo de recompensa que generaliza el principio de *reward shaping* de disrupción única presente en [40], [41] a un régimen multi-perturbación, sobre una arquitectura relacional que, como se detalla en el Capítulo 4, adopta una posición intermedia en el debate homogéneo/heterogéneo abierto por [50].

## 3.3 Tabla comparativa de trabajos (valores extraídos de fuentes primarias)

> Confianza de cada celda: los valores numéricos citados provienen del resumen, una tabla de resultados, o el repositorio de código oficial del propio trabajo; **[VERIFICAR]** indica que la fuente primaria estaba bloqueada (paywall/protección anti-bot) o que solo se localizó una afirmación cualitativa.

| Ref. | Problema | Repr. del estado | Arquitectura | Algoritmo RL | Benchmarks | Líneas base | Resultados reportados | ¿Generaliza tamaño? | ¿Multi-perturbación? |
|---|---|---|---|---|---|---|---|---|---|
| Zhang et al., NeurIPS 2020 (L2D) [23] | JSSP estático | Grafo disyuntivo homogéneo | GIN | PPO | Generadas 6×6–100×20; Taillard; DMU | SPT, MWKR, MOPNR; OR-Tools | Brecha 13.6–29.2 % vs. OR-Tools; supera PDR en 87.5–92 % de instancias | **Sí** (20×20/30×20 → 100×20, brecha 9.4 %) | No |
| Park et al., IJPR 2021 [24] | JSSP estático | Grafo (paso de mensajes) | GNN propia | PPO | Propio; ORB, SWV, FT, LA, ABZ, YN, TA | PDR; MARL; MDQN | Error relativo 14.7 % (propio); 12.6–20.9 % (público) | Parcial | No |
| Song et al., IEEE TII 2023 [25] | FJSP estático | Grafo heterogéneo (op.+máquina) | GAT heterogéneo | PPO | **[VERIFICAR]** | PDR tradicionales | Cualitativo: supera PDR en instancias no vistas **[VERIFICAR cifra]** | Reclamado, no verificado | No reportado |
| Lei et al., ESWA 2022 [26] | FJSP estático | Grafo disyuntivo multi-MDP | GIN + multi-PPO | Multi-PPO | Brandimarte; sintéticas **[VERIFICAR]** | PDR y otros DRL **[VERIFICAR]** | Cualitativo: "robustez... generalización superior" **[VERIFICAR cifra]** | Reclamado, no verificado | No reportado |
| Liu et al., ESWA 2025/2026 [36] | FJSP dinámico multi-factor | Grafo heterogéneo dinámico | HGT | DRL **[VERIFICAR]** | **[VERIFICAR]** | PDR y "otras GNN" | Cualitativo, sin cifra confirmada | Reclamado | **Sí**, eje central |
| Lassoued et al., arXiv:2601.09293 (2026) [38] | JSSP dinámico bajo incertidumbre | Red de Petri (sin GNN) | MLP + PPO enmascarable | PPO (MPPO) | Raj; Taillard 15×15–100×20 | 12 PDR | +1.1–3 % (averías); hasta 8 % (averías+llegadas) | **No** — por tamaño, sin transferencia | **Sí**, explícito |
| Smit et al., AAAI 2025, arXiv:2412.14052 [42] | FJSP con tiempos estocásticos | Grafo heterogéneo (DAN) + SPM | SPM-DAN | PPO | SD1/SD3; mk, rdata, edata, vdata | PDR; CP-SAT; CP-stoch; DAN | −7.11 % vs. CP-SAT (20×10); generaliza a 30×10/40×10 sin reentrenar | **Sí, verificado** | No — solo tiempos de proceso |
| Infantes et al. (Wheatley), CPAIOR 2024, arXiv:2404.01308 [49] | JSSP con duraciones inciertas | Grafo (GNN, Wheatley) | GNN + DRL, código abierto | DRL **[VERIFICAR algoritmo exacto]** | **[VERIFICAR]** | **[VERIFICAR]** | Declara mejoras en "generalización y escalabilidad"; cifra exacta **[VERIFICAR]** | Reclamado | No — solo duraciones |
| Zhang, Yan, Yang y Hu, AEI 2024 [40] | FJSP dinámico con transporte | POMDP heterogéneo multiagente | Redes Q por agente | D3QN + reward shaping | **[VERIFICAR]** | "Líneas base de vanguardia" **[VERIFICAR]** | Cualitativo; cifras de motor de búsqueda no verificadas **[confianza baja]** | No especificado | Parcial (un tipo) |
| Hoss, Link y Klarmann, arXiv:2604.23841 (2026) [50] | JSSP/FJSP industrial a gran escala | Grafo **homogeneizado** por características (no heterogéneo) | GNN de complejidad lineal | **[VERIFICAR]** | **[VERIFICAR]** | Arquitecturas heterogéneas/atencionales previas | Declara escalabilidad lineal manteniendo desempeño; cifra exacta **[VERIFICAR]** | Reclamado (eje de escala, no de robustez) | No |
| "PPO-Graph Explorer" (GIAN+EAE-PPO), *Machines* 2026, 14(3):310 [51] | FJSP estático | Grafo disyuntivo | GIAN (atención estructurada) | EAE-PPO (PPO + curriculum de exploración) | **[VERIFICAR]** | PPO estándar; SAC | Declara mejor equilibrio exploración/estabilidad que PPO/SAC puros; cifra exacta **[VERIFICAR]** | No reportado | No |
| Cao y Yuan (ReflecSched), arXiv:2508.01724 (2025/2026) [46] | FJSP dinámico (agente LLM) | Prompt textual (sin grafo) | LLM + reflexión jerárquica | Ninguno (zero-shot) | GEN-Bench, MK-Bench, JMS-Bench | LLM-directo; PDR; HMPSAC; IDDQN | RPD 6.09–6.83 %; tasa de victoria 71.35 % | No aplica (sin entrenamiento) | **Sí, explícito** |

## 3.4 Scheduling dinámico/estocástico con RL

Más allá de los trabajos ya discutidos en la Sección 3.2, la literatura de *scheduling* dinámico con RL distingue tres formas de incorporar incertidumbre en la formulación MDP: (i) **reactiva pura**, donde el agente simplemente vuelve a decidir en cada nueva época sin ningún mecanismo explícito de anticipación (el enfoque adoptado como base por el pipeline del Capítulo 4); (ii) **consciente del riesgo**, donde el objetivo de entrenamiento incorpora una medida de cola de la distribución de resultados (p. ej., el VaR₉₅ de Smit et al. [42], o el makespan promedio bajo incertidumbre de Wheatley [49]) en lugar del valor esperado; y (iii) **anticipatoria**, donde el agente recibe explícitamente información distribucional sobre eventos futuros — una dirección que Lassoued et al. señalan como trabajo futuro no resuelto en su propio artículo [38]. Esta tesis adopta el enfoque (i) como base, con un componente de (ii) previsto para el índice de robustez de fases posteriores del proyecto, y deja explícitamente fuera de alcance el enfoque (iii) plenamente anticipatorio.

Un patrón que se repite en los trabajos de esta sección es la tendencia a **aislar un único tipo de perturbación por diseño experimental** (solo averías, solo llegadas, o solo tiempos estocásticos), incluso cuando la arquitectura subyacente podría en principio soportar más de uno. Excepciones parciales son Lassoued et al. [38] (averías + llegadas, sin GNN ni generalización de tamaño) y ReflecSched [46] (múltiples tipos de evento simultáneos, mediante un agente LLM sin entrenamiento). Este patrón de aislamiento de una sola fuente de incertidumbre —confirmado ahora en once de los doce trabajos técnicos de la Tabla 3.1, incluidas las dos incorporaciones de esta ronda ([42]/[49] para tiempos de proceso, [38]/[40]/[41] para eventos discretos)— es precisamente el que la Sección 3.6 formaliza como la brecha central de esta tesis.

## 3.5 Generalización, arquitecturas de grafo (homogéneas, relacionales, heterogéneas) y agentes LLM

La base teórica de la generalización de tamaño en políticas basadas en GNN —invariancia a permutaciones, número de parámetros independiente de *n* y *m*, localidad del paso de mensajes— se apoya en los resultados de expresividad de Xu et al. [19]. Empíricamente, L2D [23] fue el primer trabajo en demostrar esta propiedad de forma sistemática para JSSP, y Smit et al. [42] ofrecen, de los trabajos 2024–2026 revisados, la evidencia cuantitativa más rigurosa de generalización de tamaño **en presencia de incertidumbre**. La revisión sistemática de Smit et al. (2024) sobre GNN para JSSP/FJSP [35] constituye el mapa taxonómico más actualizado disponible para clasificar arquitecturas (GCN, GAT, GIN, heterogéneas) y esta tesis la adopta como referencia de clasificación.

En cuanto a la elección arquitectónica específica, la literatura 2022–2026 revisada dibuja un **espectro**, no una dicotomía, entre tres puntos: (a) el grafo **homogéneo puro** de L2D [23], donde toda arista se agrega con un único conjunto de pesos independientemente de si conecta un par de precedencia o un par de máquina; (b) el grafo **heterogéneo** de Song et al. [25] y Liu et al. [36], donde tipos de nodo y de arista distintos reciben transformaciones y/o mecanismos de atención propios; y (c), en el extremo opuesto y de aparición muy reciente, la **homogeneización deliberada por características** de Hoss et al. [50], que proyecta explícitamente los distintos roles de nodo a un espacio latente compartido para lograr complejidad lineal en instancias industriales de gran escala, cuestionando que la heterogeneidad sea condición necesaria para escalar. La arquitectura del Capítulo 4 —una capa GIN **relacional**, con transformaciones lineales separadas para mensajes de predecesor, de sucesor y de máquina compartida, pero un único tipo de nodo— ocupa un punto intermedio de este espectro: más expresiva que (a) porque distingue el significado semántico opuesto de cada relación (un predecesor restringe, un sucesor es restringido, un competidor de máquina disputa el mismo recurso), pero sin la complejidad paramétrica ni el costo computacional de (b). Esta elección se justifica y se contrasta explícitamente contra [23], [25] y [50] en la Sección 4.1.3.

Los **agentes basados en LLM** (Sección 3.2) representan un paradigma metodológicamente distinto: no entrenan una política mediante gradiente de recompensa sobre un conjunto fijo de parámetros, sino que emplean un modelo de lenguaje preentrenado, guiado por *prompting*, simulación de rollouts, o ajuste fino supervisado [45]–[48]. Su relación con la generalización de tamaño es cualitativamente distinta a la de las GNN: en lugar de una invarianza arquitectónica a *n* y *m*, dependen de la capacidad del LLM de generalizar a partir de su preentrenamiento masivo. Esta tesis los trata como punto de contraste discursivo en el Capítulo 7, no como competidor metodológico directo, dado que el pipeline implementado (Capítulo 4) es una política GNN+PPO entrenada, no un agente LLM.

## 3.6 Análisis de brechas → preguntas de investigación

| Brecha identificada | Evidencia acumulada (2020–2026) | Pregunta de investigación asociada | Estado final |
|---|---|---|---|
| Poca evaluación conjunta de generalización en tamaño **y** robustez bajo múltiples perturbaciones | Smit et al. [42] y Wheatley [49] generalizan en tamaño bajo **una** fuente de incertidumbre (tiempos de proceso); Lassoued et al. [38] combina **dos** tipos de perturbación pero sin generalización de tamaño ni GNN; Liu et al. [36] reclama ambos ejes cualitativamente, sin cifra verificada. Ningún trabajo confirma ambos ejes de forma simultánea y cuantitativa. | **PP:** ¿Puede una política GNN+PPO generalizar en tamaño **y** ser robusta a averías/tiempos estocásticos/llegadas simultáneas? | **Brecha confirmada — sigue abierta.** Eje de novedad más defendible de la tesis. |
| Recompensas conscientes de la perturbación poco exploradas, y solo de un tipo | Zhang, Yan, Yang y Hu [40] y DCMAD3QN [41] exploran *reward shaping* consciente de disrupción, pero cada uno para un **único** tipo de evento, en un esquema multiagente sin GNN. | **SP1:** ¿Mejora una recompensa consciente de la perturbación la robustez sin dañar la optimalidad nominal? | **Brecha se estrecha, no se cierra.** El mecanismo propuesto (Sección 4.3, extensión posterior a multi-perturbación) debe diferenciarse explícitamente de [40]/[41]. |
| Comparación limitada frente a CP-SAT + GA + PDRs simultáneamente | Ningún trabajo de la ronda 2024–2026 resuelve esta limitación metodológica general; [38], [40], [46] comparan contra PDR y/o DRL/LLM previos, no simultáneamente contra CP-SAT + GA + PDR con presupuesto de tiempo equiparado. | **SP2:** ¿Cuál es la brecha real frente al techo CP-SAT y frente a GA/PDRs en cada régimen? | **Brecha confirmada — sin cambios.** El protocolo experimental de la tesis (Capítulo 5) ya contempla esta comparación simultánea, incluida en el Sprint 1 la comparación PDR/CP-SAT ya funcional (Sección 3.7). |
| Reproducibilidad y protocolo estadístico débiles | Ninguno de los trabajos 2024–2026 revisados reporta un protocolo Friedman+Nemenyi con diagramas de diferencia crítica sobre sus comparaciones principales. | **SP3:** ¿Son las diferencias estadísticamente significativas, con tamaños de efecto reportados? | **Brecha confirmada — sin cambios.** Se mantiene como contribución de rigor metodológico. |
| El supuesto de que la heterogeneidad arquitectónica es necesaria para escalar | Consenso implícito 2022–2024 ([25], [26], [35], [36]); cuestionado directamente en 2026 por Hoss et al. [50], quienes logran complejidad lineal con homogeneización por características. | *(Brecha emergente, no una pregunta de investigación original)* | **Territorio nuevo, relevante para la justificación arquitectónica del Capítulo 4.** Debe citarse [50] explícitamente al justificar la elección de una GIN relacional (ni homogénea ni heterogénea) en la Sección 4.1.3. |
| Agentes basados en LLM para *scheduling* | Área nueva y activa, casi exclusivamente en preimpresiones 2025–2026 [45]–[48]; ninguno combina agentes LLM con el eje específico de esta tesis. | *(No era una pregunta de investigación original)* | **Territorio nuevo, adyacente, no competidor directo.** Discusión comparativa recomendada en el Capítulo 7. |

**Veredicto consolidado sobre la afirmación de novedad:** la combinación específica que esta tesis reclama como contribución —generalización de tamaño **y** robustez ante múltiples tipos de perturbación simultáneos, mediante una recompensa explícitamente diseñada para ese compromiso, sobre una política GNN+PPO— continúa sin resolverse de forma conjunta en la literatura identificada hasta la fecha de esta revisión (28 de julio de 2026). Los cuatro trabajos más cercanos ([38], [40]/[41], [42], [49]) cubren cada uno un eje distinto de la contribución, pero ninguno los combina; el trabajo de Hoss et al. [50] añade una segunda dimensión de novedad potencial (la elección arquitectónica relacional del Capítulo 4 frente al debate homogéneo/heterogéneo) que no estaba presente en la iteración anterior de este capítulo. Se recomienda mantener la afirmación de novedad **matizada**, nombrando explícitamente la distinción eje-único vs. eje-conjunto en el Capítulo 1 y en la discusión final (Capítulo 7).

## 3.7 Síntesis de cierre: de la brecha a la implementación

Los seis meses de revisión bibliográfica sintetizados en este capítulo definen un espacio de investigación abierto; el primer sprint de codificación de la tesis (documentado en `VALIDATION_WEEK1.md`, código en el repositorio adjunto) es el primer paso, deliberadamente conservador, hacia ese espacio, y conviene precisar honestamente dónde se sitúa hoy frente al mapa trazado arriba. El pipeline actualmente implementado —entorno Gymnasium de grafo disyuntivo, codificador GIN relacional de tres relaciones (predecesor, sucesor, máquina) y política `MaskablePPO`, entrenado y validado sobre una única instancia estática (`ft06`, 188/188 pruebas superadas, 20 000 pasos de entrenamiento sin fallas)— reproduce, en su núcleo algorítmico, la arquitectura de L2D [23]: JSSP clásico, estático, determinista, de instancia única, sin flexibilidad de máquina y sin ningún tipo de perturbación dinámica. No implementa todavía ninguno de los tres ejes que motivan la novedad de la tesis (generalización de tamaño entre instancias, robustez multi-perturbación, o recompensa consciente de disrupciones): la propia arquitectura de la política (Sección 4.1.3) fija el tamaño de la capa de acción al número de operaciones de `ft06`, una limitación documentada explícitamente en el código (`models/policy.py`) y compartida, no por casualidad, con la limitación original de L2D antes de que Zhang et al. introdujeran su cabeza de acción por nodo.

Esta distancia entre el estado del arte revisado y el estado actual del código no es una desviación del plan, sino su cumplimiento literal: la meta declarada del primer sprint era "un pipeline que corre y está probado, no un modelo convergido" (`README.md`), es decir, validar la fontanería —sincronización entorno/recompensa/máscara/gradiente— antes de introducir la complejidad que las Secciones 3.4–3.6 identifican como la frontera real. En términos de la Tabla del Capítulo 3.6, el sprint actual cierra una precondición necesaria pero no suficiente de la brecha PP: demuestra que la arquitectura relacional elegida entrena establemente y aprende una señal genuina (makespan medio de 98.5 a 67.8 en las primeras/últimas 20 episodios, mejor makespan de 57 contra un óptimo probado de 55) sobre el caso base homogéneo-JSSP, que es exactamente el terreno ya ganado por la literatura 2020–2021 [23], [24]. Las tres extensiones que separan este punto de partida de la contribución reclamada —cabeza de acción agnóstica al tamaño (respondiendo a RQ1/Brecha 1, y a la posición intermedia frente a [23]/[50] justificada en 3.5), inyección de perturbaciones estocásticas múltiples en el entorno (RQ2), y una función de recompensa multi-perturbación que generalice el principio de [40]/[41] (SP1)— son, explícitamente, el trabajo de los sprints siguientes, no una promesa retrospectiva sobre lo ya construido. Mantener esta distinción con precisión —entre lo que el Capítulo 3 identifica como brecha y lo que el Capítulo 4 documenta como implementado hasta la fecha— es, en sí mismo, parte del rigor metodológico que esta tesis reclama frente a las once de doce referencias técnicas de la Tabla 3.1 que aíslan un solo eje de su propia contribución.

## Referencias

**Fundamentos de scheduling y complejidad**

[1] R. L. Graham, E. L. Lawler, J. K. Lenstra, A. H. G. Rinnooy Kan, "Optimization and approximation in deterministic sequencing and scheduling: a survey," *Annals of Discrete Mathematics*, vol. 5, pp. 287–326, 1979, doi:10.1016/S0167-5060(08)70356-X.

[2] M. R. Garey, D. S. Johnson, R. Sethi, "The complexity of flowshop and jobshop scheduling," *Mathematics of Operations Research*, vol. 1, no. 2, pp. 117–129, 1976, doi:10.1287/moor.1.2.117.

[3] M. L. Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 6th ed., Springer, 2022, doi:10.1007/978-3-031-05921-6.

[4] J. Carlier, É. Pinson, "An Algorithm for Solving the Job-Shop Problem," *Management Science*, vol. 35, no. 2, pp. 164–176, 1989, doi:10.1287/mnsc.35.2.164.

**Benchmarks**

[5] H. Fisher, G. L. Thompson, "Probabilistic learning combinations of local job-shop scheduling rules," in *Industrial Scheduling*, J. F. Muth and G. L. Thompson, Eds., Englewood Cliffs, NJ: Prentice-Hall, 1963, pp. 225–251.

[6] S. Lawrence, "Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques," Tech. Rep., Graduate School of Industrial Administration, Carnegie-Mellon Univ., 1984.

[7] E. Taillard, "Benchmarks for basic scheduling problems," *European Journal of Operational Research*, vol. 64, no. 2, pp. 278–285, 1993, doi:10.1016/0377-2217(93)90182-M.

[8] E. Demirkol, S. Mehta, R. Uzsoy, "Benchmarks for shop scheduling problems," *European Journal of Operational Research*, vol. 109, no. 1, pp. 137–141, 1998, doi:10.1016/S0377-2217(97)00019-2.

[9] P. Brandimarte, "Routing and scheduling in a flexible job shop by tabu search," *Annals of Operations Research*, vol. 41, no. 3, pp. 157–183, 1993, doi:10.1007/BF02023073.

[10] J. Hurink, B. Jurisch, M. Thole, "Tabu search for the job-shop scheduling problem with multi-purpose machines," *OR Spektrum*, vol. 15, no. 4, pp. 205–215, 1994, doi:10.1007/BF01719451.

[11] J. E. Beasley, "OR-Library," Brunel Univ. [Online]. Available: https://www.brunel.ac.uk/~mastjjb/jeb/info.html. Accessed 27 Jul. 2026.

**RL y GNN (fundamentos)**

[12] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Cambridge, MA: MIT Press, 2018.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[14] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2015/2016.

[15] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015, doi:10.1038/nature14236.

[16] T. N. Kipf, M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *Proc. ICLR*, 2017, arXiv:1609.02907.

[17] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs," in *Proc. NIPS (NeurIPS)*, 2017, pp. 1024–1034, arXiv:1706.02216.

[18] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph Attention Networks," in *Proc. ICLR*, 2018, arXiv:1710.10903.

[19] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?," in *Proc. ICLR*, 2019, arXiv:1810.00826.

**NCO / DRL para scheduling**

[20] O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks," in *Proc. NeurIPS*, 2015, pp. 2692–2700, arXiv:1506.03134.

[21] W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing Problems!," in *Proc. ICLR*, 2019, arXiv:1803.08475.

[22] Y.-D. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, S. Min, "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning," in *Proc. NeurIPS*, 2020, arXiv:2010.16011.

[23] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, C. Xu, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," in *Proc. NeurIPS*, 2020, arXiv:2010.12367.

[24] J. Park, J. Chun, S. H. Kim, Y. Kim, J. Park, "Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning," *International Journal of Production Research*, vol. 59, no. 11, pp. 3360–3377, 2021, doi:10.1080/00207543.2020.1870013.

[25] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 2, pp. 1600–1610, 2023, doi:10.1109/TII.2022.3189725.

[26] K. Lei, P. Guo, W. Zhao, Y. Wang, L. Qian, X. Meng, L. Tang, "A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem," *Expert Systems with Applications*, vol. 205, art. 117796, 2022, doi:10.1016/j.eswa.2022.117796.

**Entornos y benchmarks de código**

[27] P. Tassel, M. Gebser, K. Schekotihin, "A Reinforcement Learning Environment For Job-Shop Scheduling," arXiv:2104.03760, 2021.

[28] C. Waubert de Puiseau, J. Peters, C. Dörpelkus, H. Tercan, T. Meisen, "schlably: A Python framework for deep reinforcement learning based scheduling experiments," *SoftwareX*, vol. 22, art. 101383, 2023, doi:10.1016/j.softx.2023.101383.

[29] R. Reijnen, I. G. Smit, H. Zhang, Y. Wu, Z. Bukhsh, Y. Zhang, "Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods," arXiv:2308.12794, 2023.

**Simulación y estadística**

[30] A. M. Law, *Simulation Modeling and Analysis*, 6th ed., McGraw Hill, 2024.

[31] R. G. Sargent, "Verification and Validation of Simulation Models," in *Proc. 2011 Winter Simulation Conf.*, S. Jain, R. R. Creasey, J. Himmelspach, K. P. White, M. Fu, Eds., IEEE, 2011, pp. 183–198, doi:10.1109/WSC.2011.6147750.

[32] J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *Journal of Machine Learning Research*, vol. 7, pp. 1–30, 2006.

**Herramientas**

[33] L. Perron, V. Furnon, "OR-Tools," Google, v9.x, 2025. [Online]. Available: https://developers.google.com/optimization/. Referencia arbitrada del solver subyacente: L. Perron, F. Didier, S. Gay, "The CP-SAT-LP Solver," in *Proc. 29th Int. Conf. on Principles and Practice of Constraint Programming (CP 2023)*, LIPIcs, vol. 280, art. 3, 2023, doi:10.4230/LIPIcs.CP.2023.3.

**Fundamento del grafo disyuntivo**

[34] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.º 9 bis, SEMA, Montrouge, 1964.

**Incorporaciones 2024–2026 (revisión sistemática ampliada)**

[35] I. G. Smit, J. Zhou, R. Reijnen, Y. Wu, J. Chen, C. Zhang, Z. Bukhsh, Y. Zhang, W. Nuijten, "Graph Neural Networks for Job Shop Scheduling Problems: A Survey," *Computers & Operations Research*, 2024, arXiv:2406.14096.

[36] G. Liu, M. Liao, W. Chen, Z. Zhang, H. Wan, Y. Lin, "Multi-Faceted Dynamic Flexible Job Shop Scheduling via Heterogeneous Graph Transformer and Deep Reinforcement Learning," *Expert Systems with Applications*, vol. 303, art. 130532, 2025/2026, doi:10.1016/j.eswa.2025.130532. **[VERIFICAR — texto completo no accesible]**

[37] Autores no confirmados de forma independiente, "Schedule repair for flexible job shops under machine breakdowns by deep reinforcement learning," 2025. **[VERIFICAR — cita incompleta]**

[38] S. Lassoued, S. Lier, A. Schwung, "Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures," arXiv:2601.09293, 2026.

[39] R. Chen, Y. Mei, F. Zhang, M. Zhang, "DyRo-MCTS: A Robust Monte Carlo Tree Search Approach to Dynamic Job Shop Scheduling," arXiv:2509.21902, 2025.

[40] L. Zhang, Y. Yan, C. Yang, Y. Hu, "Dynamic flexible job-shop scheduling by multi-agent reinforcement learning with reward-shaping," *Advanced Engineering Informatics*, vol. 62, art. 102872, 2024, doi:10.1016/j.aei.2024.102872.

[41] Autores no confirmados de forma independiente, arquitectura "DCMAD3QN," *Expert Systems with Applications*, 2025. **[VERIFICAR — cita incompleta]**

[42] I. G. Smit, Y. Wu, P. Troubil, Y. Zhang, W. P. M. Nuijten, "Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems," in *Proc. AAAI*, 2025, arXiv:2412.14052.

[43] Autores no confirmados de forma independiente, "Learning to Optimize Job Shop Scheduling Under Structural Uncertainty" (método UP-AAC), arXiv:2601.21389, 2026. **[VERIFICAR]**

[44] "RobustGANTT" — identificador de arXiv no confirmado de forma independiente. **[VERIFICAR antes de citar]**

[45] H. Abgaryan, T. Cazenave, A. Harutyunyan, "Starjob: Dataset for LLM-Driven Job Shop Scheduling," arXiv:2503.01877, 2025.

[46] S. Cao, Y. Yuan, "ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection," arXiv:2508.01724, 2025 (v3, 2026).

[47] J. Zhang, C. Luo, Z. Su, Q. Zhang et al., "NS4S: Neighborhood Search for Scheduling Problems via Large Language Models," in *Proc. IJCAI*, 2025. **[VERIFICAR lista completa de autores]**

[48] Autores no confirmados de forma independiente, "Large Language Model-Assisted Deep Reinforcement Learning from Human Feedback for Job Shop Scheduling," *Machines* (MDPI), vol. 13, no. 5, art. 361, 2025. **[VERIFICAR]**

**Incorporaciones de esta ronda (Sección 3.2, frontera 2024–2026, búsqueda ampliada 28/07/2026)**

[49] G. Infantes, S. Roussel, P. Pereira, A. Jacquet, E. Benazera, "Learning to Solve Job Shop Scheduling under Uncertainty" (Wheatley), in *Proc. CPAIOR*, 2024, arXiv:2404.01308.

[50] J. Hoss, M. Link, N. Klarmann, "Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs," arXiv:2604.23841, 2026.

[51] Autores no confirmados de forma independiente, "PPO-Graph Explorer: A New Method for Flexible Job Shop Scheduling via Entropy-Guided Attention Networks," *Machines* (MDPI), vol. 14, no. 3, art. 310, 2026. **[VERIFICAR — lista de autores no accesible en el resumen público]**

---

## Acciones pendientes antes de la sustentación

1. Re-ejecutar las cuatro cadenas PRISMA literales directamente en Scopus, Web of Science e IEEE Xplore (acceso institucional UNIMINUTO) para reemplazar los conteos aproximados de la Sección 3.1 por conteos oficiales, y completar el cribado por texto completo.
2. Leer en su totalidad [38], [40]/[41], [42] y [49] y ampliar el párrafo de diferenciación ya presente en la Sección 3.6.
3. Confirmar directamente contra la fuente primaria las listas de autores marcadas **[VERIFICAR]**: [37], [40]/[41] (confirmación parcial ya realizada), [43], [47] y [51].
4. Obtener acceso de texto completo a [25] y [26] (vía biblioteca UNIMINUTO o solicitud a los autores) para reemplazar las cifras cualitativas de la Tabla 3.1 por valores numéricos verificados.
