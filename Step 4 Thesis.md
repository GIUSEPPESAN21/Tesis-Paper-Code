# Fundamento Teórico, Estructura de Tesis y Especificación Procedimental
## "Aprendizaje por Refuerzo Profundo Generalizable para la Programación Dinámica de Talleres de Trabajo Flexibles bajo Perturbaciones Estocásticas"

> **AVISO DE VERIFICACIÓN (CRÍTICO Y HONESTO):** En este entorno de ejecución las herramientas de búsqueda web en vivo **NO estuvieron disponibles** (todas las llamadas a `web_search`/`web_fetch`, tanto mías como del subagente, fallaron con el error "tool was not provided"). Por tanto **no pude realizar verificación bibliográfica en vivo**. La bibliografía se reconstruyó a partir de conocimiento de base y se marca con niveles de confianza [ALTA]/[MEDIA]/[BAJA]. Cada DOI, volumen, número y página **debe re-verificarse** en Scopus / Web of Science / IEEE Xplore / ScienceDirect / DOI.org / arXiv antes de la entrega de la tesis. He aplicado y señalado explícitamente las correcciones de las que tengo alta confianza (orden de autores de L2D; afiliación de *schlably*; ft10 resuelto por Carlier & Pinson 1989). Los ítems 2024–2026 son los de **mayor riesgo** y se marcan como no verificables aquí.

---

# PARTE A — ESTRUCTURA COMPLETA DE LA TESIS

Cada sección indica: contenido, páginas aproximadas, fuentes que la alimentan y su estado: **[TEÓRICO - LISTO]** (redactable ya, y en gran parte redactado en la PARTE B/C/D), **[TEÓRICO - POR REDACTAR]** (teórico pero pendiente de escritura), **[EMPÍRICO - REQUIERE EJECUCIÓN]** (requiere resultados del estudiante).

**Portada, dedicatoria, agradecimientos** — [TEÓRICO - POR REDACTAR] — 3 pp.

**Resumen / Abstract** (español + inglés, ~250 palabras c/u, 5 palabras clave) — [EMPÍRICO - REQUIERE EJECUCIÓN] (los números finales se insertan al cierre; el texto marco puede pre-redactarse) — 2 pp.

**Índice, lista de figuras, tablas, símbolos y acrónimos** — [TEÓRICO - POR REDACTAR] — 4 pp.

## Cap. 1 — Introducción — 10 pp.
- 1.1 Contexto y motivación (Industria 4.0, manufactura y mantenimiento, digitalización de talleres) — [TEÓRICO - LISTO] — 2 pp — fuentes: Pinedo (2016); surveys DRL-scheduling.
- 1.2 Planteamiento del problema (FJSP dinámico bajo perturbaciones estocásticas) — [TEÓRICO - LISTO] — 2 pp — B1, B2.3.
- 1.3 Preguntas de investigación e hipótesis — [TEÓRICO - LISTO] — 1,5 pp — C4, D7.
- 1.4 Objetivos (general y específicos) — [TEÓRICO - LISTO] — 1 p.
- 1.5 Alcance, delimitaciones y supuestos (SOLO datos simulados; un estudiante; 9 meses; GPU única) — [TEÓRICO - LISTO] — 1,5 pp.
- 1.6 Contribuciones esperadas — [TEÓRICO - LISTO] — 1 p — C4.
- 1.7 Estructura del documento — [TEÓRICO - LISTO] — 1 p.

## Cap. 2 — Marco Teórico — 35 pp. (contenido íntegro en PARTE B)
- 2.1 Fundamentos de la programación de la producción — [TEÓRICO - LISTO] — 4 pp — Graham et al. 1979; Pinedo 2016; Garey, Johnson & Sethi 1976.
  - 2.1.1 Taxonomía de problemas de scheduling; 2.1.2 Notación α|β|γ; 2.1.3 Objetivos regulares/no regulares; 2.1.4 Complejidad y NP-dureza.
- 2.2 Formulación matemática del JSSP y FJSP — [TEÓRICO - LISTO] — 6 pp.
  - 2.2.1 MILP disyuntivo (big-M); 2.2.2 Grafo disyuntivo; 2.2.3 Extensión dinámica/estocástica.
- 2.3 Procesos de decisión de Markov — [TEÓRICO - LISTO] — 4 pp — Sutton & Barto 2018.
  - 2.3.1 MDP, SMDP y CTMDP; 2.3.2 El FJSP dinámico como MDP.
- 2.4 Aprendizaje por refuerzo profundo — [TEÓRICO - LISTO] — 6 pp — Sutton & Barto 2018; Schulman et al. 2017, 2016.
  - 2.4.1 De lo tabular a la aproximación de función; 2.4.2 Valor vs. política; 2.4.3 Actor-crítico y GAE; 2.4.4 PPO.
- 2.5 Redes neuronales de grafos — [TEÓRICO - LISTO] — 6 pp — Kipf & Welling 2017; Hamilton et al. 2017; Veličković et al. 2018; Xu et al. 2019.
  - 2.5.1 Paso de mensajes; 2.5.2 GCN/SAGE/GAT/GIN; 2.5.3 Poder expresivo y test WL; 2.5.4 GNN heterogéneas para FJSP; 2.5.5 Invariancia a permutaciones y generalización.
- 2.6 Reglas de despacho por prioridad — [TEÓRICO - LISTO] — 3 pp.
- 2.7 Métodos exactos y metaheurísticas (CP-SAT, GA) — [TEÓRICO - LISTO] — 3 pp.
- 2.8 Generalización en NCO y scheduling robusto/reactivo — [TEÓRICO - LISTO] — 2 pp.
- 2.9 Simulación de eventos discretos — [TEÓRICO - LISTO] — 1 p — Law 2015; Sargent 2011.

## Cap. 3 — Estado del Arte — 22 pp. (contenido en PARTE C)
- 3.1 Protocolo de revisión sistemática PRISMA — [TEÓRICO - LISTO] — 3 pp.
- 3.2 Evolución 2018–2026 de DRL para JSSP/FJSP — [TEÓRICO - POR REDACTAR] (requiere verificar SOTA 2024–2026) — 6 pp.
- 3.3 Tabla comparativa de trabajos — [TEÓRICO - LISTO] (esqueleto listo; celdas de SOTA reciente por verificar) — 4 pp.
- 3.4 Scheduling dinámico/estocástico con RL — [TEÓRICO - POR REDACTAR] — 3 pp.
- 3.5 Generalización, GNN heterogéneas y agentes LLM — [TEÓRICO - POR REDACTAR] — 3 pp.
- 3.6 Análisis de brechas → preguntas de investigación — [TEÓRICO - LISTO] — 3 pp.

## Cap. 4 — Metodología — 25 pp. (contenido en PARTE D)
- 4.1 Enfoque general y arquitectura de solución — [TEÓRICO - LISTO] — 3 pp.
- 4.2 Entorno de simulación DES + contrato Gymnasium — [TEÓRICO - LISTO] — 6 pp.
- 4.3 Formulación MDP: estado, acciones, recompensa, enmascarado — [TEÓRICO - LISTO] — 6 pp.
- 4.4 Codificador GNN heterogéneo + política PPO — [TEÓRICO - LISTO] — 5 pp.
- 4.5 Modelos de perturbación (TTF/TTR, tiempos, llegadas) — [TEÓRICO - LISTO] — 3 pp.
- 4.6 Líneas base (PDRs, GA, CP-SAT) — [TEÓRICO - LISTO] — 2 pp.

## Cap. 5 — Diseño Experimental — 12 pp.
- 5.1 Matriz de factores y experimentos — [TEÓRICO - LISTO] — 4 pp.
- 5.2 Protocolo de evaluación y métricas — [TEÓRICO - LISTO] — 4 pp.
- 5.3 Procedimiento estadístico (Wilcoxon; Friedman + Nemenyi; diagramas CD) — [TEÓRICO - LISTO] — 3 pp.
- 5.4 Reproducibilidad, semillas, hardware — [TEÓRICO - LISTO] — 1 p.

## Cap. 6 — Resultados — 20 pp.
- 6.1 Validación del entorno de simulación — [EMPÍRICO - REQUIERE EJECUCIÓN] — 3 pp.
- 6.2 Desempeño estático (JSSP/FJSP benchmarks) — [EMPÍRICO - REQUIERE EJECUCIÓN] — 5 pp.
- 6.3 Generalización en tamaño — [EMPÍRICO - REQUIERE EJECUCIÓN] — 4 pp.
- 6.4 Robustez bajo perturbaciones — [EMPÍRICO - REQUIERE EJECUCIÓN] — 5 pp.
- 6.5 Tiempos de decisión y coste computacional — [EMPÍRICO - REQUIERE EJECUCIÓN] — 3 pp.

## Cap. 7 — Discusión — 10 pp. — [EMPÍRICO - REQUIERE EJECUCIÓN] (interpretación de resultados) + partes [TEÓRICO - POR REDACTAR] (relación con la literatura, límites teóricos de generalización).

## Cap. 8 — Conclusiones y trabajo futuro — 6 pp. — [TEÓRICO - POR REDACTAR] (armazón listo) + cierre empírico.

**Referencias** (PARTE F) — [TEÓRICO - LISTO, sujeto a re-verificación]. 
**Anexos:** A) MILP completo; B) hiperparámetros y arquitectura; C) tablas extendidas de resultados [EMPÍRICO]; D) protocolo PRISMA con diagrama de flujo; E) documento de pre-registro; F) tablas de óptimos/mejores conocidos por instancia.

**Total estimado:** ~150–165 pp.

---

# PARTE B — MARCO TEÓRICO COMPLETO Y REDACTADO

## B1. Fundamentos de la programación de la producción

La programación (*scheduling*) asigna recursos escasos (máquinas) a tareas (operaciones) a lo largo del tiempo optimizando uno o más criterios. Se emplea la **notación de tres campos α|β|γ** de Graham, Lawler, Lenstra y Rinnooy Kan (1979): **α** describe el entorno de máquinas (1 máquina; P_m idénticas paralelas; Q_m uniformes; R_m no relacionadas; F_m *flow shop*; J_m *job shop*; FJ_m *flexible job shop*); **β** las características y restricciones del proceso (r_j fechas de liberación, d_j fechas de entrega, *prmp* preemption, *brkdwn* averías, s_jk tiempos de preparación, *prec* precedencia); y **γ** el criterio objetivo (C_max makespan, ΣC_j, Σw_jC_j, L_max, T_max, ΣT_j, ΣU_j).

El **JSSP** clásico es J_m || C_max: n trabajos, cada uno con una secuencia fija de m operaciones, cada operación asignada a una máquina predeterminada; se minimiza el makespan. El **FJSP** (FJ_m) generaliza asignando a cada operación un **conjunto de máquinas elegibles**, añadiendo una decisión de **enrutamiento** (routing) además del **secuenciamiento** (sequencing).

**Objetivos regulares vs. no regulares.** Un objetivo es *regular* si es no decreciente en los tiempos de finalización C_j (p.ej. C_max, ΣT_j, L_max); para objetivos regulares existe siempre una programación *activa* óptima, lo que restringe el espacio de búsqueda. Los objetivos *no regulares* (p.ej. earliness + tardiness ponderados, ΣαE_j+βT_j) pueden requerir insertar tiempo ocioso deliberado, invalidando la optimalidad de las programaciones activas.

**Complejidad computacional.** Garey, Johnson y Sethi (1976) demostraron que el JSSP para minimizar makespan es **fuertemente NP-difícil**: J_m||C_max con m≥3 es NP-difícil en sentido fuerte, e incluso J2||C_max con más de dos operaciones por trabajo es difícil, mientras que ciertos casos particulares (p.ej. J2 con a lo sumo dos operaciones por trabajo, vía la regla de Jackson) son polinomiales. La célebre instancia **ft10** (10×10) de Fisher & Thompson (1963) fue resuelta óptimamente (C_max = 930) recién por **J. Carlier y É. Pinson en 1989** ("An Algorithm for Solving the Job-Shop Problem", *Management Science* 35(2):164–176), es decir **26 años** después de su publicación, ilustrando dramáticamente la dureza práctica del problema. El FJSP es al menos tan difícil como el JSSP, pues lo contiene como caso particular (|M_{j,h}|=1). Esta dureza intrínseca justifica el recurso a heurísticas, metaheurísticas, métodos exactos con límite de tiempo y, recientemente, **políticas aprendidas** que amortizan el coste de decisión.

## B2. Formulación matemática formal del JSSP y FJSP

### B2.1 Modelo MILP del FJSP (formulación disyuntiva con big-M)

**Conjuntos.** J={1,…,n} trabajos; para el trabajo j, O_j={O_{j,1},…,O_{j,n_j}} operaciones en orden de precedencia; M={1,…,m} máquinas; para cada operación O_{j,h}, M_{j,h}⊆M el conjunto de máquinas elegibles.

**Parámetros.** p_{j,h,k} = tiempo de proceso de O_{j,h} en la máquina k∈M_{j,h}; r_j = fecha de liberación; d_j = fecha de entrega; V = constante grande (big-M), V ≥ Σ_{j,h} max_k p_{j,h,k}.

**Variables de decisión.**
- x_{j,h,k}∈{0,1}: 1 si O_{j,h} se asigna a la máquina k.
- s_{j,h}≥0: instante de inicio de O_{j,h}.
- c_{j,h}≥0: instante de finalización de O_{j,h}.
- y_{j,h,j',h',k}∈{0,1}: 1 si O_{j,h} precede a O_{j',h'} en la máquina k.
- C_max ≥ 0: makespan.

**Función objetivo.**
$$\min\; C_{\max}$$

**Restricciones.**
1. Asignación única:  $\sum_{k\in M_{j,h}} x_{j,h,k}=1 \quad \forall j,h.$
2. Duración/finalización:  $c_{j,h}=s_{j,h}+\sum_{k\in M_{j,h}} p_{j,h,k}\,x_{j,h,k}\quad\forall j,h.$
3. Precedencia intra-trabajo:  $s_{j,h}\ge c_{j,h-1}\quad\forall j,\;h\ge 2.$
4. Liberación:  $s_{j,1}\ge r_j\quad\forall j.$
5. No solapamiento (disyuntiva) en cada máquina k, para todo par asignable de operaciones:
$$s_{j',h'}\ge c_{j,h}-V\big(3-x_{j,h,k}-x_{j',h',k}-y_{j,h,j',h',k}\big)$$
$$s_{j,h}\ge c_{j',h'}-V\big(2-x_{j,h,k}-x_{j',h',k}+y_{j,h,j',h',k}\big)$$
6. Makespan:  $C_{\max}\ge c_{j,n_j}\quad\forall j.$
7. Dominios:  $x,y\in\{0,1\},\; s,c\ge0.$

El **JSSP clásico** se recupera cuando |M_{j,h}|=1 ∀(j,h), lo que elimina las variables x y deja solo el secuenciamiento (variables y).

### B2.2 Representación por grafo disyuntivo

Siguiendo a Roy & Sussmann (1964), se define el grafo disyuntivo G=(N, C, D):
- **Nodos** N = {todas las operaciones O_{j,h}} ∪ {origen S, sumidero T}.
- **Arcos conjuntivos** C: dirigidos, unen operaciones consecutivas del mismo trabajo (precedencia fija), ponderados por la duración del nodo origen.
- **Arcos disyuntivos** D: pares no dirigidos que unen operaciones que compiten por la misma máquina. En el FJSP existe un conjunto disyuntivo por cada máquina candidata de cada operación.

**Programar** equivale a **orientar** todos los arcos disyuntivos (elegir el orden en cada máquina) —y, en FJSP, además **seleccionar la máquina**— de modo que el grafo dirigido resultante sea **acíclico**. El **makespan** es la longitud del **camino crítico** (más largo) de S a T en el grafo orientado. Esta representación es la base natural del codificador GNN: los nodos-operación llevan vectores de features, los arcos conjuntivos codifican precedencia y los disyuntivos (parcial o totalmente orientados durante la construcción) el estado de las máquinas.

### B2.3 Extensión dinámica/estocástica

Se pasa de parámetros deterministas a procesos estocásticos:
- **Llegadas dinámicas:** no todos los trabajos están disponibles en t=0; r_j es aleatorio, con llegadas modeladas como **proceso de Poisson** de tasa λ_a (intervalos entre llegadas ~ Exp(λ_a)).
- **Tiempos de proceso estocásticos:** p_{j,h,k} → variable aleatoria P_{j,h,k}=p̄_{j,h,k}·ξ, con ξ~Lognormal(μ,σ²) y E[ξ]=1.
- **Averías:** cada máquina k alterna estados *up/down*; tiempo hasta falla TTF ~ Exp(λ_k), tiempo de reparación TTR ~ Lognormal (o Exp). Disponibilidad estacionaria A_k = MTBF/(MTBF+MTTR).

El objetivo se convierte en la minimización del **valor esperado** o de un **cuantil** (para robustez): $\min\; \mathbb{E}[C_{\max}]$ o $\min\; \text{CVaR}_\alpha[C_{\max}]$. Se distingue **robustez** (insensibilidad del valor objetivo ante perturbaciones) de **estabilidad** (poca desviación respecto al plan original tras la replanificación).

## B3. Procesos de decisión de Markov

### B3.1 MDP, SMDP y CTMDP

Un **MDP** es una tupla (S, A, P, R, γ): S estados, A acciones, P(s'|s,a) probabilidad de transición, R(s,a) recompensa, γ∈[0,1] factor de descuento. Una **política** π(a|s) induce el retorno $G_t=\sum_{k\ge0}\gamma^k R_{t+k+1}$. Funciones de valor:
$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$
**Ecuaciones de Bellman:**
$$V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\big[R+\gamma V^\pi(s')\big],$$
$$V^*(s)=\max_a\sum_{s'}P(s'|s,a)\big[R+\gamma V^*(s')\big].$$
El **SMDP (semi-Markov)** añade tiempos de permanencia aleatorios τ entre decisiones; con descuento continuo el factor pasa a $e^{-\beta\tau}$. Es la formulación adecuada para eventos discretos: las épocas de decisión ocurren en instantes irregulares (fin de operación, llegada de trabajo, avería). En la práctica de scheduling suele emplearse un **MDP de eventos** con recompensa acumulada entre épocas consecutivas.

### B3.2 El FJSP dinámico como MDP

- **Épocas de decisión:** cada vez que una máquina queda libre y existe ≥1 operación elegible, o cuando la llegada de un trabajo / una avería habilita o inhabilita acciones.
- **Espacio de estados S:** el grafo disyuntivo parcialmente construido con features por nodo (ver PARTE D), el estado de las máquinas (libre/ocupada/averiada, tiempo restante), el reloj de simulación y el conjunto de operaciones listas.
- **Espacio de acciones A:** dos paradigmas. (a) **Construcción end-to-end:** elegir el par (operación elegible, máquina elegible) a programar a continuación; |A| varía con el estado y se gestiona con **enmascarado de acciones** infeasibles. (b) **Selección de reglas:** elegir una PDR de un catálogo (acción de tamaño fijo), más simple pero menos expresiva. **Esta tesis adopta (a)** por su mayor capacidad expresiva y su alineación con la representación de grafo.
- **Recompensa R:** densa vs. dispersa. La dispersa entrega −C_max solo al final (alta varianza, difícil asignación de crédito). La **densa estándar** usa el **incremento negativo de la cota inferior del makespan** entre épocas (ver B4 y PARTE D), cuya suma telescópica equivale a −C_max, alineando la optimización local con el objetivo global.

## B4. Aprendizaje por refuerzo profundo

**De lo tabular a la aproximación de función.** Cuando |S| es enorme (como en scheduling), V/Q/π se aproximan con redes de parámetros θ. **Métodos basados en valor:** Q-learning y **DQN** (Mnih et al. 2015) aprenden Q_θ minimizando el error de diferencia temporal (TD) con *replay buffer* y *target network*; funcionan bien con A discreto y fijo, pero sufren cuando A varía por estado (como en la construcción end-to-end del FJSP).

**Métodos de gradiente de política.** Parametrizan π_θ(a|s) y ascienden J(θ)=E_π[G]. El **teorema del gradiente de política**:
$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a|s)\,Q^{\pi_\theta}(s,a)\big].$$
Restar una **línea base** b(s)=V(s) reduce la varianza usando la **ventaja** A(s,a)=Q(s,a)−V(s). Los **métodos actor-crítico** aprenden simultáneamente π_θ (actor) y V_φ (crítico). El estimador **GAE(λ)** (Schulman et al. 2016, arXiv:1506.02438) balancea sesgo y varianza:
$$\hat A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\qquad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t).$$

**PPO** (Schulman et al. 2017, arXiv:1707.06347) optimiza un **objetivo sustituto recortado**. Con la razón de probabilidad $\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$:
$$L^{CLIP}(\theta)=\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big].$$
El objetivo total añade la pérdida de valor y un bono de entropía: $L=L^{CLIP}-c_1L^{VF}+c_2 H[\pi_\theta]$. PPO **domina en scheduling** por: (i) estabilidad frente a un A variable con enmascarado; (ii) compatibilidad natural con arquitecturas actor-crítico sobre *embeddings* de GNN; (iii) robustez frente a hiperparámetros; y (iv) buen rendimiento *on-policy* en episodios largos. Consideraciones clave: **asignación de crédito** en horizontes largos (mitigada por la recompensa densa), **exploración** (entropía y muestreo estocástico de la política enmascarada) y **estabilidad** (recorte del ratio + normalización de ventajas).

## B5. Redes neuronales de grafos

**Marco de paso de mensajes (MPNN).** En la capa l, cada nodo v actualiza su representación h_v:
$$m_v^{(l)}=\text{AGG}\big(\{h_u^{(l-1)}:u\in\mathcal N(v)\}\big),\qquad h_v^{(l)}=\text{UPD}\big(h_v^{(l-1)},m_v^{(l)}\big).$$

- **GCN** (Kipf & Welling 2017): $H^{(l)}=\sigma\big(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l-1)}W^{(l)}\big)$, con $\tilde A=A+I$.
- **GraphSAGE** (Hamilton et al. 2017): muestreo y agregación (mean/max/LSTM) con concatenación: $h_v^{(l)}=\sigma\big(W\cdot[h_v^{(l-1)}\Vert \text{AGG}(h_u^{(l-1)})]\big)$.
- **GAT** (Veličković et al. 2018, arXiv:1710.10903): atención sobre vecinos, $\alpha_{vu}=\text{softmax}_u\big(\text{LeakyReLU}(a^\top[Wh_v\Vert Wh_u])\big)$, $h_v^{(l)}=\sigma\big(\sum_u\alpha_{vu}Wh_u\big)$; con múltiples cabezas.
- **GIN** (Xu et al. 2019, arXiv:1810.00826): $h_v^{(l)}=\text{MLP}\big((1+\epsilon)h_v^{(l-1)}+\sum_{u\in\mathcal N(v)}h_u^{(l-1)}\big)$.

**Poder expresivo y test WL.** Xu et al. (ICLR 2019, arXiv:1810.00826) demuestran (su **Teorema 3**) que una MPNN con función de agregación **suma inyectiva** seguida de un MLP es tan discriminativa como el **test de isomorfismo de grafos de Weisfeiler-Lehman de 1 dimensión (1-WL)**, cota que las agregaciones *mean* y *max* **no** alcanzan (Lema 4 / Corolario 6). Esto motiva el uso de **GIN** como codificador cuando importa distinguir subestructuras del grafo de scheduling.

**GNN heterogéneas para FJSP.** El FJSP se modela con un **grafo heterogéneo** de dos tipos de nodo (operación, máquina) y varios tipos de arista: precedencia (op→op), elegibilidad/asignación (op↔máquina) y arco de máquina (op↔op que comparten máquina). El paso de mensajes es específico por tipo de relación (estilo R-GCN / HGT), permitiendo que cada máquina agregue información de sus operaciones candidatas y viceversa. Esta es la representación empleada por Song et al. (IEEE TII 2023).

**Invariancia a permutaciones y generalización al tamaño.** Las GNN son equivariantes a permutaciones de nodos y comparten pesos entre nodos y aristas; el número de parámetros es **independiente** de n y m. Por ello una política definida sobre *embeddings* de GNN puede evaluarse en instancias de tamaño arbitrario sin cambiar de arquitectura, habilitando la **generalización en tamaño**, hipótesis central de esta tesis.

## B6. Reglas de despacho por prioridad (PDRs)

Cuando una máquina queda libre se elige, entre las operaciones en cola, según un índice. Definiciones formales (para la operación O_{j,h} candidata en el instante t):
- **SPT** (Shortest Processing Time): min p_{j,h}.
- **LPT** (Longest PT): max p_{j,h}.
- **MWKR** (Most Work Remaining): max $\sum_{h'\ge h}p_{j,h'}$.
- **LWKR** (Least Work Remaining): min trabajo remanente.
- **MOR** (Most Operations Remaining): max (n_j−h+1); **LOR**: min.
- **FIFO/FCFS**: min tiempo de llegada a la cola.
- **EDD** (Earliest Due Date): min d_j.
- **CR** (Critical Ratio): min $\frac{d_j-t}{\sum_{h'\ge h}p_{j,h'}}$.

**Resultados teóricos:** para 1||ΣC_j, SPT es óptima; para 1||L_max, EDD es óptima (regla de Jackson); WSPT es óptima para 1||Σw_jC_j. En entornos de taller estas garantías se pierden, pero SPT y MWKR tienden a producir buenos makespans empíricos, mientras que FIFO es "justa" pero débil. Su valor: coste computacional O(1)–O(|cola|), interpretabilidad y ser el **piso de comparación** obligatorio.

## B7. Métodos exactos y metaheurísticas de referencia

**Programación por restricciones (CP) y CP-SAT (Google OR-Tools).** Modela cada operación como una **variable de intervalo** (start, size, end). Restricciones clave: `NoOverlap` sobre los intervalos de cada máquina (secuenciamiento sin solape) y, para recursos con capacidad, `Cumulative`. En FJSP se usan **intervalos opcionales** (`OptionalInterval`) más la restricción de que exactamente una alternativa de máquina esté presente por operación. CP-SAT combina propagación de restricciones, aprendizaje de cláusulas (CDCL / *lazy clause generation*) y búsqueda con reinicios; entrega óptimos con prueba (o cotas) para instancias pequeñas/medianas y buenas soluciones con límite de tiempo en grandes. Constituye el **techo de calidad** práctico y la referencia para calcular la brecha de optimalidad.

**Algoritmos genéticos (GA) para FJSP.** Codificación con dos vectores: (i) secuencia de operaciones (permutación con repetición del índice de trabajo) y (ii) asignación de máquinas. Operadores típicos: cruce POX/JBX para la secuencia y cruce uniforme para la asignación; mutación por intercambio y por reasignación de máquina. Selección por torneo, elitismo; decodificación a programación activa/semiactiva. Es el estándar metaheurístico de referencia; Brandimarte (1993) introdujo *tabu search* sobre las instancias MK.

## B8. Generalización en optimización combinatoria neuronal y scheduling robusto

**Tipos de generalización:** (a) **en tamaño** —entrenar en instancias pequeñas y evaluar en grandes—; (b) **en distribución** —cambio en las distribuciones de p_{j,h}, en la estructura de elegibilidad o en el ratio n/m—. La literatura de NCO (Bello et al. 2017; Kool et al. 2019; POMO, Kwon et al. 2020) muestra que la generalización en tamaño es el talón de Aquiles: el desempeño se degrada al alejarse del régimen de entrenamiento. Hipótesis sobre por qué las GNN generalizan: parámetros independientes del tamaño, invariancia a permutaciones y localidad del paso de mensajes; y por qué fallan: cambios de escala en grados y rangos de features (mitigables con normalización), horizontes más largos y desajuste en la distribución de los *embeddings* agregados.

**Scheduling robusto y reactivo.** Se distinguen: programación **predictiva-reactiva** (replanificar ante eventos), **robusta** (planes que minimizan la sensibilidad esperada del objetivo) y **estable** (minimizar la desviación respecto al plan base). Métricas: robustez = degradación esperada del objetivo; estabilidad = desviación de los tiempos de inicio. Una política DRL reactiva reprograma en cada época de decisión, encajando de forma natural en el paradigma predictivo-reactivo.

## B9. Simulación de eventos discretos (DES)

**Fundamentos:** reloj de simulación, **lista de eventos futuros (FEL)** ordenada por tiempo, **entidades** (trabajos, operaciones), **recursos** (máquinas), variables de estado y contadores estadísticos. El motor avanza el reloj al próximo evento (*next-event time advance*), ejecuta su **rutina de evento** (que puede programar eventos futuros) y actualiza las estadísticas.

**Verificación y validación (Sargent 2011).** Verificación = el modelo se implementó correctamente (depuración, trazas, pruebas de consistencia); validación = el modelo representa el sistema (validez conceptual, validación operacional, validación de datos de entrada). Técnicas: animación, comparación con casos analíticos (p.ej. cola M/M/1), análisis de sensibilidad y pruebas de degeneración.

**Análisis de resultados.** Sistemas **terminantes** (con evento natural de fin, p.ej. procesar N trabajos) vs. **no terminantes** (estado estacionario). Para terminantes se realizan **réplicas independientes** con semillas distintas; el número de réplicas R para un semi-ancho h del intervalo de confianza al nivel 1−α:
$$R\ge\Big(\frac{z_{1-\alpha/2}\,S}{h}\Big)^2,$$
usando la desviación estándar muestral S de una corrida piloto. Para no terminantes se aplica eliminación del periodo de calentamiento (método de Welch) y *batch means*.

---

# PARTE C — ESTADO DEL ARTE / REVISIÓN SISTEMÁTICA

## C1. Protocolo PRISMA (para que el estudiante lo reporte)

**Bases de datos:** Scopus, Web of Science, IEEE Xplore, ScienceDirect (Elsevier); arXiv/DBLP como complemento de preprints.

**Cadenas de búsqueda (literales):**
- Scopus/WoS: `TITLE-ABS-KEY(("reinforcement learning" OR "deep reinforcement learning" OR "graph neural network") AND ("job shop" OR "job-shop" OR "flexible job shop" OR "FJSP") AND (scheduling))`
- Dinámico/estocástico: `("reinforcement learning") AND ("job shop") AND ("dynamic" OR "stochastic" OR "machine breakdown" OR "disruption" OR "rescheduling")`
- Generalización: `("neural combinatorial optimization" OR "graph neural network") AND ("generalization" OR "size generalization") AND (scheduling OR "combinatorial optimization")`
- LLM: `("large language model" OR "LLM agent") AND (scheduling OR "job shop" OR "combinatorial optimization")`

**Ventana temporal:** 2015–2026 para fundamentos de NCO; foco 2018–2026 para DRL-scheduling.

**Criterios de inclusión:** artículo revisado por pares, o preprint arXiv con impacto/código; aplica RL/DRL o GNN a JSSP/FJSP; reporta métricas y benchmarks. **Exclusión:** sin evaluación cuantitativa; scheduling ajeno al taller (cloud/grid sin transferibilidad); duplicados.

**Flujo de cribado (PRISMA):** Identificación → eliminación de duplicados → cribado por título/resumen → elegibilidad por texto completo → inclusión; reportar el conteo en cada etapa mediante diagrama de flujo (Anexo D).

## C2. Narrativa 2018–2026 (por familia de método)

1. **Pioneros end-to-end (2015–2019):** *Pointer Networks* (Vinyals et al. 2015) y NCO con RL (Bello et al. 2017) para TSP; *Attention Model* (Kool et al. 2019) para problemas de ruteo. Traslado conceptual del aprendizaje de heurísticas constructivas a scheduling.
2. **GNN+RL para JSSP (2020–2021):** **L2D** (Zhang, Song, Cao et al., NeurIPS 2020) — grafo disyuntivo + GIN + PPO, recompensa por cota del makespan, generalización en tamaño. **Park et al. (IJPR 2021)** — representación por GNN y política RL para JSSP.
3. **FJSP con GNN heterogéneo (2022–2023):** **Song et al. (IEEE TII 2023)** — grafo heterogéneo operación-máquina + PPO, generaliza y compite con OR-Tools; **Lei et al. (ESWA 2022)** — DRL multi-acción/jerárquico para FJSP.
4. **Dinámico/estocástico (2022–2025):** trabajos sobre llegadas dinámicas, averías y tiempos estocásticos con GNN+DRL y esquemas jerárquicos.
5. **Frontera 2024–2026:** modelos de atención dual, formulaciones multi-objetivo (makespan + energía/tardanza), y primeros agentes basados en LLM para scheduling. **[POR VERIFICAR — la lista concreta de estos ítems requiere búsqueda en vivo con las cadenas de C1; no fue posible confirmarlos en este entorno.]**

## C3. Tabla comparativa (esqueleto a completar y verificar)

| Ref. | Problema | Repr. del estado | Arquitectura | Algoritmo RL | Objetivo | Benchmarks | Líneas base | Resultados reportados | Limitaciones |
|---|---|---|---|---|---|---|---|---|---|
| Zhang/Song/Cao 2020 (L2D) | JSSP estático | grafo disyuntivo | GIN | PPO | C_max | ta, generados | PDRs, OR-Tools | gap ~15–20% (15×15) → ~20–30% (100×20) vs. óptimo | solo estático |
| Park et al. 2021 | JSSP | grafo | GNN | Policy Gradient | C_max | ta, la | PDRs | competitivo con PDRs | estático |
| Song et al. 2023 | FJSP estático | grafo heterogéneo op-máquina | HGNN | PPO | C_max | MK, generados | PDRs, OR-Tools | gaps medios ~10–20% vs. CP-SAT; supera PDRs | no dinámico |
| Lei et al. 2022 | FJSP | grafo/estado | multi-agente | multi-PPO | C_max | MK | PDRs, GA | competitivo | estático |
| SOTA 2024–2026 | FJSP dinámico | — | — | — | multi-obj. | — | — | **[VERIFICAR]** | — |

> Los rangos de gap de L2D y Song et al. son aproximados y **deben extraerse de las tablas de las fuentes primarias** (arXiv:2010.12367, Tablas 2–3; IEEE TII 19(2):1600–1610).

## C4. Análisis de brechas → preguntas de investigación

| Brecha identificada | Evidencia | Pregunta de investigación |
|---|---|---|
| Poca evaluación conjunta de generalización en tamaño **Y** robustez bajo perturbaciones | La mayoría evalúa estático, o un solo tipo de perturbación | **PP:** ¿Puede una política GNN+PPO generalizar en tamaño **y** ser robusta a averías/tiempos estocásticos/llegadas simultáneas? |
| Recompensas conscientes de la perturbación poco exploradas | Predomina la recompensa de cota de makespan estática | **SP1:** ¿Mejora una recompensa consciente de la perturbación la robustez sin dañar la optimalidad nominal? |
| Comparación limitada frente a CP-SAT + GA + PDRs simultáneamente | Comparaciones parciales | **SP2:** ¿Cuál es la brecha real frente al techo CP-SAT y frente a GA/PDRs en cada régimen? |
| Reproducibilidad y protocolo estadístico débiles | Pocos usan Friedman/Nemenyi/CD | **SP3:** ¿Son las diferencias estadísticamente significativas, con tamaños de efecto reportados? |

---

# PARTE D — ESPECIFICACIÓN PROCEDIMENTAL (sin código)

## D1. Entorno de simulación DES + contrato Gymnasium

**Variables de estado:** reloj t; para cada máquina k: estado∈{libre, ocupada, averiada}, tiempo de fin de ocupación, próximo TTF; para cada operación: estado∈{no lista, lista, en proceso, terminada}, máquina asignada, tiempos; conjunto de operaciones listas; grafo disyuntivo parcial.

**Tipos de evento:** (E1) llegada de trabajo; (E2) fin de operación (libera la máquina y habilita la operación sucesora); (E3) avería de máquina; (E4) fin de reparación; (E5) época de decisión (el agente elige acción).

**Rutinas de evento (resumen):** E2 → marca operación terminada, libera máquina, activa sucesora si existe, dispara época de decisión; E3 → interrumpe la operación en curso (*preempt-resume* o *preempt-restart* según supuesto), reprograma su fin; E4 → máquina pasa a libre, dispara época de decisión.

**Pseudocódigo del bucle principal (DES):**
```
inicializar reloj t=0, FEL, estado; programar llegadas y primeras averías
reset() → observación inicial
mientras no fin_episodio:
    evento e ← extraer_min(FEL); t ← tiempo(e)
    ejecutar rutina(e); actualizar estado y estadísticas
    si e habilita decisión (máquina libre ∧ operaciones listas):
        obs  ← construir_observación(estado)
        mask ← acciones_factibles(estado)
        a    ← política(obs, mask)        # en entrenamiento: PPO
        aplicar(a): asignar op→máquina; programar evento E2 en t+p
        r    ← recompensa(estado_antes, estado_después)
    hasta próxima_decisión
retornar C_max y estadísticas
```

**Contrato Gymnasium:**
- `observation_space`: `Dict`/`Graph` heterogéneo — matriz de features de operación (N_op×d_op), de máquina (m×d_m), e índices de aristas por tipo.
- `action_space`: `Discrete` sobre pares (operación lista, máquina elegible) indexados, con **máscara** booleana; alternativamente `MultiDiscrete`. Las acciones infeasibles se enmascaran poniendo sus *logits* a −∞ antes del softmax.
- `reset(seed)` → (obs, info); `step(a)` → (obs, reward, terminated, truncated, info); `terminated` cuando todas las operaciones de todos los trabajos (incluidos los llegados dinámicamente hasta el horizonte) han terminado.

## D2. Vector de features por nodo (con normalización)

**Nodos-operación O_{j,h}** (normalizar a [0,1] o con z-score):
- estado (one-hot: no-lista / lista / en-proceso / terminada).
- tiempo de proceso esperado normalizado: $\hat p=p_{j,h}/\bar p_{\max}$.
- trabajo remanente normalizado: $\sum_{h'\ge h}\bar p_{j,h'}/W_{\max}$.
- número de operaciones remanentes: $(n_j-h+1)/n_{\max}$.
- tiempo de inicio más temprano estimado: $s^{LB}_{j,h}/UB$ (por camino en el grafo).
- holgura/tardanza esperada respecto a d_j (si aplica): $(d_j-\hat C_{j,h})/D$.
- grado de flexibilidad: $|M_{j,h}|/m$.
- indicador de operación en el camino crítico estimado.

**Nodos-máquina k:**
- estado (one-hot: libre / ocupada / averiada).
- tiempo hasta disponibilidad: $t^{avail}_k$ normalizado por UB.
- carga asignada pendiente: $\sum p / W_{\max}$.
- utilización acumulada hasta t.
- disponibilidad estimada $A_k$ (para la variante consciente de averías).

**Aristas:** precedencia (op→op); elegibilidad (op↔máquina, feature = p_{j,h,k} normalizado); máquina-compartida (op↔op). Las constantes de normalización se derivan de la instancia (UB = cota superior por heurística *greedy*/lista; W_max = trabajo total máximo por trabajo).

## D3. Función(es) de recompensa

**Recompensa densa estándar (incremento negativo de la cota inferior del makespan).** Sea $LB_t$ una cota inferior del makespan del estado parcial en la época t (p.ej. la longitud del camino crítico estimado, o el máximo sobre máquinas del trabajo restante). Entonces:
$$r_t = LB_{t-1} - LB_t \;(\le 0),\qquad \sum_t r_t = LB_0 - LB_{T} = -C_{\max}\ \text{(salvo constante)}.$$
Esto alinea la suma de recompensas con −C_max (formulación de L2D / Song et al.).

**Variante consciente de la perturbación (propuesta de la tesis):**
$$r_t = (LB_{t-1}-LB_t) - \kappa\cdot\Delta_{\text{inest}},$$
donde $\Delta_{\text{inest}}$ penaliza cambios inducidos por la reprogramación tras eventos (desviación de los tiempos de inicio planificados, o incremento del makespan esperado atribuible a la avería), y κ≥0 pondera robustez vs. optimalidad. Alternativa multi-objetivo escalarizada: $r_t = -w_1\Delta C_{\max} - w_2\Delta \bar T - w_3\Delta_{\text{inest}}$.

## D4. Matriz de diseño experimental

| Exp. | Hipótesis | Factores | Niveles | Semillas/réplicas | Métrica principal |
|---|---|---|---|---|---|
| E0 Validación DES | modelo válido | casos analíticos | M/M/1; instancias con óptimo conocido | 10 | error vs. teórico |
| E1 Estático JSSP | H1 | benchmark | ft, la, ta | 5 (política estocástica) | RPD, gap |
| E2 Estático FJSP | H1 | benchmark | MK01–MK10, Hurink | 5 | gap vs. CP-SAT |
| E3 Generalización tamaño | H2 | tamaño train→test | train 6×6/10×10 → test 15×15, 20×15, 30×20, ta grandes | 5 | gap por tamaño |
| E4 Robustez averías | H3 | intensidad de avería | A_k∈{0.95, 0.90, 0.80} | 30 réplicas | E[C_max], índice de robustez |
| E5 Tiempos estocásticos | H3 | σ del multiplicador | σ∈{0.1, 0.25, 0.5} | 30 | E[C_max] |
| E6 Llegadas dinámicas | H3 | tasa λ_a | utilización objetivo {0.7, 0.85, 0.95} | 30 | E[C_max], E[ΣT_j] |
| E7 Recompensa consciente | H4 | κ | {0, 0.5, 1.0} | 30 | robustez vs. optimalidad |
| E8 Tiempo de decisión | H5 | tamaño | todos | — | ms/decisión |

Líneas base en todos los experimentos: SPT, MWKR, FIFO, EDD (PDRs); GA; CP-SAT (con límite de tiempo).

## D5. Modelos de perturbación (distribuciones y parámetros propuestos)

- **Averías:** TTF ~ Exp(λ_k) con MTBF=1/λ_k; TTR ~ Lognormal(μ_r,σ_r²) con MTTR=exp(μ_r+σ_r²/2). Parametrizar por disponibilidad objetivo A_k=MTBF/(MTBF+MTTR)∈{0.95, 0.90, 0.80}, fijando MTTR como fracción del p̄ medio (p.ej. MTTR≈2·p̄) y despejando MTBF.
- **Tiempos de proceso estocásticos:** $P=p̄\cdot\xi$, con ξ~Lognormal(−σ²/2, σ²) para que E[ξ]=1, y σ∈{0.1, 0.25, 0.5} (bajo/medio/alto).
- **Llegadas dinámicas:** número de trabajos que llegan tras t=0 como Poisson; intervalos entre llegadas ~ Exp(λ_a). Fijar λ_a para lograr utilización objetivo ρ = λ_a·E[carga por trabajo]/m ∈ {0.7, 0.85, 0.95}.
- **Nivel de utilización/disponibilidad:** controlado conjuntamente por λ_a (carga) y A_k (capacidad efectiva).

## D6. Protocolo de evaluación (fórmulas exactas)

- **Makespan:** $C_{\max}=\max_j c_{j,n_j}$.
- **Brecha de optimalidad / RPD:** $\text{RPD}=100\cdot\frac{C_{\max}^{\text{método}}-C^{*}}{C^{*}}$, con $C^*$ = óptimo o mejor conocido (o solución CP-SAT con límite si no hay óptimo probado).
- **Tardanza:** $T_j=\max(0,C_j-d_j)$; $\bar T=\frac1n\sum T_j$; $T_{\max}=\max_j T_j$; $\sum U_j$ = número de trabajos tardíos.
- **Índice de robustez:** $RI=\frac{\mathbb{E}[C_{\max}^{\text{pert}}]-C_{\max}^{\text{nom}}}{C_{\max}^{\text{nom}}}$ (menor = más robusto); complementar con desviación estándar y CVaR_0.9.
- **Tiempo de decisión:** milisegundos por época y tiempo total de inferencia por instancia.

**Procedimiento estadístico:**
- Comparación de dos métodos sobre instancias emparejadas: **Wilcoxon signed-rank** (no paramétrico; no asume normalidad), α=0.05; reportar el estadístico W, el p-valor y el tamaño de efecto r=Z/√N.
- Comparación de ≥3 métodos sobre múltiples instancias: **Friedman** (χ²_F sobre rangos medios); si rechaza, post-hoc **Nemenyi** con **diferencia crítica** $CD=q_\alpha\sqrt{\frac{k(k+1)}{6N}}$ y **diagrama CD** (Demšar 2006), donde k = nº de métodos y N = nº de instancias.
- Reportar **tamaños de efecto** (Cliff's delta o r), no solo p-valores.

## D7. Pre-registro (hipótesis y reglas de decisión, ANTES de ejecutar)

- **H1 (competitividad estática):** la política supera a la mejor PDR en RPD medio sobre ta y MK. *Aceptar* si Wilcoxon p<0.05 y la mediana de mejora es >0.
- **H2 (generalización en tamaño):** el RPD en instancias 2–3× mayores que las de entrenamiento se mantiene por debajo de un umbral (p.ej. ≤ RPD de la mejor PDR). *Aceptar* si el RPD del modelo ≤ RPD de la PDR en todos los tamaños de test (Friedman + Nemenyi).
- **H3 (robustez):** el índice RI del modelo es menor que el de las PDRs bajo averías/tiempos/llegadas. *Aceptar* si RI_modelo < RI_PDR con Wilcoxon p<0.05 en cada régimen.
- **H4 (recompensa consciente):** κ>0 reduce RI sin aumentar el RPD nominal más de un margen δ (p.ej. 2 pp). *Aceptar* si ΔRI<0 significativo y ΔRPD_nominal<δ.
- **H5 (tiempo de decisión):** el tiempo por decisión del modelo es mucho menor que el de CP-SAT/GA para tamaños grandes. *Aceptar* si la mediana de ms/instancia del modelo es < 1% del de CP-SAT.

---

# PARTE E — RESULTADOS PARCIALES / ESPERADOS

> **Todo lo referente al modelo propio del estudiante en esta parte son HIPÓTESIS, NO resultados.** Los números "publicados" provienen de la literatura y deben re-verificarse contra las fuentes primarias.

## E1. Números de benchmark publicados (referencia y objetivos de reproducción)

- **Óptimos confirmados (Fisher & Thompson):** ft06 = 55; ft10 = 930 (resuelto por Carlier & Pinson 1989, *Management Science* 35(2):164–176); ft20 = 1165. Tabulados en la **OR-Library** de J. E. Beasley (Brunel University). Sirven como *sanity check* del entorno. [ALTA]
- **la01–la40 (Lawrence 1984):** óptimos/mejores conocidos tabulados en la literatura y en OR-Library. [re-verificar tabla]
- **ta01–ta80 (Taillard 1993):** el conjunto consta de **80 instancias en 8 grupos de tamaño** (de 15×15 a 100×20), introducidas en Taillard, *EJOR* 64(2):278–285, 1993 (doi:10.1016/0377-2217(93)90182-M); mejores conocidos publicados y actualizados a lo largo de los años; las instancias grandes aún presentan brechas. [re-verificar tabla]
- **MK01–MK10 (Brandimarte 1993):** conjunto de **10 instancias FJSP** (de 10×6 a 20×15), de Brandimarte, *Annals of Operations Research* 41:157–183, 1993 (doi:10.1007/BF02023073); valores de referencia FJSP ampliamente citados. [re-verificar tabla]
- **Brechas DRL publicadas:** **L2D** (Zhang et al., NeurIPS 2020, arXiv:2010.12367) reporta gaps vs. óptimo que **crecen con el tamaño**, aproximadamente de ~15–20% en 15×15 hasta ~20–30% en instancias 100×20 de Taillard (extraer los valores exactos de las Tablas 2–3 del preprint). **Song et al. 2023** (IEEE TII 19(2):1600–1610) reportan que su HGNN+PPO obtiene gaps medios del orden de ~10–20% respecto a OR-Tools CP-SAT en instancias MK/generadas, superando consistentemente a las PDRs (verificar la tabla exacta en la fuente primaria). **[Los valores exactos por instancia deben copiarse de las fuentes; no se pudieron confirmar en vivo en este entorno.]**

## E2. Rangos de desempeño esperados del método propio (HIPÓTESIS)

| Régimen | Métrica | Rango esperado (hipótesis) | Razonamiento |
|---|---|---|---|
| Estático JSSP (tamaño de entrenamiento) | RPD vs. óptimo | mejor que la mejor PDR; por encima de CP-SAT | GNN+PPO captura la estructura mejor que reglas fijas |
| Generalización 2–3× | RPD | degradación moderada, aún ≤ PDR | invariancia a permutación + parámetros indep. del tamaño |
| Robustez (averías/tiempos) | RI | RI menor que el de las PDRs | política reactiva que reprograma en cada época |
| Tiempo de decisión | ms | órdenes de magnitud menor que CP-SAT | inferencia de una pasada vs. búsqueda |

Estos rangos son **objetivos a contrastar** (targets de las hipótesis H1–H5), **no** mediciones.

---

# PARTE F — BIBLIOGRAFÍA (formato IEEE; confianza y verificación marcadas)

> Recordatorio: sin acceso web en este entorno; toda cita es reconstruida de memoria y debe re-verificarse. Correcciones aplicadas señaladas en negrita.

**Fundamentos de scheduling y complejidad**
- [1] R. L. Graham, E. L. Lawler, J. K. Lenstra, A. H. G. Rinnooy Kan, "Optimization and approximation in deterministic sequencing and scheduling: a survey," *Annals of Discrete Mathematics*, vol. 5, pp. 287–326, 1979, doi:10.1016/S0167-5060(08)70356-X. [ALTA]
- [2] M. R. Garey, D. S. Johnson, R. Sethi, "The complexity of flowshop and jobshop scheduling," *Mathematics of Operations Research*, vol. 1, no. 2, pp. 117–129, 1976, doi:10.1287/moor.1.2.117. [ALTA]
- [3] M. L. Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 5.ª ed., Springer, 2016. [MEDIA — verificar edición/año]
- [4] J. Carlier, É. Pinson, "An Algorithm for Solving the Job-Shop Problem," *Management Science*, vol. 35, no. 2, pp. 164–176, 1989, doi:10.1287/mnsc.35.2.164. [ALTA — resolvió ft10]

**Benchmarks**
- [5] H. Fisher, G. L. Thompson, "Probabilistic learning combinations of local job-shop scheduling rules," en *Industrial Scheduling*, J. F. Muth y G. L. Thompson, Eds., Prentice-Hall, 1963, pp. 225–251. [ALTA — sin DOI; origen de ft06/ft10/ft20]
- [6] S. Lawrence, "Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques," Tech. Rep., GSIA, Carnegie-Mellon Univ., 1984. [ALTA — sin DOI; origen de la01–la40]
- [7] E. Taillard, "Benchmarks for basic scheduling problems," *European Journal of Operational Research*, vol. 64, no. 2, pp. 278–285, 1993, doi:10.1016/0377-2217(93)90182-M. [ALTA]
- [8] E. Demirkol, S. Mehta, R. Uzsoy, "Benchmarks for shop scheduling problems," *European Journal of Operational Research*, vol. 109, no. 1, pp. 137–141, 1998, doi:10.1016/S0377-2217(97)00019-2. [MEDIA]
- [9] P. Brandimarte, "Routing and scheduling in a flexible job shop by tabu search," *Annals of Operations Research*, vol. 41, no. 3, pp. 157–183, 1993, doi:10.1007/BF02023073. [ALTA — origen de MK01–MK10]
- [10] J. Hurink, B. Jurisch, M. Thole, "Tabu search for the job-shop scheduling problem with multi-purpose machines," *OR Spektrum*, vol. 15, no. 4, pp. 205–215, 1994, doi:10.1007/BF01719451. [MEDIA — origen de edata/rdata/vdata]
- [11] J. E. Beasley, "OR-Library," http://people.brunel.ac.uk/~mastjjb/jeb/info.html. [ALTA]

**RL y GNN (fundamentos)**
- [12] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2.ª ed., MIT Press, 2018. [ALTA]
- [13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. [ALTA]
- [14] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2016. [ALTA]
- [15] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015, doi:10.1038/nature14236. [ALTA]
- [16] T. N. Kipf, M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017, arXiv:1609.02907. [ALTA]
- [17] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs" (GraphSAGE), NeurIPS 2017, arXiv:1706.02216. [ALTA]
- [18] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, "Graph Attention Networks," ICLR 2018, arXiv:1710.10903. [ALTA]
- [19] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?" (GIN), ICLR 2019, arXiv:1810.00826. [ALTA]

**NCO / DRL para scheduling**
- [20] O. Vinyals, M. Fortunato, N. Jaitly, "Pointer Networks," NeurIPS 2015, arXiv:1506.03134. [ALTA]
- [21] W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing Problems!," ICLR 2019, arXiv:1803.08475. [ALTA]
- [22] Y.-D. Kwon et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning," NeurIPS 2020, arXiv:2010.16011. [MEDIA]
- [23] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, C. Xu, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," NeurIPS 2020, arXiv:2010.12367; repo github.com/zcaicaros/L2D. [ALTA — **CORRECCIÓN:** el orden correcto de autores es Zhang, **Song, Cao** (la tarea indicaba "Zhang, Cao, Song")]
- [24] J. Park, J. Chun, S. H. Kim, Y. Kim, J. Park, "Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning," *International Journal of Production Research*, vol. 59, no. 11, pp. 3360–3377, 2021, doi:10.1080/00207543.2020.1870013. [MEDIA — **NOTA:** el título completo es más largo que el breve dado en la tarea; verificar]
- [25] W. Song, X. Chen, Q. Li, Z. Cao, "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 2, pp. 1600–1610, 2023, doi:10.1109/TII.2022.3189725. [MEDIA — verificar vol./nº/páginas y DOI]
- [26] K. Lei, P. Guo, Y. Wang, X. Wu, W. Zhao, "A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem," *Expert Systems with Applications*, vol. 205, art. 117796, 2022, doi:10.1016/j.eswa.2022.117796. [MEDIA]

**Entornos y benchmarks de código**
- [27] P. Tassel, M. Gebser, K. Schekotihin, "A Reinforcement Learning Environment For Job-Shop Scheduling," arXiv:2104.03760, 2021; repo github.com/prosysscience/JSSEnv. [ALTA]
- [28] C. Waubert de Puiseau, J. Peters, C. Dörpelkus, H. Tercan, T. Meisen, "schlably: A Python framework for deep reinforcement learning based scheduling experiments," *SoftwareX*, vol. 22, art. 101383, 2023, doi:10.1016/j.softx.2023.101383; repo github.com/tmdt-buw/schlably. [MEDIA — **CORRECCIÓN:** la afiliación es el grupo **TMDT de la Bergische Universität Wuppertal (buw)**, NO Fraunhofer IML / TU Dortmund como indicaba la tarea]
- [29] R. Reijnen, K. van Straaten, Z. Bukhsh, Y. Zhang, "Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods," arXiv:2308.12794, 2023. [MEDIA — verificar arXiv ID y repo]

**Simulación y estadística**
- [30] A. M. Law, *Simulation Modeling and Analysis*, 5.ª ed., McGraw-Hill, 2015. [MEDIA]
- [31] R. G. Sargent, "Verification and validation of simulation models," *Proc. Winter Simulation Conference*, 2011. [MEDIA]
- [32] J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *Journal of Machine Learning Research*, vol. 7, pp. 1–30, 2006. [ALTA]

**Herramientas**
- [33] L. Perron, V. Furnon, "OR-Tools" (CP-SAT), Google, https://developers.google.com/optimization. [ALTA]

**Fundamento del grafo disyuntivo**
- [34] B. Roy, B. Sussmann, "Les problèmes d'ordonnancement avec contraintes disjonctives," Note DS n.º 9 bis, SEMA, Montrouge, 1964. [MEDIA — clásico, sin DOI]

**No verificables en este entorno (frontera 2024–2026):** surveys recientes de DRL-scheduling; modelos de atención dual para FJSP (posible IEEE TNNLS 2023–2024); DRL multi-objetivo (makespan + energía/tardanza); y agentes basados en LLM para scheduling **NO pudieron verificarse** por ausencia de búsqueda en vivo. Deben localizarse con las cadenas de búsqueda PRISMA de la sección C1 antes de la defensa. **[SIN VERIFICAR — riesgo alto]**

---

# Recomendaciones

1. **Cerrar la verificación bibliográfica primero.** Re-verificar toda la bibliografía en Scopus/WoS/IEEE Xplore/DOI.org. Confirmar de inmediato las correcciones ya señaladas: (a) orden de autores de L2D = Zhang, Song, Cao; (b) afiliación de *schlably* = TMDT, Univ. de Wuppertal; (c) ft10 resuelto por Carlier & Pinson 1989. *Umbral que cambia el plan:* si Song et al. 2023 resulta tener otro vol./nº/DOI, actualizar [25] antes de citarlo en el marco teórico.
2. **Cerrar la frontera 2024–2026 con búsqueda sistemática.** Ejecutar las cuatro cadenas PRISMA de C1; el argumento de novedad de la tesis (generalización + robustez conjuntas + recompensa consciente de perturbación) depende de confirmar que no existe ya un trabajo 2024–2026 que lo haga. Si aparece uno idéntico, reorientar hacia el diferencial (multi-perturbación simultánea, o el enfoque de mantenimiento/calidad propio de Ingeniería Industrial).
3. **Secuencia de ejecución experimental.** (i) E0: validar el DES reproduciendo ft06=55, ft10=930, ft20=1165 antes de cualquier entrenamiento; (ii) reproducir L2D y Song et al. como líneas base neuronales; (iii) recién entonces innovar con la recompensa consciente de perturbación (E7). *Umbral:* si no se reproduce el orden de magnitud del gap publicado de L2D en E1, depurar el entorno antes de continuar.
4. **Congelar el pre-registro (D7) antes de generar cualquier resultado**, para blindar la defensa metodológica (evita HARKing).
5. **Respetar el presupuesto de cómputo (GPU única):** entrenar en instancias pequeñas (6×6, 10×10) y explotar la generalización en tamaño para evaluar en grandes; limitar CP-SAT y GA por tiempo de pared para comparaciones justas.

# Caveats

- **Sin acceso web en este entorno de ejecución:** ni yo ni el subagente pudimos ejecutar búsquedas en vivo (todas fallaron). Toda cita es reconstruida de memoria y **debe** verificarse; los niveles [ALTA]/[MEDIA]/[BAJA] reflejan mi confianza interna, no una verificación.
- **Los números del modelo propio (PARTE E2) son hipótesis, no resultados.** No se fabricó ningún dato experimental del estudiante.
- **Las tablas de gaps de L2D y Song et al.** se dan como rangos aproximados de memoria; los valores exactos por instancia deben copiarse de las fuentes primarias (arXiv:2010.12367; IEEE TII 19(2)).
- **Ítems de mayor riesgo a re-verificar:** DOIs y vol./nº/páginas de [3], [8], [10], [24], [25]; arXiv ID y repo de [29]; y **toda** la frontera 2024–2026 (surveys, atención dual, multi-objetivo, LLM-agentes), que quedó sin verificar.
- **Recomendación operativa:** re-ejecutar la fase de verificación bibliográfica en un entorno con `web_search`/`web_fetch` funcionales antes de dar por definitiva la bibliografía.