# IA en Ingeniería Industrial: Investigación de Tendencias, Recomendación de Tema de Tesis y Plan Metodológico Completo

> **Nota importante sobre las herramientas (leer primero):** La búsqueda y la recuperación web en vivo no estuvieron disponibles durante esta investigación. En consecuencia, los conteos exactos de publicaciones 2023–2026, los porcentajes de crecimiento, las cifras actuales de los *leaderboards*, los DOI/URL y las citas textuales **no** pudieron verificarse contra fuentes en vivo. Todas las identidades de los artículos (p. ej., Zhang et al., "Learning to Dispatch", NeurIPS 2020; Song et al., FJSP-GNN, IEEE TII 2023; Bergmann et al., MVTec AD, CVPR 2019; Saxena y Goebel, C-MAPSS, PHM 2008), los contenidos de los conjuntos de datos y las identidades de las herramientas de software provienen de conocimiento consolidado del dominio. **Cada cita, URL y conteo de citaciones debe verificarse manualmente antes de usarse en la tesis.** Considera esto un plan experto rigurosamente razonado, no una auditoría bibliométrica con fuentes en vivo.

## Resumen ejecutivo (TL;DR)
- El tema con mayor impulso y mejor ajuste para una tesis de maestría basada únicamente en simulación, en la intersección de IA + (Manufactura/Producción y Calidad/Mantenimiento), es el **Aprendizaje por Refuerzo Profundo (DRL) para la Programación de Talleres de Trabajo Flexibles (JSSP/FJSP)** — específicamente, **políticas basadas en redes neuronales de grafos (GNN) para programación *dinámica/estocástica***: lo bastante madura para tener *benchmarks* estándar y entornos Gym abiertos, pero con brechas abiertas claramente declaradas (generalización, eventos dinámicos, multiobjetivo, rigor en las líneas base).
- Alternativas fuertes: (2) **aumento de datos sintéticos con difusión/GAN para detección de defectos superficiales con pocos ejemplos** (Calidad); (3) **estimación de RUL/PHM sobre el benchmark C-MAPSS con cuantificación calibrada de incertidumbre** (Mantenimiento).
- Todas las rutas recomendadas funcionan enteramente con *benchmarks* públicos + simulación (instancias Taillard/DMU/OR-Library y entornos Gym JSSEnv/schlably/JobShopLib, o C-MAPSS / MVTec AD), requieren una sola GPU de consumo, y apuntan a revistas como *Journal of Manufacturing Systems*, *IEEE T-ASE*, *Computers & Industrial Engineering*, IEEE CASE y la Winter Simulation Conference.

## Hallazgos clave
1. Entre 2023 y 2026, cuatro temas dominan la publicación sobre IA en ingeniería industrial: **mantenimiento predictivo/RUL**, **detección de defectos y anomalías visuales con aprendizaje profundo**, **gemelos digitales** y **DRL para programación de la producción**. El mantenimiento predictivo y la detección de defectos son los más **saturados** (mayor volumen, más difícil reclamar novedad). El DRL para programación y los datos generativos/sintéticos son los grupos de **mayor impulso y menor saturación**, con brechas abiertas claras.
2. El DRL para programación es especialmente adecuado para una tesis basada solo en simulación porque **los "datos" SON la simulación**: bibliotecas de instancias estándar (Taillard, Demirkol/DMU, Lawrence, OR-Library) y entornos Gymnasium abiertos (JSSEnv, schlably, JobShopLib) permiten experimentación reproducible sin datos reales de planta.
3. La familia de métodos dominante es un **grafo disyuntivo codificado por una GNN con una política de Optimización de Política Proximal (PPO)**, establecida por "Learning to Dispatch" (L2D, NeurIPS 2020) de Zhang et al. y extendida al taller flexible por Song et al. (IEEE TII 2023).
4. Brechas recurrentes declaradas: **mala generalización al tamaño; manejo débil de eventos dinámicos/estocásticos** (averías de máquina, tiempos de proceso estocásticos, llegadas de trabajos nuevos); **sesgo hacia un solo objetivo (makespan); y falta de estandarización en las comparaciones** frente a OR-Tools CP-SAT y metaheurísticas.
5. La tesis recomendada se concreta en: **un programador DRL basado en GNN para FJSP dinámico bajo perturbaciones estocásticas, evaluado en generalización y comparado rigurosamente contra reglas de despacho, un algoritmo genético y CP-SAT** — una contribución defendible, publicable y ligera en recursos.

## Detalle

# DOCUMENTO 1 — Investigación del Tema, Introducción a la Materia y Pregunta de Investigación

## Parte 1 — Investigación Bibliométrica / de Tendencias

### 1.1 El panorama general (2023–2026)
La literatura sobre IA aplicada a la Ingeniería Industrial se ha consolidado en un puñado de temas de alto volumen, visible en el flujo de revisiones sistemáticas en *Journal of Manufacturing Systems*, *Journal of Intelligent Manufacturing*, *Computers & Industrial Engineering*, *IEEE Transactions on Automation Science and Engineering (T-ASE)*, *IEEE Transactions on Industrial Informatics (TII)* y *Reliability Engineering & System Safety*. Los grupos dominantes y su ajuste a tus restricciones:

| Tema | Volumen | Trayectoria de crecimiento | Saturación | Viabilidad con datos simulados/sintéticos | Ajuste (Manuf. / Cal. y Mant.) |
|---|---|---|---|---|---|
| Mantenimiento predictivo / PHM / RUL | Muy alto | Estable, maduro | Alta (saturado) | Excelente (C-MAPSS es simulado) | Mantenimiento |
| Detección de defectos superficiales y anomalías con DL | Muy alto | Creciente | Alta (saturado) | Buena (imágenes sintéticas, MVTec) | Calidad |
| Gemelos digitales | Alto | Explosivo pero con muchas revisiones | Media; a menudo conceptual | Buena | Ambos |
| **DRL para programación de producción (JSSP/FJSP)** | Alto | **Crecimiento rápido desde 2020** | **Baja–media (brechas abiertas)** | **Excelente (benchmarks + entornos Gym)** | Manufactura |
| IA generativa / LLM en manufactura | Emergente | El de mayor crecimiento relativo | Muy baja; sobre todo artículos de posición | Mixta | Ambos |
| Datos sintéticos (GAN/difusión) para datos industriales | Emergente | Crecimiento rápido | Baja | Excelente por definición | Calidad/Mantenimiento |
| Redes neuronales informadas por física (PINN) en PHM/procesos | Emergente | Creciente | Baja–media | Buena | Mantenimiento |
| GNN para programación/sistemas productivos | Emergente | Creciente (columna vertebral del DRL en programación) | Baja | Excelente | Manufactura |
| Detección de defectos con pocos ejemplos / autosupervisada | Emergente | Creciente | Baja–media | Buena | Calidad |
| Aprendizaje por transferencia / adaptación de dominio | Alto | Estable | Media | Buena | Ambos |
| Industria 5.0 / IA centrada en el humano | Alto (conceptual) | Creciente | Conceptual | Débil para una tesis técnica | Ambos |

**Interpretación.** Para una tesis de maestría que necesita una *afirmación de novedad defendible* bajo una restricción de *solo simulación*, la zona ideal es un tema que sea (a) de alto impulso, (b) aún no saturado, (c) nativamente basado en simulación y (d) dotado de *benchmarks* estándar para que los resultados sean comparables con trabajos previos. El **DRL para programación** se sitúa exactamente en esa zona; la **detección de defectos con datos sintéticos** y la **RUL con cuantificación de incertidumbre** son las alternativas más fuertes.

### 1.2 Caracterización tema por tema

**Mantenimiento predictivo / PHM / RUL.** Una de las literaturas más grandes y maduras de IA en manufactura. Se organiza en torno al *benchmark* de degradación de turbofán C-MAPSS de la NASA (Saxena y Goebel, PHM 2008) y a conjuntos de datos de rodamientos (CWRU, FEMTO-ST/PRONOSTIA, IMS). Métodos dominantes: regresores CNN y LSTM, después *Transformers* y, recientemente, modelos informados por física y conscientes de la incertidumbre. Trabajos seminales: Babu et al. (CNN profunda para RUL, DASFAA 2016); Zheng et al. (LSTM para RUL, IEEE ICPHM 2017). Foros: *Reliability Engineering & System Safety*, *Mechanical Systems and Signal Processing*, IEEE TII/TIE, PHM Society. **La saturación es alta**: superar el estado del arte en C-MAPSS por un margen pequeño es una afirmación débil para una tesis. Brechas abiertas: cuantificación de incertidumbre, adaptación de dominio entre condiciones de operación, escasez de datos.

**Detección de defectos superficiales y anomalías industriales con aprendizaje profundo.** Volumen muy grande, impulsado por el *benchmark* MVTec AD (Bergmann et al., CVPR 2019) y sus sucesores MVTec LOCO (IJCV 2022) y VisA (Zou et al., ECCV 2022). Conjuntos de defectos específicos de manufactura: base de datos NEU de defectos superficiales (fleje de acero, 6 clases, 1.800 imágenes), DAGM 2007 (texturas sintéticas), Severstal steel defect (Kaggle 2019). Métodos dominantes: detección de anomalías no supervisada (PatchCore, PaDiM, *student-teacher*/EfficientAD) más segmentación supervisada. **La saturación es alta** en MVTec AD (las puntuaciones están cerca del techo). Brechas abiertas: aprendizaje con pocos ejemplos/autosupervisado, anomalías lógicas/contextuales, cambio de dominio y **generación de datos sintéticos para superar la escasez de etiquetas** — esta última es donde queda novedad genuina.

**Gemelos digitales.** Crecimiento explosivo de publicaciones, pero con muchas revisiones y a menudo conceptual; una tesis técnica rigurosa necesita un núcleo de modelado concreto (p. ej., un gemelo digital basado en simulación que alimente un programador DRL), lo que reconduce a programación o a PHM.

**DRL para programación de la producción (JSSP/FJSP).** El tema *técnico* de crecimiento más rápido con brechas abiertas nítidas. Catalizado por "Learning to Dispatch" (L2D, NeurIPS 2020) de Zhang et al., que introdujo un estado de grafo disyuntivo con un codificador Graph Isomorphism Network entrenado por PPO para construir horarios de extremo a extremo. Park et al. (IJPR 2021) formalizaron GNN+RL para JSSP; Song et al. (IEEE TII 2023) extendieron GNN+DRL al taller flexible mediante un grafo heterogéneo. Los enfoques de atención/*pointer networks* derivan de Vinyals et al. (Pointer Networks, NeurIPS 2015) y Kool et al. ("Attention, Learn to Solve Routing Problems!", ICLR 2019). Foros: IEEE T-ASE, IEEE TII, *International Journal of Production Research*, *Computers & Industrial Engineering*, *Journal of Manufacturing Systems*; los aportes metodológicos en NeurIPS/ICML/ICLR. **La saturación es baja–media**, con brechas ampliamente declaradas.

**IA generativa / LLM y datos sintéticos.** Lo más nuevo y de crecimiento relativo más rápido; el trabajo con LLM es en su mayoría de posición o de marcos conceptuales, pero los **datos sintéticos con GAN/difusión para conjuntos de defectos y PHM** son técnicamente sustanciales y nativos de la simulación.

### 1.3 Lo más investigado frente a lo más defendible
- **Lo más investigado ahora mismo (mayor volumen bruto):** mantenimiento predictivo/RUL y detección de defectos/anomalías con DL.
- **Mayor impulso con brechas abiertas defendibles:** DRL para programación (especialmente FJSP dinámico/estocástico) y generación de datos sintéticos para inspección de calidad.
- **La mejor elección para TU caso:** DRL para FJSP dinámico — alto impulso, no saturado, nativamente basado en simulación, plenamente dentro de Manufactura y Producción, y con un vínculo secundario fuerte con Calidad y Mantenimiento cuando se modelan perturbaciones como las averías de máquina.

### 1.4 Brechas de investigación recurrentes declaradas
- **DRL en programación:** generalización a tamaños/distribuciones de instancia no vistos; eventos dinámicos/estocásticos (averías, tiempos de proceso estocásticos, llegadas dinámicas de trabajos); multiobjetivo más allá del makespan (tardanza, energía); comparación estandarizada contra OR-Tools CP-SAT y metaheurísticas; transferencia simulación-realidad; escalabilidad y costo de entrenamiento.
- **RUL/PHM:** cuantificación de incertidumbre, adaptación de dominio, escasez de datos, interpretabilidad.
- **Detección de defectos:** escasez y desbalance de etiquetas, cambio de dominio, anomalías lógicas, despliegue en tiempo real/borde, realismo de los datos sintéticos.

### 1.5 Conjuntos de datos de referencia y entornos de simulación abiertos

**Programación (ruta recomendada):**
- **Instancias de Taillard (1993)** (ta01–ta80) — conjunto canónico de JSSP/*flow-shop* (EJOR 64(2):278–285).
- **Demirkol, Mehta y Uzsoy (1998)**, instancias DMU (EJOR 109(1):137–141).
- **Lawrence (1984)** la01–la40; los clásicos **Fisher–Thompson** ft06/ft10/ft20.
- **Específicos de FJSP:** Brandimarte (MK01–MK10); Hurink (edata/rdata/vdata).
- **OR-Library** (J. E. Beasley) aloja instancias de JSSP.
- **Entornos Gym/Gymnasium abiertos:** JSSEnv (Tassel et al., 2021, arXiv:2104.03760, `prosysscience/JSSEnv`); schlably (Fraunhofer IML / TU Dortmund, SoftwareX 2023, `tmdt-buw/schlably`); JobShopLib (entorno Gymnasium con representaciones de grafo); la base de código consolidada "Job Shop Scheduling Benchmark" de Reijnen et al. (2023).
- **Líneas base de solucionadores/metaheurísticas:** Google OR-Tools CP-SAT; algoritmos genéticos; reglas de despacho por prioridad (SPT, MWKR, MOR, FIFO, LWKR).

**Mantenimiento/PHM:** NASA C-MAPSS (FD001–FD004) y N-CMAPSS; retos de datos de la PHM Society; rodamientos FEMTO-ST/PRONOSTIA (IEEE PHM 2012); rodamientos IMS (Univ. de Cincinnati); CWRU.

**Calidad/defectos:** MVTec AD; MVTec LOCO; VisA; DAGM 2007; NEU surface defect; Severstal steel defect; SECOM (UCI, 590 variables, pasa/falla); Bosch Production Line Performance (Kaggle).

**Anomalías en series temporales industriales:** SMD (OmniAnomaly, KDD 2019); SWaT y WADI (bancos de prueba iTrust de SUTD).

**Herramientas de simulación y datos sintéticos:** SimPy y salabim (simulación de eventos discretos de código abierto en Python); AnyLogic (multimétodo, versión PLE gratuita); Siemens Tecnomatix Plant Simulation; Arena; NVIDIA Isaac Sim + Omniverse Replicator (simulación física y datos visuales sintéticos con aleatorización de dominio); Blender + BlenderProc para imágenes sintéticas etiquetadas de defectos.

## Parte 2 — Recomendación y Selección del Tema

### 2.1 Tema recomendado
**"Aprendizaje por Refuerzo Profundo Generalizable para la Programación Dinámica de Talleres de Trabajo Flexibles bajo Perturbaciones Estocásticas."**
Una política DRL basada en una red neuronal de grafos (codificador GNN + PPO) que programa en línea un taller de trabajo *flexible*, entrenada y evaluada explícitamente para (a) generalizar entre tamaños y distribuciones de instancia y (b) ser robusta ante eventos estocásticos (averías de máquina, tiempos de proceso estocásticos, llegadas dinámicas de trabajos), comparada rigurosamente con reglas de despacho, una metaheurística (AG) y OR-Tools CP-SAT.

**Justificación frente a los seis criterios:**
- (a) *Actualidad/impulso:* uno de los temas técnicos de crecimiento más rápido desde 2020; el FJSP dinámico/estocástico se señala explícitamente como poco explorado.
- (b) *Viabilidad solo con simulación:* el entorno ES el dato — las instancias Taillard/DMU/Brandimarte más una capa de simulación de eventos discretos (SimPy/JSSEnv/schlably) generan todos los episodios de entrenamiento y prueba; no se necesitan datos reales de planta.
- (c) *Ajuste:* núcleo de Manufactura y Producción; las perturbaciones incorporan Calidad y Mantenimiento (dinámica de averías y reparaciones).
- (d) *Brecha defendible:* generalización + robustez dinámico-estocástica + rigor en las líneas base son brechas recurrentes y explícitamente declaradas.
- (e) *Cómputo:* entrenable en una sola GPU de consumo (clase RTX, 8–12 GB) en horas o días; las GNN de programación son ligeras.
- (f) *Potencial de publicación:* encaja en IEEE T-ASE, *Computers & Industrial Engineering*, *Journal of Manufacturing Systems*, IEEE CASE y WSC.

### 2.2 Temas alternativos
1. **Aumento de datos sintéticos con difusión/GAN para detección de defectos superficiales con pocos ejemplos** (Calidad). Nativo de la simulación (tú generas los datos), alta novedad, usa MVTec AD/NEU/DAGM. Queda en segundo lugar porque el entrenamiento de generación de imágenes exige más GPU y evaluar "¿realmente ayudan los datos sintéticos?" es metodológicamente sutil.
2. **Estimación de RUL en C-MAPSS con cuantificación calibrada de incertidumbre** (Mantenimiento). *Benchmark* totalmente simulado, métricas limpias. Segundo porque el *benchmark* está saturado; la novedad debe venir de la cuantificación/calibración de incertidumbre, no de la precisión bruta.
3. **Optimización vía simulación con modelos sustitutos / optimización bayesiana para una línea de producción simulada** (Manufactura). Elegante y nativo de la simulación, pero con literatura más pequeña y menor superficie de publicación.

## Parte 3 — Introducción a la Materia (Primer sobre DRL para Programación)

### 3.1 Fundamentos conceptuales y terminología
- **Problema de Programación de Taller de Trabajo (JSSP):** *n* trabajos, cada uno una secuencia fija de operaciones, procesados en *m* máquinas; cada operación requiere una máquina específica durante un tiempo fijo; se minimiza un objetivo (clásicamente el **makespan**, C_max). NP-difícil.
- **JSSP Flexible (FJSP):** cada operación puede ejecutarse en cualquiera de un *conjunto* de máquinas elegibles (flexibilidad de ruteo) → añade una decisión de asignación de máquina sobre la de secuenciación; más difícil y más realista.
- **Programación dinámica/estocástica:** los eventos se despliegan en el tiempo — llegan trabajos nuevos, las máquinas se averían, los tiempos de proceso son aleatorios — lo que exige (re)programación en línea.
- **Grafo disyuntivo:** la representación estándar del JSSP; nodos = operaciones, arcos conjuntivos = precedencia dentro de un trabajo, arcos disyuntivos = restricciones de máquina compartida. Resolver = orientar los arcos disyuntivos.
- **Formulación como MDP:** *estado* = horario parcial/grafo actual; *acción* = elegir la siguiente operación (y, en FJSP, la máquina); *recompensa* = derivada del makespan/tardanza; *política* = red neuronal que selecciona acciones.
- **Reglas de despacho:** heurísticas de prioridad simples (SPT = menor tiempo de proceso, MWKR = mayor trabajo restante, MOR, FIFO) — las líneas base clásicas que un agente de RL debe superar.

### 3.2 Evolución histórica
Los métodos exactos (*branch-and-bound*, MILP, programación por restricciones) resuelven instancias pequeñas de forma óptima pero no escalan. Las metaheurísticas (algoritmos genéticos, búsqueda tabú, recocido simulado, ACO) dominaron entre los años 1990 y 2010 para instancias mayores. Las reglas de despacho por prioridad ofrecían decisiones en línea rápidas pero miopes. Desde ~2018, el **DRL de extremo a extremo** replanteó la construcción del horario como toma de decisiones secuencial: *Pointer Networks* (Vinyals, 2015) y modelos de atención (Kool, 2019) para ruteo; luego L2D de Zhang et al. (NeurIPS 2020) trajo GNN+PPO al JSSP; Park et al. (IJPR 2021) y Song et al. (IEEE TII 2023, FJSP) consolidaron el paradigma GNN+DRL hoy estándar.

### 3.3 Métodos técnicos y familias de modelos centrales
- **Codificadores de estado:** redes neuronales de grafos (GIN, GAT, GNN heterogéneas para FJSP) sobre el grafo disyuntivo; alternativamente, codificadores de atención/*Transformer*.
- **Algoritmos de RL:** PPO (el más común, estable), además de DQN, A2C/A3C; REINFORCE con líneas base para construcción.
- **Dos familias de diseño:** (1) *construcción de extremo a extremo* — la política construye el horario operación por operación; (2) *selección de reglas de despacho* — el agente elige entre reglas de prioridad en cada época de decisión (más fácil, natural para entornos dinámicos/en línea).
- **Modelado de la recompensa:** recompensas intermedias densas (p. ej., makespan incremental negativo) frente a recompensa terminal dispersa.

### 3.4 Métricas de evaluación y forma de reportar
- **Valores objetivo:** makespan C_max; tardanza total/ponderada; tiempo de flujo; utilización de máquinas; energía (para multiobjetivo).
- **Brecha de optimalidad/calidad:** brecha porcentual respecto al mejor conocido o al óptimo de CP-SAT: (obj − obj*) / obj*.
- **Generalización:** entrenar en instancias pequeñas, probar en instancias mayores/distribuciones no vistas; reportar la degradación de la brecha.
- **Eficiencia:** tiempo de inferencia/decisión; tiempo de entrenamiento en reloj; eficiencia muestral.
- **Robustez (dinámica):** desempeño con averías/tiempos estocásticos inyectados frente a la línea base determinista.
- **Reporte estadístico:** múltiples semillas, media ± desviación estándar, y pruebas de significancia no paramétricas (Wilcoxon de rangos con signo por pares; Friedman + Nemenyi entre múltiples algoritmos) — el estándar del campo para comparar algoritmos de programación.

### 3.5 Problemas abiertos y controversias
- ¿Generalizan realmente las políticas aprendidas, o se sobreajustan a las distribuciones de entrenamiento? (Controversia central.)
- ¿Son los programadores DRL de verdad mejores que un AG bien ajustado o que CP-SAT con el mismo presupuesto de cómputo? (El rigor de las líneas base está en disputa.)
- Modelado de recompensa frente a recompensa dispersa; asignación de crédito en horizontes largos.
- La transferencia simulación-realidad sigue en gran medida sin abordarse.

### 3.6 Lista de lectura priorizada (orden sugerido)
1. **Primero una revisión** — una revisión reciente (2023–2024) de DRL para JSSP/FJSP en IEEE T-ASE / *Journal of Intelligent Manufacturing* / *Computers & Industrial Engineering* (para la taxonomía y el mapa de brechas).
2. **Zhang et al., L2D, NeurIPS 2020** (arXiv:2010.12367) — el método seminal GNN+PPO de extremo a extremo; léelo junto con el código `zcaicaros/L2D`.
3. **Park et al., IJPR 2021** — formalización GNN+RL para JSSP.
4. **Song et al., IEEE TII 2023** — DRL con GNN heterogénea para FJSP (tu línea base más cercana).
5. **Kool et al., ICLR 2019** y **Vinyals et al., NeurIPS 2015** — fundamentos de atención y *pointer networks*.
6. **Tassel et al., 2021** (JSSEnv) y **schlably** (SoftwareX 2023) — los entornos sobre los que construirás.
7. **Reijnen et al., 2023** — base de código unificada para *benchmarking* de talleres de trabajo.
8. **Libros de texto:** Pinedo, *Scheduling: Theory, Algorithms, and Systems*; Sutton y Barto, *Reinforcement Learning: An Introduction* (2.ª ed.).
9. **Cursos/tutoriales:** Hugging Face Deep RL Course; OpenAI Spinning Up; Stanford CS234.
10. **Bibliotecas:** Stable-Baselines3 / CleanRL (PPO), PyTorch Geometric (GNN), Gymnasium, Google OR-Tools (línea base CP-SAT), SimPy (capa de simulación de eventos discretos).

### 3.7 Revistas y congresos objetivo
Revistas: *Journal of Manufacturing Systems*; IEEE T-ASE; IEEE TII; *Computers & Industrial Engineering*; *International Journal of Production Research*; *Journal of Intelligent Manufacturing*; IISE Transactions. Congresos: IEEE CASE; Winter Simulation Conference (WSC); IFAC MIM; talleres de NeurIPS/ICML sobre aprendizaje automático para optimización combinatoria.

## Parte 4 — Pregunta de Investigación

**Pregunta principal:** *¿Puede una política de aprendizaje por refuerzo profundo basada en redes neuronales de grafos, entrenada sobre instancias pequeñas y deterministas de taller de trabajo flexible, generalizar a instancias mayores y mantenerse robusta ante perturbaciones estocásticas (averías de máquina, tiempos de proceso estocásticos, llegadas dinámicas de trabajos), superando a las reglas de despacho por prioridad e igualando o aproximándose a la calidad de solución de las metaheurísticas y de CP-SAT con una fracción del costo en tiempo de decisión?*

**Subpreguntas e hipótesis:**
- **SP1:** ¿Cómo escala la brecha de generalización al tamaño a medida que las instancias de prueba superan el tamaño de entrenamiento? **H1:** la brecha de optimalidad de la política aprendida se degrada de forma suave (sublineal) y se mantiene por debajo de la de las reglas de despacho ajustadas.
- **SP2:** ¿Cuánto mejora la robustez el entrenamiento explícito con perturbaciones frente a una política entrenada de forma determinista? **H2:** el entrenamiento consciente de perturbaciones reduce la degradación de makespan/tardanza bajo perturbaciones inyectadas en un margen estadísticamente significativo.
- **SP3:** ¿Cómo se compara la política DRL con OR-Tools CP-SAT y con un AG en calidad frente a tiempo de decisión? **H3:** el DRL es mucho más rápido en tiempo de decisión con una brecha de calidad modesta y cuantificable que se reduce bajo presupuestos de tiempo ajustados y en condiciones dinámicas.
- **SP4 (opcional):** ¿Produce una recompensa multiobjetivo (makespan + tardanza) horarios competitivos en el sentido de Pareto? **H4:** una recompensa escalarizada/ponderada genera compromisos no dominados frente a las líneas base de un solo objetivo.

**Contribución/novedad:** Un estudio sistemático y reproducible de *generalización + robustez ante perturbaciones* para FJSP con GNN-DRL, con comparación estandarizada frente a reglas de despacho, AG y CP-SAT sobre instancias públicas — abordando directamente las tres brechas más citadas, enteramente con datos generados por simulación.

**Alcance y delimitaciones:** FJSP (no programación general de proyectos con recursos restringidos); objetivos limitados a makespan (+ tardanza opcional); perturbaciones limitadas a los tres tipos nombrados; PPO de agente único; solo simulación (sin despliegue simulación-realidad).

**Mapeo de brechas:** Pregunta principal → brecha de generalización; SP2 → brecha dinámico-estocástica; SP3 → brecha de rigor en las líneas base; SP4 → brecha multiobjetivo.

---

# DOCUMENTO 2 — Plan de Desarrollo Metodológico

## M1. Diseño y encuadre de la investigación
Investigación cuantitativa, experimental y computacional dentro de un marco de **Investigación en Ciencias del Diseño (Design Science Research, DSR)**: el "artefacto" es la política de programación DRL; la evaluación es experimentación controlada por simulación. Postura positivista/empírica; hipótesis contrastadas mediante experimentos factoriales controlados con inferencia estadística.

## M2. Estrategia de recolección de datos (simulación)
- **Instancias:** usar bibliotecas públicas de referencia — Taillard (ta01–ta80), Demirkol/DMU, Lawrence (la), Fisher–Thompson — para JSSP, y conjuntos estándar de FJSP (Brandimarte MK01–MK10; Hurink edata/rdata/vdata) para la flexibilidad. Generar instancias sintéticas adicionales con un generador aleatorio controlado (variando n trabajos, m máquinas, distribución de tiempos de proceso, razón de flexibilidad) para entrenamiento y pruebas de generalización.
- **Capa de simulación:** envolver las instancias en un entorno de eventos discretos — extender JSSEnv/schlably o construir un entorno Gymnasium basado en SimPy — para inyectar eventos dinámicos: averías de máquina (tiempo hasta el fallo exponencial + reparación), tiempos de proceso estocásticos (multiplicador log-normal), llegadas dinámicas de trabajos según Poisson.
- **Diseño de escenarios (factores experimentales):** tamaño de instancia (pequeño/mediano/grande), razón de flexibilidad, régimen de perturbación (ninguno / averías / tiempos estocásticos / llegadas / combinado), objetivo (makespan / makespan+tardanza).
- **Volumen:** entrenar con miles de episodios (PPO durante millones de pasos de entorno); evaluar sobre instancias de referencia reservadas con ≥30 réplicas estocásticas cada una para estimar la varianza.
- **Verificación y validación de la simulación:** *verificación* — pruebas unitarias de que los horarios generados respetan precedencias y restricciones de máquina; reproducir con CP-SAT los makespan óptimos conocidos en ft06/ft10. *Validación* — confirmar que las líneas base de reglas de despacho reproducen los valores publicados; confirmar que CP-SAT recupera los mejores makespan conocidos en las instancias de Taillard.
- **Justificación de los datos simulados:** las instancias de referencia son la verdad de campo aceptada por la disciplina; CP-SAT provee referencias óptimas o casi óptimas, de modo que los resultados simulados son directamente comparables con trabajos publicados previamente.

## M3. Canalización de preprocesamiento
- **Caracterización del estado:** atributos de nodo (tiempo de proceso de la operación, trabajo restante, elegibilidad de máquina, tiempo de disponibilidad), normalizados (min-max o z-score por atributo); construir el grafo disyuntivo/heterogéneo en cada paso de decisión.
- **Secuenciación/ventaneo:** los episodios del MDP son horarios completos de la instancia; no se requiere ventaneo manual (a diferencia de las series temporales).
- **Manejo de clases/etiquetas:** el RL no tiene etiquetas; asegurar que el escalado de la recompensa sea estable (normalización de recompensa/ventaja en PPO).
- **Particionado:** entrenar en una distribución (p. ej., ≤15×15), validar en un conjunto disjunto, **probar en instancias estrictamente mayores y no vistas y en todas las bibliotecas de referencia** — esta partición fuera de distribución es el núcleo de la afirmación sobre generalización.

## M4. Desarrollo del modelo
- **Líneas base:** reglas de despacho por prioridad (SPT, MWKR, MOR, FIFO, LWKR); un algoritmo genético; OR-Tools CP-SAT (con presupuestos de tiempo); y la reproducción de L2D / Song et al. como línea base de aprendizaje.
- **Modelo propuesto:** codificador GNN heterogéneo (nodos de operación + de máquina) → actor-crítico PPO; ablaciones sobre el codificador (GIN frente a GAT), la recompensa (densa frente a dispersa, un objetivo frente a multiobjetivo) y el entrenamiento consciente de perturbaciones frente al determinista.
- **Ajuste de hiperparámetros:** búsqueda en malla/bayesiana (Optuna) sobre tasa de aprendizaje, capas/anchura de la GNN, recorte de PPO, coeficiente de entropía y descuento; selección únicamente sobre el conjunto de validación.

## M5. Diseño experimental
- **Factores/niveles:** como en M2; factorial completo o fraccionado.
- **Réplicas:** ≥5 semillas de entrenamiento por configuración; ≥30 réplicas estocásticas de evaluación por instancia.
- **Aleatorización:** control de semillas para el entorno, torch y numpy; orden aleatorizado de instancias.
- **Potencia estadística:** con 30 réplicas y tamaños de efecto típicos, Wilcoxon tiene potencia amplia; reportar tamaños de efecto (p. ej., diferencia mediana, A12 de Vargha–Delaney).

## M6. Evaluación
- **Métricas:** makespan, brecha de optimalidad respecto a CP-SAT, tardanza (si es multiobjetivo), tiempo de decisión, tiempo de entrenamiento, curvas de brecha de generalización.
- **Pruebas de significancia:** Wilcoxon de rangos con signo para comparaciones por pares DRL frente a línea base; prueba de Friedman + post-hoc de Nemenyi entre múltiples algoritmos (con diagramas de diferencia crítica) — el protocolo estándar del campo.
- **Validación cruzada:** entre bibliotecas de instancias y semillas.
- **Sensibilidad/robustez:** variar la intensidad de las perturbaciones; pruebas de esfuerzo en las instancias más grandes; ablación de recompensa y codificador.

## M7. Herramientas y pila de software
Python; PyTorch + PyTorch Geometric (GNN); Gymnasium; Stable-Baselines3 o CleanRL (PPO); Google OR-Tools CP-SAT (línea base/oráculo); SimPy (capa de eventos discretos) y/o JSSEnv/schlably; Optuna (optimización de hiperparámetros); NumPy/pandas; Matplotlib/Seaborn; scipy.stats/scikit-posthocs (estadística); Weights & Biases o MLflow (seguimiento de experimentos); Git + DVC (versionado).

## M8. Recursos computacionales
Una sola GPU de consumo (8–12 GB, clase RTX) es suficiente; las GNN de programación son ligeras. El conjunto completo de experimentos corre en horas o días. La capa gratuita en la nube (Colab/Kaggle) es una alternativa viable. Esto es plenamente viable para un estudiante.

## M9. Reproducibilidad
Semillas fijas; entorno anclado (conda/requirements + Docker); ejecuciones dirigidas por configuración (Hydra); seguimiento de experimentos; repositorio público en GitHub con instancias, entorno y scripts; archivado en Zenodo con DOI; reportar hardware y tiempo de reloj.

## M10. Amenazas a la validez
- **Interna:** comparación injusta de hiperparámetros → ajustar por igual todas las líneas base de aprendizaje; varianza por semilla → múltiples semillas + intervalos de confianza.
- **Externa:** sobreajuste a distribuciones tipo Taillard → probar en múltiples bibliotecas y distribuciones sintéticas; solo simulación → declararlo explícitamente, sin afirmaciones sobre planta real.
- **De constructo:** sesgo hacia solo makespan → incluir tardanza; artefactos del modelado de recompensa → ablacionar.
- **De conclusión:** inflación por comparaciones múltiples → Friedman/Nemenyi con corrección; reportar tamaños de efecto y no solo valores p.

## M11. Consideraciones éticas
Riesgo bajo (sin datos humanos ni personales). Cabe señalar implicaciones más amplias: automatización de decisiones de programación, impacto laboral y la responsabilidad de reportar resultados negativos con honestidad (evitar seleccionar líneas base a conveniencia).

## M12. Cronograma (≈9 meses)

| Fase | Meses | Actividades | Entregables |
|---|---|---|---|
| 1. Fundamentos y revisión de literatura | 1–1,5 | Lectura de revisiones, mapa de brechas, reproducir L2D | Capítulo de revisión de literatura; línea base funcionando |
| 2. Construcción del entorno y V&V | 1,5–3 | Entorno SimPy/Gym, inyección de perturbaciones, verificación | Entorno de simulación validado; pruebas unitarias |
| 3. Líneas base | 3–4 | Reglas de despacho, AG, arnés para CP-SAT | Tabla de resultados de líneas base |
| 4. Desarrollo del modelo | 4–6 | GNN-DRL, ablaciones de recompensa/codificador, ajuste de hiperparámetros | Modelos entrenados; resultados de ablación |
| 5. Experimentos y análisis | 6–8 | Experimentos de generalización y perturbaciones, estadística | Resultados completos; análisis de significancia |
| 6. Redacción y envío | 8–9 | Redacción de la tesis, artículo de congreso | Borrador de tesis; artículo enviado |

## M13. Registro de riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| El DRL no supera al AG/CP-SAT ajustados | Media | Medio | Enmarcar la contribución en la ventaja de tiempo de decisión y robustez; reportar honestamente como hallazgo |
| Inestabilidad del entrenamiento | Media | Medio | Usar valores por defecto de SB3/CleanRL, normalización de recompensa, más semillas |
| Errores en el entorno (horarios inválidos) | Media | Alto | Verificación rigurosa contra CP-SAT; aserciones de restricciones |
| Ampliación descontrolada del alcance (demasiados factores) | Alta | Medio | Congelar el conjunto de factores desde el inicio; multiobjetivo como opcional |
| Sobrecostos de cómputo | Baja | Medio | Limitar tamaños de instancia; alternativa en la nube |

## M14. Resultados esperados y estructura de la tesis
Capítulos: (1) Introducción y pregunta de investigación; (2) Revisión de literatura y brecha; (3) Metodología (este plan); (4) Entorno de simulación y V&V; (5) Líneas base; (6) Modelo GNN-DRL propuesto; (7) Experimentos y resultados; (8) Discusión y amenazas a la validez; (9) Conclusiones y trabajo futuro. Artículo objetivo: una ponencia en IEEE CASE o WSC, extendida a revista (IEEE T-ASE / *Computers & Industrial Engineering*).

## Recomendaciones
1. **Comprométete con el DRL para FJSP dinámico** como tema de tesis. Empieza la Fase 1 reproduciendo L2D de Zhang et al. sobre instancias de Taillard — **si no logras reproducir una línea base publicada en ~3 semanas, ese es tu punto de decisión de continuar o no.**
2. **Fija el alcance temprano:** makespan + tardanza opcional; tres tipos de perturbación; PPO de agente único. Añade el multiobjetivo solo si las Fases 1–4 terminan a tiempo.
3. **El rigor en las líneas base es tu foso de novedad:** compara siempre contra reglas de despacho ajustadas, un AG y OR-Tools CP-SAT con presupuestos de tiempo equiparados, usando estadística de Friedman/Nemenyi. Esto por sí solo distingue tu tesis de los muchos artículos de DRL en programación que comparan de forma insuficiente.
4. **Detonantes de plan alternativo:** si el DRL no se aproxima a la calidad de CP-SAT tras la Fase 4, reorienta la contribución hacia *robustez ante perturbaciones y tiempo de decisión* (sigue siendo publicable); si toda la ruta de DRL se estanca, cambia a la Alternativa 2 (detección de defectos con datos sintéticos sobre MVTec/NEU), que reutiliza buena parte de las herramientas.
5. **Publica el entorno** como código abierto — un entorno Gym de FJSP limpio y con capacidad de perturbaciones es en sí mismo una contribución citable y refuerza la narrativa de reproducibilidad.

**Umbrales que cambiarían la recomendación:** (i) si una búsqueda actualizada en Scopus/Semantic Scholar (2025–2026) muestra que la combinación exacta "generalización + FJSP dinámico + GNN-DRL" se ha publicado a fondo en los últimos 12 meses, estrecha la novedad hacia un tipo de perturbación poco cubierto o un escenario multiobjetivo; (ii) si la reproducción de L2D falla en el punto de decisión de la Fase 1, cambia a la familia de diseño de *selección de reglas de despacho* (más simple y robusta) o a las Alternativas 2/3.

## Advertencias
- **Limitación de herramientas:** la búsqueda y recuperación web en vivo no estuvieron disponibles, por lo que los conteos exactos de publicaciones 2023–2026, los porcentajes de crecimiento, las cifras actuales de los *leaderboards*, los DOI/URL y las citas textuales no pudieron verificarse. Todas las identidades de artículos (Zhang L2D NeurIPS 2020; Song FJSP-GNN IEEE TII 2023; Bergmann MVTec CVPR 2019; Saxena C-MAPSS PHM 2008), contenidos de conjuntos de datos e identidades de herramientas provienen de conocimiento consolidado del dominio y deben verificarse manualmente antes de citarse.
- **Riesgo de desactualización:** la frontera 2025–2026 (especialmente la programación con agentes LLM y los datos sintéticos por difusión) evoluciona rápido; confirma las revisiones más recientes antes de fijar la afirmación sobre la brecha.
- **Verificación de novedad:** antes de comprometerte, ejecuta una búsqueda en Scopus/Semantic Scholar con "generalization" + "dynamic flexible job shop" + "deep reinforcement learning" (2024–2026) para asegurar que la combinación específica sigue abierta.
- **Honestidad en las comparaciones:** los programadores DRL no siempre superan a CP-SAT/AG en calidad bruta; la contribución defendible y honesta es el compromiso entre generalización, robustez y tiempo de decisión, no necesariamente un nuevo mejor makespan.
