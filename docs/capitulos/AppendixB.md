# Anexo B — Hiperparámetros y Arquitectura

> **Nota de trazabilidad.** Este anexo documenta la arquitectura **realmente implementada** a la fecha (Semana 18, `TRAINING_REPORT_WEEK18.md`; código en `models/gin.py`, `models/policy.py`, `envs/jssp_env.py`, `envs/fjsp_env.py`), no la especificación teórica de `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` tomada de forma independiente — donde el código coincide exactamente con la especificación, se cita la especificación como respaldo; donde el código añade algo que la especificación no cubría (la cabeza FJSP, la extensión del espacio de acciones), se documenta como lo que es, una extensión, no un cumplimiento literal. **Los valores de la Tabla B.6 no son finales**: son los valores de `../docs/logs/TrainingPlan_Week17.md` §2, usados sin ajuste en la corrida de la Semana 18 por instrucción explícita del sprint ("prioritize getting a correct... training loop over premature tuning", `TRAINING_REPORT_WEEK18.md` §2). Los valores finales dependen de la Fase 2 de `HPO_TechSpec.md`, todavía no ejecutada — cada celda pendiente se marca `[PENDIENTE: HPO_TechSpec.md Fase 2]`, no se deja en blanco ni se anticipa.

> **Nota adicional de conciliación:** `../docs/capitulos/Chapter4_sections_4.1_4.3.md` describe la arquitectura **homogénea** del Sprint 1 (`GINEncoder`, `JSSPGraphFeaturesExtractor`, sin nodos de máquina) como si fuera la arquitectura vigente del pipeline. Ese capítulo necesita su propia actualización de trazabilidad para Semana 18 (fuera del alcance de este anexo) — la ruta homogénea sigue existiendo en el código como configuración de ablación (`use_machine_edges=False`, `../docs/planificacion/AblationPlan.md` §2.1), no ha sido eliminada, pero ya no es la configuración por defecto que este anexo documenta.

---

## B.1 Espacio de observación

### B.1.1 Nodos de operación — 8 dimensiones, sin cambios desde el Sprint 1

**Tabla B.1.** `NUM_NODE_FEATURES = 8` (`envs/jssp_env.py`, `NODE_FEATURE_NAMES`), compartido sin modificación entre `JSSPEnv` y `FJSPEnv`.

| # | Nombre | Descripción |
|---:|---|---|
| 0 | `processing_time` | $p_o$ / duración máxima de la instancia |
| 1 | `is_scheduled` | 1 si la operación ya fue despachada |
| 2 | `is_ready` | 1 si es legal despachar ahora (frontera de su trabajo) |
| 3 | `is_job_complete` | 1 si el trabajo propietario ya está completo |
| 4 | `earliest_start_estimate` | estimación normalizada del inicio más temprano |
| 5 | `remaining_job_work` | trabajo no despachado restante en el trabajo, normalizado |
| 6 | `job_progress` | posición / operaciones por trabajo |
| 7 | `machine_remaining_load` | trabajo no despachado restante en la máquina de la operación, normalizado |

### B.1.2 Nodos de máquina — 7 dimensiones, nuevo en Semana 18

**Tabla B.2.** `NUM_MACHINE_FEATURES = 7` (`envs/jssp_env.py`, re-exportado desde `envs/fjsp_env.py` para simetría), `MACHINE_FEATURE_NAMES`.

| # | Nombre | Descripción |
|---:|---|---|
| 0 | `is_free` | componente one-hot 1/3 |
| 1 | `is_busy` | componente one-hot 2/3 |
| 2 | `is_broken` | componente one-hot 3/3 — siempre 0.0 bajo `JSSPEnv`/`FJSPEnv` (estáticos); se activa al cablear `DynamicJSSPEnv` (pendiente) |
| 3 | `time_to_available` | 0 si libre; si no, (tiempo de liberación − t)/horizonte |
| 4 | `assigned_pending_load` | suma de duraciones de operaciones pendientes que requieren esta máquina / carga máxima |
| 5 | `utilization_accum` | tiempo ocupado / **extensión realizada del cronograma** (no `t` actual — ver B.10, bug corregido) |
| 6 | `estimated_availability` | estimación empírica de disponibilidad; 1.0 bajo los entornos estáticos (sin averías) |

### B.1.3 Relaciones del grafo

**Tabla B.3.** Cuatro relaciones consumidas por el codificador heterogéneo.

| Relación | Adyacencia | Forma | Dirección |
|---|---|---|---|
| Precedencia (predecesor) | `adj_precedence` (transpuesta) | $(N_{op}, N_{op})$ | operación → operación (misma cadena de trabajo) |
| Precedencia (sucesor) | `adj_precedence` | $(N_{op}, N_{op})$ | operación → operación |
| Máquina compartida | `adj_machine` | $(N_{op}, N_{op})$ | operación ↔ operación (simétrica) |
| Elegibilidad | `adj_eligibility` | $(N_{op}, N_{mach})$ | operación ↔ máquina — **uno-hot** por fila en `JSSPEnv` (ruta fija), **multi-hot** por fila en `FJSPEnv` (enrutamiento flexible) |

---

## B.2 Codificador GNN heterogéneo (`models/gin.py`)

**Tabla B.4.** `HeteroRelationalGINLayer` / `HeteroGINEncoder`, valores realmente usados en la corrida de la Semana 18 (idénticos a `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` Tabla 2.1).

| Parámetro | Valor | Nota |
|---|---:|---|
| `in_dim_ope` | 8 | Tabla B.1 |
| `in_dim_mach` | 7 | Tabla B.2 |
| `hidden_dim` (ambos tipos de nodo, cada capa) | 64 | igual al `GINEncoder` homogéneo validado y a L2D |
| `num_layers` (rondas de paso de mensajes) | 3 | igual al valor validado y a L2D |
| `mlp_layers` (profundidad del MLP interno de cada capa) | 2 | igual al valor validado y a L2D |
| Agregación | suma + MLP, uniforme en las 4 relaciones | sin atención — justificación teórica y de precedentes en `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §2.1; **no probado empíricamente contra la alternativa GAT-híbrida todavía** — ver `../docs/planificacion/AblationPlan.md` §2.1 |
| Matrices de peso, lado operación | 4 (`lin_pred`, `lin_succ`, `lin_machine`, `lin_elig`) | una por relación entrante a un nodo-operación |
| Matrices de peso, lado máquina | 1 (`lin_cand`) | agrega los candidatos operación→máquina |
| `learn_eps` | `True` | $\epsilon$ de GIN como parámetro aprendible, no fijo en 0 |
| Pooling de grafo | media, concatenada entre los dos tipos de nodo | `graph_embedding = concat(mean(h_ope), mean(h_mach))`, dimensión 128 |

---

## B.3 Cabezas de actor y crítico (`models/policy.py`)

**Tabla B.5.** Dos rutas de acción, seleccionadas automáticamente según el entorno (`JSSPEnv` → `PerNodeActorHead`; `FJSPEnv` → `PerPairActorHead`); crítico compartido en ambas.

| Componente | Ruta JSSP | Ruta FJSP | Notas |
|---|---|---|---|
| Espacio de acción | `Discrete(num_operations)` | `Discrete(num_operations × num_machines)`, aplanado como `op_id × num_machines + machine` | la codificación aplanada mantiene el mecanismo de enmascarado de `sb3_contrib` idéntico entre ambas rutas |
| Cabeza de actor | `PerNodeActorHead` — MLP sobre el embedding de un solo nodo-operación | `PerPairActorHead` — MLP sobre `concat(h_ope, h_mach)` por par elegible | la primera capa de `PerPairActorHead` es el doble de ancha (entrada de 128 en vez de 64) porque concatena dos embeddings de 64 |
| `hidden_dim` de la cabeza | 64 | 64 | `head_num_layers=2` en ambas |
| Cabeza de crítico | `PooledCriticHead`, `in_dim=128` (concat de los dos poolings) | igual, compartida | ve tanto la señal "trabajo restante" (operaciones) como "carga/disponibilidad" (máquinas) — no disponible en el crítico homogéneo del Sprint 1 |
| Parámetros totales de la política | **97,528** | **101,624** | independiente de `num_operations`/`num_machines` — verificado por `test_per_node_actor_head_parameter_count_independent_of_n`; la diferencia entre columnas es exclusivamente el ancho de entrada de la primera capa de la cabeza de actor |
| Clase de política SB3 | `PerNodeMaskableActorCriticPolicy` | misma clase, cabeza intercambiada según el entorno | subclase de `MaskableActorCriticPolicy`, sobrescribe `_get_action_dist_from_latent` — **verificado contra `sb3_contrib==2.9.0`** exactamente, per `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §3.3; re-verificar si la versión instalada cambia |
| Extractor de características | `JSSPHeteroGraphFeaturesExtractor` | misma clase | devuelve el embedding de grafo pooled como `features` (contrato SB3) y expone los embeddings por nodo/par vía `self.last_node_embeddings` para la cabeza de actor |

---

## B.4 Hiperparámetros de entrenamiento (PPO)

**Tabla B.6.** Configuración usada en la corrida de la Semana 18 (`../docs/logs/TrainingPlan_Week17.md` §2, sin ajuste) frente a la búsqueda pendiente (`HPO_TechSpec.md`).

| Hiperparámetro | Valor actual (pre-HPO, Semana 18) | Rango de búsqueda (`HPO_TechSpec.md` §3) | Valor final |
|---|---:|---|---|
| `gamma` | 1.0 | **no se busca** — estructural (identidad telescópica de la recompensa, verificada por `test_reward_telescopes_to_h0_minus_makespan`); brazo diagnóstico opcional $\{0.99, 0.995, 1.0\}$ solo sobre `ft06` | **1.0, fijo** (no depende de HPO) |
| `learning_rate` | $2\times10^{-4}$ | log-uniforme $[1\times10^{-5},\ 5\times10^{-4}]$ | `[PENDIENTE: HPO_TechSpec.md Fase 2]` |
| `clip_range` ($\epsilon$) | 0.2 | categórico $\{0.1, 0.15, 0.2, 0.25, 0.3\}$ | `[PENDIENTE: HPO_TechSpec.md Fase 2]` |
| `ent_coef` | 0.01 (constante, sin programación) | programación de 3 parámetros: `ent_coef_start` $[5\times10^{-3},5\times10^{-2}]$, `ent_coef_end` $[1\times10^{-4},\text{start}]$, `decay_fraction` $[0.2,0.8]$ | `[PENDIENTE: HPO_TechSpec.md Fase 2]` — motivado directamente por el diagnóstico de meseta de entropía en `ft06` (`TrainingDiagnosis.md` §2.1) |
| `gae_lambda` | 0.95 | fijo, fuera de alcance de esta búsqueda | 0.95 (sin cambio previsto) |
| `n_epochs` | 3 | fijo, fuera de alcance | 3 (sin cambio previsto) |
| `vf_coef` | 0.5 | fijo, fuera de alcance | 0.5 (sin cambio previsto) |
| `n_steps` (por entorno) | 216 (`ft06`, 6×36); 360 (`mk01`, 6×60) | fijo, fuera de alcance | sin cambio previsto |
| `n_envs` | 4 | fijo, fuera de alcance | sin cambio previsto |
| `batch_size` | 64 | fijo, fuera de alcance | sin cambio previsto |
| `num_layers` (GNN) | 3 | categórico $\{2, 3, 4\}$ | `[PENDIENTE: HPO_TechSpec.md Fase 2]` |
| `hidden_dim` (GNN, ambos tipos de nodo) | 64 | categórico $\{32, 64, 96, 128\}$ — 128 motivado por el diagnóstico de posible sub-parametrización de `PerPairActorHead` en `mk01` (`TrainingDiagnosis.md` §3, hipótesis rango 4) | `[PENDIENTE: HPO_TechSpec.md Fase 2]` |

---

## B.5 Configuración de entrenamiento por corrida (Semana 18, evidencia real)

**Tabla B.7.** Presupuesto y rendimiento reales, no estimados (`TRAINING_REPORT_WEEK18.md` §2, `summary_hetero_stage{A,F}_*.json`).

| | `ft06` (JSSP) | `mk01` (FJSP) |
|---|---:|---:|
| Timesteps entrenados | 400,000 | 300,000 (`TrainingDiagnosis.md` recomienda repetir a 1,000,000 antes de reportar en el Capítulo 6) |
| Episodios completados | 11,112 | 5,016 |
| Tiempo de pared de entrenamiento | 2,216.1 s (36.9 min) | 2,528.9 s (42.1 min) |
| Rendimiento aislado | 275 pasos/s | 134 pasos/s |
| Semilla(s) | 0 únicamente — **`TrainingDiagnosis.md` §3/§4 exige ≥3 semillas antes de reportar en el Capítulo 6** | 0 únicamente, mismo requisito pendiente |
| Hardware | CPU, 8 núcleos, `torch==2.12.1+cpu`, sin CUDA — no la GPU única que `../docs/logs/TrainingPlan_Week17.md` §1 asume; ambas corridas concurrentes a 3 hilos de `torch` cada una (cifras de tiempo de pared son cota superior, no aislada) | igual |

---

## B.6 Decisiones de diseño con desviación documentada (resumen de `TRAINING_REPORT_WEEK18.md` §1.1)

**Tabla B.8.** Tres decisiones donde la implementación se apartó de una lectura literal de la especificación/encargo — resumidas aquí para referencia rápida del anexo; la justificación completa está en la fuente citada.

| Decisión | Qué se implementó | Por qué | Fuente |
|---|---|---|---|
| Paso de mensajes denso (`bmm`) en vez de `torch_geometric.HeteroConv` | Cinco términos `nn.Linear` + `bmm`, sin dependencia de PyG | PyG no está instalado; el entorno es CPU-only; a estos tamaños de grafo (36–225 nodos de operación) el producto denso es además más rápido; PyG es la causa más común de fallas de instalación en repos DRL-scheduling publicados (docstring de `models/gin.py`) | `TRAINING_REPORT_WEEK18.md` §1.1(a) |
| `stable-baselines3`/`sb3_contrib` en vez de una implementación de PPO propia | `PerNodeMaskableActorCriticPolicy`, subclase de `MaskableActorCriticPolicy` | El agrupamiento heterogéneo no fue un obstáculo (todo entorno de una corrida comparte tamaño de grafo fijo); reutiliza la máscara `MaskableCategorical` y la implementación de GAE/pérdida PPO ya probadas | `TRAINING_REPORT_WEEK18.md` §1.1(b) |
| Espacio de acción FJSP `(operación, máquina)` — fuera de alcance en `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §0 | `FJSPEnv`, `PerPairActorHead`, `envs/fjsp_loader.py` | El encargo del sprint (STEP 3/STEP 5) exigía lo contrario a la especificación vigente; se siguió el encargo y se extendió la especificación en vez de contradecir al encargo — la ruta JSSP permanece sin cambios | `TRAINING_REPORT_WEEK18.md` §1.1(c) — **acción pendiente:** reconciliar `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §0 con este alcance ya implementado (también señalado en `TrainingDiagnosis.md` §4, ítem 7) |

---

## B.7 Dependencias de software y reproducibilidad

**Tabla B.9.** `pyproject.toml`, dependencias fijadas por versión exacta.

| Paquete | Versión | Rol |
|---|---|---|
| `numpy` | 2.4.6 | cómputo numérico |
| `pandas` | 2.2.2 | manejo de CSV de resultados |
| `matplotlib` | 3.8.4 | curvas de entrenamiento (`Agg`, sin dependencia de display) |
| `gymnasium` | 1.3.0 | contrato de entorno |
| `torch` | 2.12.1 | red neuronal, CPU-only en la corrida de la Semana 18 |
| `stable_baselines3` | 2.9.0 | PPO base |
| `sb3_contrib` | 2.9.0 | `MaskablePPO`, `MaskableCategorical` |
| `ortools` | 9.15.6755 | CP-SAT (línea base, cota $C^*$) |
| `pytest` | 9.1.1 (grupo `dev`) | 585 pruebas, 0 regresiones (Semana 18) |
| `tensorboard` | 2.21.0 (grupo `monitoring`, opcional) | registro de curvas |
| `optuna` | **no listado todavía** | requerido por `HPO_TechSpec.md` — **acción pendiente:** añadir a un grupo opcional (`hpo`) antes de ejecutar la Fase 1 de la búsqueda |

**Checkpoints de referencia citables desde este anexo:** `experiments/results/checkpoints/ppo_hetero_stageA_ft06_final.zip`, `experiments/results/checkpoints/ppo_hetero_stageF_mk01_final.zip` — ambos entrenados **después** de la corrección del bug de B.8 (ningún checkpoint citado en este anexo proviene de la versión con la característica `utilization_accum` mal escalada).

---

## B.8 Nota sobre integridad de datos: bug corregido durante el sprint

`utilization_accum` (característica de máquina #5, Tabla B.2) se calculó inicialmente como `tiempo_ocupado / t_actual`, siguiendo el texto literal de la especificación. Bajo *list scheduling*, una operación puede colocarse en el momento más temprano factible en su máquina, que puede ser **posterior** a la siguiente época de decisión — el tiempo ocupado comprometido excedía rutinariamente `t_actual`, llegando a 3.0 donde el rango válido es $[0,1]$. Corregido: el denominador ahora es la extensión real del cronograma construido hasta el momento, que acota por construcción cada intervalo comprometido. **Ambas corridas documentadas en este anexo y en el Capítulo 6 se reentrenaron desde cero después de la corrección** (`TRAINING_REPORT_WEEK18.md` §6.3); ningún resultado citado en este proyecto proviene de la versión con el error. Prueba de regresión: `test_utilization_never_exceeds_one`.
