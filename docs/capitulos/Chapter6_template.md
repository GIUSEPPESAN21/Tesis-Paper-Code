# Capítulo 6 — Resultados (plantilla de reporte)

> **Nota de trazabilidad.** Este capítulo se redacta como **andamiaje de reporte** (Semana 20 del plan mensual), antes de que existan resultados finales, siguiendo el mismo principio que ya rige el pre-registro (`PreRegistration_FINAL.md`, congelado 2026-07-29T05:22:26Z): fijar la estructura, las tablas, las figuras y el criterio de aceptación **antes** de generar los datos que los llenarán, para que ningún resultado empírico pueda influir retroactivamente en cómo se decide reportarlo. Cada subsección (6.1–6.5) reproduce, en el orden y con el presupuesto de páginas de `../docs/planificacion/Step 4 Thesis.md` Parte A ("Cap. 6 — Resultados — 20 pp."), la hipótesis que atiende, el estado actual de ejecución, y marcadores explícitos `[TABLA: …]`/`[FIGURA: …]` donde el resultado final se inserta sin reestructurar el texto alrededor.

> **Nota de conciliación (obligatoria antes de cerrar este capítulo).** `../docs/capitulos/Section5.1_5.2.md` §5.1.4 fija un criterio de aceptación de H1 *provisional*, basado en el bloque de $N=13$ instancias evaluado en la Semana 9/12, y nombra a **FIFO** explícitamente como "la mejor PDR" de ese bloque. `PreRegistration_FINAL.md` §2 (congelado con posterioridad, 2026-07-29) fija el criterio **definitivo**: el comparador es la mejor PDR **por instancia** (no una regla fija nombrada de antemano) sobre las $N=43$ instancias `ft`/`la` bundladas, más el brazo FJSP-sintético — una definición más estricta y, además, la que rige por ser el documento de pre-registro congelado en último lugar. **Este capítulo sigue el criterio de `PreRegistration_FINAL.md` como autoridad**; `../docs/capitulos/Section5.1_5.2.md` §5.1.4 debe marcarse como superada por ese documento (acción pendiente de edición en el Capítulo 5, fuera del alcance de esta plantilla, pero señalada aquí para que nadie cite el criterio de FIFO como si fuera el vigente).

> **Estado de ejecución agregado, a la fecha de esta plantilla:** un único corte de entrenamiento existe (`TRAINING_REPORT_WEEK18.md`, `ft06` y `mk01`, semilla única cada uno) — evidencia de que el pipeline aprende correctamente (`TrainingDiagnosis.md`), **no** una ejecución del protocolo de las Secciones 6.2–6.5, que exigen el conjunto de instancias, las semillas y las pruebas estadísticas ya congeladas en `PreRegistration_FINAL.md`. Ningún número de `TRAINING_REPORT_WEEK18.md` debe copiarse directamente a las tablas de este capítulo como si fuera ya el resultado final — ver la advertencia específica en 6.2.

---

## 6.1 Validación del entorno de simulación

**Hipótesis que atiende:** ninguna de H1–H5 directamente — esta subsección corresponde a **E0** (`../docs/planificacion/Step 4 Thesis.md` Parte D4), el experimento de validación que condiciona la confiabilidad de todo lo que sigue. Un E0 que no reproduce las referencias conocidas invalida cualquier RPD calculado en 6.2–6.5, exactamente por eso `../docs/planificacion/Step 4 Thesis.md` (Recomendaciones, ítem 3) exige ejecutarlo "antes de cualquier entrenamiento."

**Estado de ejecución:** **el único de los cinco bloques que ya tiene evidencia empírica casi completa**, dispersa en documentos previos y lista para consolidarse aquí sin trabajo experimental adicional:

- Óptimos JSSP reproducidos exactamente por CP-SAT (`ft06`=55, `ft10`=930, `ft20`=1165), confirmado en `BASELINES_RESULTS_WEEK12.md` y re-confirmado en `TRAINING_REPORT_WEEK18.md` §5.1.
- Óptimo FJSP `mk01`=40 reproducido por CP-SAT sobre la instancia rectangularizada, verificando simultáneamente el parser, la indexación de máquinas y la neutralidad del relleno de operaciones ragged (`TRAINING_REPORT_WEEK18.md` §1.2, `test_cpsat_reaches_the_literature_optimum_on_the_padded_instance`).
- Calibración estadística de los tres generadores de perturbación (averías, tiempos estocásticos, llegadas dinámicas) ya ejecutada y reportada con $N=5000$ muestras por aserción (`VALIDATION_WEEK6.md`, Tablas de la Sección (a)): error relativo <2.6% en todos los parámetros frente al valor teórico, prueba KS sin rechazo de la hipótesis exponencial ($p=0.929$).

**Lo que falta:** consolidar estas tres fuentes en las tablas de esta sección con formato uniforme, y el caso analítico M/M/1 que D4/E0 exige como cuarto punto de referencia (no ejecutado en ninguno de los documentos citados arriba — verificar si `../docs/planificacion/VandV_Protocol.md` ya lo cubre antes de tratarlo como pendiente nuevo).

**Tabla 6.1.1.** Errores de validación E0: óptimos publicados/probados por CP-SAT frente a los reproducidos por el pipeline.

`[TABLA: instancia, óptimo publicado (OR-Library/Brandimarte), óptimo reproducido por CP-SAT, error relativo (%), estado ("proven"/"best_known")]`

**Tabla 6.1.2.** Calibración de los generadores de perturbación (consolidada de `VALIDATION_WEEK6.md`).

`[TABLA: generador (averías/tiempos estocásticos/llegadas), parámetro objetivo, valor empírico, error (%), prueba de bondad de ajuste y p-valor]`

**Figura 6.1.1.** Ajuste empírico de los tres generadores de perturbación frente a sus distribuciones teóricas.

`[FIGURA: histogramas TTF/TTR/tiempo de proceso/inter-llegada superpuestos a la densidad teórica correspondiente (Exp/Lognormal/Poisson), un panel por generador]`

---

## 6.2 Desempeño estático (JSSP/FJSP)

**Hipótesis que atiende:** **H1** (`PreRegistration_FINAL.md` §2) — la política supera a la mejor PDR por instancia en RPD medio, con significancia de Wilcoxon $p<0.05$ y mediana de mejora $>0$.

**Advertencia explícita, para que no se pierda al copiar y pegar resultados en esta plantilla:** `TRAINING_REPORT_WEEK18.md` §5.1 ya muestra `ft06`: RL=55 (0% RPD), superando las nueve PDR y empatando con CP-SAT/GA. **Este es un único par (una instancia, una semilla) y no constituye una prueba de H1** — el criterio congelado exige $N=43$ instancias y una prueba de Wilcoxon pareada; un solo resultado favorable, por convincente que parezca, es exactamente el tipo de evidencia parcial que el pre-registro existe para evitar que se sobre-interprete (`PreRegistration_FINAL.md`, encabezado). La Sección 6.2 solo se completa cuando las 43 instancias (más el brazo FJSP-sintético) están evaluadas.

**Tabla 6.2.1.** RPD por instancia y método, arm JSSP ($N=43$: `ft06`, `ft10`, `ft20`, `la01`–`la40`).

`[TABLA: instancia, familia, n_jobs×n_machines, RPD por método (9 PDR + GA + CP-SAT + política DRL), makespan, factible]`

**Tabla 6.2.2.** Resultado de la prueba de Wilcoxon de rangos con signo, política DRL vs. mejor PDR por instancia (criterio de aceptación de H1, `PreRegistration_FINAL.md` §2).

`[TABLA: N efectivo, estadístico W, p-valor, tamaño de efecto r=Z/√N, mediana de la diferencia RPD_mejorPDR − RPD_RL, decisión (aceptar/rechazar H1)]`

**Tabla 6.2.3.** Comparación de makespan bruto, brazo FJSP-sintético (sin RPD publicable — `envs.fjsp_instance.generate_fjsp_instance`, seed=0; ver `PreRegistration_FINAL.md` §1 sobre por qué este brazo no tiene RPD contra un óptimo de literatura).

`[TABLA: instancia base, makespan GA, makespan CP-SAT-FJSP, makespan política DRL, factible]`

**Figura 6.2.1.** Diagrama de diferencia crítica (Friedman + post-hoc de Nemenyi), rango medio de las nueve PDR, GA y la política DRL sobre las 43 instancias — **CP-SAT excluido del ranking conjunto** por el motivo ya fijado en `../docs/capitulos/Section5.1_5.2.md` §5.2.3/`../docs/capitulos/Section5.3.md` §5.3.6 (fija $C^*$, no compite como método par).

`[FIGURA: diagrama de diferencia crítica de Demšar (2006), eje horizontal de rango medio, barras uniendo los subconjuntos estadísticamente indistinguibles al nivel CD]`

**Figura 6.2.2.** Curvas de entrenamiento del/de los modelo(s) final(es) reportado(s) en esta sección (formato de seis paneles ya establecido en `TRAINING_REPORT_WEEK18.md` §3: retorno por episodio, makespan por episodio vs. óptimo de referencia, entropía, pérdida de valor, varianza explicada, diagnósticos de tamaño de paso de PPO).

`[FIGURA: panel de 6 gráficos, uno por instancia de entrenamiento reportada, con el checkpoint final ya seleccionado tras HPO_TechSpec.md]`

---

## 6.3 Generalización en tamaño

**Hipótesis que atiende:** **H2** (`PreRegistration_FINAL.md` §3) — el RPD en instancias 2–3× más grandes que el tamaño de entrenamiento se mantiene, por nivel (tier), no significativamente peor que el de la mejor PDR (Friedman + Nemenyi por tier).

**Estado de ejecución:** bloqueado únicamente por la disponibilidad del checkpoint de la Etapa B (`../docs/logs/TrainingPlan_Week17.md`, entrenamiento en `ft10`/`la16`–`la20`) — `TRAINING_REPORT_WEEK18.md` §8 confirma que el mecanismo de transferencia de pesos ya está implementado y probado (`test_weight_transfer_across_instance_sizes`), solo falta ejecutar la Etapa B/C. No depende de ninguna pieza de código pendiente, a diferencia de 6.4.

**Tabla 6.3.1.** Niveles (tiers) de generalización, tamaños e instancias, frente al tamaño de entrenamiento (`PreRegistration_FINAL.md` §3).

`[TABLA: tier, instancias, num_operations, razón frente al tamaño de entrenamiento (100 operaciones, bloque 10×10)]`

**Tabla 6.3.2.** RPD por instancia, método y tier (transferencia de pesos sin reentrenamiento — `deterministic=True`, un rollout por instancia, más 4 rollouts estocásticos descriptivos por `PreRegistration_FINAL.md` §2).

`[TABLA: tier, instancia, RPD política DRL (zero-shot), RPD mejor PDR, rango Friedman de cada uno]`

**Tabla 6.3.3.** Resultado de Friedman + Nemenyi por tier.

`[TABLA: tier, k métodos, N instancias, χ²_F, p-valor, CD, ¿política DRL significativamente peor que la mejor PDR? (sí/no/inconcluso)]`

**Figura 6.3.1.** Gradiente de degradación: RPD de la política DRL en función del tamaño de instancia, con el tamaño de entrenamiento marcado como referencia.

`[FIGURA: eje x = num_operations (36 a 300), eje y = RPD (%), una serie para la política DRL, una para la mejor PDR de cada tier, línea vertical en el tamaño de entrenamiento]`

**Figura 6.3.2.** Diagramas de diferencia crítica, uno por tier (o un panel de 3, `la21`–`la30`/`la36`–`la40`/`la31`–`la35`).

`[FIGURA: 3 diagramas de diferencia crítica, un panel por tier, mismo formato que la Figura 6.2.1]`

---

## 6.4 Robustez bajo perturbaciones

**Hipótesis que atiende:** **H3** (`PreRegistration_FINAL.md` §4) y, si el bloqueador se resuelve antes del cierre de este capítulo, **H4** (§5 del mismo documento — recompensa consciente de la perturbación).

**Estado de ejecución — bloqueado, y así debe permanecer marcado hasta que cambie:** `PreRegistration_FINAL.md` §0 y `TrainingDiagnosis.md` (Semana 18) coinciden en que **ni H3 ni H4 son ejecutables todavía**: la política entrenada solo cubre el contrato de observación estático de `JSSPEnv`/`FJSPEnv`; el cableado contra `DynamicJSSPEnv` (estado de máquina averiada, contrato de arreglos crecientes) sigue pendiente, y el término $\kappa\cdot\Delta_{\text{inest}}$ de H4 no está implementado más allá de la instrumentación (`info["planned_start_deviation"]`). **Esta subsección se deja con las tablas y figuras especificadas pero explícitamente marcadas como no ejecutables**, en vez de simular una estructura de "resultado pendiente de rellenar" que sugiera que solo falta tiempo de cómputo — lo que falta es una pieza de ingeniería no programada aún (`../docs/planificacion/AblationPlan.md` §2.3 rastrea el mismo bloqueador para la ablación equivalente).

**Tabla 6.4.1.** Índice de robustez ($RI$) por régimen de perturbación y método (9 regímenes: $A_k\in\{0.95,0.90,0.80\}$ / $\sigma\in\{0.1,0.25,0.5\}$ / $\rho\in\{0.7,0.85,0.95\}$, `ft06` y `la01`, 30 réplicas).

`[TABLA — BLOQUEADA hasta que exista el cableado DynamicJSSPEnv para la política entrenada: régimen, método (PDR arrival-safe, GA/CP-SAT predictivo-reactivo, política DRL), RI medio, desviación estándar, CVaR_0.9]`

**Tabla 6.4.2.** Resultado de Wilcoxon, $RI$ política DRL vs. mejor PDR, por régimen (criterio de H3, `PreRegistration_FINAL.md` §4).

`[TABLA — BLOQUEADA: régimen, N efectivo, W, p-valor, mediana ΔRI, ¿soporta H3 en este régimen? (sí/no) — decisión final: H3 aceptada solo si "sí" en los 9 regímenes]`

**Tabla 6.4.3.** Barrido de $\kappa$ (H4): efecto sobre $RI$ y sobre el RPD nominal (sin perturbación), frente al margen $\delta=2$ pp.

`[TABLA — BLOQUEADA, adicionalmente al término Δ_inest sin implementar: κ, ΔRI (Wilcoxon, por régimen), ΔRPD_nominal (umbral, no prueba de hipótesis — ver PreRegistration_FINAL.md §5), decisión por κ]`

**Figura 6.4.1.** Diagramas de caja de $RI$ por régimen y método.

`[FIGURA — BLOQUEADA: boxplot de RI, un panel por eje de perturbación (averías/tiempos/llegadas), cajas agrupadas por método dentro de cada panel]`

**Figura 6.4.2.** Compensación robustez–optimalidad del barrido de $\kappa$ (H4): $\Delta$RPD nominal en el eje x, $\Delta RI$ en el eje y, un punto por $\kappa\in\{0.5,1.0\}$, línea vertical en el margen $\delta=2$ pp.

`[FIGURA — BLOQUEADA: gráfico de dispersión robustez-vs-costo nominal, cuadrantes etiquetados según la Sección 5 de PreRegistration_FINAL.md ("robustez ganada a costo nominal inaceptable" / "sin beneficio medible")]`

---

## 6.5 Tiempos de decisión y coste computacional

**Hipótesis que atiende:** **H5** (`PreRegistration_FINAL.md` §6) — el tiempo mediano por decisión de la política es menor al 1% del de CP-SAT, en todo el rango de tamaños, con la brecha ensanchándose (no constante) al crecer el tamaño de instancia.

**Estado de ejecución:** ejecutable en cuanto exista el checkpoint de referencia (no depende del bloqueador de 6.4). **Advertencia heredada de `TrainingDiagnosis.md` §2.2:** no reutilizar la cifra "≈200×" de `TRAINING_REPORT_WEEK18.md` §5.2 en esta sección — esa cifra compara el tiempo de inferencia de la política contra el **presupuesto de tiempo fijo** de 60 s de GA, no contra el tiempo de convergencia real de GA ni contra un tiempo por época normalizado; es una comparación de conveniencia, no la medición controlada que exige esta sección.

**Tabla 6.5.1.** Tiempo de decisión (ms/época) y tiempo de pared total, por método e instancia, rango completo de tamaño (`ft06` a `la31`–`la35`, 5 réplicas cronometradas por instancia).

`[TABLA: instancia, num_operations, ms/decisión mediana (política DRL, mejor PDR), tiempo de pared total mediano (política DRL, GA, CP-SAT), hardware (CPU/GPU, núcleos)]`

**Tabla 6.5.2.** Razón CP-SAT/política DRL (ms/decisión), y verificación del umbral del 1% por instancia.

`[TABLA: instancia, razón CP-SAT/DRL, ¿razón < 100× (umbral 1%)? (sí/no), decisión H5]`

**Figura 6.5.1.** Razón de tiempo de decisión (CP-SAT / política DRL) en función del tamaño de instancia, escala logarítmica en el eje y.

`[FIGURA: eje x = num_operations, eje y (log) = razón CP-SAT/DRL, línea horizontal en 100× (umbral del 1%), verificando si la razón se ensancha con el tamaño per PreRegistration_FINAL.md §6]`

---

## Síntesis del capítulo: estado de H1–H5

**Tabla 6.6.1.** Estado de las cinco hipótesis pre-registradas al cierre de este capítulo — adaptada de la tabla de bloqueo de `PreRegistration_FINAL.md` §0, actualizada con el resultado real una vez ejecutado cada bloque.

| Hipótesis | Subsección | Estado a la fecha de esta plantilla | Decisión final |
|---|---|---|---|
| H1 (competitividad estática) | 6.2 | No ejecutado sobre el conjunto congelado ($N=43$); un punto de datos preliminar existe (`ft06`, Semana 18) | `[PENDIENTE]` |
| H2 (generalización en tamaño) | 6.3 | No ejecutado — mecanismo de transferencia probado, sweep pendiente | `[PENDIENTE]` |
| H3 (robustez) | 6.4 | **Bloqueado** — cableado `DynamicJSSPEnv` pendiente | `[BLOQUEADO]` |
| H4 (recompensa consciente de la perturbación) | 6.4 | **Bloqueado** — mismo bloqueador que H3, más el término $\Delta_{\text{inest}}$ sin implementar | `[BLOQUEADO]` |
| H5 (tiempo de decisión) | 6.5 | No ejecutado — ejecutable en cuanto exista el checkpoint de referencia | `[PENDIENTE]` |

## Acciones pendientes antes de completar este capítulo

1. Reconciliar `../docs/capitulos/Section5.1_5.2.md` §5.1.4 con el criterio de H1 congelado en `PreRegistration_FINAL.md` §2 (nota de conciliación, encabezado de este documento) — marcar la sección anterior como superada, no eliminarla, siguiendo la misma disciplina de trazabilidad que `../docs/capitulos/Chapter4_sections_4.1_4.3.md` ya aplica a su propia versión anterior.
2. Ejecutar el bloque de 43 instancias (6.2) y el barrido de generalización (6.3) sobre el checkpoint que resulte de `HPO_TechSpec.md` Fase 2 — no sobre el checkpoint preliminar de la Semana 18, para que las cifras publicadas en este capítulo reflejen la configuración final, no una configuración de diagnóstico.
3. Resolver el bloqueador de `DynamicJSSPEnv` antes de que 6.4 pueda dejar de estar marcada como bloqueada — acción de ingeniería, no de redacción.
4. Verificar si `../docs/planificacion/VandV_Protocol.md` ya cubre el caso analítico M/M/1 de E0 (6.1) o si debe añadirse como ejecución nueva.
5. Renderizar las figuras marcadas `[FIGURA: …]` a partir de los CSV/JSON ya producidos por `experiments/train_hetero.py` y `run_evaluation.py`, reutilizando el formato de seis paneles ya validado en `TRAINING_REPORT_WEEK18.md`, en vez de definir un formato nuevo para este capítulo.
