# Auditoría pre-sustentación — `../../archive/capitulos/Thesis_FullDraft_v1.md`

**Rol:** asesor doctoral embebido, auditoría de defensa completa. **Calibración de estándar:** el encargo pide evaluar contra estándares de una **tesis de maestría en ingeniería industrial** — un listón distinto (más exigente en ejecución honesta y rigor metodológico, menos exigente en originalidad teórica per se) del que aplicaría a una tesis doctoral; esta auditoría se calibra en consecuencia y lo señala explícitamente donde la distinción importa (§5).

**Relación con `../docs/auditorias/ConsistencyCheckReport.md`:** ese informe ya cubrió mecánica de ensamblaje — numeración, referencias cruzadas rotas, marcas `[VERIFICAR]`, deriva terminológica. Esta auditoría no repite esos hallazgos salvo cuando son prerrequisito directo de un juicio académico de fondo (p. ej., el Capítulo 2 ausente se retoma porque condiciona la evaluabilidad del rigor metodológico, no por repetir el hallazgo). El foco aquí es sustantivo: coherencia lógica, rigor metodológico ejecutado-vs-especificado, honestidad y viabilidad de las afirmaciones, y alineación con estándares académicos de maestría.

---

## 0. Veredicto ejecutivo

**La tesis es defendible tal como está, con trabajo de preparación específico, no con una reescritura de fondo.** Su mayor activo — el protocolo de pre-registro congelado antes de la ejecución, la disciplina estadística no paramétrica con tamaños de efecto, y el reporte honesto de cuatro rechazos de cinco hipótesis con causa raíz diagnosticada en cada caso — es precisamente el tipo de rigor metodológico que un comité de maestría en ingeniería debe premiar, no penalizar. El riesgo real de esta sustentación no es "los resultados son negativos"; es que **partes del andamiaje metodológico que sostienen esas conclusiones viven fuera del documento de tesis formal**, en archivos de trabajo internos no anexados (§1), y que dos comparaciones muy visibles en las propias tablas del Capítulo 6 no se discuten en el texto (§2.1, §2.2) — ambos son defectos de **presentación y completitud documental**, no de la validez de los hallazgos subyacentes, y ambos son corregibles en días, no en semanas.

---

## 1. Hallazgo crítico — la tesis depende de documentos de trabajo externos no anexados para sostener decisiones metodológicas centrales

**Severidad: crítica. Es el hallazgo de mayor impacto de esta auditoría porque afecta la *evaluabilidad* del rigor metodológico, no solo su presentación.**

Un lector (o un miembro del comité) que solo tenga acceso a `../../archive/capitulos/Thesis_FullDraft_v1.md` —el documento de tesis formal— no puede verificar independientemente varias decisiones que determinan directamente los resultados reportados en el Capítulo 6, porque su justificación vive en archivos internos de trabajo que el Capítulo 1 (§1.7) **no** lista entre los anexos previstos (que son: formulación MILP, hiperparámetros/arquitectura, tablas extendidas de resultados, diagrama PRISMA, documento de pre-registro, tablas de óptimos por instancia). En particular:

- **La justificación del diseño de entrenamiento de H4** (aleatorización de dominio "Opción B", la decisión que directamente causa el colapso de RPD nominal reportado en la Tabla 6.4.4 y discutido extensamente en el Capítulo 7, §7.4.3) se atribuye a `../../thesis/E7_E8_Plan.md` §2.2 — un documento de planificación de esprint, no un anexo de la tesis.
- **La exclusión de `mk06`** del brazo Brandimarte de H1 se atribuye a "Finding #2" de `../docs/logs/EXPERIMENT_LOG_WEEK24.md` — un registro de bitácora de trabajo, sin que el Capítulo 6 explique en sus propias palabras qué fue ese *mismatch* de datos.
- **La especificación técnica completa del arnés de evaluación** (`../docs/especificaciones/Baselines_TechSpec.md`), **del entorno dinámico** (`../docs/especificaciones/DynamicEnv_TechSpec.md`), **de la arquitectura GNN/PPO** (`../docs/especificaciones/GNN_PPO_Architecture_Spec.md`) y **del pipeline estadístico** (`../../thesis/StatsAnalysis_TechSpec.md`) se citan repetidamente en los Capítulos 4–6 como la fuente de verdad de detalles de implementación que el texto de la tesis no reproduce.
- **El propio protocolo de E7/E8** —el diseño experimental exacto que produjo los resultados de H4 y H5— nunca se especifica de forma prospectiva en el Capítulo 4 o 5 (Metodología/Diseño experimental); solo aparece narrado retrospectivamente en el Capítulo 6 (Resultados), citando `../../thesis/E7_E8_Plan.md`. Esto invierte el orden lógico esperado de una tesis: **el diseño debe especificarse en el capítulo de metodología antes de reportarse como resultado ejecutado**, no reconstruirse a partir de la narrativa de resultados.

**Por qué esto es crítico y no solo cosmético:** un comité de maestría evalúa explícitamente si "la metodología ejecutada coincide con la especificada" — la pregunta que este mismo encargo pide auditar. Tal como está el documento, esa comparación **no puede hacerse de forma autocontenida** para E7/E8 específicamente, porque el lado "especificado" de la comparación no está en ningún capítulo de metodología del documento formal.

**Arreglo recomendado (concreto, acotado):**
1. Añadir una Sección 5.4 ("Diseño de E7: recompensa consciente de la perturbación" — o similar) al Capítulo 5, que formalice prospectivamente la rejilla de $\kappa$, la justificación de la Opción B (aleatorización de dominio) frente a la alternativa de régimen fijo, y el protocolo de re-medición limpia de E8 — trasladando el contenido sustantivo de `../../thesis/E7_E8_Plan.md` a prosa de tesis, no solo citándolo.
2. Formalizar como Anexos explícitos (actualizando la lista de §1.7) al menos `../docs/especificaciones/Baselines_TechSpec.md`, `../docs/especificaciones/DynamicEnv_TechSpec.md`, `../docs/especificaciones/GNN_PPO_Architecture_Spec.md` y `../../thesis/StatsAnalysis_TechSpec.md` — o, si el volumen lo desaconseja, extraer de cada uno un resumen de una página que sí se incorpore al cuerpo del Capítulo 4/5, dejando el documento técnico completo como anexo de referencia.
3. Inline el contenido de "Finding #2" de `../docs/logs/EXPERIMENT_LOG_WEEK24.md` directamente en la Tabla 6.0/síntesis del Capítulo 6, como ya se hace correctamente con la anomalía de `la40` (§6.5.1) — ese es el estándar de auto-contención que el resto del Capítulo 6 ya cumple; la exclusión de `mk06` es la única excepción notable.

---

## 2. Hallazgos moderados

### 2.1 La superioridad del GA sobre RL, visible en la propia Tabla 6.2.1, nunca se discute narrativamente

**Sección afectada:** Capítulo 6, §6.2.1 (Tabla 6.2.1); Capítulo 7, §7.2.

La Tabla 6.2.1 incluye una columna `ga` con valores de RPD dramáticamente mejores que los de RL en toda la muestra (verificado de forma independiente en esta auditoría contra `e1_rpd_summary.md`: RPD medio del GA = 5.52 %, frente al 24.82 % de RL y al 18.96 % de la mejor PDR por instancia, $N=43$) — es decir, el GA no solo supera a RL, lo supera por un margen aún mayor que el de la propia PDR que el criterio pre-registrado de H1 usa como comparador. Ni el Capítulo 6 ni el Capítulo 7 comentan este hecho explícitamente: la discusión de H1 se centra, correctamente desde el punto de vista del criterio pre-registrado, en RL-vs-PDR, pero **un comité que lea la Tabla 6.2.1 con atención hará esta pregunta en la sustentación**, y no tener una respuesta preparada en el propio texto es un riesgo evitable.

**Por qué importa:** no es un defecto del pre-registro (H1 se definió correctamente contra PDR, el comparador de menor costo computacional, que es la comparación metodológicamente más informativa para "¿la política aprendió algo mejor que una heurística barata?"). Es un defecto de **discusión incompleta**: el dato está en la tabla, es sorprendente, y el texto no lo nombra.

**Arreglo recomendado:** añadir 3–4 líneas al final de §6.2.1 o al inicio de §7.2 reconociendo explícitamente que el GA, con presupuesto de tiempo equiparado a CP-SAT (60 s), domina tanto a RL como a la mejor PDR por un margen aún mayor, y ofrecer la lectura ya disponible en el propio Capítulo 7 (§7.2, punto 2) como explicación: RL opera bajo un presupuesto de entrenamiento acotado a un solo estudiante/GPU de consumo, mientras que GA explota directamente 60 s de búsqueda poblacional por instancia en el momento de la evaluación — dos paradigmas de asignación de cómputo no directamente comparables, pero la comparación numérica está ahí y merece una frase, no un silencio.

### 2.2 La Tabla 7.1 del Capítulo 7 introduce estadísticos (RPD medio de los brazos FJSP) que no aparecen en ninguna tabla del Capítulo 6

**Sección afectada:** Capítulo 7, §7.2.1 (Tabla 7.1); Capítulo 6, §6.2.

Las cifras de RPD medio de los brazos FJSP sintético (48.76 %) y Brandimarte (89.65 %) que la Tabla 7.1 usa para la comparación con L2D/Song et al. se calcularon como parte de la auditoría del Capítulo 7 directamente contra los CSV crudos (`e2_armA_merged.csv`, `e2_armB_mk_seed0.csv`) — son correctas, pero **no aparecen en ninguna tabla del Capítulo 6**, que solo reporta el estadístico de Wilcoxon (W, p, r) para esos brazos, no su RPD medio. Un lector que intente trazar la Tabla 7.1 hasta su origen dentro del propio documento de tesis no encontrará el número en el Capítulo 6 — debe confiar en la nota de procedencia de la Tabla 7.1 sin poder verla tabulada donde "debería" estar.

**Arreglo recomendado:** añadir una fila o una nota de RPD medio a la Tabla 6.2.2 (o una tabla 6.2.3 nueva) del Capítulo 6 para los brazos FJSP sintético y Brandimarte, replicando exactamente lo que ya se reporta para JSSP — cierra el trazado dentro del propio documento sin depender de la auditoría externa que produjo esas cifras.

### 2.3 La Tabla 6.3.3 excluye al GA de la clasificación conjunta de Friedman/Nemenyi sin justificación en el propio texto

**Sección afectada:** Capítulo 6, §6.3 (Tabla 6.3.3); cf. Capítulo 4, §4.6.4.

La Tabla 6.3.3 fija $k=10$ (RL + 9 PDR) para H2, sin GA. El Capítulo 4 (§4.6.4) sí anticipa que el GA "entra en esa clasificación conjunta, porque su presupuesto de tiempo de pared es explícitamente igualado al de CP-SAT... para poder compararlo con la política DRL bajo el mismo criterio de equidad." La Tabla 6.3.3 no dice si el GA se excluyó de H2 específicamente por diseño (p. ej., porque H2 evalúa generalización de tamaño y el GA, al no entrenarse, no tiene una noción de "tamaño de entrenamiento" que generalizar, lo cual sería una justificación razonable) o si es una omisión. Cualquiera de las dos respuestas es aceptable — lo que no es aceptable para una sustentación es que el lector tenga que adivinar cuál de las dos es.

**Arreglo recomendado:** una frase en §6.3 (o en la nota metodológica de la Tabla 6.3.3) que declare explícitamente el criterio de inclusión/exclusión del GA en esta prueba específica.

### 2.4 Parámetros de reproducibilidad declarados como "configurables" en el Capítulo 4 nunca se fijan a un valor concreto en el Capítulo 6

**Sección afectada:** Capítulo 4, §4.6.1 (regla TWK, factor de holgura $k$ para EDD/CR) y §4.6.2 (tamaño de población/generaciones del GA); Capítulo 6, en general.

El Capítulo 4 documenta correctamente que EDD/CR requieren sintetizar una fecha de entrega ($d_j=k\cdot\sum_h p_{j,h}$, factor $k$ configurable) y que el GA usa una población/número de generaciones calibrados empíricamente contra el presupuesto de tiempo de pared — ambos señalados explícitamente como "decisión de modelado no dictada por B6" y "se calibra empíricamente... no se fija a priori", respectivamente. El Capítulo 6, al reportar los resultados de EDD/CR y del GA, **no indica qué valor de $k$ ni qué configuración de población/generaciones se usó finalmente**. Esto no invalida los resultados, pero impide su reproducción exacta por un tercero — un estándar básico de rigor metodológico incluso a nivel de maestría.

**Arreglo recomendado:** una nota de una línea en §6.2 (p. ej., en el pie de la Tabla 6.2.1) fijando el valor de $k$ usado y la configuración final del GA (población, generaciones o criterio de parada temprana efectivamente alcanzado).

### 2.5 Modo de cálculo del $p$-valor de Wilcoxon (exacto vs. aproximación normal) para $N\geq20$ no se declara

**Sección afectada:** Capítulo 5, §5.3.2 (regla: aproximación normal si $N\gtrsim20$, exacta en caso contrario); Capítulo 6, §6.2.2 (H1-JSSP, $N=43$).

La Sección 5.3.2 fija explícitamente la regla de decisión sobre qué aproximación usar según $N$. Con $N=43$ para H1-JSSP, la propia regla del Capítulo 5 indicaría aproximación normal, no exacta — pero el Capítulo 6 no declara cuál se usó realmente para ninguna de las pruebas con $N\geq20$. Dado que el resultado de H1-JSSP es de todas formas contundente ($p=4.03\times10^{-5}$), es improbable que la elección cambie la decisión, pero la propia Sección 5.3.2 crea una expectativa de transparencia que el Capítulo 6 no satisface explícitamente.

**Arreglo recomendado:** una frase en la nota metodológica del pipeline estadístico (Capítulo 6, encabezado o Tabla 6.2.2) declarando el modo de cálculo efectivamente usado — previsiblemente ya registrado en `stats_manifest.json`, sería un traslado directo, no un cálculo nuevo.

### 2.6 Expectativa de validación aplicada/industrial — riesgo específico del programa de ingeniería industrial

**Sección afectada:** Capítulo 1, §1.5 (alcance); Capítulo 8, §8.5.

La tesis declara explícitamente, y de forma consistente en todo el documento, que toda validación es por simulación, sin datos ni contacto con una planta real (§1.5, reiterado en §8.5 y en las amenazas a la validez externa del Capítulo 7). Esto es metodológicamente honesto y suficiente para un alcance de maestría bien delimitado — **pero un comité de un programa de ingeniería industrial, a diferencia de uno de ciencias de la computación puras, frecuentemente espera al menos un gesto de conexión aplicada**: un caso de estudio con datos de un proceso real (aunque sea para calibrar los generadores de perturbación, no para entrenar), una entrevista con un responsable de programación de planta, o una discusión más extensa de la Sección 8.5 sobre la brecha *sim-to-real*. El Capítulo 8 (§8.5, §8.6 punto 11) ya reconoce esto como trabajo futuro, lo cual es correcto — el riesgo no es que falte, es que la defensa oral debe estar preparada para justificar activamente por qué el alcance puramente simulado es apropiado para esta maestría específica, no asumir que la Sección 1.5 "ya lo cubre" pasivamente.

**Arreglo recomendado:** ninguno al documento en sí (el alcance está correctamente delimitado y declarado) — se recomienda preparar una respuesta oral de 60–90 segundos que anticipe esta pregunta, apoyada directamente en la Sección 1.5 y en el punto 11 del Capítulo 8.

---

## 3. Hallazgos menores

### 3.1 Título de portada provisional

El bloque de portada de `../../archive/capitulos/Thesis_FullDraft_v1.md` usa un título de trabajo construido durante la compilación de este borrador integrado ("Política de programación de taller de trabajo flexible dinámico mediante...") — debe confirmarse contra el título oficialmente registrado ante UNIMINUTO antes de cualquier entrega formal; no se encontró un título institucional alternativo en el corpus de trabajo consultado.

### 3.2 Tono motivacional del §1.1 podría beneficiarse de una frase de calibración de expectativas

El §1.1 (Contexto y motivación) mantiene, sin cambios, el tono asertivo propio de una introducción escrita *antes* de conocer los resultados ("esta tesis se posiciona precisamente en esa intersección" — motivación doble académica e industrial). Esto es normal y aceptable en la estructura estándar de una tesis (la introducción plantea la pregunta, no anticipa la respuesta), y el Resumen/Abstract de este mismo borrador ya expone los resultados negativos con total honestidad antes de que el lector llegue al Capítulo 1 — por lo que el riesgo de una impresión de "cebo y cambio" (*bait-and-switch*) es bajo, no nulo. Una frase opcional al cierre de §1.1 (p. ej., anticipando que "los resultados, presentados con transparencia metodológica completa en los Capítulos 6–8, no confirman todas las hipótesis planteadas") reforzaría la coherencia de tono de punta a punta sin restar fuerza a la motivación.

### 3.3 Nomenclatura `MOR`/`mopnr` (ya documentada en `../docs/auditorias/ConsistencyCheckReport.md` §5.3)

Se reitera aquí solo para que el plan de acción priorizado de este documento (§4) sea autocontenido; ver el informe de consistencia para el detalle completo.

### 3.4 Confirmar explícitamente que las enmiendas al pre-registro no tocaron reglas de decisión

Las enmiendas fechadas de `../../thesis/PreRegistration.md` (Enmienda 1, sobre la estructura de instancias; Enmienda 2, sobre el colapso de la distinción GPU/CPU) se citan en varios puntos del documento como parte del historial metodológico, y por lo que se pudo verificar en esta auditoría **ninguna modifica una regla de decisión ni un criterio de aceptación de H1–H5** — son aclaraciones de alcance y de medición, exactamente el tipo de enmienda que un protocolo de pre-registro bien administrado permite sin comprometer su validez frente al riesgo de HARKing. Se recomienda incluir una frase explícita en algún punto del Capítulo 5 o 6 afirmando esto directamente ("ninguna enmienda alteró una regla de decisión ya congelada"), en vez de dejar que el lector deba confirmarlo por inferencia — es una de las defensas más fuertes de esta tesis y merece decirse en voz alta, no solo ser cierta.

### 3.5 Lista de anexos del §1.7 desactualizada frente a lo que el documento realmente necesita (ver también Hallazgo Crítico §1)

Actualizar la lista de anexos de la Sección 1.7 una vez se decida el tratamiento de los documentos técnicos complementarios (§1 de esta auditoría) es una acción menor de redacción, pero con una dependencia directa del hallazgo crítico — se lista aquí como recordatorio de cierre de ciclo, no como un hallazgo nuevo independiente.

---

## 4. Fortalezas a destacar activamente en la defensa

Una auditoría honesta debe nombrar las fortalezas con el mismo rigor que las debilidades — estas son, en opinión de este auditor, los tres argumentos más sólidos que el documento ya sostiene y que deben liderar la narrativa oral de la sustentación, no quedar implícitos:

1. **El protocolo de pre-registro (`../../thesis/PreRegistration.md`, congelado 2026-07-29) con reglas de decisión exactas para las cinco hipótesis es, dentro del corpus de literatura revisado en el propio Capítulo 3, una práctica que ninguno de los trabajos comparables reporta** (Capítulo 3, Tabla 3.2, fila "Reproducibilidad y protocolo estadístico débiles"). Esto no es un detalle procedimental — es la razón por la que los cuatro rechazos de hipótesis pueden presentarse con total autoridad, sin sospecha de HARKing.
2. **Cada rechazo tiene una causa raíz diagnosticada, no solo un $p$-valor reportado** (colapso nominal por aleatorización de dominio en H4; desajuste de legalidad *completion-gated* en H3; artefacto de construcción de la métrica en H5, con su propia anomalía de `la40` detectada, investigada y neutralizada mediante análisis de sensibilidad dentro del propio Capítulo 6). Este nivel de diagnóstico es infrecuente incluso en tesis con resultados positivos, y es, específicamente, lo que un comité de maestría en ingeniería debe reconocer como la competencia técnica real que se está evaluando — no "¿ganó el modelo?", sino "¿entendió el candidato por qué pasó lo que pasó?".
3. **La disciplina de marcado `[VERIFICAR]` sostenida desde el Capítulo 3 hasta el Capítulo 7** (38 marcas activas, ninguna oculta) es exactamente el tipo de honestidad metodológica que distingue una tesis defendible de una que colapsa bajo la primera pregunta incómoda del comité.

---

## 5. Nota sobre el estándar aplicado (maestría, no doctorado)

Calibrado explícitamente contra un estándar de maestría en ingeniería industrial, este auditor no penaliza — y recomienda que el comité tampoco penalice — los siguientes puntos que sí serían objetables en una tesis doctoral: una sola semilla de entrenamiento por condición de $\kappa$ (Capítulo 7, §7.7.1), la ausencia de una ablación arquitectónica GIN-vs-GAT-vs-heterogénea completa (Capítulo 8, Tabla 8.1, objetivo 3), o el no haber reimplementado L2D/Song et al. como líneas base propias (Capítulo 8, Tabla 8.1, objetivo 5). Estas son limitaciones de recursos explícitamente declaradas (Sección 1.5, "único estudiante, una GPU de consumo, nueve meses") y correctamente registradas como trabajo futuro (Capítulo 8, §8.6) — el criterio de una maestría es que la ejecución sea honesta, rigurosa dentro del alcance declarado, y que las limitaciones se nombren, no que el alcance sea exhaustivo. Bajo ese criterio, el documento cumple.

---

## 6. Plan de acción priorizado antes de la sustentación

| # | Prioridad | Acción | Esfuerzo estimado |
|---|---|---|---|
| 1 | Crítica | Redactar el Capítulo 2 (bloqueante, ya identificado en `../docs/auditorias/ConsistencyCheckReport.md` §1) | Alto (encargo propio) |
| 2 | Crítica | Añadir §5.4 (diseño de E7/E8) al Capítulo 5 y formalizar como anexos los documentos técnicos complementarios citados como fuente de verdad (§1 de esta auditoría) | Medio |
| 3 | Moderada | Añadir discusión explícita de GA-vs-RL en §6.2.1/§7.2 (§2.1) | Bajo |
| 4 | Moderada | Añadir RPD medio de los brazos FJSP a una tabla del Capítulo 6 (§2.2) | Bajo |
| 5 | Moderada | Justificar en texto la exclusión del GA de la Tabla 6.3.3 (§2.3) | Bajo |
| 6 | Moderada | Fijar y reportar el valor de $k$ (EDD/CR) y la configuración final del GA (§2.4) | Bajo |
| 7 | Moderada | Declarar el modo de cálculo del $p$-valor de Wilcoxon para $N\geq20$ (§2.5) | Bajo |
| 8 | Moderada | Preparar respuesta oral sobre el alcance puramente simulado (§2.6) | Bajo (preparación, no redacción) |
| 9 | Menor | Confirmar título oficial de portada (§3.1) | Trivial |
| 10 | Menor | Frase de calibración de expectativas en §1.1 (§3.2) | Trivial |
| 11 | Menor | Afirmación explícita de que las enmiendas no tocaron reglas de decisión (§3.4) | Trivial |
| 12 | Menor | Actualizar lista de anexos de §1.7 (§3.5, depende de la acción #2) | Trivial |

Ninguna de estas doce acciones requiere reabrir resultados experimentales ni recalcular una sola prueba estadística — todas son de redacción, trazabilidad documental o preparación oral. **El documento está, en el fondo empírico, listo para defenderse; lo que falta es que su propia superficie documental sostenga, sin fisuras, el rigor que su ejecución ya tiene.**
