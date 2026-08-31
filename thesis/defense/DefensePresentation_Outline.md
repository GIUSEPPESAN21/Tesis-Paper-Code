# Esquema de presentación de sustentación (20 diapositivas)

**Fuente:** `../Thesis.md`. **Fuente de las notas del orador:** `docs/auditorias/PreDefenseAudit.md` (hallazgo crítico §1 y hallazgos moderados 2.1–2.6, ya resueltos en la versión final vía la Sección 5.4 y las tablas/notas añadidas en v2 — las notas de esta diapositiva indican explícitamente cuándo un hallazgo ya fue corregido en el documento y cuándo requiere solo una respuesta oral preparada).

**Duración sugerida:** 20 diapositivas × ~1.5–2 min = 30–40 min de exposición, dejando 20–30 min para preguntas del comité en una sustentación típica de 60–75 min. Ajustar recortando las diapositivas 6–8 (arquitectura/metodología) si el tiempo asignado es menor — el comité de maestría ya tiene el documento completo; la exposición oral debe priorizar resultados y su interpretación, no relitigar la metodología en detalle.

---

## Diapositiva 1 — Portada

**Contenido:**
- Título completo de la tesis (confirmar contra el título oficial registrado en UNIMINUTO antes de finalizar — ver `docs/auditorias/PreDefenseAudit.md` §3.1, hallazgo menor no resuelto de forma independiente en esta pasada).
- Joseph Sanchez — UNIMINUTO — 2026.
- Programa, director/asesor, jurado (completar).

**Notas del orador (ES):** Abrir con el nombre completo, el programa y agradecer al comité — protocolo estándar, sin contenido técnico. Aprovechar los primeros 15 segundos para fijar el tono: esta es una tesis que pre-registró sus hipótesis y reporta con honestidad los resultados, sean o no los esperados. Ese encuadre en la primera diapositiva previene que el comité interprete los rechazos de hipótesis (diapositivas 9–13) como una tesis fallida en lugar de una tesis rigurosa con hallazgos negativos bien diagnosticados.

---

## Diapositiva 2 — Motivación: por qué importa la programación robusta y generalizable

**Contenido:**
- Industria 4.0 → talleres intensivos en datos, pero programación de producción sigue siendo difícil de automatizar de forma robusta.
- FJSP: NP-difícil en sentido fuerte (≥3 máquinas); métodos exactos no escalan, PDR/metaheurísticas no se adaptan nativamente a eventos dinámicos.
- DRL+GNN (L2D, 2020): política entrenada una vez, aplicada de forma amortizada — pero casi toda la literatura la evalúa en condiciones estáticas y deterministas, o con un único tipo de perturbación a la vez.

**Notas del orador (ES):** Explicar la motivación en 60–90 segundos, sin tecnicismos todavía. Pregunta anticipada: *"¿Por qué no simplemente usar CP-SAT o un algoritmo genético, que ya funcionan bien?"* — Respuesta preparada: CP-SAT y el GA son excelentes cuando hay tiempo de cómputo disponible en el momento de la decisión (segundos a minutos); el interés de una política aprendida es la decisión amortizada en milisegundos, útil quando el taller necesita reaccionar de inmediato a un evento — la tesis evalúa precisamente si esa promesa se sostiene, y la respuesta honesta (adelanto: parcialmente, y con costos medibles) es el contenido de las diapositivas 9–13.

---

## Diapositiva 3 — Planteamiento del problema y preguntas de investigación

**Contenido:**
- FJSP dinámico y estocástico: tres fuentes de incertidumbre — averías de máquina, tiempos de proceso estocásticos, llegadas dinámicas de trabajos.
- **PP:** ¿Puede una política GNN+PPO, entrenada en instancias pequeñas y deterministas, generalizar a instancias más grandes **y**, simultáneamente, mantenerse robusta ante múltiples tipos de perturbación?
- SP1 (¿ayuda una recompensa consciente de la perturbación?), SP2 (¿cuál es la brecha real frente a CP-SAT/GA?), SP3 (¿son las diferencias estadísticamente significativas?).

**Notas del orador (ES):** Enfatizar la palabra "simultáneamente" — es el eje exacto de la contribución (Capítulo 3, Sección 3.6: ningún trabajo revisado combina generalización de tamaño y robustez multi-perturbación bajo un protocolo estadístico riguroso). Pregunta anticipada: *"¿Probaron alguna vez los tres tipos de perturbación combinados en el mismo episodio?"* — Respuesta honesta preparada: no; el protocolo experimental (E4–E6) evalúa cada eje de perturbación **individualmente** (así lo fija el propio criterio pre-registrado de H3), variando un eje a la vez con los otros dos en su nivel nominal. "Multi-perturbación" describe que se modelan y evalúan **múltiples tipos distintos** de perturbación bajo el mismo entorno y el mismo mecanismo de recompensa, no que se estresen los tres ejes a la vez dentro de un mismo episodio — esa combinación explícita queda como extensión natural no ejecutada en este alcance. Es mejor decir esto de forma proactiva y clara que dejar que el comité lo descubra preguntando.

---

## Diapositiva 4 — La brecha en la literatura (revisión sistemática, Capítulo 3)

**Contenido:**
- Protocolo PRISMA 2015–2026, 51 referencias verificadas.
- Trabajos más cercanos cubren un solo eje: Smit et al. [AAAI 2025] generaliza en tamaño bajo **un** tipo de incertidumbre; Lassoued et al. [2026] combina dos tipos de perturbación pero **sin** generalización de tamaño ni GNN.
- Ningún trabajo identificado combina ambos ejes de forma simultánea y cuantitativa → brecha confirmada (Tabla 3.2).

**Notas del orador (ES):** Esta diapositiva es la base de la defensa de originalidad — mantenerla breve pero firme. Pregunta anticipada: *"¿No es esto simplemente aplicar técnicas ya conocidas?"* — Respuesta preparada: la arquitectura (GIN relacional, PPO enmascarado) reutiliza componentes conocidos de forma deliberada — la contribución no es un componente nuevo, es la **caracterización conjunta y estadísticamente rigurosa** de una combinación de propiedades que la literatura solo había abordado por separado, con un protocolo de pre-registro que ningún trabajo comparable de la Tabla 3.1 reporta. Si el comité presiona sobre esto, remitir directamente a la Tabla 3.2 (análisis de brechas) del Capítulo 3.

---

## Diapositiva 5 — Marco teórico: por qué una GNN debería generalizar en tamaño

**Contenido:**
- Invariancia a permutaciones + número de parámetros independiente de *n*/*m* + localidad del paso de mensajes (Xu et al., 2019, expresividad tipo Weisfeiler-Lehman).
- Paradigmas de *scheduling* dinámico: predictivo-reactivo, robusto, estable — la política de esta tesis opera en el paradigma predictivo-reactivo; el índice de robustez (RI) y la penalización de inestabilidad ($\Delta_{\text{inest}}$) operacionalizan robustez y estabilidad respectivamente.
- **Nota honesta:** la generalización de tamaño es una propiedad **arquitectónica** (transfiere el *rango relativo*), no una garantía de competitividad absoluta — esta distinción es la clave para entender por qué H1 se rechaza y H2 se acepta sin contradicción (diapositivas 10–11).

**Notas del orador (ES):** Esta diapositiva previene la pregunta más peligrosa de toda la sustentación — "si H1 se rechaza, ¿cómo puede H2 aceptarse, no es contradictorio?" — explicándola *antes* de mostrar los resultados, no después. Insistir en que la teoría predice exactamente esta separación: la arquitectura transfiere lo que ya tenía, ni más ni menos. Nota del Capítulo 2: el desarrollo formal completo de este marco teórico está previsto para el Capítulo 2 de la tesis, que a la fecha de esta compilación permanece sin redactar en prosa — ver diapositiva 16 y `QA_Prep.md`, pregunta sobre el Capítulo 2, para la respuesta preparada si el comité lo señala.

---

## Diapositiva 6 — Arquitectura de la solución: GIN relacional + PPO enmascarado

**Contenido:**
- Grafo heterogéneo operación–máquina; tres relaciones tipadas (predecesor, sucesor, máquina compartida) con transformación lineal separada por relación — punto intermedio entre el grafo homogéneo de L2D y el heterogéneo de Song et al.
- Cabeza actor-crítico por nodo → política agnóstica al tamaño de instancia por construcción.
- Enmascaramiento de acciones dentro de la distribución categórica (no como penalización de recompensa) vía `MaskablePPO`.

**Notas del orador (ES):** Explicar en un minuto, apoyándose en un diagrama si es posible (Figura 4.1, descrita en prosa en el documento, pendiente de renderizado formal — confirmar si ya se generó antes de la sustentación). Pregunta anticipada: *"¿Por qué no una arquitectura totalmente heterogénea, como Song et al.?"* — Respuesta preparada: fue una decisión deliberada de costo-beneficio (menor complejidad paramétrica, sin dependencias frágiles de `torch-geometric`), reforzada retroactivamente por Hoss et al. (2026), quienes muestran que la heterogeneidad plena no es condición necesaria para escalar. Reconocer honestamente que **no se ejecutó una ablación que aísle el efecto de esta elección específica** (GIN relacional vs. homogénea vs. heterogénea) — es una limitación declarada, no oculta (Capítulo 7, §7.2; Capítulo 8, objetivo 3, "parcialmente cumplido"), y forma parte del trabajo futuro de mediano plazo.

---

## Diapositiva 7 — Entorno dinámico y modelos de perturbación

**Contenido:**
- Simulación de eventos discretos sobre contrato Gymnasium; tres generadores independientes (averías TTF~Exp/TTR~Lognormal; tiempos de proceso ~Lognormal de media 1; llegadas ~Poisson calibrado a utilización objetivo ρ).
- Validación estadística de los tres generadores contra sus parámetros teóricos: error <2.6 % en todos los casos, prueba KS no rechaza la hipótesis exponencial de TTF ($p=0.929$).
- Política de preempción *preempt-resume*, justificada por realismo del dominio y consistencia con la literatura más cercana (Lassoued et al.; Zhang et al.).

**Notas del orador (ES):** Diapositiva técnica breve — el objetivo es mostrar que el simulador está verificado (los generadores reproducen exactamente sus propios parámetros objetivo), no que está validado contra un proceso industrial real (esa es una distinción de Sargent, 2011, que conviene nombrar si el comité pregunta). Pregunta anticipada: *"¿Estos parámetros de perturbación corresponden a algún taller real?"* — Respuesta honesta preparada: no; son modelos estilizados calibrados a niveles de severidad operacionalmente interpretables (p. ej., disponibilidad de máquina de 95/90/80 %), consistentes con los rangos usados en la literatura más cercana revisada, pero sin contraste contra datos de planta — la tesis lo declara explícitamente como límite de validez externa (Capítulo 7, §7.7.2) y como dirección de trabajo futuro (Capítulo 8, punto 11).

---

## Diapositiva 8 — Diseño experimental: protocolo E0–E8 y pre-registro

**Contenido:**
- **Pre-registro congelado** (`../PreRegistration.md`, 2026-07-29T05:22:26Z UTC): reglas de decisión de H1–H5 fijadas **antes** de generar cualquier resultado — blindaje explícito contra HARKing.
- Nueve experimentos (E0 validación → E8 latencia limpia); líneas base: 9 PDR, GA (presupuesto de 60 s igualado a CP-SAT), CP-SAT (60 s).
- Protocolo estadístico no paramétrico: Wilcoxon (método exacto, verificado incluso para $N=43$), Friedman + Nemenyi con diagramas de diferencia crítica, tamaños de efecto reportados en toda comparación.

**Notas del orador (ES):** Esta es la diapositiva que debe presentarse con más confianza, no con más detalle — es la fortaleza metodológica más fuerte de la tesis (`docs/auditorias/PreDefenseAudit.md` §4, punto 1: "ninguno de los trabajos comparables del Capítulo 3 reporta un protocolo de pre-registro similar"). Pregunta anticipada: *"¿Cómo garantizan que el pre-registro no se ajustó después de ver resultados preliminares?"* — Respuesta preparada: el archivo de pre-registro tiene timestamp UTC verificable, y las únicas dos enmiendas posteriores (Enmienda 1: estructura de instancias; Enmienda 2: colapso de la distinción GPU/CPU) están fechadas, documentadas explícitamente como adendas, y **ninguna modificó una regla de decisión de H1–H5** — se puede afirmar esto con seguridad total en la sustentación (`docs/auditorias/PreDefenseAudit.md` §3.4).

---

## Diapositiva 9 — Síntesis de resultados: el panorama honesto

**Contenido:**
- **Cuatro de cinco hipótesis se rechazan; una se acepta (H2).**
- Tabla resumen (Tabla 6.0): H1 rechazada (JSSP, FJSP sintético, FJSP Brandimarte), H2 aceptada, H3 rechazada (9/9 regímenes), H4 rechazada (κ=0.5 y κ=1.0), H5 rechazada bajo umbral estricto.
- Mensaje central: cada rechazo tiene una **causa raíz diagnosticada**, no solo un $p$-valor — ese es el estándar que esta tesis se impuso a sí misma.

**Notas del orador (ES):** Decir el resultado global **antes** de que el comité lo infiera de las diapositivas siguientes — no hay nada que ganar ocultándolo, y decirlo primero con seguridad transmite control sobre los propios hallazgos. Frase sugerida de apertura de esta diapositiva: *"Antes de entrar al detalle, quiero ser directo: de las cinco hipótesis pre-registradas, cuatro se rechazan. Lo que voy a mostrar en las siguientes diapositivas no es que esto sea un resultado negativo sin más, sino que cada rechazo viene acompañado de un diagnóstico causal específico — y eso, dentro de la literatura revisada en el Capítulo 3, es en sí mismo un aporte poco común."* Pregunta anticipada, la más dura de toda la sustentación: *"Si cuatro de cinco hipótesis fallaron, ¿por qué debería aprobarse esta tesis?"* — ver `QA_Prep.md`, pregunta 1, para la respuesta completa preparada.

---

## Diapositiva 10 — H1: competitividad estática (rechazada, la más decisiva estadísticamente)

**Contenido:**
- JSSP: RL 24.82 % RPD vs. 18.96 % mejor PDR por instancia ($N=43$, $p=4.03\times10^{-5}$, $r=-0.626$).
- FJSP sintético: $W=0$ (RL pierde en las 43 instancias sin excepción), $p=1.12\times10^{-8}$.
- FJSP Brandimarte: $N=7$ (`mk06` excluido por desajuste de datos de 15 vs. 10 máquinas en el espejo del benchmark), $p=0.0156$ — el $p$-valor mínimo posible con ese tamaño muestral.
- El GA (5.52 % RPD medio) supera tanto a RL como a la mejor PDR por un margen aún mayor.

**Notas del orador (ES):** Nombrar la superioridad del GA de forma proactiva, sin esperar la pregunta — es exactamente el Hallazgo moderado 2.1 de `docs/auditorias/PreDefenseAudit.md`, ya incorporado al texto final (Capítulo 6, §6.2.1; Capítulo 7, §7.2), y mencionarlo primero transmite control del dato en lugar de omisión. Explicación preparada: RL opera bajo presupuesto de entrenamiento acotado a un solo estudiante con una GPU de consumo, mientras que el GA explota 60 s de búsqueda poblacional **en el momento de la evaluación** de cada instancia — dos paradigmas de asignación de cómputo no directamente comparables, pero la comparación numérica es real y no debe minimizarse. Pregunta anticipada: *"¿Por qué no compararon simplemente contra L2D o Song et al. directamente, entrenados por ustedes?"* — Respuesta preparada: es una limitación reconocida (Capítulo 8, objetivo 5, "no cumplido como se planteó") — la comparación de la Tabla 7.1 usa las cifras publicadas de ambos trabajos, no una reimplementación propia bajo el mismo protocolo; queda como primera acción de trabajo futuro de corto plazo (Capítulo 8, §8.6.1).

---

## Diapositiva 11 — H2: generalización en tamaño (aceptada — el resultado más defendible de la tesis)

**Contenido:**
- Cuatro *tiers* de tamaño (1.5×–3.0× el tamaño de entrenamiento): en los cuatro, el rango de RL **no es significativamente peor** que el de la mejor PDR (Friedman + Nemenyi, CD=6.058).
- Matiz honesto: el RPD absoluto de RL (23.9–32.9 %) permanece por encima de `fifo` (12.7–26.8 %) en los cuatro *tiers* — la aceptación es sobre **rango estadístico**, no paridad de desempeño.
- Coincide cualitativamente con L2D (2020): la arquitectura transfiere su comportamiento relativo, no su nivel absoluto de competitividad.

**Notas del orador (ES):** Presentar este resultado con la misma seriedad estadística que los rechazos — no inflarlo como una "victoria" que compense los rechazos, porque el propio matiz (rango vs. paridad) es parte de lo que hace a esta tesis rigurosa. Pregunta anticipada, una de las dos citadas explícitamente en el encargo: *"¿Cómo saben que esto generaliza más allá de sus benchmarks?"* — ver `QA_Prep.md`, pregunta 2, para la respuesta completa; en síntesis oral breve: la generalización demostrada es **dentro** de la familia Lawrence (`la21`–`la40`), no **entre** familias de instancias distintas (Taillard, Hurink nunca se incorporaron al conjunto de prueba) — es una limitación de validez externa declarada explícitamente (Capítulo 7, §7.7.2), no una afirmación no examinada.

---

## Diapositiva 12 — H3/H4: robustez y recompensa consciente de la perturbación (rechazadas, con causa raíz)

**Contenido:**
- H3: RI de RL peor que la mejor PDR en los **9 de 9** regímenes evaluados ($p<1.7\times10^{-11}$ en todos) — la política nunca fue entrenada contra la semántica de legalidad *completion-gated* del entorno dinámico.
- H4: κ=0.5 y κ=1.0 rechazadas — ambas colapsan la competitividad nominal (+106.0 pp y +46.6 pp de RPD, respectivamente).
- **Causa raíz diagnosticada:** entrenamiento con aleatorización de dominio sobre los 9 regímenes, **sin exponer nunca a la política a un episodio sin perturbación** — no es un error de implementación, es una consecuencia de una decisión de diseño (Opción B) tomada y documentada *antes* del entrenamiento.

**Notas del orador (ES):** Esta es la diapositiva donde el diagnóstico causal debe brillar — no defenderse del rechazo, sino mostrar que se entendió exactamente por qué ocurrió y qué lo corregiría (mezclar un régimen "sin perturbación" en el muestreo de dominio, ya propuesto como trabajo futuro concreto). Pregunta anticipada: *"¿Por qué eligieron la aleatorización de dominio (Opción B) si sabían que arriesgaba el desempeño nominal?"* — Respuesta preparada: la Opción A (régimen fijo único durante el entrenamiento) habría sido más barata mueber pero no habría sostenido la propia regla de decisión de H4, que exige una política consciente de la perturbación *en general*, no de un solo régimen — la decisión se documentó y fechó *antes* de entrenar (Sección 5.4.2), siguiendo la misma disciplina anti-HARKing del resto del protocolo; el costo que produjo se reporta con la misma honestidad con la que se reporta cualquier otro resultado.

---

## Diapositiva 13 — H5: tiempo de decisión (rechazada bajo el umbral estricto, ventaja cualitativa real)

**Contenido:**
- Criterio pre-registrado: mediana(ms/decisión RL) < 1 % de CP-SAT, **para las 43 instancias**. Solo 1/43 pasa (`la40`) → RECHAZADA.
- Anomalía detectada y neutralizada: el tiempo de CP-SAT en `la40` (7216.67 s) es ~120× el de una instancia estructuralmente idéntica (`la38`, 60.06 s) — probable artefacto de medición, no una propiedad genuina del solver. Corrección de sensibilidad: el veredicto se vuelve **más decisivo** (0/43), no menos.
- Lectura cualitativa correcta: para instancias grandes y difíciles, RL es 1–2 órdenes de magnitud más rápido por decisión — un margen real, aunque no alcance el umbral extremo del 1 %.

**Notas del orador (ES):** Presentar la anomalía de `la40` de forma proactiva y con la investigación ya hecha — es exactamente el tipo de hallazgo que, bien manejado, genera confianza del comité en el rigor del candidato en lugar de sospecha. Pregunta anticipada: *"¿No debilita esta anomalía la credibilidad de todo el Capítulo 6?"* — Respuesta preparada: al contrario — la anomalía se detectó, se investigó (comparación contra instancias estructuralmente idénticas, revisión de logs de ejecución) y se corrigió mediante un análisis de sensibilidad documentado dentro del propio capítulo, **sin cambiar el veredicto final**; es evidencia de que el protocolo de verificación funciona, no de que los datos sean poco confiables.

---

## Diapositiva 14 — Esta tesis frente a L2D y Song et al. (comparación cuantitativa)

**Contenido:**
- Tabla comparativa (Tabla 7.1): en cada fila donde una comparación directa es posible, la cifra de esta tesis es numéricamente peor que la de L2D (brecha estática 24.82 % vs. 13.6–26.0 %; % instancias ganadas 18.6 % vs. 87.5–92 %).
- Comparador de esta tesis es intrínsecamente más exigente: mínimo de **9** PDR por instancia, no 3 reglas fijas — `fifo` (la PDR dominante en el 93 % de las instancias de esta tesis) ni siquiera está entre las reglas de comparación de L2D.
- Song et al. (FJSP heterogéneo): comparación cualitativa únicamente — texto completo bloqueado (IEEE Xplore), acción pendiente antes de la sustentación.

**Notas del orador (ES):** No suavizar el hecho de que las cifras son peores que L2D — explicar por qué la comparación no es de "igual a igual" (comparador más estricto, presupuesto de entrenamiento no verificable de L2D) sin que suene a excusa. Pregunta anticipada: *"Si L2D es mejor en todo, ¿qué aporta entonces esta tesis frente a L2D?"* — Respuesta preparada: la aportación no es superar a L2D en competitividad estática — es la caracterización conjunta de generalización y robustez multi-perturbación bajo un protocolo de pre-registro y significancia estadística que L2D (2020) no reporta; son contribuciones en ejes distintos, no una carrera por el mismo número.

---

## Diapositiva 15 — El hallazgo teórico central: generalización ≠ robustez

**Contenido:**
- Patrón H2-aceptada / H3-H4-rechazadas: la generalización de tamaño se transfiere **arquitectónicamente**; la robustez ante perturbación **no emerge como subproducto** de esa misma arquitectura ni se induce con éxito mediante el término de recompensa bajo el régimen de entrenamiento probado.
- Primera caracterización cuantitativa, en la literatura revisada, de esta tensión — contraste directo con Lassoued et al. (2026), que logra superar a las PDR bajo perturbación **cuando la robustez es el único objetivo**, sin exigir generalización de tamaño simultánea.
- Valor del hallazgo: diagnóstico, no solo el resultado — información que ningún trabajo que aísla un solo eje podía producir.

**Notas del orador (ES):** Esta es la diapositiva de síntesis intelectual de la tesis — el lugar para mostrar que el candidato entiende el significado del patrón de resultados más allá de reportar números. Practicar decir esto sin leer la diapositiva. Pregunta anticipada: *"¿Cómo saben que esto no es solo un problema de esta arquitectura específica, y no una limitación general del enfoque GNN+PPO?"* — Respuesta preparada: no se puede afirmar con los datos disponibles que sea una limitación general — el Capítulo 7 (§7.2, §7.6) es explícito en que la ablación arquitectónica necesaria para aislar "objetivo conjunto costoso" de "esta elección arquitectónica específica compite mal" no se ejecutó en este alcance; se nombra como la pregunta abierta más importante que deja la tesis, no se disfraza de conclusión general.

---

## Diapositiva 16 — Amenazas a la validez y limitaciones reconocidas

**Contenido:**
- **Interna:** una sola semilla de entrenamiento por condición de κ; asimetría estructural κ=0 (política reutilizada) vs. κ>0 (entrenada nativamente); anomalía de `la40` (investigada, no afecta el veredicto).
- **Externa:** generalización demostrada dentro de la familia Lawrence, no entre familias distintas; validación exclusivamente por simulación, sin datos de planta real.
- **Constructo:** métrica "ms/decisión" no es una unidad natural para CP-SAT (no delibera secuencialmente); convención de línea base compartida para RI favorece mecánicamente a `cr` como "mejor PDR" en algunos regímenes.
- **Conclusión:** sin corrección formal por comparaciones múltiples entre los 9 regímenes de H3/H4 (mitigado por la regla de decisión conjuntiva, más estricta que cualquier corrección de Bonferroni razonable).

**Notas del orador (ES):** No leer las cuatro categorías como una lista de disculpas — presentarlas como evidencia de que el candidato conoce los límites exactos de sus propias conclusiones, que es precisamente lo que un comité de maestría busca. Recordatorio explícito y honesto sobre el estado del Capítulo 2: el desarrollo teórico formal completo (marco MDP, teoría de generalización GNN, notación α|β|γ) está especificado y anticipado desde el Capítulo 1 y el Capítulo 4, pero el Capítulo 2 de la tesis permanece sin redactar en prosa a la fecha de esta compilación — es la única acción de esfuerzo "alto" pendiente identificada en `docs/auditorias/PreDefenseAudit.md`, y debe nombrarse aquí de forma directa si no se ha resuelto antes de la fecha de sustentación (ver `QA_Prep.md` para la respuesta completa preparada). Es preferible que el candidato lo mencione primero a que el comité lo descubra en la mitad de una pregunta sobre otro tema.

---

## Diapositiva 17 — Cumplimiento de objetivos y contribuciones

**Contenido:**
- 5 de 7 objetivos específicos cumplidos en su totalidad; 1 parcialmente (objetivo 3 — ablación GIN/GAT pendiente); 1 no cumplido como se planteó (objetivo 5 — L2D/Song et al. no reimplementados, solo comparados contra cifras publicadas).
- 3 de 5 contribuciones esperadas logradas plenamente; 1 lograda con carácter distinto al anticipado (caracterización de la tensión, no una combinación exitosa); 1 con componente administrativo pendiente (publicación del código como código abierto).
- Mensaje: ninguna brecha invalida los veredictos de H1–H5 — son honestamente declaradas, no descubiertas por el comité.

**Notas del orador (ES):** Mostrar la Tabla 8.1/8.2 tal como está en el documento — no ocultar los "parcialmente cumplido"/"no cumplido". Pregunta anticipada: *"¿Por qué prometieron en el Capítulo 1 algo que no cumplieron completamente?"* — Respuesta preparada: los objetivos originales se fijaron al inicio del proyecto, antes de conocer las restricciones reales de tiempo/cómputo de un solo estudiante con una GPU de consumo en nueve meses; ajustar el alcance ejecutado frente al planeado es normal en un proyecto de investigación, y lo que se evalúa no es si el plan original se ejecutó al pie de la letra, sino si las desviaciones se declaran honestamente — que es exactamente lo que la Tabla 8.1 hace.

---

## Diapositiva 18 — Implicaciones prácticas: arquitectura híbrida propuesta

**Contenido:**
- No desplegar esta política como programador autónomo primario en su estado actual — `fifo` iguala o supera su desempeño a costo computacional casi nulo.
- El eje de disponibilidad de máquina (averías) es donde la política es **más** frágil, no menos — justo donde un programador robusto sería más valioso para mantenimiento.
- Propuesta motivada por los propios resultados: **puente reactivo de baja latencia** (decisión en milisegundos tras un evento de disrupción) + **re-optimizador exacto/metaheurístico en paralelo** (CP-SAT/GA recalculando un plan de mayor calidad) — no evaluada en esta tesis, dirección de trabajo futuro de largo plazo.

**Notas del orador (ES):** Esta diapositiva responde de forma proactiva al Hallazgo moderado 2.6 de `docs/auditorias/PreDefenseAudit.md` (expectativa de validación aplicada en un programa de ingeniería industrial) — tener lista la respuesta oral de 60–90 segundos que el propio informe recomienda preparar. Guion sugerido: *"Esta tesis es una validación exclusivamente por simulación, y lo declaramos así desde el Capítulo 1. Elegimos ese alcance porque permite aislar y diagnosticar con precisión estadística por qué el objetivo conjunto de generalización y robustez no se logró en este primer intento — algo que un estudio de campo, con muchas más variables no controladas, no habría permitido hacer con el mismo rigor. La conexión aplicada que sí ofrecemos es esta arquitectura híbrida, motivada directamente por el contraste entre la debilidad de calidad y la fortaleza de velocidad que medimos — es una hipótesis de ingeniería derivada de evidencia simulada rigurosa, no un resultado validado en planta, y así lo mantenemos declarado en todo el documento."*

---

## Diapositiva 19 — Trabajo futuro

**Contenido:**
- Corto plazo: reproducir L2D/Song et al. dentro del propio *pipeline*; re-medir `la40` de forma aislada.
- Mediano plazo: ablación arquitectónica GIN homogénea/relacional/heterogénea; corrección de aleatorización de dominio (mezclar episodios sin perturbación); múltiples semillas para H4; métrica de tardanza esperada para el régimen de llegadas.
- Largo plazo: arquitectura híbrida puente-reactivo/re-optimizador; módulo de procesamiento de escenarios (estilo SPM-DAN); validación *sim-to-real* parcial.

**Notas del orador (ES):** Enmarcar cada ítem como derivado de un hallazgo específico, no como una lista genérica de cierre — el propio Capítulo 8 lo hace así deliberadamente, y vale la pena decirlo en voz alta: *"cada una de estas líneas responde a un hallazgo diagnosticado en los Capítulos 6 y 7, no a una sugerencia estándar de 'más datos, más cómputo'."*

---

## Diapositiva 20 — Cierre: mensaje final

**Contenido:**
- Una hipótesis aceptada, cuatro rechazadas — un resultado pre-registrado, no ajustado a posteriori.
- El valor de esta tesis: la primera caracterización cuantitativa y diagnosticada de que generalización de tamaño y robustez multi-perturbación **no son la misma propiedad** para una política GNN+PPO.
- Agradecimientos y apertura a preguntas.

**Notas del orador (ES):** Cerrar con la misma seguridad con la que se abrió la diapositiva 9 — repetir, casi palabra por palabra, el mensaje central para que sea lo último que el comité escuche antes de las preguntas: *"No demostramos que la política propuesta sea superior. Demostramos, con el mismo rigor estadístico con el que se habría reportado un resultado positivo, exactamente dónde y por qué no lo es — y eso es, en última instancia, lo que un protocolo de pre-registro está diseñado para permitir decir con total tranquilidad."* Transición directa a preguntas — no añadir contenido nuevo después de esta frase.
