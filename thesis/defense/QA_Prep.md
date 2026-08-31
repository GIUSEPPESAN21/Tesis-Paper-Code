# Preparación de preguntas y respuestas — sustentación

**Fuentes:** `../Thesis.md` (todas las cifras citadas están verificadas contra el Capítulo 6 y son consistentes con el Resumen/Abstract) y `docs/auditorias/PreDefenseAudit.md` (hallazgos crítico y moderados; la mayoría ya resueltos en el documento final vía la Sección 5.4 y las tablas/notas añadidas en la revisión v2 — se indica explícitamente en cada respuesta si el hallazgo ya está corregido en el texto o si requiere solo una respuesta oral).

**Cómo usar este documento:** cada pregunta incluye (1) por qué es probable que el comité la haga, (2) una respuesta preparada de 30–90 segundos, lista para decirse casi textualmente, y (3) la referencia exacta del documento donde ampliar si el comité pide más detalle. Practicar las respuestas en voz alta, no solo leerlas — el objetivo es sonar seguro, no memorizado.

---

## A. Sobre el resultado global (la pregunta que abre la sustentación)

### A1. "De las cinco hipótesis pre-registradas, cuatro se rechazaron. ¿Por qué debería aprobarse esta tesis?"

**Por qué la harán:** es la pregunta más obvia y la más importante de anticipar — si no se responde bien en los primeros minutos, condiciona negativamente todo lo que sigue.

**Respuesta preparada:** "Porque lo que se evalúa en un pre-registro no es si la hipótesis se confirma, sino si la pregunta se responde con rigor. Fijamos las cinco reglas de decisión de H1 a H5 antes de generar un solo resultado, con un archivo de pre-registro con marca de tiempo verificable, precisamente para poder reportar rechazos con la misma autoridad con la que se reportaría un resultado positivo. Y cada uno de los cuatro rechazos no se quedó en un $p$-valor: tiene una causa raíz diagnosticada — el desajuste de legalidad entre el entorno estático de entrenamiento y el dinámico de evaluación para H3, el colapso de competitividad nominal por aleatorización de dominio sin episodios limpios para H4, un artefacto de construcción de métrica para H5. Ese nivel de diagnóstico es, en la literatura que revisamos en el Capítulo 3, más raro que un resultado positivo sin explicar. Eso es lo que se está evaluando aquí: si el candidato entendió por qué pasó lo que pasó, no si el modelo ganó."

**Ampliar en:** Capítulo 6, Tabla 6.0 (síntesis); Capítulo 7, §7.1; `docs/auditorias/PreDefenseAudit.md` §0 (veredicto ejecutivo) y §4 (fortalezas).

---

## B. Sobre resultados específicos

### B1. "¿Por qué su modelo tiene un desempeño tan por debajo de CP-SAT y del algoritmo genético aquí?" *(pregunta citada explícitamente en el encargo)*

**Por qué la harán:** la Tabla 6.2.1 muestra al GA (5.52 % RPD medio) y a CP-SAT (0 % por construcción) claramente por encima de RL (24.82 %) — es visualmente imposible de ocultar y probablemente la primera pregunta técnica de la sesión.

**Respuesta preparada:** "Dos razones, ninguna de las cuales excusa el resultado, pero ambas lo contextualizan. Primero, CP-SAT no es un método par — es un solver exacto sin restricción de presupuesto de entrenamiento, es literalmente la cota de calidad contra la que medimos a todos los demás; no esperábamos ni pretendíamos superarlo. Segundo, y más relevante para el GA: el algoritmo genético invierte 60 segundos de búsqueda poblacional en el momento mismo de resolver cada instancia, mientras que nuestra política invierte todo su cómputo por adelantado, durante el entrenamiento, con un presupuesto acotado a un solo investigador con una GPU de consumo, y luego decide en milisegundos sin ninguna búsqueda adicional en el momento de la evaluación. Son dos paradigmas de asignación de cómputo distintos — tiempo de entrenamiento acotado vs. tiempo de búsqueda en evaluación — y bajo ese paradigma de decisión amortizada, perder frente al GA en calidad absoluta es exactamente el tipo de costo que esperaríamos, no una sorpresa que invalide el diseño. Lo que si es un resultado genuino y no una excusa: la política tampoco superó a la mejor regla de despacho simple, `fifo`, en más del 20% de las instancias — eso sí es un rechazo de H1 sin atenuantes de presupuesto, y lo reportamos así."

**Ampliar en:** Capítulo 6, §6.2.1 (incluye la discusión explícita GA-vs-RL, ya incorporada en la versión final resolviendo el Hallazgo moderado 2.1); Capítulo 7, §7.2.

### B2. "¿Cómo saben que esto generaliza más allá de sus benchmarks?" *(pregunta citada explícitamente en el encargo)*

**Por qué la harán:** es la pregunta natural tras presentar H2 como el único resultado aceptado — un comité atento notará que "generalización" se está usando en un sentido específico y limitado.

**Respuesta preparada:** "Con la evidencia que tenemos, la respuesta honesta es: sabemos que generaliza *dentro* de la familia Lawrence, de instancias 1.5 a 3 veces el tamaño de entrenamiento, en el sentido de que el rango relativo de la política frente a nueve reglas de despacho no empeora significativamente. No sabemos si generaliza *entre* familias de instancias estructuralmente distintas — Taillard, Hurink, ABZ — porque esos bloques, aunque estaban planificados en el diseño experimental original, nunca se ejecutaron, una desviación que documentamos explícitamente como amenaza a la validez externa en el Capítulo 7. Tampoco sabemos si generaliza a distribuciones de instancias con una estructura de taller distinta a la de estos benchmarks académicos. Lo que sí podemos afirmar con rigor estadístico es la generalización de tamaño dentro de una misma familia — y esa es, precisamente, la afirmación que hacemos, ni más ni menos."

**Ampliar en:** Capítulo 6, §6.3 (Tabla 6.3.2/6.3.3); Capítulo 7, §7.7.2 (validez externa); Capítulo 8, §8.6.2 punto 8 (trabajo futuro: extender a Taillard/Hurink).

### B3. "Si H1 se rechaza pero H2 se acepta, ¿no es eso contradictorio? ¿Cómo puede ser mala en desempeño absoluto pero buena en generalización?"

**Por qué la harán:** es la aparente paradoja lógica más visible del patrón de resultados, y un comité que no maneje la distinción teórica la señalará de inmediato.

**Respuesta preparada:** "No es contradictorio, y la razón es exactamente la teoría de generalización de GNN que desarrollamos en el marco teórico: la invariancia a permutaciones y el número de parámetros independiente del tamaño de instancia son propiedades arquitectónicas que preservan el *comportamiento relativo* de la política al transferirse a instancias más grandes — no propiedades que mejoran su nivel absoluto de competitividad. Una política mediocre en su régimen de entrenamiento transfiere una posición relativa mediocre; una política fuerte, como L2D, transfiere una posición relativa fuerte. Nuestra política aceptó H2 porque su rango frente a las PDR nunca empeoró significativamente al crecer el tamaño — pero su RPD absoluto siguió siendo peor que `fifo` en los cuatro *tiers*. Es, de hecho, el mismo patrón cualitativo que reporta L2D, solo que partiendo de un punto de competitividad inicial más bajo."

**Ampliar en:** Capítulo 7, §7.3 (desarrollo teórico completo de esta distinción, con la comparación explícita contra L2D).

### B4. "La recompensa consciente de la perturbación (H4) empeoró drásticamente el desempeño nominal — más de 100 puntos porcentuales en un caso. ¿No es eso evidencia de que el diseño experimental estaba mal planteado?"

**Por qué la harán:** +106.0 pp es una cifra dramática que puede leerse como un error grave si no se explica su causa con precisión.

**Respuesta preparada:** "Es una consecuencia real y cuantificada de una decisión de diseño específica, no un error de implementación ni un síntoma de que el planteamiento general estaba mal concebido. Entrenamos las políticas de κ mayor que cero con aleatorización de dominio: cada episodio de entrenamiento muestrea uno de los nueve regímenes de perturbación, sin incluir nunca un episodio sin perturbación. Elegimos esa opción — documentada y fechada antes de entrenar — precisamente porque era la única que sostenía la propia regla de decisión de H4, que exige una política consciente de la perturbación en general, no de un régimen específico. El costo que produjo, ahora medido con precisión, es un hallazgo genuino y transferible: cualquier intento de generalizar un mecanismo de recompensa de un solo tipo de perturbación a un régimen multi-perturbación mediante aleatorización de dominio necesita preservar una fracción de episodios normales en el entrenamiento, algo que la literatura de referencia en este eje nunca tuvo que resolver porque nunca entrenó sobre un régimen tan amplio. Ya identificamos la corrección concreta — mezclar un régimen sin perturbación con peso no nulo — como el primer paso de trabajo futuro de mediano plazo."

**Ampliar en:** Capítulo 6, §6.4.2 (diagnóstico completo); Capítulo 5, §5.4.2 (la decisión de diseño Opción A vs. Opción B, documentada prospectivamente); Capítulo 7, §7.4.3.

### B5. "¿Por qué excluyeron la instancia `mk06`? ¿No podrían simplemente haber usado el óptimo publicado de todas formas?"

**Por qué la harán:** reduce el tamaño muestral del brazo Brandimarte de 8 a 7, visible en cualquier tabla que reporte $N$.

**Respuesta preparada:** "Al cargar las instancias de Brandimarte desde el espejo de datos que usamos, detectamos que los metadatos del espejo describen `mk06` con 15 máquinas — la cifra canónica de Brandimarte 1993, la que corresponde al óptimo publicado de 58 — pero el archivo de datos real de esa instancia en el mismo espejo usa solo 10 índices de máquina distintos, y su propia línea de cabecera lo confirma. Es una inconsistencia genuina entre los metadatos y el archivo de datos del espejo, no un error nuestro de carga ni un problema transitorio de descarga — lo verificamos recargando el archivo de forma independiente. La instancia que efectivamente se ejecutó no es la `mk06` canónica de 15 máquinas contra la que el óptimo de 58 es válido, así que calcular su RPD contra ese óptimo habría sido comparar contra una referencia incorrecta. Preferimos excluirla del cálculo de RPD, con la explicación completa en el propio Capítulo 6, a reportar una cifra que sabíamos que no correspondía a la instancia real evaluada."

**Ampliar en:** Capítulo 6, §6.2.1 (nota "Por qué N=7 y no N=8", con la explicación técnica completa incorporada directamente en el texto).

### B6. "Detectaron una anomalía de medición en `la40` que hace que sea 120 veces más lenta que una instancia comparable. ¿Cómo confiamos en el resto de sus mediciones de tiempo si esto pasó sin que nadie lo notara al principio?"

**Por qué la harán:** es la anomalía más visible del documento y pone a prueba directamente si el candidato confía en sus propios datos.

**Respuesta preparada:** "La anomalía se detectó precisamente porque construimos el protocolo para que fuera detectable: comparamos cada instancia contra otras de tamaño y dificultad estructural equivalente, y `la40` — con 7216 segundos de tiempo de pared de CP-SAT frente a 60 segundos en `la38`, una instancia de tamaño y estructura idénticos — se separó de inmediato del resto del conjunto. Investigamos la causa: no hay ningún error o advertencia registrada en los logs de ejecución, lo que apunta a interferencia del sistema operativo durante la corrida, no a una propiedad genuina del solver sobre esa instancia. Hicimos un análisis de sensibilidad sustituyendo esa medición por el valor de una instancia comparable, y el veredicto de H5 no solo se mantiene — se vuelve más decisivo, de 1 de 43 instancias que pasan el umbral a 0 de 43. El hecho de que la anomalía se haya detectado, investigado y neutralizada dentro del propio capítulo de resultados, con el veredicto reportado sin depender de ella, es evidencia de que el protocolo de verificación funciona, no de que los datos sean poco confiables."

**Ampliar en:** Capítulo 6, §6.5.1 (análisis completo de la anomalía y de sensibilidad).

---

## C. Sobre limitaciones metodológicas

### C1. "Solo entrenaron una semilla por cada valor de κ en H4. ¿Cómo saben que estos resultados no son solo ruido de una corrida particular?"

**Por qué la harán:** una sola semilla por condición experimental es una limitación estadística real, ya señalada en el propio Capítulo 7.

**Respuesta preparada:** "Es una limitación real que declaramos explícitamente, no una que el comité está descubriendo por primera vez. Con una sola semilla, no podemos reportar un intervalo de confianza sobre la magnitud exacta del colapso nominal o del beneficio de robustez. Dicho esto, la magnitud del efecto observado — entre 46 y 106 puntos porcentuales de colapso de RPD nominal, y cero de nueve regímenes con mejora significativa de robustez para κ=0.5 — es tan grande que es poco probable que una segunda semilla revierta la dirección cualitativa del hallazgo; lo que sí podría cambiar con más semillas es la magnitud precisa, y en particular si el único resultado parcialmente favorable de κ=1.0 —un régimen de nueve con mejora significativa— se replica o no. Repetir el entrenamiento de H4 con múltiples semillas es la acción de trabajo futuro de mediano plazo que priorizamos explícitamente para cerrar esta brecha."

**Ampliar en:** Capítulo 7, §7.7.1 (validez interna); Capítulo 8, §8.6.2 punto 6; `docs/auditorias/PreDefenseAudit.md` §5 (nota de calibración: esta limitación es aceptable, no objetable, bajo un estándar de maestría con los recursos declarados de un solo estudiante/GPU/nueve meses).

### C2. "Su arquitectura es un punto intermedio entre homogénea y heterogénea, pero nunca compararon directamente contra esas dos alternativas. ¿Cómo saben que su elección arquitectónica es la correcta?"

**Por qué la harán:** el objetivo 3 de la tesis (Tabla 8.1) está marcado explícitamente como "parcialmente cumplido" por esta razón exacta.

**Respuesta preparada:** "No lo sabemos con los datos que tenemos, y lo decimos así en el Capítulo 7: el rechazo de H1 no nos permite distinguir entre dos explicaciones igualmente compatibles con los datos — que el objetivo conjunto de generalización y robustez es intrínsecamente costoso en capacidad, sin importar la arquitectura específica, o que esta elección arquitectónica relacional en particular compite mal con reglas de despacho bien afinadas para este tipo de benchmark. La ablación que separaría ambas explicaciones — homogénea pura, relacional, heterogénea completa, bajo el mismo protocolo — está especificada en nuestro plan de ablaciones pero no se ejecutó dentro del alcance de esta tesis. La justificamos con el argumento de costo-beneficio disponible en la literatura — más expresiva que una GNN homogénea porque distingue el significado semántico de cada relación, más barata que una heterogénea completa — y reforzada retroactivamente por un trabajo de 2026 que muestra que la heterogeneidad plena no es condición necesaria para escalar. Pero es una justificación de diseño, no una validación empírica comparativa, y lo nombramos como la pregunta abierta más importante que deja esta tesis."

**Ampliar en:** Capítulo 4, §4.1.2 (justificación de la elección); Capítulo 7, §7.2, §7.6; Capítulo 8, Tabla 8.1 objetivo 3, §8.6.2 punto 4.

### C3. "El pre-registro es una fortaleza que ustedes mismos destacan, pero tuvo dos enmiendas posteriores. ¿No socava eso la garantía anti-HARKing que reclaman?"

**Por qué la harán:** cualquier modificación posterior a un documento "congelado" invita naturalmente a la sospecha, incluso si es legítima.

**Respuesta preparada:** "Es una pregunta justa, y la respuesta corta es no, por una razón verificable: ninguna de las dos enmiendas modificó una regla de decisión ni un criterio de aceptación de H1 a H5. La Enmienda 1 aclaró la estructura de instancias disponibles; la Enmienda 2 documentó que la ruta de evaluación de nuestro pipeline es exclusivamente CPU, independientemente de la disponibilidad de GPU, resolviendo una ambigüedad sobre qué hardware reportar — ninguna de las dos toca qué constituye aceptar o rechazar una hipótesis. Ambas están fechadas y documentadas como adendas explícitas, no como ediciones silenciosas del documento original, que es exactamente la disciplina que un protocolo de pre-registro bien administrado permite sin comprometer su validez."

**Ampliar en:** `../PreRegistration.md` (Enmiendas 1–2, con fecha); Capítulo 5, §5.4.5 (Enmienda 2 in extenso); `docs/auditorias/PreDefenseAudit.md` §3.4.

### C4. "¿Por qué no calcularon la tardanza esperada para el régimen de llegadas, si el propio pre-registro la señala como la métrica primaria más apropiada para ese eje?"

**Por qué la harán:** es una omisión que el propio documento reconoce explícitamente en la sección de validez de constructo — un comité que lea el Capítulo 7 con cuidado lo encontrará.

**Respuesta preparada:** "Es una omisión real que reconocemos directamente, no un descuido que estemos ocultando. El índice de robustez que sí calculamos, basado en el makespan, evita parcialmente la distorsión de comparar programaciones de alcance distinto bajo llegadas dinámicas, porque usa el propio makespan nominal del método como referencia en lugar de un óptimo externo de alcance fijo — pero no sustituye a una métrica de tardanza, que captura una dimensión del desempeño, trabajos que llegan tarde, que el makespan no captura en absoluto. El rechazo de H3 en el eje de llegadas es, por tanto, una conclusión sólida sobre el índice de robustez específicamente, pero una conclusión incompleta sobre el desempeño real del sistema bajo llegadas dinámicas. Priorizamos completar los nueve regímenes con la métrica ya especificada antes que añadir una métrica adicional a mitad de la ejecución experimental — calcular la tardanza esperada es la cuarta acción de nuestro trabajo futuro de mediano plazo."

**Ampliar en:** Capítulo 7, §7.7.3 (validez de constructo); Capítulo 8, §8.6.2 punto 7.

---

## D. Sobre generalización y aplicabilidad práctica

### D1. "Todo esto es simulación. ¿Qué aplicabilidad real tiene para un taller de manufactura de verdad?"

**Por qué la harán:** es la pregunta que `docs/auditorias/PreDefenseAudit.md` identifica explícitamente como riesgo específico de un programa de ingeniería industrial (Hallazgo moderado 2.6) — casi garantizada en algún momento de la sesión.

**Respuesta preparada:** "Con los datos que tenemos, la aplicabilidad directa es limitada, y lo declaramos así desde el Capítulo 1: todos los datos son simulados, sobre benchmarks académicos, sin ningún contraste contra un taller físico. Elegimos deliberadamente ese alcance —no por falta de interés en la aplicación real, sino porque nueve meses con un solo investigador y una GPU de consumo no permiten sostener simultáneamente un estudio de campo y el nivel de rigor estadístico que quisimos aplicar aquí. Lo que sí ofrecemos con validez práctica son dos cosas: primero, una lección de ingeniería de datos de entrenamiento directamente transferible a cualquier equipo que quiera entrenar una política consciente de mantenimiento — que aleatorizar sobre múltiples regímenes de perturbación sin preservar operación normal en el entrenamiento colapsa la competitividad nominal, algo que descubrimos empíricamente y que cualquiera enfrentaría al intentar la misma extensión. Segundo, una arquitectura híbrida motivada directamente por nuestros propios resultados — puente reactivo de baja latencia más re-optimización exacta en paralelo — que no evaluamos en esta tesis, pero que es una hipótesis de ingeniería razonable derivada de evidencia simulada rigurosa. Una validación *sim-to-real*, aunque sea parcial, es la primera línea de trabajo futuro de largo plazo que proponemos."

**Ampliar en:** Capítulo 1, §1.5 (delimitación de alcance); Capítulo 8, §8.5 (implicaciones prácticas, incluye el reconocimiento explícito de esta brecha); §8.6.3 punto 11.

### D2. "Si `fifo`, una regla trivial de asignar por orden de llegada, iguala o supera a su política en el 93 % de las instancias, ¿no sugiere eso que el enfoque de aprendizaje por refuerzo no vale la pena para este problema?"

**Por qué la harán:** es la lectura más escéptica posible del resultado de H1, y un comité crítico puede plantearla directamente.

**Respuesta preparada:** "Para esta instanciación específica —este presupuesto de entrenamiento, esta arquitectura, este régimen de entrenamiento— la evidencia dice que sí, `fifo` es más eficiente en el sentido costo-beneficio: prácticamente gratis computacionalmente e igual o mejor en calidad. Pero eso no es lo mismo que decir que el enfoque GNN+PPO no vale la pena para el problema en general. Nuestra política aceptó H2: transfiere su comportamiento relativo a instancias de hasta el triple de tamaño de entrenamiento sin reentrenar, algo que `fifo`, al ser una regla fija, no necesita demostrar porque no tiene una noción de tamaño de entrenamiento que transferir. Y bajo instancias grandes y difíciles, nuestra política decide en milisegundos donde un re-optimizador exacto puede tardar minutos u horas — una propiedad que `fifo` comparte, pero con calidad de solución generalmente inferior a la de una política aprendida bien entrenada. La lectura correcta no es 'RL no sirve', es 'esta política, con este presupuesto de entrenamiento y sin especializar la arquitectura para robustez, no cierra la brecha frente a alternativas más baratas o más maduras' — una distinción que el propio Capítulo 7 hace explícitamente para no sobre-generalizar el rechazo de H1."

**Ampliar en:** Capítulo 7, §7.2 (segunda explicación: presupuesto de entrenamiento, no arquitectura per se); Capítulo 8, §8.7 (reflexión de cierre).

---

## E. Sobre aspectos estructurales del documento

### E1. "El Capítulo 2 (Marco Teórico) no está redactado. ¿Cómo se puede evaluar el rigor teórico de una tesis sin su capítulo teórico?"

**Por qué la harán:** es el hallazgo crítico no resuelto más visible de `docs/auditorias/PreDefenseAudit.md` — si el Capítulo 2 sigue sin redactarse a la fecha de la sustentación, esta pregunta es prácticamente segura.

**Respuesta preparada (usar solo si el Capítulo 2 sigue pendiente en la fecha de sustentación; de lo contrario, omitir esta pregunta del guion):** "Es una brecha real y la reconocemos directamente, sin minimizarla: el Capítulo 2 no se redactó en prosa de tesis con aparato de citas formal. El contenido teórico que le correspondería —la formulación MDP completa, la teoría de invariancia a permutaciones de Xu et al. que sustenta la generalización de tamaño, los fundamentos de simulación de eventos discretos— sí se desarrolló y se aplicó a lo largo de la tesis: la formulación MDP completa está en el Capítulo 4, §4.3; la teoría de generalización se retoma y se usa activamente para interpretar H2 en el Capítulo 7, §7.3; los fundamentos de simulación se aplican en el Capítulo 4, §4.2. Lo que falta no es el contenido teórico en sí, que sostiene cada interpretación del Capítulo 7, sino su presentación consolidada y autónoma como un capítulo formal de marco teórico. Es la única acción de esfuerzo alto que dejamos pendiente de las doce identificadas en nuestra propia auditoría pre-sustentación, y la nombramos aquí de forma directa en lugar de esperar a que el comité la señale."

**Nota para el candidato, no para decir en voz alta:** si es posible redactar el Capítulo 2 antes de la fecha de sustentación —es la acción de mayor prioridad de `docs/auditorias/PreDefenseAudit.md` §6, ítem 1—, hacerlo elimina esta pregunta por completo. Si no es posible, esta respuesta debe decirse con la misma seguridad que cualquier otra, no a la defensiva.

**Ampliar en:** Nota de estado al inicio del Capítulo 2 (dentro de `../Thesis.md`); `docs/auditorias/PreDefenseAudit.md` §1 y §6, ítem 1.

### E2. "¿Por qué el título de la tesis en este documento parece un título de trabajo? ¿Es el título oficial?"

**Por qué la harán:** solo si el comité compara el documento contra el registro institucional; riesgo bajo pero trivial de resolver.

**Respuesta preparada:** "Confirmamos el título contra el registro oficial de UNIMINUTO antes de esta entrega" — **acción pendiente para el candidato:** verificar este punto antes de la sustentación; si el título cambió, actualizar la portada es un ajuste de minutos, no de contenido.

**Ampliar en:** `docs/auditorias/PreDefenseAudit.md` §3.1.

---

## Resumen de una página: las cinco respuestas que hay que poder decir sin dudar

1. **"¿Por qué cuatro rechazos?"** → Pre-registro congelado + causa raíz diagnosticada en cada rechazo, no solo un $p$-valor.
2. **"¿Generaliza de verdad?"** → Sí, dentro de la familia Lawrence, en rango relativo, no en desempeño absoluto; no probado entre familias distintas — declarado, no descubierto.
3. **"¿Por qué pierde contra el GA/CP-SAT?"** → Paradigmas de cómputo distintos (entrenamiento acotado vs. búsqueda en evaluación); el GA se nombra proactivamente, no se esconde.
4. **"¿Esto sirve para un taller real?"** → No validado en campo, por diseño; arquitectura híbrida propuesta como hipótesis de ingeniería, no como resultado.
5. **"¿Qué le falta a esta tesis?"** → Decirlo primero, con la lista ya lista: Capítulo 2 sin redactar (si aplica), ablación arquitectónica pendiente, una sola semilla por κ, sin reimplementación de L2D/Song et al. — todo ya reconocido en el propio documento, nada nuevo que el comité esté "descubriendo".
