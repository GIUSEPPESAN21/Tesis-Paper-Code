# Auditoría doctoral — Borrador consolidado, Capítulos 1–5

**Fecha de auditoría:** 28 de julio de 2026
**Rol:** auditor de tesis doctoral (metodología, coherencia, citación)
**Alcance:** el borrador que declaraste como "Capítulos 1 a 5" no existe como un único archivo consolidado; se ensambló a partir de los archivos individuales de tu carpeta, según confirmaste. La tabla siguiente es el mapa exacto usado para esta auditoría.

## 0. Tabla de contenidos del borrador auditado

| Capítulo declarado | Archivo(s) fuente | Estado |
|---|---|---|
| Cap. 1 — Introducción | `../../archive/capitulos/Chapter1.md` | Completo, §1.1–1.7 |
| Cap. 2 — Marco Teórico | **ningún archivo** | **No existe como capítulo redactado** (ver Hallazgo C2-1) |
| Cap. 3 — Estado del Arte | `../../archive/capitulos/Chapter3_final.md` | Completo, §3.1–3.7 + referencias [1]–[51] |
| Cap. 4 — Metodología | `../../archive/capitulos/Chapter4_sections_4.1_4.3.md` (4.1, 4.1.1, 4.1.3, 4.3) + `../../archive/capitulos/Section4.2.md` (4.2) + `../../archive/capitulos/Section4.4.md` (4.4) + `../../archive/capitulos/Section4.5.md` (4.5) + `../../archive/capitulos/Section4.6.md` (4.6) | Completo 4.1–4.6, pero con tres capas temporales distintas (ver §3 de este informe) |
| Cap. 5 — Diseño Experimental | `../../archive/capitulos/Section5.1_5.2.md` (5.1, 5.2) + `../../archive/capitulos/Section5.3.md` (5.3) | Completo 5.1–5.3 |

Esta tabla es en sí misma el primer hallazgo: **no hay un Capítulo 2 redactado en ninguna parte del proyecto**. El contenido teórico existe únicamente como notas de framework ("Parte B" de `../docs/planificacion/Step 4 Thesis.md`), nunca convertido a prosa de tesis. Todo el resto de esta auditoría trata al Cap. 2 como ausente, no como "pendiente de pulir".

---

## 1. Resumen ejecutivo

| Severidad | N.º de hallazgos |
|---|---|
| Crítico | 6 |
| Moderado | 7 |
| Menor | 5 |

Los dos problemas de mayor impacto son estructurales, no de redacción: (a) el Capítulo 2 no existe, pese a que el Capítulo 1 lo cita seis veces como si ya estuviera escrito; y (b) las Secciones 4.1–4.3 documentan honestamente un entorno **estático, sin averías, sin κ, con enmascarado plano** (Sprint 1), mientras que las Secciones 4.4–4.6, escritas después, ya asumen la arquitectura heterogénea, el generador de disrupciones y el arnés de baselines como si existieran — un capítulo que, leído de corrido, dobla de tiempo verbal sin transición. Ninguno de los dos es un defecto de contenido individual: cada sección, tomada aislada, es honesta y está bien anclada al código. El problema es de ensamblaje.

---

## 2. Capítulo 1 — Introducción

### [CRÍTICO] C1-1. La ecuación de recompensa con κ, prometida en "Sección 4.3", no está en la Sección 4.3

`../../archive/capitulos/Chapter1.md` §1.2, punto 2, y §1.3 (SP1) presentan como ya diseñado "el diseño propuesto en esta tesis (**Sección 4.3**, ecuación de recompensa consciente de la perturbación) ... mediante un término de penalización de inestabilidad ponderado por un factor **κ**". La Sección 4.3.3 real (`../../archive/capitulos/Chapter4_sections_4.1_4.3.md`) documenta exactamente:

```
r_t = ( H(s_t) − H(s_{t+1}) ) × reward_scale
```

Sin término κ, sin penalización de inestabilidad, sin mención alguna de perturbación — porque, como la propia sección declara explícitamente, corresponde al entorno estático sin averías del Sprint 1. La Sección 4.3.3 ni siquiera anticipa que κ vendrá en una revisión futura de esa misma sección (a diferencia de 4.3.1, que sí declara explícitamente sus desviaciones de alcance). Un lector que llega a 4.3 buscando la ecuación prometida en el Capítulo 1 no la encuentra en ninguna parte del borrador actual.
**Acción:** o (a) se añade a 4.3.3 un apartado explícito "extensión prevista: término κ (sprint posterior)" con la fórmula objetivo, o (b) se reescribe Cap. 1 §1.2/1.3 para dejar de citar "Sección 4.3" como si la ecuación ya existiera, y en su lugar decir "se especifica en un sprint posterior (Sección 4.3, revisión pendiente)".

### [CRÍTICO] C1-2. Referencias cruzadas a secciones inexistentes

- §1.3 (SP1): "índice de robustez RI, **Sección 4.2/5.2**" — `../../archive/capitulos/Section4.2.md` fue releída íntegra para esta auditoría y **no define RI** en ninguna parte (es una sección sobre DES y el contrato Gymnasium). RI solo está definido en `../../archive/capitulos/Section5.1_5.2.md` §5.2.2. La referencia a "4.2" es errónea y debe eliminarse, dejando solo "Sección 5.2".
- §1.3: "hipótesis pre-registradas (**Sección 4.7**/D7...)" — el propio §1.7 de este mismo capítulo describe la estructura del Capítulo 4 como "el entorno de simulación..., la formulación MDP..., el codificador GNN..., los modelos paramétricos de perturbación y las líneas base" — es decir, 4.1 a 4.6, sin mención de una 4.7. Y de hecho el contenido de pre-registro/ajuste de hipótesis ya se redactó como **Sección 5.1.4** (`../../archive/capitulos/Section5.1_5.2.md`), no como 4.7. "Sección 4.7" es una sección fantasma.
- §1.4, objetivo 1: cita "Sección 2.2, 2.3, 4.2–4.3" para la "representación de estado en forma de grafo heterogéneo operación-máquina" — pero 4.2–4.3, tal como están redactadas hoy, describen explícitamente el grafo **homogéneo** de un solo tipo de nodo (Sección 4.3.1: "ocho características... sin averías ni flexibilidad de máquina todavía"). El grafo heterogéneo se formaliza recién en la Sección **4.4**, no citada aquí.

**Acción:** pasar todo el capítulo por una revisión de referencias cruzadas sección-por-sección contra el índice real del documento (tarea mecánica, pero debe hacerse antes de cualquier lectura de comité).

### [CRÍTICO] C1-3. El objetivo específico 5 promete reproducir L2D y Song et al. como líneas base neuronales; ninguna sección de Metodología o Diseño Experimental lo contempla

§1.4, objetivo 5: "Reproducir como líneas base neuronales **L2D** [23] y el **esquema heterogéneo de Song et al.** [25], además de las líneas base clásicas (PDR, algoritmo genético, CP-SAT...)". La Sección 4.6 (Líneas base) formaliza únicamente PDR, GA y CP-SAT — no menciona en ningún punto una reimplementación de L2D o de Song et al. como baseline neuronal. La matriz de experimentos E0–E8 (Sección 5.1.3) tampoco incluye ningún experimento de reproducción de L2D/Song et al. Esto no es una desviación menor: es un objetivo específico completo, con dos referencias bibliográficas nombradas, que no tiene ningún correlato en el capítulo que se supone lo implementa. O se añade formalmente a 4.6/5.1, o se retira del objetivo 5 con una nota explícita de por qué se abandonó (probablemente presupuesto de cómputo, Sección 1.5).

### [CRÍTICO] C1-4. H1 exige resultados sobre MK (Brandimarte/FJSP), que el propio Capítulo 5 marca como bloqueado

`../../archive/capitulos/Chapter1.md`, H1: "la política propuesta supera a la mejor PDR ... sobre los conjuntos de referencia **ta** (Taillard) **y MK** (Brandimarte)." Pero `../../archive/capitulos/Section5.1_5.2.md`, Tabla 5.1.3, fila E2: "Estático FJSP | H1 | ... MK01–MK10, Hurink | **— (bloqueado: carga de instancias FJSP fuera de alcance de este sprint...)**". Una hipótesis pre-registrada que exige datos de un experimento que el propio diseño experimental declara fuera de alcance no es ejecutable tal como está escrita. Esto debe resolverse **antes** de congelar el pre-registro (el principio anti-HARKing que la propia Sección 5.1.4 invoca exige precisamente que esta clase de inconsistencia no llegue viva al momento de generar resultados): o se remueve MK de H1 y se dejan ta+la como el conjunto de aceptación, con MK como una extensión condicional explícita, o se prioriza destrabar el alcance FJSP antes de la ejecución de H1.

### [MODERADO] C1-5. El objetivo 3 promete comparar variantes GIN y GAT; la Sección 4.4 ya descartó GAT como configuración por defecto

§1.4, objetivo 3: "política actor-crítico con codificador GNN heterogéneo (**comparando variantes GIN y GAT**)". La Sección 4.4.2 dedica un apartado completo a justificar por qué **no** se adopta atención (GAT/estilo Song et al.) como configuración por defecto, dejándola "como una ablación futura concreta y bien definida ... no como una alternativa genérica sin especificar". Esto no es necesariamente incorrecto — una ablación futura sigue siendo, en sentido amplio, una comparación — pero la redacción del objetivo 3 sugiere una comparación empírica planificada dentro del alcance actual, mientras que 4.4.2 la trata como trabajo futuro fuera de alcance. Conviene alinear el verbo del objetivo ("comparar" → "justificar la elección de GIN frente a la alternativa GAT, documentada como ablación futura") para que ambos textos digan lo mismo.

### [MENOR] C1-6. Afirmaciones sin fuente marcadas como [VERIFICAR], correctamente señaladas pero aún pendientes

§1.1 tiene dos marcas [VERIFICAR] (penetración de tecnologías 4.0; frecuencia empírica de disrupciones en manufactura). Estas ya están correctamente autoseñaladas por el propio autor — no es un hallazgo nuevo, pero se listan aquí para que queden en el mismo inventario de pendientes que el resto de la auditoría, de cara al checklist final antes de sustentación.

---

## 3. Capítulo 2 — Marco Teórico

### [CRÍTICO] C2-1. El capítulo no existe como documento de tesis

No hay ningún archivo `Chapter2*.md` ni equivalente en la carpeta de trabajo. El contenido teórico correspondiente (notación α|β|γ, MDP/semi-Markov, fundamentos de PPO, invariancia a permutaciones de GNN, PDR, CP-SAT/GA, DES) existe **solo** como notas de framework en la "Parte B" de `../docs/planificacion/Step 4 Thesis.md` — un documento de trabajo interno, no redactado en prosa de tesis ni con el aparato de citas que sí tienen los Capítulos 1, 3 y 4.

Esto es grave porque **seis referencias explícitas** al Capítulo 2 ya están incrustadas en el Capítulo 1 como si el contenido existiera:
- §1.2: "véase la Sección 2.2 para la formulación completa"
- §1.2: "siguiendo la extensión estocástica descrita en la Sección 2.2.3"
- §1.4, objetivo 1: "Sección 2.2, 2.3, 4.2–4.3"
- §1.4, objetivo 2: "verificación y validación, **Sección 2.9**"
- §1.7: describe el contenido completo de nueve subsecciones (2.1–2.9) de un capítulo que no existe

No se puede evaluar coherencia, consistencia terminológica ni completitud de citas de un capítulo que no está escrito. **Este es el hallazgo de mayor prioridad de toda la auditoría**: no se puede defender con rigor un Capítulo 4 que dice "esto ya se formalizó en el Capítulo 2" cuando el Capítulo 2 no existe. Antes de continuar con desarrollo de modelo (la pregunta explícita del encargo — "qué falta antes de poder avanzar con credibilidad al desarrollo del modelo"), este es el elemento bloqueante número uno.

**Acción prioritaria:** redactar el Capítulo 2 a partir de la Parte B de `../docs/planificacion/Step 4 Thesis.md`, con el mismo estándar de trazabilidad código-a-prosa ya aplicado en los Capítulos 3 y 4 (cada afirmación teórica anclada a una cita primaria, no a una paráfrasis). Dado el volumen (9 subsecciones), se recomienda tratarlo como un encargo propio, no como un añadido rápido.

---

## 4. Capítulo 3 — Estado del Arte

Este capítulo es, en general, el más sólido del borrador: la Sección 3.7 es honesta hasta el punto de la autocrítica (reconoce que el código actual solo reproduce L2D a nivel Sprint 1, sin ninguno de los tres ejes de novedad todavía), y la mayoría de las cifras cuantitativas citadas de L2D [23] se verificaron contra la fuente primaria (arXiv:2010.12367, texto completo, incluyendo el material suplementario) durante esta auditoría, con el siguiente resultado:

| Afirmación (../../archive/capitulos/Chapter3_final.md, §3.2) | Verificación contra fuente primaria | Veredicto |
|---|---|---|
| "17.7 % (6×6) a 29.2 % (30×20) sobre instancias generadas" | Tabla S.1: 6×6=17.7 %, 10×10=22.3 %, 15×15=26.7 %, 20×20=29.0 %, 30×20=29.2 % | **Correcto** |
| "22.1 %/9.4 % al evaluarse en instancias de 50×20/100×20" | Tabla S.2, columna "Ours(30×20)": 50×20=22.1 %, 100×20=9.4 % | **Correcto** |
| "87.5 % de las instancias DMU evaluadas" | Texto: "outperforms baselines over 87.5 % (35 out of 40) of these instances" | **Correcto** |
| "13.6 % a 26.0 % sobre el benchmark de Taillard" | Tabla S.3 (columna "Ours" completa, las 7 bandas de tamaño Ta15×15 a Ta100×20): mínimo real = 13.6 % (100×20), **máximo real = 33.6 % (30×20)**, no 26.0 % — 26.0 % es el valor de la banda Ta15×15, un punto intermedio, no el extremo superior del rango | **[MODERADO] Impreciso — subestima el rango real** |
| "superando a las PDR clásicas en el **92 %** de las instancias detalladas de Taillard" | No se localizó ninguna cifra de "92 %" asociada a Taillard en el texto completo ni en las tablas S.3/S.5/S.6. Las únicas cifras de "% de instancias superadas" encontradas en el paper son 87.5 % y 77.5 %, y ambas corresponden al benchmark **DMU**, no a Taillard | **[CRÍTICO] No verificable — posible atribución errónea de benchmark** |
| "muy por debajo de la brecha de la mejor PDR (**33.7–54.9 %**) en esos mismos tamaños [50×20/100×20]" | Tabla S.2: la mejor PDR (MOPNR, la de menor brecha) es 33.7 % en 50×20 y **14.7 %**, no 54.9 %, en 100×20. El valor 54.9 % corresponde a SPT — la **peor** PDR, no la mejor — en 50×20. El rango correcto de "la mejor PDR" a través de ambos tamaños es 14.7 %–33.7 % | **[CRÍTICO] Dato transcrito incorrectamente — mezcla la peor PDR con la mejor** |

### [CRÍTICO] C3-1 / C3-2. Dos cifras cuantitativas de L2D no resisten la verificación directa contra la fuente primaria

Ambas están en el mismo párrafo de apertura de §3.2 (el párrafo más citado y con más probabilidad de aparecer también, parafraseado, en el Capítulo 1 o en una futura introducción de artículo). Dado que esta tesis se autodefine explícitamente por su rigor de verificación bibliográfica ("esta tesis prefiere una brecha de verificación declarada a un dato inventado", nota de trazabilidad del propio capítulo), estas dos imprecisiones son más dañinas de lo que serían en un capítulo sin esa declaración de estándar. Nótese la diferencia con lo que ya cubrió `../../archive/Bibliographic_Verification_PRISMA_GapUpdate.md`: ese documento verificó **metadatos** de referencia (DOI, autores, venue); esta auditoría verificó las **cifras citadas en el cuerpo del texto**, un nivel de chequeo distinto y complementario que no estaba cubierto.
**Acción:** corregir "13.6 % a 26.0 %" → "13.6 % a 33.6 %" (o, si se prefiere reportar el mismo rango de tamaños que Taillard 15×15–30×20 sin las bandas grandes, aclarar explícitamente qué subconjunto de bandas se está citando); eliminar o sustituir la cifra "92 %" por una atribuida correctamente a DMU (87.5 %/77.5 %) o remover la oración si no se localiza una cifra equivalente para Taillard específicamente; corregir "33.7–54.9 %" → "14.7–33.7 %" (mejor PDR) o, si la intención original era citar el rango completo de PDR (mejor a peor), reformular la oración para que diga explícitamente "frente al rango completo de PDR clásicas (14.7–54.9 %)" en vez de atribuirlo a "la mejor PDR".

### [MENOR] C3-3. Nombre de sección con terminología mixta ("perturbación" vs. "disrupción")

Ver hallazgo transversal T-1 (§6).

---

## 5. Capítulo 4 — Metodología

### [CRÍTICO] C4-1. El capítulo mezcla dos capas temporales de implementación sin marcarlo en la estructura

Las Secciones 4.1–4.3 (`../../archive/capitulos/Chapter4_sections_4.1_4.3.md`) y 4.2 (`../../archive/capitulos/Section4.2.md`) documentan, de forma explícita y honesta, el estado del código **al cierre del Sprint 1**: JSSP estático, sin averías, sin FJSP, ocho características de nodo, cabeza de acción aplanada, seis PDR. Las Secciones 4.4–4.6, redactadas en sesiones posteriores de este mismo proyecto de tesis, ya **asumen como validado** el generador de disrupciones (4.5, correspondiente a `VALIDATION_WEEK6.md`), el arnés de nueve PDR + GA + CP-SAT (4.6, correspondiente a `BASELINES_RESULTS_WEEK9/12.md`) y la arquitectura heterogénea (4.4, todavía no implementada según el propio `../../archive/capitulos/Section4.4.md`, que la describe como "arquitectura objetivo... la que se implementa en el sprint siguiente"). Es decir: dentro del mismo capítulo, 4.3 dice "no hay averías todavía" y 4.5 dice "aquí está el generador de averías ya validado estadísticamente (KS p=0.929)" — sin ninguna oración de transición entre 4.3 y 4.4 que le indique al lector que está cruzando una frontera temporal de dos a tres sprints de desarrollo.

Cada sección, leída aisladamente, es internamente honesta (todas declaran su propio estado de implementación con precisión). El problema es de **montaje**, no de contenido: un comité que lea el capítulo de corrido experimentará una discontinuidad narrativa no señalada. La Sección 4.2 incluso contiene una inconsistencia temporal interna: su nota de trazabilidad remite a `../docs/especificaciones/DynamicEnv_TechSpec.md` como "documento técnico complementario" para una extensión **futura**, cuando, en el momento en que se redactó posteriormente la Sección 4.5, esa especificación ya se había implementado y validado — es decir, 4.2 sigue hablando en futuro de algo que, en el estado real del proyecto al momento de esta auditoría, ya es pasado.

**Acción:** añadir un párrafo de apertura al Capítulo 4 completo (antes de 4.1) que declare explícitamente la cronología de sprints y qué secciones documentan qué corte temporal — por ejemplo: "las Secciones 4.1–4.3 documentan el estado del código al cierre del Sprint 1 (semana 1); las Secciones 4.4–4.6 formalizan, respectivamente, la arquitectura objetivo del sprint siguiente y los resultados ya validados de los Sprints 6 y 9–12". Esto convierte una discontinuidad no señalada en una estructura deliberada y defendible — el contenido ya lo permite, solo falta la señalización explícita.

### [MODERADO] C4-2. Lista de PDR: 4.1 dice "seis reglas", 4.6 dice "nueve reglas" — consistente con C4-1 pero vale la pena señalarlo aparte

Sección 4.1, componente 5: "**Seis** reglas de despacho por prioridad (aleatoria, SPT, LPT, FIFO, MWKR, MOPNR)". Sección 4.6.1: "las **nueve** reglas requeridas por el diseño experimental" (añade LWKR, LOR, EDD, CR). Ambas cifras son correctas para su propio corte temporal (Sprint 1 vs. especificación completa de B6), y 4.6.1 incluso explica con precisión cuáles de las seis originales quedan "congeladas" por incompatibilidad con llegadas dinámicas. No es un error de dato, es la misma discontinuidad de C4-1 manifestada en un número concreto — se lista aparte porque es el ejemplo más fácil de citar ante un comité si preguntan "¿cuántas PDR implementan?".

### [MODERADO] C4-3. La Sección 4.4 no tiene lista de referencias, a diferencia de sus secciones hermanas

`../../archive/capitulos/Section4.4.md` cita explícitamente a Xu et al. (2019, GIN), Song et al. (IEEE TII 2023), Zhang et al. (L2D, 2020) y Huang & Ontañón (2022, action masking) en el cuerpo del texto, pero — a diferencia de `../../archive/capitulos/Chapter4_sections_4.1_4.3.md`, `../../archive/capitulos/Section4.2.md` y `../../archive/capitulos/Chapter3_final.md`, que sí cierran con una sección "Referencias empleadas en este capítulo/esta sección" — no incluye ninguna lista de referencias formal al final. Esto no significa que las citas sean inválidas (los números de referencia [23], [25], [19], [52] coinciden con los ya definidos en el Capítulo 3), pero rompe el patrón de completitud bibliográfica por sección que el resto del capítulo sí sigue, y dificulta la compilación automática de la bibliografía si el flujo de ensamblaje depende de listas por sección.
**Acción:** añadir a `../../archive/capitulos/Section4.4.md` el mismo bloque "Referencias empleadas en esta sección" que ya tienen 4.1–4.3 y 4.2, reutilizando los números ya asignados en el Capítulo 3.

### [MODERADO] C4-4. Verificar el mismo patrón (lista de referencias por sección) en 4.5 y 4.6 antes de la entrega

No se releyó el bloque final de `../../archive/capitulos/Section4.5.md` y `../../archive/capitulos/Section4.6.md` en esta ronda de auditoría con el mismo nivel de detalle que 4.4; dado que 4.4 sí presentó el hueco de C4-3, se recomienda una pasada de verificación mecánica (buscar la cadena "## Referencias" en los tres archivos) antes de considerar el capítulo cerrado.

### [MENOR] C4-5. Numeración de subsección faltante

`../../archive/capitulos/Chapter4_sections_4.1_4.3.md` salta de "4.1.1 Descripción del diagrama de arquitectura" a "4.1.3 Justificación de la elección arquitectónica" — no existe una subsección "4.1.2" en el archivo. Puede ser deliberado (4.1.2 reservada para una futura subsección de hiperparámetros de entrenamiento, por ejemplo) o un salto accidental de numeración; en cualquier caso, conviene resolverlo explícitamente antes de la versión final, porque un hueco de numeración sin explicación es lo primero que un revisor de forma señala.

---

## 6. Capítulo 5 — Diseño Experimental

Este es, junto con el Capítulo 3, el capítulo mejor anclado a evidencia empírica real: la Sección 5.1.4 hace exactamente lo que un pre-registro riguroso debe hacer — corrige explícitamente la expectativa teórica original ("la mejor PDR" se asumía MWKR por default de la literatura) con el dato observado (es FIFO, 17.80 % de RPD media, `../docs/planificacion/BaselineAnalysis.md`), y lo hace **antes** de generar resultados de la política DRL, exactamente el principio anti-HARKing que la propia sección invoca. Esta es una fortaleza metodológica real que vale la pena señalar explícitamente en la defensa, no solo un dato correcto.

### [CRÍTICO] Ver C1-4 (H1 exige MK, que E2/Tabla 5.1.3 marca como bloqueado) — la inconsistencia vive en la frontera Cap. 1/Cap. 5, se referencia aquí para que quede junto al resto del capítulo.

### [MODERADO] C5-1. Bloques de instancias "planificados" en 5.1.1 no tienen una fecha ni un criterio de decisión explícito para ejecutarse

La Tabla 5.1.1 marca los bloques "Generalización extendida" y "Generalización de gran escala" como **No ejecutado**, con una nota de que la ejecución de `la11`–`la40` es "un prerequisito... no un paso que pueda omitirse" (§5.1.2). Esto es metodológicamente correcto, pero el capítulo no fija un criterio de **cuándo** se considera que el presupuesto de cómputo (Sección 1.5, GPU única, nueve meses) permite o no absorber el bloque de gran escala (`ta`, hasta 2000 operaciones). Sin ese criterio explícito, existe el riesgo de que la decisión de incluir o no `ta01+` se tome de forma ad hoc más adelante, lo cual sí sería una desviación real del principio de pre-registro que el resto del capítulo defiende con cuidado.
**Acción:** añadir a 5.1.2 un criterio de decisión explícito y cuantitativo (p. ej., "se ejecuta el bloque `ta` completo si el tiempo total medido en el bloque `la11`–`la40` proyecta un presupuesto de cómputo total por debajo de X horas-GPU; en caso contrario, se ejecuta un subconjunto estratificado de N instancias por banda de tamaño, fijado antes de ver los resultados").

---

## 7. Hallazgos transversales (aplican a varios capítulos)

### [MODERADO] T-1. Terminología: "perturbación" domina, pero "disrupción" se cuela en frases clave sobre el mecanismo central de la tesis

Conteo verificado sobre los nueve archivos del borrador: "perturbación/perturbaciones" aparece ~48 veces frente a solo 5 apariciones de "disrupción/disrupciones", concentradas en `../../archive/capitulos/Chapter1.md` (3) y `../../archive/capitulos/Chapter3_final.md` (3, incluida la síntesis de cierre §3.7). El patrón más delicado es que la frase que nombra la **contribución metodológica central** de la tesis aparece en ambas variantes según el archivo: "recompensa consciente de la **perturbación**" (Capítulo 1 §1.2 punto 2, mayoritario en todo el borrador) frente a "recompensa consciente de **disrupciones**" (Capítulo 3 §3.7, cierre). Dado que este es el nombre propio del mecanismo que la tesis reclama como novedad (parámetro κ), la inconsistencia terminológica en su propio nombre es más grave que una simple variación estilística.
**Acción:** fijar "perturbación" como término único (ya es el dominante, 90 % de las apariciones) y hacer una pasada de reemplazo global de "disrupción(es)" → "perturbación(es)" en los cuatro puntos donde aparece, salvo si se decide, de forma deliberada y señalada, reservar "disrupción" específicamente para citar la terminología original de Lassoué et al. [38]/Zhang et al. [40] cuando se les referencia directamente.

### [MENOR] T-2. Notación LaTeX de subíndices con coma vs. punto y coma

Se observa uso consistente de $O_{j,h}$, $M_{j,h}$, $p_{j,h}$ a lo largo de los capítulos 1, 4.4 y 4.6 — no se detectó inconsistencia real aquí; se incluye como verificación positiva más que como hallazgo, dado que la notación matemática es exactamente el tipo de inconsistencia silenciosa que un comité detecta rápido y que en este borrador **no** ocurre.

### [MENOR] T-3. Bibliografía verificada vs. bibliografía citada en el cuerpo: dos niveles de verificación distintos, ninguno reemplaza al otro

`../../archive/Bibliographic_Verification_PRISMA_GapUpdate.md` (documento ya entregado en una ronda anterior) verifica **metadatos** de [1]–[34] y hace una búsqueda proxy PRISMA para [35]–[51]. Esta auditoría verificó, adicionalmente y por primera vez, **cifras citadas textualmente** de la fuente [23] (L2D) contra el PDF completo, encontrando los dos problemas de C3-1/C3-2. Se recomienda dejar constancia explícita en el propio Capítulo 3 (nota de trazabilidad) de que existen dos niveles de verificación distintos y que el segundo (cifras en el cuerpo del texto) solo se ha aplicado hasta ahora a la referencia [23]; las demás fuentes numéricas del capítulo (Tabla 3.3, filas [24], [38], [40], [42], [46], etc.) **no** han pasado por este mismo nivel de verificación de cifra-contra-fuente-primaria y deberían pasar por él antes de la sustentación, dado el estándar de rigor que el propio capítulo se autoimpone.

---

## 8. Lista de acciones priorizada

### Crítico (bloqueante antes de avanzar a desarrollo de modelo)

1. Redactar el Capítulo 2 (Marco Teórico) como documento propio a partir de la Parte B de `../docs/planificacion/Step 4 Thesis.md`, con el mismo estándar de trazabilidad que ya tienen los Capítulos 3 y 4. (C2-1)
2. Resolver la brecha entre la promesa de Cap. 1 (ecuación κ "ya en Sección 4.3") y el contenido real de 4.3.3 (recompensa plana, sin κ): añadir el apartado de extensión prevista o reescribir Cap. 1 para no presentarlo como ya resuelto. (C1-1)
3. Corregir las tres cifras de L2D con problema de verificación en ../../archive/capitulos/Chapter3_final.md §3.2: rango Taillard (13.6–26.0 % → 13.6–33.6 %), la cifra "92 % de instancias Taillard" (no localizada, probable confusión con DMU 87.5 %/77.5 %), y "33.7–54.9 % de la mejor PDR" (→ 14.7–33.7 %, o reformular como "rango completo de PDR"). (C3-1, C3-2)
4. Resolver la contradicción entre H1 (exige resultados sobre MK) y la Tabla 5.1.3 (E2/MK marcado como bloqueado/fuera de alcance) antes de congelar el pre-registro. (C1-4)
5. Añadir un párrafo de apertura al Capítulo 4 que declare explícitamente la cronología de sprints que separa 4.1–4.3 (Sprint 1) de 4.4–4.6 (arquitectura objetivo + sprints 6/9–12 ya validados). (C4-1)
6. Formalizar en el Capítulo 4 (Sección 4.6 o una nueva subsección) o retirar del objetivo 5 del Capítulo 1 la promesa de reproducir L2D y Song et al. como líneas base neuronales — actualmente no aparece en ninguna sección de metodología ni en la matriz experimental E0–E8. (C1-3)

### Moderado

7. Corregir las referencias cruzadas rotas del Capítulo 1: "Sección 4.2/5.2" → "Sección 5.2" (RI); "Sección 4.7/D7" → "Sección 5.1.4"; objetivo 1 debe citar también la Sección 4.4, no solo 4.2–4.3. (C1-2)
8. Alinear el objetivo 3 del Capítulo 1 ("comparando variantes GIN y GAT") con el estado real de la Sección 4.4.2 (GAT descartado como configuración por defecto, reservado como ablación futura). (C1-5)
9. Añadir sección de referencias a `../../archive/capitulos/Section4.4.md`, siguiendo el patrón ya usado en 4.1–4.3 y 4.2; verificar que 4.5 y 4.6 también la tengan. (C4-3, C4-4)
10. Fijar un criterio de decisión explícito y cuantitativo para determinar si el bloque de gran escala (`ta`, hasta 2000 operaciones) se ejecuta o no, antes de que la decisión se tome de forma ad hoc. (C5-1)
11. Estandarizar la terminología del mecanismo de recompensa: "perturbación" en todo el documento, incluida la síntesis de cierre del Capítulo 3. (T-1)
12. Extender la verificación de cifra-contra-fuente-primaria (aplicada en esta ronda solo a [23]) al resto de referencias numéricas citadas en la Tabla 3.3 del Capítulo 3, dado el estándar de rigor que el propio capítulo se autoimpone. (T-3)
13. Resolver la inconsistencia de conteo de PDR (6 en 4.1 vs. 9 en 4.6) con la misma nota de cronología del punto 5. (C4-2)

### Menor

14. Resolver el hueco de numeración "4.1.2" faltante en `../../archive/capitulos/Chapter4_sections_4.1_4.3.md`. (C4-5)
15. Confirmar identificador arXiv y lista de autores de Gymnasium [55], ya autoseñalado con [VERIFICAR] en dos archivos distintos.
16. Completar los dos [VERIFICAR] de §1.1 (penetración Industria 4.0, frecuencia empírica de disrupciones) o mantenerlos explícitamente como limitación reconocida en el capítulo de discusión.

---

## 9. Respuesta directa a las cinco preguntas del encargo

**Coherencia lógica entre capítulos:** parcialmente. Cap. 1 → Cap. 3 fluye bien (las preguntas de investigación nacen limpiamente del análisis de brechas). Cap. 3 → Cap. 4 tiene la discontinuidad de capas temporales ya descrita (C4-1). Cap. 4 → Cap. 5 fluye bien y de hecho contiene la mejor práctica metodológica del borrador (el ajuste pre-registrado de H1 en 5.1.4). El eslabón roto real es Cap. 1 ↔ Cap. 2 (capítulo inexistente citado como si existiera).

**Consistencia de estado/acción/recompensa entre teoría, metodología y código:** la Sección 4.3 está perfectamente alineada con el código real (auditado línea por línea contra `envs/jssp_env.py`). El problema no es metodología-vs-código, sino Cap.1-vs-metodología: Cap. 1 promete una recompensa con κ que la Sección 4.3 (fiel al código) todavía no tiene.

**Consistencia terminológica en español:** alta en general (notación matemática, "cronograma" para *schedule*, "avería" para *breakdown*); el punto débil concreto es perturbación/disrupción en el nombre del mecanismo central (T-1).

**Completitud de citas:** sólida en Cap. 3 y 4 (aparato [VERIFICAR] ya disciplinado); esta auditoría añadió una capa nueva de verificación (cifra contra fuente primaria, no solo metadato) y encontró dos problemas concretos en L2D que deben corregirse antes de que ese párrafo se reutilice en una publicación o en la defensa oral.

**Qué falta antes de avanzar con credibilidad al desarrollo de modelo:** en orden — (1) el Capítulo 2, (2) resolver la contradicción H1/MK antes de que el pre-registro se congele formalmente, (3) decidir si κ se implementa ya o se pospone con esa postergación señalada explícitamente en Cap. 1, y (4) la nota de cronología de sprints en la apertura del Capítulo 4. Ninguno de estos cuatro puntos requiere reescribir contenido ya validado; todos son adiciones o correcciones de encuadre.
