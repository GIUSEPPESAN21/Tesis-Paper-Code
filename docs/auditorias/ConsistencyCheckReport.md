# Informe de consistencia — `../../archive/capitulos/Thesis_FullDraft_v1.md`

**Alcance:** auditoría de ensamblaje realizada al compilar `../../archive/capitulos/Chapter1.md`, `../../archive/capitulos/Chapter3_final.md`, `../../archive/capitulos/Chapter4_sections_4.1_4.3.md` + `Section4.2/4.4/4.5/4.6.md`, `../../archive/capitulos/Section5.1_5.2.md` + `../../archive/capitulos/Section5.3.md`, `../../thesis/Thesis.md`, `../../thesis/Thesis.md` y `../../archive/capitulos/Chapter8.md` en un único documento, más `../../thesis/FIGURES_INDEX.md` para la inserción de figuras. No cubre anexos (`../../archive/capitulos/AppendixB.md`, `../../thesis/PreRegistration.md`, tablas de óptimos) ni portada/agradecimientos — fuera del alcance explícito de "Capítulo 1 a 8" del encargo (ver §6 de este informe).

**Resumen ejecutivo:** se encontraron **2 hallazgos estructurales mayores** (un capítulo completo ausente, y una fórmula central citada en tres capítulos posteriores sin definición formal previa), **1 contradicción interna directa** (Capítulo 8 se autocontradice citando el capítulo ausente como evidencia de cumplimiento), **38 marcas `[VERIFICAR]` sin resolver**, y **3 casos de deriva terminológica/de alcance** entre capítulos — dos de ellos ya mitigados en el borrador integrado mediante notas editoriales explícitas (no correcciones silenciosas del contenido original). Ningún hallazgo invalida los resultados numéricos de los Capítulos 6–8; todos son problemas de completitud documental o de ensamblaje.

---

## 1. Hallazgo estructural mayor #1 — el Capítulo 2 (Marco Teórico) no existe

**Severidad: alta. Bloqueante para la sustentación, no para la validez de los resultados.**

Ningún archivo `Chapter2*.md` ni equivalente existe en el corpus de trabajo. El contenido teórico correspondiente (notación α|β|γ, formulación MDP/semi-Markov, fundamentos de PPO, invariancia a permutaciones de GNN, PDR, CP-SAT/GA, DES) existe únicamente como notas internas de framework (Parte B de `../docs/planificacion/Step 4 Thesis.md`), sin redactar en prosa de tesis ni con aparato de citas propio. Este hallazgo ya había sido identificado de forma independiente en `../docs/auditorias/AuditoriaCapitulos1-5.md` ("no hay un Capítulo 2 redactado en ninguna parte del proyecto... hallazgo de mayor prioridad de toda la auditoría") — esta compilación lo confirma y lo hace visible estructuralmente por primera vez dentro del propio documento de tesis, en vez de en un documento de auditoría separado.

**Referencias cruzadas rotas resultantes (ubicaciones exactas en `../../archive/capitulos/Thesis_FullDraft_v1.md`):**

| Línea aprox. | Capítulo de origen | Cita rota |
|---|---|---|
| 198 | Cap. 1, §1.2 | "véase la Sección 2.2 para la formulación completa"; "la extensión estocástica descrita en la Sección 2.2.3" |
| 236–237 | Cap. 1, §1.4 (objetivos 1–2) | "Sección 2.2, 2.3, 4.2–4.3"; "verificación y validación, Sección 2.9" |
| 271 | Cap. 1, §1.7 | Nueve referencias a Secciones 2.1–2.9 en la descripción de la estructura del documento |
| 497 | Cap. 4, §4.3.1 | "la especificación teórica original de la tesis (marco conceptual, Capítulo 1–2)" |
| 1589 | Cap. 8, Tabla 8.1 (objetivo 1) | "Capítulo 2 (formulación completa)" citado como **evidencia de que el objetivo 1 está "Cumplido"** — ver hallazgo #3, más abajo, que trata esta línea específicamente |

**Tratamiento aplicado en este borrador:** se insertó un Capítulo 2 explícito, marcado sin ambigüedad como *placeholder* (no como contenido real), que documenta el hallazgo, enumera las nueve subsecciones esperadas según la estructura ya anticipada por el Capítulo 1 §1.7, y remite a la Parte B de `../docs/planificacion/Step 4 Thesis.md` como fuente para su redacción futura. Esto evita dos errores peores: (a) omitir el capítulo silenciosamente, dejando que el lector descubra la ausencia solo al llegar a una referencia rota, o (b) fabricar contenido teórico no verificado para "rellenar" el hueco.

**Acción recomendada:** redactar el Capítulo 2 como encargo propio antes de la sustentación — ya identificado como la acción de mayor prioridad en `../docs/auditorias/AuditoriaCapitulos1-5.md` y reafirmado aquí.

---

## 2. Hallazgo estructural mayor #2 — la fórmula de recompensa consciente de κ nunca se define formalmente en el Capítulo 4

**Severidad: alta. Afecta la trazabilidad de H4, aunque no su resultado.**

Los Capítulos 6, 7 y 8 citan repetidamente la fórmula $r_t = (LB_{t-1}-LB_t) - \kappa\cdot\Delta_{\text{inest}}$ como el mecanismo de recompensa central de H4 (p. ej. `../../archive/capitulos/Thesis_FullDraft_v1.md`, línea 1492, Capítulo 7 §7.4.3; también líneas 1477, 1514, 1540, 1560). Sin embargo, **el Capítulo 4 tal como está compilado nunca introduce esta fórmula**: la Sección 4.3.3 formaliza únicamente la recompensa densa del Sprint 1 ($r_t=(H(s_t)-H(s_{t+1}))\times\text{reward\_scale}$, sin término de $\kappa$), y las Secciones 4.4–4.6 (arquitectura objetivo, perturbaciones, líneas base) no contienen ninguna subsección dedicada a extender esa recompensa con el término de inestabilidad. La única mención de $\kappa$ en todo el Capítulo 4/5 es una celda de la matriz experimental (Tabla 5.1.3, fila "E7 Recompensa consciente"), que nombra el factor pero no define su fórmula.

En otras palabras: el objetivo específico 4 de la Sección 1.4 ("diseñar y ablacionar una función de recompensa consciente de la perturbación") se **ejecuta y se reporta** en los Capítulos 6–8, pero su **diseño formal** —la parte metodológica que correspondería al Capítulo 4— falta en el corpus compilado, de forma análoga (aunque menos grave) a la ausencia del Capítulo 2. Es probable que esta formalización exista en el código (`envs/dynamic_jssp_env.py`, ya referenciado en Chapter 8) o en un documento técnico complementario no incluido en el alcance "Capítulo 1 a 8" de este encargo (p. ej. `../docs/especificaciones/DynamicEnv_TechSpec.md`), pero no está en ningún capítulo compilado.

**Acción recomendada:** añadir una subsección (p. ej. "4.4.5 Recompensa consciente de la perturbación") que formalice la ecuación, defina $\Delta_{\text{inest}}$ explícitamente (la Nomenclatura de este borrador ya la glosa como "penalización de inestabilidad (desviación del plan base)", pero eso no sustituye su definición matemática completa) y documente la verificación de no-operación en $\kappa=0$ (`TestKappaReward`, ya citada en el Capítulo 7 sin haberse presentado antes).

---

## 3. Contradicción interna directa — Tabla 8.1 del Capítulo 8 cita el Capítulo 2 (ausente) como evidencia de cumplimiento

**Severidad: media-alta. Es una contradicción lógica dentro del propio documento, generada por el ensamblaje.**

La Tabla 8.1 (línea 1589) evalúa el objetivo específico 1 de la Sección 1.4 ("Formalizar el FJSP dinámico y estocástico como MDP...") como **"Cumplido"**, citando como evidencia "Capítulo 2 (formulación completa)". Esta celda se escribió, en su momento, asumiendo la existencia del Capítulo 2 — exactamente el mismo supuesto no verificado que el Capítulo 1 hace seis veces (§1 de este informe). Con el Capítulo 2 confirmado como ausente, **esta celda es autocontradictoria**: no puede citarse como "cumplido, ver Capítulo 2" un objetivo cuya única evidencia textual formal no existe en el documento.

**Nota importante:** esto no significa que el objetivo 1 esté necesariamente incumplido — la formalización MDP probablemente existe en el código (`envs/dynamic_jssp_env.py`, `models/policy.py`, ya citados en esa misma celda) y en las notas informales de la Parte B de `../docs/planificacion/Step 4 Thesis.md`. El problema es específicamente que la **evidencia documental citada** (Capítulo 2) no existe, no que el trabajo subyacente no se haya hecho.

**Acción recomendada:** corregir la Tabla 8.1 (Capítulo 8) para que la columna "Evidencia" del objetivo 1 cite la Parte B de `../docs/planificacion/Step 4 Thesis.md` y el código (`envs/`, `models/`) en lugar de "Capítulo 2", y opcionalmente reclasificar el estado de "Cumplido" a "Cumplido en el código; pendiente de formalización en prosa de tesis (ver Capítulo 2)" — coherente con la distinción que el propio Capítulo 3 (§3.7) ya hace en otro contexto entre "lo que el código implementa" y "lo que el documento de tesis formaliza en prosa."

---

## 4. Marcas `[VERIFICAR]` sin resolver (38 en total)

Todas las ocurrencias localizadas en los ocho capítulos compilados. Ninguna se eliminó ni se resolvió durante el ensamblaje — resolverlas requiere acceso a fuentes primarias (IEEE Xplore, arXiv, confirmación de listas de autores) fuera del alcance de una compilación editorial.

**Capítulo 1** (3 ocurrencias): una en la nota de trazabilidad de apertura (genérica, ya explicada); una sobre "cifra específica de adopción... de tecnologías 4.0 en manufactura discreta" (§1.1); una sobre "cita de un estudio empírico sobre frecuencia de disrupciones en manufactura discreta" (§1.1) — ambas afirmaciones motivacionales sin respaldo estadístico primario verificado.

**Capítulo 3** (33 ocurrencias): concentradas en la Tabla 3.1 (§3.3) y la narrativa de §3.2, marcando benchmarks, algoritmos, listas de autores y cifras de resultados no confirmadas para las referencias [25], [26], [36]–[39], [41], [43], [44], [47], [48], [51]. Ya priorizadas en la propia sección "Acciones pendientes antes de la sustentación" del capítulo (4 acciones concretas, incluida la obtención de texto completo de [25]/[26], directamente relevante para la Tabla 7.1 del Capítulo 7).

**Capítulo 4** (2 ocurrencias): la identidad exacta y lista de autores de la referencia [55] (Gymnasium), marcada de forma idéntica en dos puntos distintos del capítulo (§4.1/4.3 y §4.2) — **posible duplicación de la misma verificación pendiente**, candidata a resolverse una sola vez y propagarse a ambos puntos.

**Capítulo 7** (2 ocurrencias, cualitativamente distintas de las anteriores): no son marcas heredadas sin resolver, sino **hallazgos de la propia auditoría del Capítulo 7** que documentan por qué una cifra de Song et al. [25] no puede tratarse como verificada (§7.2, §7.2.1) — es decir, el capítulo *usa* la disciplina `[VERIFICAR]` correctamente para señalar su propia limitación, no la hereda sin resolver.

**Capítulos 2 (placeholder), 5, 6, 8:** sin marcas `[VERIFICAR]` — consistente con que el Capítulo 6 reporta datos ya ejecutados y verificados, y el Capítulo 8 sintetiza sin introducir afirmaciones nuevas no verificadas.

---

## 5. Deriva terminológica y de alcance entre capítulos

### 5.1 Capítulo 4, Secciones 4.1–4.3 vs. 4.4–4.6: salto de alcance Sprint-1 → arquitectura objetivo (mitigado)

Las Secciones 4.1–4.3 documentan, en tiempo presente, un GIN **homogéneo** sobre JSSP **estático** sin averías ni FJSP (el pipeline real del Sprint 1). Las Secciones 4.4–4.6, inmediatamente después, formalizan —también en tiempo presente— un grafo **heterogéneo** operación-máquina, cabezas actor-crítico independientes del tamaño, y el entorno dinámico completo, sin transición explícita en el material original. Ya señalado en `../docs/auditorias/AuditoriaCapitulos1-5.md` como "un capítulo que, leído de corrido, dobla de tiempo verbal sin transición." **Mitigado en este borrador** mediante una nota editorial insertada entre ambos bloques (`../../archive/capitulos/Thesis_FullDraft_v1.md`, línea 614) que aclara la progresión Sprint-1 → versión final sin alterar el contenido técnico original de ninguna de las dos secciones.

### 5.2 Capítulo 5 completo: matriz experimental de la Semana 9–12 vs. la matriz final ejecutada (mitigado)

`../../archive/capitulos/Section5.1_5.2.md`/`../../archive/capitulos/Section5.3.md` describen un estado muy temprano del diseño experimental ($N=13$ instancias, límites de tiempo de pared escalonados por banda de tamaño no adoptados finalmente, criterio de H1 anclado literalmente a "RPD de FIFO < 17.80 %"), todos superados por `../../thesis/PreRegistration.md`/`../docs/planificacion/FinalExperimentMatrix.md` y por lo efectivamente reportado en el Capítulo 6 ($N=43$ instancias JSSP, límite único de 60 s, criterio de H1 = mejor PDR *por instancia*). **Mitigado** mediante una nota editorial al inicio del Capítulo 5 (línea 877) que documenta explícitamente las tres discrepancias concretas y remite al Capítulo 6 para el protocolo realmente ejecutado, preservando intacto el aparato formal de fórmulas (Wilcoxon, Friedman+Nemenyi, RPD, RI), que sigue siendo válido y es, de hecho, el que el Capítulo 6 aplica.

### 5.3 Nomenclatura `MOR` vs. `mopnr` (no mitigado — inconsistencia menor, no bloqueante)

El Capítulo 4 (§4.1, §4.6.1) se refiere a la regla de despacho "Most Operations Remaining" usando el nombre de código `mopnr` (heredado del módulo `baselines/dispatching_rules.py`), mientras que el Capítulo 6 (todas las tablas de resultados, p. ej. Tabla 6.2.1) y el Capítulo 7 usan consistentemente `mor`. Ambos nombres designan la misma regla (confirmado en la propia Sección 4.6.1: "una regla equivalente a MOR bajo el nombre `mopnr`"), por lo que no hay ambigüedad de contenido, pero un lector que busque "mopnr" en las tablas del Capítulo 6 no lo encontrará. **No se corrigió** en este borrador (alteraría nombres de columna citados directamente de archivos de datos reales); se recomienda unificar la nomenclatura de código a `mor` en una revisión futura del pipeline, o añadir una nota de equivalencia en la Tabla 4.6.1.

---

## 6. Fuera de alcance de esta compilación (declarado, no un defecto)

- **Anexos** referenciados por el Capítulo 1 (§1.7): formulación MILP completa, hiperparámetros y arquitectura del modelo (`../../archive/capitulos/AppendixB.md` existe como archivo separado), tablas extendidas de resultados, diagrama de flujo PRISMA oficial, documento de pre-registro (`../../thesis/PreRegistration.md` existe) y tablas de óptimos por instancia. Ninguno se integró en `../../archive/capitulos/Thesis_FullDraft_v1.md`, consistente con el encargo explícito de "Capítulo 1 a 8". Recomendación: una segunda pasada de compilación para anexos, una vez el Capítulo 2 exista.
- **Portada formal, agradecimientos, declaración de originalidad, actas de aprobación**: no generados — administrativos, no derivables de los capítulos fuente.
- **Renderizado de la Figura 4.1** (diagrama de arquitectura): permanece como descripción en prosa (Sección 4.1.1), tal como el propio Capítulo 4 ya señalaba como pendiente ("Acciones pendientes antes de la sustentación", punto 1). No se generó un diagrama real en este paso.

---

## 7. Correcciones de numeración aplicadas durante el ensamblaje (documentadas, no silenciosas)

| Elemento | Antes | Después | Justificación |
|---|---|---|---|
| Subsección del Capítulo 4 | `### 4.1.3 Justificación...` (saltaba de 4.1.1 a 4.1.3, sin 4.1.2 en ningún archivo fuente) | `### 4.1.2 Justificación...` | No existe ningún `4.1.2` en el resto del corpus que este renombrado pudiera romper; se verificó por búsqueda antes de renombrar. |
| Tablas sin numerar | Tabla comparativa de trabajos (Cap. 3, §3.3); tabla de brechas (Cap. 3, §3.6); tabla de claves de observación (Cap. 4, §4.3.1); tabla de síntesis final (Cap. 6); tabla de veredictos (Cap. 8, §8.4) | Tabla 3.1, Tabla 3.2, Tabla 4.1, Tabla 6.6.1, Tabla 8.3 | Asignadas siguiendo la convención ya usada en el resto del documento (`Capítulo.Sección.Índice`), para que el Índice de Tablas del frente del documento sea completo. |
| Ecuaciones sin numerar | Fórmulas centrales de recompensa (Cap. 4, §4.3.3), paso de mensajes (§4.4.2), política del actor (§4.4.3), averías/tiempos/llegadas (§4.5.1–4.5.3), RPD/RI (Cap. 5, §5.2.2), Wilcoxon/Friedman/CD (Cap. 5, §5.3) | Numeradas (4.1)–(4.9), (5.1)–(5.5) | Se numeraron las ecuaciones "titulares" de cada subsección (la fórmula que el texto cita después por nombre), no cada línea de derivación intermedia — convención estándar de tesis. |
| Figuras citadas sin insertar | Figuras 6.3.1(a–d), 6.3.2, 6.4.1(a–c), 6.5.1 (el Capítulo 6 original las describía en texto/pie de figura, con los archivos ya generados en `figures/` y catalogados en `../../thesis/FIGURES_INDEX.md`, pero sin insertar la imagen) | Insertadas como bloques `![...](../../figures/...)` inmediatamente después de cada pie de figura ya redactado | Las 9 imágenes (`Fig_6.3.1a`–`Fig_6.5.1`) existen en `figures/`, generadas y validadas (paleta *colorblind-safe*) según `../../thesis/FIGURES_INDEX.md`; se insertaron en las nueve ubicaciones correctas sin alterar los pies de figura ya aprobados en `../../thesis/Thesis.md`. |

---

## 8. Consistencia ya verificada, sin hallazgos (para que quede constancia de lo que también se revisó)

- **Estilo de citación:** IEEE numérico `[n]`, consistente en los ocho capítulos; los Capítulos 1, 6, 7 y 8 citan por número sin bibliografía local propia (correcto, dependen de la bibliografía maestra); el Capítulo 3 introduce `[1]`–`[51]` y el Capítulo 4 añade `[52]`–`[55]`; no se encontró ningún número de referencia usado en el texto sin entrada correspondiente en la lista consolidada, ni ninguna entrada de la lista sin uso en el texto.
- **Calibración de perturbaciones:** las cifras de la Tabla 4.5.1/§4.5.6 (Capítulo 4) coinciden exactamente, cifra por cifra, con la Tabla 6.1.2 (Capítulo 6) — sin deriva entre la especificación y el resultado reportado.
- **Convención de signo de $RI$:** consistente entre la definición de §5.2.2 (Capítulo 5) y su uso e interpretación en el Capítulo 6 (§6.4) y Capítulo 7 (§7.4.1) — positivo = peor, sin inversiones de signo entre capítulos.
- **Hipótesis H1–H5 y subpreguntas SP1–SP3:** nomenclatura estable desde su introducción en el Capítulo 1 (§1.3) hasta su uso en los Capítulos 6, 7 y 8, sin renombrados ni renumeraciones a lo largo del documento.
- **Cifras de H1–H5 citadas en el Capítulo 8** (síntesis, Tabla 8.3): verificadas contra la Tabla 6.0 del Capítulo 6 — coinciden exactamente en los cinco veredictos y sus estadísticos clave.
