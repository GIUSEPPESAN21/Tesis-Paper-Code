# Bitácora de la pasada final de pulido — `Thesis_FullDraft_v2.md` → `Thesis_FINAL.md`

**Alcance de esta pasada:** proofreading de calidad del español académico, verificación de cada cita contra la lista de referencias, verificación de cada referencia cruzada y numeración de figuras/tablas/ecuaciones, verificación de que el resumen/abstract refleja los resultados finales, y producción de un Markdown final listo para conversión a PDF/Word. No se reabrió ningún resultado experimental, estadística o veredicto de hipótesis — todos los cambios son de exposición, trazabilidad y consistencia estructural.

**Método:** lectura completa del documento (1763 líneas, ~45 400 palabras) más verificación programática (Python) de citas y de la lista de referencias, para no depender solo de una revisión visual en un documento de este tamaño.

---

## 1. Hallazgo estructural más importante: front matter faltante

`Thesis_FullDraft_v2.md` **no contenía** el Índice general, el Índice de figuras, el Índice de tablas ni la Nomenclatura y abreviaturas — los cuatro sí existían en `Thesis_FullDraft_v1.md`, pero se perdieron durante el script de ensamblaje de la v2 (`assemble_v2.py`), que reconstruyó el documento a partir de los capítulos fuente sin volver a insertar ese bloque de front matter. Esto no estaba señalado en `ConsistencyCheckReport.md` ni en `PreDefenseAudit.md` (ambos auditaron v1); es un hallazgo nuevo de esta pasada.

**Corrección aplicada:** se reconstruyó el front matter completo a partir de `Thesis_FullDraft_v1.md`, actualizado para reflejar el contenido nuevo de la v2:
- Índice general: se añadió la entrada "NOTA DE VERSIÓN — REGISTRO DE CAMBIOS (v1 → v2)" y la nueva Sección 5.4.
- Índice de tablas: se añadió la Tabla 6.2.3 (nueva en v2).
- Índice de figuras y Nomenclatura: verificados contra el contenido real de v2 (conteo de figuras y símbolos sin cambios) y reinsertados sin alteración de contenido.

## 2. Reordenamiento estructural del Capítulo 4

El Capítulo 4, tal como estaba ensamblado, presentaba las secciones fuera de orden numérico: **4.1 → 4.3 → 4.2 → 4.4** (con el bloque "Acciones pendientes antes de la sustentación" y la nota editorial de transición intercalados en los mismos puntos de la secuencia incorrecta). Esto es un artefacto de que las secciones 4.1–4.3 y 4.2 provenían de archivos fuente distintos (`Chapter4_sections_4.1_4.3.md` y `Section4.2.md`) concatenados en el orden en que se leyeron, no en orden lógico.

**Corrección aplicada:** se reordenó el capítulo a la secuencia correcta **4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6**, preservando el bloque "Acciones pendientes" y la nota editorial de transición inmediatamente después de 4.3 (su posición original relativa al final del contenido "4.1–4.3"). Se corrigió también el nivel de encabezado de la Sección 4.2, que estaba marcado como encabezado de capítulo (`#`) en lugar de encabezado de sección (`##`), inconsistente con 4.1/4.3/4.4. **Ningún texto se modificó ni se perdió** en el reordenamiento — se verificó programáticamente que el conjunto de líneas de contenido antes y después del reordenamiento es idéntico (recuento de palabras: 45 438 antes, 45 439 después, diferencia atribuible únicamente al propio encabezado corregido).

## 3. Sección 1.7 (Estructura del documento): tiempo verbal desactualizado

La Sección 1.7 seguía describiendo los Capítulos 6, 7 y 8 en tiempo futuro y como contenido no incluido ("El Capítulo 6 **reportará** los resultados... y **no se incluye** en el presente documento de fundamento"; "El Capítulo 7 **discutirá**..."; "El Capítulo 8 **presentará**...") — una redacción que tenía sentido cuando el Capítulo 1 se escribió como documento de fundamento previo a la ejecución experimental, pero que en el documento integrado final es factualmente incorrecta: los tres capítulos existen y están completos.

**Corrección aplicada:** se reescribió el párrafo en tiempo presente, describiendo el contenido real de cada capítulo (los ocho experimentos E0–E8, la comparación cuantitativa con L2D/Song et al., las amenazas a la validez, el contraste objetivos-vs-logros).

## 4. Autocontradicción en la Tabla 8.1 (objetivo 1) sobre el Capítulo 2

La Tabla 8.1 marcaba el objetivo 1 ("formalizar el FJSP... como MDP") como **Cumplido**, citando como evidencia el "Capítulo 2 (formulación completa)" — pero el propio documento declara explícitamente, en la nota de estado al inicio del Capítulo 2, que ese capítulo **no existe redactado en prosa de tesis**. Esta contradicción ya había sido detectada en `ConsistencyCheckReport.md` y en `PreDefenseAudit.md`, pero en ambos casos se dejó señalada sin resolver porque requería una decisión editorial (a qué evidencia real citar en su lugar).

**Corrección aplicada:** se cambió la celda de evidencia para citar la fuente real y verificable del cumplimiento — Capítulo 4, §4.3 (formulación MDP completa) y §4.4 (representación de grafo heterogéneo) — y se añadió una nota de consistencia explícita reconociendo que el objetivo se formuló originalmente esperando su fundamento en el Capítulo 2, que permanece pendiente.

## 5. Citas nunca usadas en el cuerpo del texto

Verificación programática: de las 55 referencias de la bibliografía, **8 nunca aparecían citadas por número** en el cuerpo del documento. De esas 8, 3 ([17], [18] y parte de [16]–[19]) sí estaban cubiertas por una cita de rango (`[16]–[19]`) no detectada por una búsqueda literal de corchetes individuales — no eran un problema real. Las **5 restantes** sí eran huérfanas:

| Ref. | Referencia | Corrección aplicada |
|---|---|---|
| [11] | OR-Library (Beasley) | Se citó explícitamente en el Capítulo 5 (§5.2.1), donde el texto ya mencionaba "los óptimos publicados en OR-Library" sin el corchete de cita. |
| [8] | Demirkol, Mehta y Uzsoy (benchmark DMU) | Se citó en el Capítulo 3 (§3.2), donde el texto ya mencionaba "las instancias DMU evaluadas" por L2D sin el corchete de cita — es exactamente la fuente de ese benchmark. |
| [15] | Mnih et al. (DQN) | Se añadió a la Sección 1.7, en la enumeración de fundamentos de aprendizaje por refuerzo profundo previstos para el Capítulo 2. |
| [4] | Carlier y Pinson (algoritmo exacto para JSSP) | Se añadió a la Sección 1.7, junto a la mención de métodos exactos y metaheurísticos. |
| [39] | Chen, Mei, Zhang y Zhang, "DyRo-MCTS" (2025) | No tenía un punto de inserción natural sin fabricar una afirmación nueva no verificada; se optó por **declarar explícitamente** en la lista "Acciones pendientes antes de la sustentación" del Capítulo 3 que esta referencia está localizada pero pendiente de integrarse a la narrativa — más honesto que forzar una cita superficial. |
| [44] | "RobustGANTT" | Mismo tratamiento que [39]; además, su identificador arXiv nunca se confirmó de forma independiente (ya marcado `[VERIFICAR]` en la bibliografía), por lo que se señaló explícitamente que no debe citarse en la sustentación hasta resolver esa verificación. |

Verificación final: **0 citas huérfanas** (citadas en el cuerpo pero sin entrada en la bibliografía) y **0 referencias sin ningún uso** en el cuerpo del texto (directo o por rango).

## 6. Otras correcciones menores

- **Carácter espurio:** se encontró y eliminó un carácter de espacio de ancho completo (U+3000, un artefacto de codificación) al final de una línea en la Sección 4.3.1 (nota de desviación arquitectónica).
- **Barrido de codificación:** se verificó el documento completo en busca de caracteres de control, BOM y espacios Unicode no estándar; no se encontró ningún otro caso.

## 7. Verificaciones realizadas sin encontrar errores

- **Numeración de ecuaciones:** (4.1)–(4.9) en el Capítulo 4 y (5.1)–(5.5) en el Capítulo 5 — ambas secuencias completas, sin saltos ni duplicados, consistentes con lo que el Índice general siempre había prometido.
- **Numeración de tablas y figuras:** se extrajeron programáticamente todas las declaraciones `**Tabla X.Y.**` y `**Figura X.Y.**` del documento (30 tablas, 6 entradas de figura incluyendo los grupos 6.3.1(a–d) y 6.4.1(a–c)) y se verificaron contra el nuevo Índice de tablas/figuras — coinciden exactamente.
- **Coherencia numérica resumen↔capítulos:** se verificaron programáticamente los 15 valores numéricos clave del RESUMEN/ABSTRACT (RPD de RL y de la mejor PDR en los tres brazos de H1, los cuatro *p*-valores, el rango de RPD de H2 por *tier*, los deltas de RPD nominal de H4, y 42/43 de H5) contra sus apariciones en el Capítulo 6 — los 15 coinciden exactamente (las diferencias de formato encontradas inicialmente, p. ej. `1.12\times10^{-8}` en el resumen contra `1.12×10⁻⁸` en las tablas del Capítulo 6, son la misma cifra en notación LaTeX vs. Unicode, el mismo estilo dual usado consistentemente en todo el documento, no un error).
- **Citas huérfanas:** cero citas en el cuerpo sin entrada correspondiente en la bibliografía (ver §5, arriba).

## 8. Fuera de alcance de esta pasada (declarado explícitamente, no una omisión)

- Los **28 marcadores `[VERIFICAR]`** restantes en el documento no se resolvieron — representan afirmaciones genuinamente no verificables contra una fuente primaria a la fecha de esta revisión (acceso bloqueado a Scopus/WoS/IEEE Xplore, identificadores de arXiv no confirmados, etc.). Resolverlos requiere acceso bibliotecario o contacto con autores, no una pasada editorial; están listados en las respectivas secciones de "Acciones pendientes antes de la sustentación" de los Capítulos 3 y 4.
- El **Capítulo 2 (Marco Teórico)** permanece sin redactar. Esto se reconoce explícitamente en el propio documento (nota de estado al inicio del Capítulo 2) y no es competencia de una pasada de proofreading — es la única acción de esfuerzo "alto" pendiente identificada en `PreDefenseAudit.md` §6.
- No se re-verificó ningún resultado estadístico ni número crudo contra los CSV originales — esa verificación ya se realizó en las auditorías previas (`Chapter7_FINAL.md`, `PreDefenseAudit.md`) y no se repite aquí.

---

## Notas para la conversión a PDF/Word

- El documento usa notación matemática LaTeX inline (`$...$`) y en bloque (`$$...$$`) de forma consistente en los Capítulos 4 y 5 (formulación MDP, modelos de perturbación, fórmulas estadísticas) — un conversor Markdown→PDF/Word debe soportar renderizado de LaTeX (p. ej. Pandoc con `--pdf-engine=xelatex` y un filtro de matemáticas, o un pipeline vía Word con un complemento de ecuaciones) para que estas se muestren correctamente.
- Las tablas están en formato Markdown estándar (`|---|---|`); se recomienda revisar el ajuste de ancho de columna tras la conversión, dado que varias tablas del Capítulo 6 (p. ej. Tabla 6.2.1, con 12 columnas) son anchas y pueden requerir orientación horizontal o reducción de tamaño de fuente en el documento final.
- Las seis referencias a figuras (`![...](figures/Fig_...)`) apuntan a la carpeta `figures/` ya presente en el proyecto (9 archivos PNG+PDF, confirmados en `FIGURES_INDEX.md`); deben copiarse junto con el Markdown si la conversión se hace fuera de esta carpeta de trabajo.
- Los encabezados usan una jerarquía de tres niveles (`#` capítulo, `##` sección, `###` subsección) ya consistente en todo el documento tras la corrección del §4.2 (punto 2, arriba) — un generador de tabla de contenidos automática (Pandoc `--toc`) debería producir una tabla de contenidos idéntica en estructura a la ya incluida manualmente en el Índice general.
