# archive/

Material archivado el 2026-08-20 durante una limpieza de la raíz del repositorio, y
reducido en dos pasadas de limpieza más el 2026-08-31. Lo que queda aquí se movió con
`git mv` (historial intacto, recuperable con `git log --follow -- archive/<archivo>`)
porque no tiene ninguna referencia activa desde `README.md`, `../thesis/CODE_APPENDIX_SUMMARY.md`,
`../thesis/Thesis.md`, `../thesis/PreRegistration.md` ni ningún otro documento del núcleo activo
del repositorio — verificado por búsqueda de texto completo, no supuesto.

**Eliminado el 2026-08-31** (recuperable en el historial de git, no en el árbol de
trabajo actual):
- `borradores_tempranos/` (25 borradores tempranos en inglés, cada uno superado por su
  versión final en español ya citada en el repo).
- `JournalArticle.docx` / `JournalArticle_OR_DRAFT.docx` (superados en su momento por
  `JournalArticle_LaTeX/`, que a su vez se archivó aquí mismo el 2026-08-31 — ver la
  sección de esa fecha más abajo), `HPO_ABLATION_REPORT_WEEK21.html` (copia sin referencias
  del mismo reporte que ya existe como `.md`/`.pdf`), y los dos documentos de entrega
  institucional en Word (`Tesis Joseph Español.docx`,
  `Revision_y_ajustes_Tesis_Joseph_TNR12_Negro_Justificado.docx`) — superados por
  `../thesis/Thesis.md`.
- Documentos de planificación mensual internos, sin referencias entrantes desde ningún
  documento del repo (a diferencia de `docs/planificacion/Step 2 Thesis.md` y
  `Step 4 Thesis.md`, que **sí** siguen siendo fuente primaria activa — ver la nota en
  `../docs/README.md` antes de tocar esos dos).

## Lo que queda en la raíz de `archive/`

- `ProofreadingChangelog.md` — registro de correcciones de redacción, proceso cerrado.
- `Chapter3.md` — borrador de capítulo anterior a `../thesis/Thesis.md` (ver también
  `capitulos/Chapter3_final.md`, la revisión posterior de este mismo borrador — ambos
  viven ahora en `archive/`, ver la sección 2026-08-31 más abajo).

## 2026-08-31 (segunda pasada, reorganización por audiencia) — archivado, no eliminado

Reorganización mayor: la raíz del repo se separó en `thesis/` (núcleo citable),
`paper/` (envío vigente) y `docs/` (apoyo activo no duplicado). Todo lo que perdió su
condición de "vigente" en ese proceso se movió aquí, con `git mv` (historial intacto):

- **`Chapter6_FINAL.md`, `Chapter7_FINAL.md`** — verificados por diff como duplicado
  byte-a-byte de `../thesis/Thesis.md` §6 y §7 respectivamente (solo diferían en
  terminador de línea CRLF/LF y una línea de encabezado de tabla de contenidos). El
  documento íntegro (`../thesis/Thesis.md`) es ahora la única fuente citable; toda
  referencia cruzada en el repo que citaba `Chapter6_FINAL.md §6.X` o
  `Chapter7_FINAL.md §7.X` se reescribió a `thesis/Thesis.md §6.X`/`§7.X`.
- **`ConferencePaper_DRAFT.md`, `PAPER_SUPPLEMENT_NOTES.md`, `JournalArticle_LaTeX/`,
  `figures_paper/`** (antes `figures/paper/`) — los cuatro usan el índice de robustez
  (RI) **sin la corrección del artefacto** que `../paper/PaperCIE_LaTeX/Paper_CIE_DecisionRecord.md`
  documenta como el análisis que el tutor descartó explícitamente ("El tutor descartó
  los resultados anteriores"). `JournalArticle_LaTeX/README.txt` además nunca declaró
  un venue objetivo real ("No specific target journal was specified"). No son
  duplicados de `paper/PaperCIE_LaTeX/` — son el intento previo a la corrección
  metodológica que dio lugar a ese paper.
- **`capitulos/`** (16 archivos, antes `docs/capitulos/`) — borradores de
  capítulos/secciones ya absorbidos verbatim o casi-verbatim en `../thesis/Thesis.md`
  (confirmado por diff donde se verificó explícitamente: Ch.1, Ch.3, §4.2, §4.4,
  §5.1-5.2, §5.3, Ch.8), más `Thesis_FullDraft_v1.md`/`_v2.md` (compilaciones previas
  a la versión final) y `Chapter6_template.md` (plantilla vacía, superada dos veces,
  sin dato real). Auditado por agente dedicado antes de mover — ver detalle completo
  en la conversación que produjo este commit si se necesita el razonamiento por archivo.
- **`Bibliographic_Verification_PRISMA_GapUpdate.md`** (antes
  `docs/auditorias/`) — cero citas activas; sus correcciones ya están absorbidas en la
  sección `REFERENCIAS` de `../thesis/Thesis.md`.

**Nota de baja confianza, pendiente de revisión manual:** `capitulos/AppendixB.md`
(arriba) se movió aquí junto con el resto del lote, pero el agente auditor no pudo
confirmar con certeza si todo su contenido no fusionado (notas de "desviación" §B.6,
"bug corregido" §B.8) ya vive en `docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md`
con otra redacción — revisar antes de asumir que no hay información única ahí; si la
hay, rescatarla de vuelta a `docs/especificaciones/` en vez de dejarla solo en `archive/`.
