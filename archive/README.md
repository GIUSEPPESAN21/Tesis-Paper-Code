# archive/

Material archivado el 2026-08-20 durante una limpieza de la raíz del repositorio, y
reducido en dos pasadas de limpieza más el 2026-08-31. Lo que queda aquí se movió con
`git mv` (historial intacto, recuperable con `git log --follow -- archive/<archivo>`)
porque no tiene ninguna referencia activa desde `README.md`, `CODE_APPENDIX_SUMMARY.md`,
`Thesis_FINAL.md`, `PreRegistration_FINAL.md` ni ningún otro documento del núcleo activo
del repositorio — verificado por búsqueda de texto completo, no supuesto.

**Eliminado el 2026-08-31** (recuperable en el historial de git, no en el árbol de
trabajo actual):
- `borradores_tempranos/` (25 borradores tempranos en inglés, cada uno superado por su
  versión final en español ya citada en el repo).
- `JournalArticle.docx` / `JournalArticle_OR_DRAFT.docx` (superados por
  `../JournalArticle_LaTeX/`), `HPO_ABLATION_REPORT_WEEK21.html` (copia sin referencias
  del mismo reporte que ya existe como `.md`/`.pdf`), y los dos documentos de entrega
  institucional en Word (`Tesis Joseph Español.docx`,
  `Revision_y_ajustes_Tesis_Joseph_TNR12_Negro_Justificado.docx`) — superados por
  `../Thesis_FINAL.md`.
- `Step 1 Thesis.md`, `Step 3 Thesis.md`, `PROMPT_AsistenteIA_MejorasTutor.md` —
  prompts/planes de trabajo mensual usados para producir la tesis, sin ninguna
  referencia entrante desde ningún documento del repo (a diferencia de
  `docs/planificacion/Step 2 Thesis.md` y `Step 4 Thesis.md`, que **sí** siguen siendo
  fuente primaria activa — ver la nota en `../docs/README.md` antes de tocar esos dos).

**Corrección, 2026-08-31 (auditoría posterior a la misma limpieza):** `Step 5 Thesis
(Month Plan With All Promts).md` se había eliminado en la misma pasada bajo la misma
justificación de "cero referencias entrantes" — esa verificación fue incompleta.
`docs/planificacion/FinalExperimentMatrix.md` lo cita por nombre exacto dos veces
("The master project plan (`Step 5 Thesis (Month Plan With All Prompts).md`)
schedules..." y "The master plan (`Step 5 Thesis (Month Plan)`) labels..."). Restaurado
con `git checkout <commit-previo> -- archive/Step 5...md` (mismo objeto git, historial
intacto). Sigue archivado, no es fuente primaria activa como Step 2/4, pero no es
huérfano.

## Lo que queda en la raíz de `archive/`

- `ProofreadingChangelog.md` — registro de correcciones de redacción, proceso cerrado.
- `Chapter3.md` — borrador de capítulo anterior a `Thesis_FINAL.md` (ver también
  `../docs/capitulos/Chapter3_final.md`, la revisión posterior de este mismo borrador).
- `Step 5 Thesis (Month Plan With All Promts).md` — restaurado el 2026-08-31, ver nota
  de corrección arriba.
