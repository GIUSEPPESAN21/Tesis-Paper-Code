# docs/

Documentos activos (referenciados desde `Thesis_FINAL.md`, `CODE_APPENDIX_SUMMARY.md`
u otros documentos del núcleo) pero que no son el texto final citable — reorganizados
el 2026-08-20 desde la raíz del repositorio para reducir el desorden visual, con
`git mv` (historial intacto) y cada referencia cruzada reescrita a mano/script para
apuntar a la nueva ruta. Nada se borró.

**Si vas a citar algo de la tesis, cita `../Thesis_FINAL.md` (o
`../Chapter6_FINAL.md`/`../Chapter7_FINAL.md` para citas internas puntuales), no los
archivos de aquí** — son el historial de versiones/planificación que produjo el
documento final, no el documento final en sí.

## `capitulos/`

Borradores de capítulos/secciones anteriores a `Thesis_FINAL.md`, y las dos versiones
de "borrador completo" (`Thesis_FullDraft_v1.md`, `_v2.md`) que precedieron la versión
final. Incluye `Chapter1.md`, `Chapter3_final.md`, `Chapter4_sections_4.1_4.3.md`,
`Chapter6_sections_6.1-6.4_DRAFT.md`, `Chapter6_template.md`, `Chapter7_DRAFT.md`,
`Chapter8.md`, `AppendixB.md`, y los `Section*.md` sueltos (4.2, 4.4, 4.5, 4.6,
5.1_5.2, 5.3).

## `especificaciones/`

Documentos técnicos de especificación/arquitectura: `Baselines_TechSpec.md`,
`DynamicEnv_TechSpec.md`, `GA_TechSpec.md`, `GNN_PPO_Architecture_Spec.md` y su
sucesora `_v2.md` (ambas activamente citadas — v2 no reemplaza a v1 en las citas,
documentan etapas arquitectónicas distintas, ver `CODE_APPENDIX_SUMMARY.md`).
`Baselines_TechSpec.md`/`DynamicEnv_TechSpec.md`/`GNN_PPO_Architecture_Spec.md` son
los Anexos E–G de la tesis (§1.7).

## `planificacion/`

Documentos de planificación y proceso: `AblationPlan.md`,
`BaselineAnalysis.md`/`BaselineComparativeAnalysis.md`, `VandV_Protocol.md`,
`FinalExperimentMatrix.md`, y los prompts de plan mensual `Step 2 Thesis.md`/
`Step 4 Thesis.md` (Step 1, 3, 5 y `PROMPT_AsistenteIA_MejorasTutor.md` se eliminaron el
2026-08-31 — cero referencias entrantes, ver `../archive/README.md`).

**`Step 2 Thesis.md` y, sobre todo, `Step 4 Thesis.md` NO son prompts descartables** —
pese al nombre, son el documento de marco metodológico (`Thesis_FINAL.md` los llama
"Parte B/D1/D3/D4/D5/D7/E1" del "documento de marco") del que `PreRegistration_FINAL.md`
deriva literalmente su disciplina anti-HARKing y su rejilla experimental congelada, y del
que partes de la tesis **todavía no redactadas en prosa** (p. ej. el contenido teórico
del Capítulo 2, Parte B) siguen dependiendo directamente como única fuente. Más de 15
citas activas en `Thesis_FINAL.md`/`PreRegistration_FINAL.md` apuntan a `Step 4
Thesis.md` por ruta exacta — borrarlo rompería esas citas y dejaría sin fuente contenido
de tesis aún por escribir. Verificado por búsqueda de texto completo el 2026-08-31, no
supuesto.

**Corrección, 2026-08-31:** `Step 5 Thesis (Month Plan With All Promts).md` se eliminó
en la misma pasada bajo la misma verificación de "cero referencias", pero esa
verificación fue incompleta — `FinalExperimentMatrix.md` (este mismo directorio) lo cita
por nombre dos veces. Restaurado en `../archive/`; ver la nota de corrección en
`../archive/README.md`.

## `auditorias/`

`AuditoriaCapitulos1-5.md`, `ConsistencyCheckReport.md`, `PreDefenseAudit.md`,
`Bibliographic_Verification_PRISMA_GapUpdate.md`.

## `logs/`

Reportes de experimentos por semana: `E0_CLOSEOUT_WEEK24.md`,
`EXPERIMENT_LOG_WEEK24.md`, `HPO_ABLATION_REPORT_WEEK21.md`/`.pdf`,
`TrainingPlan_Week17.md`.

## Qué se quedó en la raíz y por qué

- El núcleo citable: `Thesis_FINAL.md`, `Chapter6_FINAL.md`, `Chapter7_FINAL.md`,
  `FIGURES_INDEX.md`, `PreRegistration_FINAL.md`, `StatsAnalysis_TechSpec.md`,
  `E7_E8_FinalPlan.md`, `CODE_APPENDIX_SUMMARY.md`, `README.md`,
  `FINAL_REPRODUCIBILITY_REPORT.md`, `CITATION.cff`, `.zenodo.json`.
- `papers/ConferencePaper_DRAFT.md` y `papers/PAPER_SUPPLEMENT_NOTES.md` —
  movidos a `papers/` (junto con `JournalArticle_LaTeX/` y `PaperCIE_LaTeX/`)
  durante la profesionalización de la estructura, 2026-08-31 — no son borrador
  superado, siguen citables, solo cambiaron de ubicación.
- `defense/DefensePresentation_Outline.md` y `defense/QA_Prep.md` — material
  activo de preparación para la sustentación (decisión explícita del usuario,
  2026-08-20: no son un borrador superado, se siguen usando), movidos a
  `defense/` en la misma pasada de profesionalización.
