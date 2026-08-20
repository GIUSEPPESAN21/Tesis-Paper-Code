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
`Step 4 Thesis.md` (Step 1, 3 y 5 ya estaban en `../archive/` desde la limpieza
anterior).

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
  `E7_E8_FinalPlan.md`, `ConferencePaper_DRAFT.md`, `CODE_APPENDIX_SUMMARY.md`,
  `README.md`, `FINAL_REPRODUCIBILITY_REPORT.md`, `PAPER_SUPPLEMENT_NOTES.md`,
  `CITATION.cff`, `.zenodo.json`.
- `DefensePresentation_Outline.md` y `QA_Prep.md` — material activo de preparación
  para la sustentación (decisión explícita del usuario, 2026-08-20: no son un
  borrador superado, se siguen usando).
