# docs/

Documentos de apoyo activos y **no duplicados** (especificaciones técnicas,
planeación, auditorías, logs semanales, trabajo futuro) — todo lo que en algún
momento fue borrador/versión previa ya absorbida en `../thesis/Thesis.md` vive en
`../archive/` en vez de aquí (ver `../archive/README.md`, sección 2026-08-31).

**Si vas a citar algo de la tesis, cita `../thesis/Thesis.md` por número de sección
(§6.X, Tabla 6.0, etc.), no un archivo de capítulo separado** — ya no existe uno; era
un duplicado byte-a-byte de `../thesis/Thesis.md` y se archivó (ver
`../archive/README.md`).

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
`FinalExperimentMatrix.md`, y los documentos de plan mensual `Step 2 Thesis.md`/
`Step 4 Thesis.md` (los demás documentos de plan se retiraron del árbol de trabajo —
sin referencias entrantes, ver `../archive/README.md`).

**`Step 2 Thesis.md` y, sobre todo, `Step 4 Thesis.md` NO son documentos descartables** —
pese al nombre, son el documento de marco metodológico (`../thesis/Thesis.md` los llama
"Parte B/D1/D3/D4/D5/D7/E1" del "documento de marco") del que `../thesis/PreRegistration.md`
deriva literalmente su disciplina anti-HARKing y su rejilla experimental congelada, y del
que partes de la tesis **todavía no redactadas en prosa** (p. ej. el contenido teórico
del Capítulo 2, Parte B) siguen dependiendo directamente como única fuente. Más de 15
citas activas en `../thesis/Thesis.md`/`../thesis/PreRegistration.md` apuntan a `Step 4
Thesis.md` por ruta exacta — borrarlo rompería esas citas y dejaría sin fuente contenido
de tesis aún por escribir. Verificado por búsqueda de texto completo el 2026-08-31, no
supuesto.

## `auditorias/`

`AuditoriaCapitulos1-5.md`, `ConsistencyCheckReport.md`, `PreDefenseAudit.md`.
(`Bibliographic_Verification_PRISMA_GapUpdate.md` se archivó el 2026-08-31 — cero
citas activas, sus correcciones ya están en `../thesis/Thesis.md`'s `REFERENCIAS`;
ver `../archive/README.md`.)

## `logs/`

Reportes de experimentos por semana: `E0_CLOSEOUT_WEEK24.md`,
`EXPERIMENT_LOG_WEEK24.md`, `HPO_ABLATION_REPORT_WEEK21.md`/`.pdf`,
`TrainingPlan_Week17.md`.

## `future_work/`

`ROADMAP.md` — trabajo pendiente consolidado desde el §8.6 ("Trabajo futuro") de
`../thesis/Thesis.md` y desde la sección "Threats to validity" de
`../paper/PaperCIE_LaTeX/manuscript/en/main.tex`. No es una lista inventada: cada
punto cita su fuente exacta.

## Qué vive fuera de `docs/` ahora, y por qué (reorganización 2026-08-31)

El repo se reorganizó por audiencia (ver `../README.md`). Lo que antes vivía en la
raíz o en `docs/capitulos/` ahora vive en:
- `../thesis/` — el núcleo citable (`Thesis.md`, `PreRegistration.md`,
  `RESULTS_REPORT.md`, `REPRODUCIBILITY_REPORT.md`, `FIGURES_INDEX.md`,
  `StatsAnalysis_TechSpec.md`, `E7_E8_Plan.md`, `CODE_APPENDIX_SUMMARY.md`,
  `defense/`).
- `../paper/PaperCIE_LaTeX/` — el envío vigente a Computers & Industrial Engineering.
- `../archive/` — `docs/capitulos/` completo (absorbido en `Thesis.md`, verificado
  por diff), `Chapter6_FINAL.md`/`Chapter7_FINAL.md` (duplicado byte-a-byte de
  `Thesis.md` §6/§7), y el paper cluster superado por la corrección metodológica del
  artefacto RI (`ConferencePaper_DRAFT.md`, `PAPER_SUPPLEMENT_NOTES.md`,
  `JournalArticle_LaTeX/`, `figures_paper/`).
