# `results/` — resultados finales, organizados para citar en la tesis

Este directorio es la vista **limpia y numerada por tesis** de los resultados de
E1–E8. No reemplaza `code/experiments/results/` (el árbol de trabajo
completo, con *checkpoints*, logs de entrenamiento, TensorBoard, y todo CSV
intermedio) — es una copia derivada, generada por scripts reproducibles, pensada
para citarse directamente desde `../thesis/Thesis.md` y para
adjuntarse como material suplementario de la sustentación.

## Estructura

```
results/
    tablas/           -- tablas ya agregadas, numeradas Tabla_6.X.Y, en .csv/.md/.tex
    datos_crudos/      -- CSV fuente (una fila por instancia/método/semilla), por experimento
    README.md          -- este archivo
```

## `tablas/` — tablas citadas en el Capítulo 6

Generadas por `code/scripts/export_thesis_tables.py`, que **convierte
de formato** (Markdown → CSV/TeX) las tablas ya producidas y probadas por el
*pipeline* de estadística (`experiments/stats/`, 84/84 pruebas superadas) —
**no recalcula ningún número**, así que estas copias son, por construcción,
idénticas a las citadas en `../thesis/Thesis.md`.

| Archivo | Tabla en el capítulo | Contenido |
|---|---|---|
| `Tabla_6.0_veredicto_hipotesis.*` | Tabla 6.0 | Veredicto H1–H5, estadístico, p-valor, efecto |
| `Tabla_6.2.1_rpd_jssp_e1.*` | Tabla 6.2.1 | RPD (%) por instancia y método, E1 (43×12) |
| `Tabla_6.3.2_rpd_generalizacion_e3.*` | Tabla 6.3.2 | RPD medio±SD por *tier* y método, E3 |
| `Tabla_6.4.1_ri_por_regimen.*` | Tabla 6.4.1 | RI por régimen y serie (mejor PDR, κ∈{0,0.5,1.0}) |
| `Tabla_6.5.1_latencia_e8.*` | Tabla 6.5.1 | ms/decisión mediana, RL vs. CP-SAT, 43 instancias |

**Tablas del capítulo sin archivo fuente separado** (no producidas por el
*pipeline* como tabla independiente — o bien son manuales/E0, o se derivan
directamente de las de arriba): Tablas 6.1.1–6.1.3 (validación E0, no pasa por
`experiments/stats/`), 6.2.2 (subconjunto de la fila H1 de la Tabla 6.0), 6.3.1/6.3.3
(definición de *tiers* y estadístico Friedman/Nemenyi — este último está
embebido en los diagramas de diferencia crítica, `figures/Fig_6.3.1*`, y en
`stats_manifest.json`), 6.4.2–6.4.4 (subconjunto de las filas H3/H4 de la Tabla
6.0), 6.5.2 (umbral, derivable de `Tabla_6.5.1`).

## `datos_crudos/` — CSV fuente, una fila por observación

Copia directa (no transformada) de los CSV que el pipeline de estadística
consume, organizados por lote de ejecución — **no** por número de experimento,
porque así es como se generaron y así referencia `stats_manifest.json` su
procedencia:

| Carpeta | Experimentos | Origen |
|---|---|---|
| `E1_E2_E3_E4_E5_E6_semana24/` | E1, E2a, E2b, E3, E4, E5, E6 (brazo κ=0) | `experiments/results/week24/` |
| `E8_semana25_limpio/` | E8 (y E1 comparte fila con E8, ver abajo) | `experiments/results/week25/e8_clean/` |
| `E7_dynamic/` | E7 (κ=0.5, κ=1.0) | `experiments/results/dynamic/*.csv` |

**Nota de procedencia importante (per `../thesis/StatsAnalysis_TechSpec.md` §2.3):** E1 y
E8 comparten una sola pasada de ejecución — la columna `wall_clock_s` de los CSV
de E1 (`E8_semana25_limpio/e1_e8_clean_*.csv`) **es** el dato crudo de E8, no
existe un CSV de E8 separado. Esto es intencional, no un archivo faltante.

**Lo que NO está aquí, y dónde encontrarlo:** *checkpoints* de los modelos
entrenados (`.zip`), logs de TensorBoard, curvas de entrenamiento (`.png`), y
logs de ejecución (`.log`) permanecen en
`code/experiments/results/` — son artefactos de ingeniería/
reproducibilidad, no datos que una tabla o figura del Capítulo 6 cite
directamente. `stats_manifest.json` (`code/experiments/results/stats/`)
es la fuente de verdad para trazar cualquier número específico hasta su CSV y
fila de origen exactos.

## Cómo regenerar este directorio

Ambos scripts son deterministas dado el mismo estado de
`experiments/results/` — no dependen de aleatoriedad ni de fecha/hora:

```bash
cd "code"
./.venv/Scripts/python.exe scripts/export_thesis_tables.py --out-dir ../results/tablas
./.venv/Scripts/python.exe scripts/generate_thesis_figures.py --out-dir ../figures
```

(La copia de `datos_crudos/` es manual, un `cp` de los CSV fuente — no tiene
script propio porque no transforma nada; ver el historial de esta sesión para
el comando exacto si hace falta repetirla tras una nueva corrida de E1–E8.)
