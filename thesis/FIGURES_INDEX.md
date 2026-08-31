# Índice de figuras — Capítulo 6

Mapa de cada figura final (`../figures/`) a su sección de tesis, tabla/dato fuente,
y el pie de figura exacto ya citado en `Thesis.md`. Todas las figuras se
generaron con `../code/scripts/generate_thesis_figures.py`, en español,
paleta segura para daltonismo (validada con `validate_palette.js` del *skill*
`dataviz` — ver notas de validación al final), fuente legible, 300dpi PNG +
PDF vectorial (fuentes incrustadas, `pdf.fonttype=42`, aptas para inclusión
directa en LaTeX/Word).

Cada figura existe en dos formatos, mismo nombre base:
- `Fig_X.Y.Z_descripcion.png` — inclusión rápida, previsualización
- `Fig_X.Y.Z_descripcion.pdf` — vectorial, preferida para la versión impresa/PDF final de la tesis

---

## 6.3 Generalización en tamaño (H2)

### Figura 6.3.1 — Diagramas de diferencia crítica, uno por *tier*

| Archivo | *Tier* |
|---|---|
| `../figures/Fig_6.3.1a_dc_tier_1.5x.png` / `.pdf` | 1.5× (`la21`–`la25`) |
| `../figures/Fig_6.3.1b_dc_tier_2.0x.png` / `.pdf` | 2.0× (`la26`–`la30`) |
| `../figures/Fig_6.3.1c_dc_tier_2.25x.png` / `.pdf` | 2.25× (`la36`–`la40`) |
| `../figures/Fig_6.3.1d_dc_tier_3.0x.png` / `.pdf` | 3.0× (`la31`–`la35`) |

**Pie de figura (Thesis.md):** *"Diagramas de diferencia crítica, uno
por tier ... En los cuatro diagramas, `fifo` y `rl` caen dentro del mismo grupo
estadísticamente indistinguible (la barra horizontal de agrupamiento conecta
`fifo` con `rl`, `mwkr`, `mor`, `spt` y `lpt`), separados con significancia del
segundo grupo (`edd`, `lwkr`, `lor`, `cr`)."*

**Fuente de datos:** Friedman + Nemenyi (`experiments.stats.omnibus`) sobre
`experiments/results/week24/e3_generalization_la21_la40.csv`, recomputado
directamente por el script generador (misma llamada que usa el *pipeline*
oficial) — ver también `../results/tablas/Tabla_6.3.2_rpd_generalizacion_e3.*`
para las medias RPD subyacentes.

---

### Figura 6.3.2 — RPD de RL vs. tamaño de instancia

`../figures/Fig_6.3.2_rpd_vs_tamano.png` / `.pdf`

**Pie de figura:** *"RPD de RL en función de `num_operations`."*

**Fuente de datos:** mismo CSV de E3; RL (azul) resaltado contra las 9 PDR
(gris, contexto). `../results/tablas/Tabla_6.3.2_rpd_generalizacion_e3.*`.

---

## 6.4 Robustez y recompensa consciente de la perturbación (H3/H4)

### Figura 6.4.1 — RI vs. nivel de perturbación, un panel por eje

| Archivo | Eje |
|---|---|
| `../figures/Fig_6.4.1a_ri_averias.png` / `.pdf` | Averías ($A_k \in \{0.95, 0.90, 0.80\}$) |
| `../figures/Fig_6.4.1b_ri_tiempos_estocasticos.png` / `.pdf` | Tiempos estocásticos ($\sigma \in \{0.10, 0.25, 0.50\}$) |
| `../figures/Fig_6.4.1c_ri_llegadas.png` / `.pdf` | Llegadas ($\rho \in \{0.70, 0.85, 0.95\}$) |

**Pie de figura:** *"RI vs. nivel de perturbación, un panel por eje, las
cuatro series (mejor PDR, κ∈{0,0.5,1.0}) superpuestas ... la serie κ=0.5
(verde) queda visiblemente por encima de κ=0 y κ=1.0, que casi se superponen;
la mejor PDR (azul) permanece muy por debajo de las tres variantes de RL en
todo el rango."*

**Colores (fijos por el pie de figura, verificados en el render):** mejor
PDR = azul `#2a78d6`; RL κ=0 = violeta `#4a3aa7`; RL κ=0.5 = verde `#008300`;
RL κ=1.0 = rojo `#e34948`. Marcadores y estilos de línea distintos por serie
como codificación secundaria (círculo/sólido, cuadrado/discontinuo,
triángulo/sólido, rombo/punto-raya).

**Fuente de datos:** `experiments/results/dynamic/eval_kappa{0.5,1.0}_*.csv`
(rejilla E7) + `experiments/results/week24/e4e5e6_rl/*.csv` (κ=0, reutilizado)
+ `experiments/results/week24/e1_seed0_baselines.csv` (denominador nominal de
RI). `../results/tablas/Tabla_6.4.1_ri_por_regimen.*`.

---

## 6.5 Tiempo de decisión (H5)

### Figura 6.5.1 — Razón CP-SAT/RL vs. tamaño de instancia

`../figures/Fig_6.5.1_latencia_vs_tamano.png` / `.pdf`

**Pie de figura:** *"Razón CP-SAT/RL (ms/decisión) vs. tamaño de instancia,
escala log ... El panel superior muestra RL (azul) esencialmente plano entre
4–18 ms/decisión ... CP-SAT (naranja) es errático ... El panel inferior
(razón) muestra un pico aislado y extremo exactamente en `la40`."*

**Advertencia ya documentada en §6.5.1 del capítulo, repetida aquí para que no
se pierda al citar solo la figura:** el punto de `la40` en esta figura refleja
una medición de CP-SAT anómala (~120× el presupuesto de tiempo configurado,
sin error registrado) — **no se ha corregido ni excluido de la figura**,
exactamente como pide la disciplina de "reportar el dato crudo, señalar la
anomalía en el texto" que sigue el capítulo. Si esta figura se reutiliza fuera
de §6.5 (p. ej. en una presentación de sustentación), debe acompañarse de esa
advertencia — ver §6.5.1 de `Thesis.md` para el análisis de
sensibilidad completo.

**Colores:** RL = azul `#2a78d6`; CP-SAT = naranja `#eb6834`; razón (panel
inferior) = verde `#008300`.

**Fuente de datos:** `../results/datos_crudos/E8_semana25_limpio/*.csv`.
`../results/tablas/Tabla_6.5.1_latencia_e8.*`.

---

## Fase 1/5 (mejoras del tutor, 2026-08-20) — figuras nuevas, `[PENDIENTE-CORRIDA]`

Escritas esta sesión (`AUDIT_PREFLIGHT.md`, `RUN_PLAN.md`, `MEJORAS_TUTOR_IMPLEMENTADAS.md`),
no generadas — ningún archivo `.png`/`.pdf` de esta sección existe todavía.

### Fig_holgura_vs_RI — holgura (CPM) vs. Índice de Robustez, un panel por eje de disrupción

`../figures/Fig_holgura_vs_RI.png` / `.pdf` — `[PENDIENTE-CORRIDA: experiments/run_slack_study.py + scripts/generate_thesis_figures.py --slack-csv ...]`

**Generador:** `scripts/generate_thesis_figures.py::generate_fig_holgura_vs_ri`.
**Fuente de datos:** `../results/datos_crudos/slack/slack_metrics.csv` (nuevo,
`experiments/run_slack_study.py`) + E4/E5/E6 (RI). Un panel por eje (averías, tiempos
estocásticos, llegadas), dispersión de `slack_ratio` vs. RI con ajuste lineal. Reporta
signo y magnitud tal como salgan de los datos — el generador no fuerza la dirección
"más holgura → más robustez" (`experiments/slack_vs_robustness.py`'s propio criterio
de diseño).

---

### Fig_costo_amortizado — costo acumulado GNN-DRL vs. CP-SAT en función de N

`../figures/Fig_6.5.2_costo_amortizado.png` / `.pdf` — `[PENDIENTE-CORRIDA: experiments/results/week24/e8_cpu_timing.py (ampliado a 43 instancias) + scripts/generate_thesis_figures.py --training-summary-glob ...]`

**Generador:** `scripts/generate_thesis_figures.py::generate_fig_6_5_2`.
**Fuente de datos:** `experiments/cost_accounting.py::amortization_curve` sobre
`e8_cpu_timing.csv` (costo de inferencia por instancia) + `summary_*.json` (costo de
entrenamiento). Marca el punto de cruce N* donde GNN-DRL deja de ser más caro que
reoptimizar con CP-SAT desde cero en cada instancia. `../results/tablas/Tabla_6.5.2_costo_computacional.*`.

---

## Validación de paleta (colorblind-safe), *skill* `dataviz`

Ejecutado con `node scripts/validate_palette.js "<hex,...>" --mode light --pairs all`:

| Conjunto de colores | Uso | Resultado |
|---|---|---|
| `#2a78d6` (azul), `#eb6834` (naranja) | Fig. 6.5.1 (RL vs. CP-SAT) | **PASA todo**, ΔE par-completo 24.7 (CVD), 33.6 (visión normal) |
| `#2a78d6` (azul), `#4a3aa7` (violeta), `#008300` (verde), `#e34948` (rojo) | Fig. 6.4.1 (4 series RI) | **PASA todo** (peor par CVD ΔE 7.2, banda de aviso 6–8 — legal con codificación secundaria; aquí, marcador+estilo de línea distintos por serie) |
| Diagramas de diferencia crítica (Fig. 6.3.1) | Monocromo (convención Demsar 2006) | No aplica verificación categórica — no hay codificación de identidad por color |

Los ocho colores base provienen de la paleta categórica de referencia del
*skill* `dataviz` (`references/palette.md`), no elegidos a mano — solo el
subconjunto de 2 y 4 colores usado en cada figura fue seleccionado y
verificado explícitamente para esta tesis.

---

## Cómo regenerar

```bash
cd "code"
./.venv/Scripts/python.exe scripts/generate_thesis_figures.py --out-dir ../figures
```

Determinista dado el estado actual de `experiments/results/` — reusa
directamente las funciones de cómputo ya probadas de `experiments.stats`
(`io.load_results`, `report.add_row_ri`/`nominal_makespan_lookup`/
`best_pdr_for_regime`, `omnibus.friedman_test`/`critical_difference`), así que
los números graficados son, por construcción, los mismos que
`experiments/results/stats/tables/*.md` y `Thesis.md` ya citan — este
script solo cambia el estilo de render, no ningún dato.
