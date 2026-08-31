# EXPERIMENT_LOG_ABLACION_2x2.md — Plantilla (Fase 3, T3.3)

**Estado: plantilla, sin ejecutar.** Este documento se llena la primera vez que se
corre `experiments/run_2x2_ablation.py --execute`. Ningún número en este archivo es
real hasta entonces — todo lo que sigue es la estructura a completar, no un resultado.

**Propósito:** aislar el efecto del entorno de entrenamiento (estático JSSPEnv vs.
dinámico DynamicJSSPEnv) del efecto de la recompensa (nominal vs. robusta,
kappa·Δ_inest) en el colapso de RPD nominal que `../../thesis/RESULTS_REPORT.md` §6.4
diagnostica para H4 (+106.0pp κ=0.5, +46.6pp κ=1.0). Ver `experiments/run_2x2_ablation.py`
para la definición exacta de las cuatro celdas.

---

## 1. Configuración de la corrida

| Campo | Valor |
|---|---|
| Fecha de inicio | `[PENDIENTE-CORRIDA]` |
| Instancia | `[PENDIENTE-CORRIDA]` (ft06 por defecto en `run_2x2_ablation.py`) |
| Checkpoint estático (celda A) | `[PENDIENTE-CORRIDA]` |
| Checkpoint κ>0 dinámico (celda D, si se reutiliza) | `[PENDIENTE-CORRIDA]` |
| κ para celdas B/D | `[PENDIENTE-CORRIDA]` (0.5 por defecto) |
| `--nominal-episode-fraction` (celdas C/D) | `[PENDIENTE-CORRIDA]` (0.2 por defecto, T3.2) |
| Semillas de entrenamiento (T2.1) | `[PENDIENTE-CORRIDA]` |
| Timesteps por celda | `[PENDIENTE-CORRIDA]` |
| Máquina / hardware | `[PENDIENTE-CORRIDA]` |

## 2. Comandos ejecutados

```
[PENDIENTE-CORRIDA: pegar aquí la salida de
 `python -m experiments.run_2x2_ablation --static-checkpoint ... ` sin --execute
 (el plan impreso), y luego confirmar que la corrida real con --execute usó
 exactamente esos comandos]
```

## 3. Resultado por celda

| Celda | Entorno | Recompensa | Checkpoint final | train_wall_clock_s | Estado |
|---|---|---|---|---|---|
| A | estático | nominal | `[PENDIENTE-CORRIDA]` | — (reutilizado) | `[PENDIENTE-CORRIDA]` |
| B | estático | robusta | `[PENDIENTE-CORRIDA]` | `[PENDIENTE-CORRIDA]` | `[PENDIENTE-CORRIDA]` |
| C | dinámico | nominal | `[PENDIENTE-CORRIDA]` | `[PENDIENTE-CORRIDA]` | `[PENDIENTE-CORRIDA]` |
| D | dinámico | robusta | `[PENDIENTE-CORRIDA]` | — (reutilizado, si aplica) | `[PENDIENTE-CORRIDA]` |

## 4. Efectos factoriales (T3.3, `experiments/stats/ablation_2x2.py`)

Ver [`results/tablas/Tabla_ablacion_2x2.md`](../results/tablas/Tabla_ablacion_2x2.md)
(generada por `python -m experiments.stats.ablation_2x2`) para la tabla completa.
Resumen a completar aquí:

- **Efecto de entorno** (dinámico − estático): `[PENDIENTE-CORRIDA]`
- **Efecto de recompensa** (robusta − nominal): `[PENDIENTE-CORRIDA]`
- **Interacción** ((D−C)−(B−A)): `[PENDIENTE-CORRIDA]`
- **Friedman (4 celdas)**: χ²=`[PENDIENTE-CORRIDA]`, p=`[PENDIENTE-CORRIDA]`

## 5. Lectura — ¿de dónde viene el colapso de RPD nominal?

Marcar la conclusión que los datos reales sostengan (no elegir de antemano):

- [ ] El efecto de entorno domina → el colapso viene de entrenar contra
      `DynamicJSSPEnv` en sí (domain randomization sobre 9 regímenes), no del término
      κ·Δ_inest.
- [ ] El efecto de recompensa domina → el colapso viene del término κ·Δ_inest en sí,
      independientemente del entorno.
- [ ] La interacción domina → el término κ·Δ_inest solo colapsa el RPD nominal
      *cuando además* se entrena contra el entorno dinámico (los efectos no son
      aditivos).
- [ ] Ningún efecto es significativo → con los datos de esta corrida, el diseño 2×2
      no logra separar las dos causas candidatas; reportar como inconcluso, no forzar
      una lectura.

**Nota de diseño de la celda B** (`experiments/run_2x2_ablation.py`'s propio
docstring, T3.1): sin averías reales, el término Δ_inest podría acercarse a un no-op
— si B ≈ A, es evidencia consistente con "el término necesita disrupción real para
tener efecto"; si B ≠ A, el término tiene efecto incluso sin disrupción (vía
contención de máquina/elección de la política), lo cual también es un hallazgo
reportable, no un error de diseño.

## 6. Limitaciones y supuestos no verificados por esta corrida

- `[PENDIENTE-CORRIDA]` — listar aquí cualquier desviación entre el plan de
  `run_2x2_ablation.py` y lo que realmente se ejecutó (p. ej. si una celda se corrió
  con menos semillas que las planeadas por presupuesto de tiempo).
