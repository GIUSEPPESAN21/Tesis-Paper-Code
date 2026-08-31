"""Fase 6 (T6.3): generador de results/tablas/Tabla_politicas_reprogramacion.

Lee las filas de ``run_evaluation.py`` producidas bajo las tres politicas de
ejecucion (T6.2: ``right_shift``, ``periodic``, ``event_triggered``) para el mismo
metodo (ga o cpsat), mismo regimen de disrupcion, y responde explicitamente la
pregunta que motiva la Fase 6: ¿la fragilidad de los metodos fuera de linea viene de
la OPTIMALIDAD del solver o del COMPROMISO con una secuencia fija?

Lectura (no una opinion de este modulo, una regla de decision fija sobre los datos):
- Si ``periodic``/``event_triggered`` recuperan una fraccion sustancial de la brecha
  de RPD frente a ``right_shift`` (mismo solver, misma calidad de optimizacion
  disponible, solo cambia si se compromete con una secuencia fija o se reoptimiza),
  la fragilidad viene principalmente del COMPROMISO CON LA SECUENCIA, no de la
  optimalidad del solver.
- Si reoptimizar no recupera RPD de forma sustancial, la fragilidad no se explica por
  el compromiso con la secuencia -- es mas probable que venga de otro factor (p. ej.
  el propio horizonte/calidad del solver bajo el presupuesto reducido de la
  reoptimizacion), reportado como tal, no forzado a encajar en la lectura esperada.
"""

from __future__ import annotations

import pandas as pd


def compare_execution_policies(df: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Mean RPD and reoptimization cost per execution policy, for one method.

    Args:
        df: A ``run_evaluation.py`` CSV (or concatenation of several), already
            restricted by the caller to one disruption regime.
        method: ``"ga"`` or ``"cpsat"``.

    Returns:
        One row per ``execution_policy`` present: ``execution_policy``,
        ``mean_rpd_pct``, ``mean_n_reoptimizations``, ``mean_reoptimization_wall_clock_s``,
        ``n_rows``.
    """
    subset = df[df["method"] == method]
    if subset.empty:
        return pd.DataFrame(columns=["execution_policy", "mean_rpd_pct", "mean_n_reoptimizations", "mean_reoptimization_wall_clock_s", "n_rows"])
    grouped = subset.groupby("execution_policy").agg(
        mean_rpd_pct=("rpd_pct", "mean"),
        mean_n_reoptimizations=("n_reoptimizations", "mean"),
        mean_reoptimization_wall_clock_s=("reoptimization_wall_clock_s", "mean"),
        n_rows=("rpd_pct", "size"),
    ).reset_index()
    return grouped


def diagnose_fragility_source(comparison: pd.DataFrame, *, recovery_threshold_pp: float = 2.0) -> dict:
    """T6.3's headline verdict, computed from :func:`compare_execution_policies`'s
    output for one method/regime.

    Args:
        recovery_threshold_pp: How many percentage points of RPD improvement over
            ``right_shift`` counts as "sustancial" -- a threshold, stated explicitly
            here so the verdict is reproducible, not eyeballed. 2.0pp default (matches
            H4's own margin convention elsewhere in this pipeline, not re-derived).

    Returns:
        ``{"right_shift_rpd", "best_reopt_policy", "best_reopt_rpd",
        "recovered_pp", "diagnosis"}`` where ``diagnosis`` is one of
        ``"compromiso_con_secuencia"`` (reoptimizing recovers >= threshold),
        ``"no_explicado_por_secuencia"`` (recovers less), or ``"sin_datos"``.
    """
    if comparison.empty or "right_shift" not in set(comparison["execution_policy"]):
        return {"diagnosis": "sin_datos"}
    right_shift_rpd = float(comparison.loc[comparison["execution_policy"] == "right_shift", "mean_rpd_pct"].iloc[0])
    reopt_rows = comparison[comparison["execution_policy"] != "right_shift"]
    if reopt_rows.empty:
        return {"right_shift_rpd": right_shift_rpd, "diagnosis": "sin_datos"}
    best_row = reopt_rows.loc[reopt_rows["mean_rpd_pct"].idxmin()]
    recovered = right_shift_rpd - float(best_row["mean_rpd_pct"])
    diagnosis = "compromiso_con_secuencia" if recovered >= recovery_threshold_pp else "no_explicado_por_secuencia"
    return {
        "right_shift_rpd": right_shift_rpd,
        "best_reopt_policy": best_row["execution_policy"],
        "best_reopt_rpd": float(best_row["mean_rpd_pct"]),
        "recovered_pp": recovered,
        "diagnosis": diagnosis,
    }


def tabla_politicas_reprogramacion_markdown(
    comparisons_by_method: dict[str, pd.DataFrame], diagnoses_by_method: dict[str, dict]
) -> str:
    lines = ["| Metodo | Politica de ejecucion | RPD medio (%) | Reoptimizaciones (media) | Costo reopt (s, media) | N |", "|---|---|---|---|---|---|"]
    for method, comparison in comparisons_by_method.items():
        if comparison.empty:
            lines.append(f"| {method} | -- | [PENDIENTE-CORRIDA: run_evaluation.py --execution-policy ...] | -- | -- | -- |")
            continue
        for _, row in comparison.iterrows():
            lines.append(
                f"| {method} | {row['execution_policy']} | {row['mean_rpd_pct']:.3f} | "
                f"{row['mean_n_reoptimizations']:.1f} | {row['mean_reoptimization_wall_clock_s']:.3f} | {int(row['n_rows'])} |"
            )

    lines += ["", "## Diagnostico: ¿optimalidad o compromiso con la secuencia?", ""]
    lines += ["| Metodo | RPD right_shift | Mejor politica reopt | RPD mejor reopt | Recuperado (pp) | Diagnostico |", "|---|---|---|---|---|---|"]
    for method, diag in diagnoses_by_method.items():
        if diag.get("diagnosis") == "sin_datos":
            lines.append(f"| {method} | -- | -- | -- | -- | [PENDIENTE-CORRIDA] |")
        else:
            lines.append(
                f"| {method} | {diag['right_shift_rpd']:.3f} | {diag['best_reopt_policy']} | "
                f"{diag['best_reopt_rpd']:.3f} | {diag['recovered_pp']:+.3f} | {diag['diagnosis']} |"
            )
    return "\n".join(lines)
