"""Fase 3 (T3.3): analisis de la ablacion 2x2 entorno x recompensa.

Friedman sobre las cuatro celdas (A/B/C/D, ``experiments.run_2x2_ablation``) con
efecto principal de entorno, efecto principal de recompensa, e interaccion --
la descomposicion estandar de un diseno factorial 2x2 aplicada sobre la metrica
primaria (RPD realizada, T8.2) por instancia/semilla, mas el generador de
``results/tablas/Tabla_ablacion_2x2.{csv,md,tex}``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.stats.omnibus import FriedmanResult, critical_difference, friedman_test  # noqa: E402

CELL_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class FactorialEffects:
    """2x2 factorial decomposition of four cell means (A=estatico/nominal,
    B=estatico/robusta, C=dinamico/nominal, D=dinamico/robusta).

    Standard contrasts (each is a difference of averages of two cells, the textbook
    2x2-factorial decomposition -- not a novel formula):
        env_effect     = mean(C, D) - mean(A, B)   -- effect of dinamico vs. estatico
        reward_effect  = mean(B, D) - mean(A, C)   -- effect of robusta vs. nominal
        interaction    = (D - C) - (B - A)         -- does the reward effect change
                                                       depending on the environment?
    """

    env_effect: float
    reward_effect: float
    interaction: float
    cell_means: dict[str, float]


def compute_factorial_effects(cell_means: dict[str, float]) -> FactorialEffects:
    """Args:
        cell_means: ``{"A": ..., "B": ..., "C": ..., "D": ...}``, e.g. mean realized
            RPD (T8.2) per cell.

    Raises:
        KeyError: If any of A/B/C/D is missing.
    """
    missing = [c for c in CELL_LABELS if c not in cell_means]
    if missing:
        raise KeyError(f"cell_means missing cell(s): {missing}")
    a, b, c, d = (cell_means[k] for k in CELL_LABELS)
    return FactorialEffects(
        env_effect=(c + d) / 2 - (a + b) / 2,
        reward_effect=(b + d) / 2 - (a + c) / 2,
        interaction=(d - c) - (b - a),
        cell_means=dict(cell_means),
    )


def friedman_four_cells(long_df: pd.DataFrame, *, value_col: str) -> FriedmanResult:
    """Friedman test over the four cells, blocked by ``instance`` (or
    ``instance``+``seed`` if the caller has already collapsed to one row per block).

    Args:
        long_df: Columns ``cell`` (in ``{"A","B","C","D"}``), ``block`` (the blocking
            key, e.g. instance name or "instance_seed"), ``value_col``.
        value_col: Metric column to rank on (lower = better, matching D6's convention
            elsewhere in this pipeline).

    Raises:
        ValueError: Propagated from :func:`~experiments.stats.omnibus.friedman_test`
            if any block has a missing cell (incomplete design).
    """
    pivot = long_df.pivot_table(index="block", columns="cell", values=value_col, aggfunc="first")
    pivot = pivot[[c for c in CELL_LABELS if c in pivot.columns]]
    return friedman_test(pivot)


def tabla_ablacion_2x2_markdown(
    cell_means: dict[str, float] | None, friedman: FriedmanResult | None, *, metric_name: str = "RPD realizada (%)"
) -> str:
    """T3.3's table. Any missing input renders as ``[PENDIENTE-CORRIDA]`` rather than
    a fabricated number."""
    lines = ["## Medias por celda", "", "| Celda | Entorno | Recompensa | " + metric_name + " |", "|---|---|---|---|"]
    descriptions = {"A": ("estatico", "nominal"), "B": ("estatico", "robusta"), "C": ("dinamico", "nominal"), "D": ("dinamico", "robusta")}
    if cell_means is None:
        for cell, (env, reward) in descriptions.items():
            lines.append(f"| {cell} | {env} | {reward} | [PENDIENTE-CORRIDA: experiments/run_2x2_ablation.py] |")
    else:
        for cell, (env, reward) in descriptions.items():
            value = cell_means.get(cell)
            lines.append(f"| {cell} | {env} | {reward} | {value:.4f} |" if value is not None else f"| {cell} | {env} | {reward} | [PENDIENTE-CORRIDA] |")

    lines += ["", "## Efectos factoriales (2x2)", ""]
    if cell_means is not None and all(c in cell_means for c in CELL_LABELS):
        effects = compute_factorial_effects(cell_means)
        lines += [
            "| Efecto | Valor | Lectura |",
            "|---|---|---|",
            f"| Entorno (dinamico - estatico) | {effects.env_effect:+.4f} | ¿el entorno de entrenamiento por si solo cambia la metrica? |",
            f"| Recompensa (robusta - nominal) | {effects.reward_effect:+.4f} | ¿el termino kappa*Delta_inest por si solo cambia la metrica? |",
            f"| Interaccion (D-C)-(B-A) | {effects.interaction:+.4f} | ¿el efecto de la recompensa depende del entorno de entrenamiento? |",
        ]
    else:
        lines.append("[PENDIENTE-CORRIDA: se necesitan las 4 celdas para calcular los efectos factoriales]")

    lines += ["", "## Prueba de Friedman (4 celdas)", ""]
    if friedman is not None:
        cd = critical_difference(friedman.k_methods, friedman.n_blocks)
        lines += [
            f"chi2={friedman.chi_square:.4f}, df={friedman.df}, p={friedman.p_value:.4g}, "
            f"n_bloques={friedman.n_blocks}, DC(Nemenyi)={cd:.4f}",
            "",
            "| Celda | Rango medio |",
            "|---|---|",
        ]
        for cell in CELL_LABELS:
            if cell in friedman.average_ranks:
                lines.append(f"| {cell} | {friedman.average_ranks[cell]:.3f} |")
    else:
        lines.append("[PENDIENTE-CORRIDA: experiments/run_2x2_ablation.py + experiments/stats/ablation_2x2.py]")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI:

        python -m experiments.stats.ablation_2x2 \\
            --cell-csv A experiments/results/ablation_2x2/eval_cellA_none.csv \\
            --cell-csv B experiments/results/ablation_2x2/eval_cellB_none.csv \\
            --cell-csv C experiments/results/ablation_2x2/eval_cellC_none.csv \\
            --cell-csv D experiments/results/ablation_2x2/eval_cellD_none.csv \\
            --out results/tablas/Tabla_ablacion_2x2
    """
    import argparse

    from experiments.stats.tables import to_latex_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-csv", nargs=2, action="append", metavar=("CELL", "CSV"), default=[],
        help="repeat for A/B/C/D. Each CSV needs 'instance' and 'rpd_pct' columns "
        "(run_evaluation.py's own schema).",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "results" / "tablas" / "Tabla_ablacion_2x2")
    args = parser.parse_args(argv)

    frames = []
    for cell, csv_path in args.cell_csv:
        path = Path(csv_path)
        if not path.exists():
            print(f"[ablation_2x2] {cell}: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        frames.append(df.assign(cell=cell, block=df["instance"]))

    cell_means = None
    friedman = None
    if len(frames) == 4:
        long_df = pd.concat(frames, ignore_index=True)
        cell_means = long_df.groupby("cell")["rpd_pct"].mean().to_dict()
        try:
            friedman = friedman_four_cells(long_df, value_col="rpd_pct")
        except ValueError as exc:
            print(f"[ablation_2x2] Friedman skipped: {exc}")
    else:
        print(f"[ablation_2x2] only {len(frames)}/4 cells available; table will show [PENDIENTE-CORRIDA]")

    md = tabla_ablacion_2x2_markdown(cell_means, friedman)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    if cell_means is not None:
        pd.DataFrame([cell_means]).to_csv(args.out.with_suffix(".csv"), index=False)
        args.out.with_suffix(".tex").write_text(
            to_latex_table(md, caption="Ablacion 2x2: entorno x recompensa", label="ablacion-2x2"), encoding="utf-8"
        )
    print(f"wrote {args.out}.md" + (" + .csv/.tex" if cell_means is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
