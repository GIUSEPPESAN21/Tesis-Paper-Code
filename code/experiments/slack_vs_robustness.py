"""Fase 1 (Recomendacion 2 del tutor), T1.3: correlaciona las seis metricas de holgura
(``experiments.slack_analysis``, via ``experiments.run_slack_study``'s
``slack_metrics.csv``) con el Indice de Robustez (RI) y con la RPD realizada bajo
disrupcion, por instancia/metodo, y emite
``results/tablas/Tabla_holgura_vs_robustez.{csv,md,tex}``.

Criterio de diseno explicito (instruccion del tutor): el codigo debe reportar signo y
magnitud de la correlacion AUNQUE contradigan la hipotesis del mecanismo ("mas holgura
-> mas robustez"). Ninguna funcion de este modulo descarta ni re-etiqueta un
coeficiente porque su signo sea "el que no se esperaba" -- si la holgura no explica la
fragilidad, eso es un resultado que este pipeline debe poder decir, no ocultar.

No se ejecuta en esta sesion (Regla Cero).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

SLACK_METRIC_COLUMNS = [
    "total_slack", "mean_slack", "slack_ratio", "critical_path_length",
    "critical_op_fraction", "machine_idle_fraction",
]


def correlate_slack_vs_target(
    slack_df: pd.DataFrame, target_df: pd.DataFrame, *, target_col: str, regime_label: str
) -> pd.DataFrame:
    """One row per slack metric: Spearman rho/p and Pearson r/p against ``target_col``.

    Args:
        slack_df: ``instance, method`` + the six slack-metric columns (nominal, one
            row per (instance, method) -- ``run_slack_study.py``'s output).
        target_df: ``instance, method``, ``target_col`` -- e.g. mean RI or mean RPD
            realizada, one row per (instance, method) within one disruption regime.
        target_col: Column name in ``target_df`` to correlate against.
        regime_label: Free-text label for the output row (e.g.
            ``"E4_breakdowns_A0.90"`` or ``"pooled"``).

    Returns:
        One row per slack metric: ``regime``, ``metric``, ``target``, ``spearman_rho``,
        ``spearman_p``, ``pearson_r``, ``pearson_p``, ``n``. Sign and magnitude are
        always reported as computed -- no metric is dropped or flipped because its
        sign contradicts the "more slack -> more robust" hypothesis.

    Raises:
        ValueError: If the join on ``(instance, method)`` produces fewer than 3 paired
            observations (Spearman/Pearson are not meaningful below that).
    """
    merged = slack_df.merge(target_df[["instance", "method", target_col]], on=["instance", "method"], how="inner")
    if len(merged) < 3:
        raise ValueError(
            f"only {len(merged)} paired (instance, method) observations for regime "
            f"{regime_label!r} -- need at least 3 to compute a correlation"
        )

    rows = []
    for metric in SLACK_METRIC_COLUMNS:
        x = merged[metric].to_numpy(dtype=float)
        y = merged[target_col].to_numpy(dtype=float)
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        rows.append(
            {
                "regime": regime_label,
                "metric": metric,
                "target": target_col,
                "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "pearson_r": float(r),
                "pearson_p": float(p_r),
                "n": len(merged),
            }
        )
    return pd.DataFrame(rows)


def build_slack_vs_robustness_table(
    slack_df: pd.DataFrame, ri_by_regime: dict[str, pd.DataFrame], rpd_by_regime: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Run :func:`correlate_slack_vs_target` for every regime x {RI, RPD realizada}
    combination and concatenate into one table (T1.3's full deliverable).

    Args:
        slack_df: ``run_slack_study.py``'s nominal slack metrics (one row per
            instance/method, regime-independent).
        ri_by_regime: ``{regime_label: DataFrame(instance, method, ri)}``.
        rpd_by_regime: ``{regime_label: DataFrame(instance, method, rpd_realized_pct)}``.
    """
    frames: list[pd.DataFrame] = []
    for regime_label, ri_df in ri_by_regime.items():
        try:
            frames.append(correlate_slack_vs_target(slack_df, ri_df, target_col="ri", regime_label=regime_label))
        except ValueError as exc:
            print(f"[slack_vs_robustness] skipping RI correlation for {regime_label}: {exc}", file=sys.stderr)
    for regime_label, rpd_df in rpd_by_regime.items():
        try:
            frames.append(
                correlate_slack_vs_target(slack_df, rpd_df, target_col="rpd_realized_pct", regime_label=regime_label)
            )
        except ValueError as exc:
            print(f"[slack_vs_robustness] skipping RPD correlation for {regime_label}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame(
            columns=["regime", "metric", "target", "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "n"]
        )
    return pd.concat(frames, ignore_index=True)


def paired_points(
    slack_df: pd.DataFrame, target_df: pd.DataFrame, *, metric: str, target_col: str
) -> pd.DataFrame:
    """Raw ``(metric_value, target_value)`` pairs for one slack metric, for scatter
    plotting (T1.4's ``Fig_holgura_vs_RI``) -- the per-point data
    :func:`correlate_slack_vs_target` summarizes into a single rho/r, exposed here
    separately since a figure needs the points, not just the statistic.

    Raises:
        KeyError: If ``metric`` is not a column of ``slack_df``.
    """
    if metric not in slack_df.columns:
        raise KeyError(f"{metric!r} is not a column of slack_df")
    merged = slack_df.merge(target_df[["instance", "method", target_col]], on=["instance", "method"], how="inner")
    return merged[["instance", "method", metric, target_col]].rename(columns={metric: "metric_value", target_col: "target_value"})


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(sin datos -- ver [PENDIENTE-CORRIDA])"
    lines = ["| Regimen | Metrica de holgura | Objetivo | Spearman rho | Spearman p | Pearson r | Pearson p | N |", "|---|---|---|---|---|---|---|---|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['regime']} | {row['metric']} | {row['target']} | {row['spearman_rho']:+.4f} | "
            f"{row['spearman_p']:.2e} | {row['pearson_r']:+.4f} | {row['pearson_p']:.2e} | {int(row['n'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slack-csv", type=Path, required=True, help="run_slack_study.py's output CSV")
    parser.add_argument(
        "--ri-csv", type=Path, required=True,
        help="long-format CSV with columns instance, method, regime, ri (one row per "
        "(instance, method, regime), e.g. pre-aggregated from E4-E6 via report.add_row_ri)",
    )
    parser.add_argument(
        "--rpd-realized-csv", type=Path, default=None,
        help="optional long-format CSV with instance, method, regime, rpd_realized_pct "
        "(T8.2's realized-quality metric; omit to correlate against RI only)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "results" / "tablas" / "Tabla_holgura_vs_robustez.csv")
    args = parser.parse_args(argv)

    slack_df = pd.read_csv(args.slack_csv)
    ri_long = pd.read_csv(args.ri_csv)
    ri_by_regime = {regime: g for regime, g in ri_long.groupby("regime")}

    rpd_by_regime: dict[str, pd.DataFrame] = {}
    if args.rpd_realized_csv is not None and args.rpd_realized_csv.exists():
        rpd_long = pd.read_csv(args.rpd_realized_csv)
        rpd_by_regime = {regime: g for regime, g in rpd_long.groupby("regime")}

    table = build_slack_vs_robustness_table(slack_df, ri_by_regime, rpd_by_regime)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    md = to_markdown(table)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    if not table.empty:
        from experiments.stats.tables import to_latex_table

        tex = to_latex_table(md, caption="Holgura (CPM) vs. Indice de Robustez y RPD realizada, por regimen", label="holgura-vs-robustez")
        args.out.with_suffix(".tex").write_text(tex, encoding="utf-8")
    print(f"wrote {args.out} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
