"""Fase 8 (T8.2/T8.4): calidad realizada como metrica primaria junto al RI.

The manuscript's own discussion already recommends comparing realized quality
(RPD against the certified optimum, shared denominator across methods) alongside RI,
but never applies it to H3 -- this module closes that gap: realized RPD per
(instance, method, regime), plus the Wilcoxon comparison against the three T4.1
comparators (never fused, per Recomendaciones 6/7), producing
``results/tablas/Tabla_calidad_realizada_disrupcion.{csv,md,tex}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.stats.comparators import oracle_best_per_instance, pdr_global, pdr_por_familia
from experiments.stats.metrics import compute_realized_quality
from experiments.stats.pairwise import align_paired, wilcoxon_signed_rank


def realized_quality_table(
    regime_df: pd.DataFrame, *, certified_optimum: dict[str, float], regime_label: str
) -> pd.DataFrame:
    """Realized RPD (mean/std/CVaR_90) per method, for one disruption regime.

    Args:
        regime_df: Rows for one regime (e.g. ``"E4_breakdowns_A0.90"``), one row per
            (instance, method, seed).
        certified_optimum: ``{instance: certified_optimum}`` -- E1's ``known_optimum``
            column (proven, per ``data/known_optima.csv``'s ``status == "proven"``
            rows, per ``PreRegistration_FINAL.md`` Sec 1) is the certified optimum for
            every bundled JSSP instance in this thesis.
        regime_label: Free-text label carried into the output ``regime`` column.

    Returns:
        One row per (instance, method): ``regime``, ``instance``, ``method``,
        ``mean_rpd_realized_pct``, ``std_rpd_realized_pct``, ``cvar_90_rpd_realized_pct``,
        ``n_replicates``. Rows for an instance with no known certified optimum are
        skipped (printed note), never silently assigned a fabricated denominator.
    """
    rows = []
    for (instance, method), group in regime_df.groupby(["instance", "method"]):
        optimum = certified_optimum.get(instance)
        if optimum is None or optimum <= 0:
            print(f"[realized_quality] skipping {instance}/{method}: no certified optimum")
            continue
        result = compute_realized_quality(group["makespan"].to_numpy(dtype=float), optimum)
        rows.append(
            {
                "regime": regime_label,
                "instance": instance,
                "method": method,
                "mean_rpd_realized_pct": result.mean_rpd_realized_pct,
                "std_rpd_realized_pct": result.std_rpd_realized_pct,
                "cvar_90_rpd_realized_pct": result.cvar_90_rpd_realized_pct,
                "n_replicates": result.n_replicates,
            }
        )
    return pd.DataFrame(rows)


def compare_rl_vs_three_comparators(
    regime_realized_df: pd.DataFrame,
    e1_df: pd.DataFrame,
    *,
    pdr_candidates: tuple[str, ...],
    regime_label: str,
) -> pd.DataFrame:
    """T8.4: RL's realized RPD vs. each of the three T4.1 comparators (pdr_global,
    pdr_por_familia, oraculo_best_per_instance), Wilcoxon signed-rank + effect size
    ``r = Z/sqrt(N)``, reported as three SEPARATE rows -- never fused into one
    "vs. best PDR" number (Recomendaciones 6/7's own separation carried into T8.4).

    Args:
        regime_realized_df: One regime's :func:`realized_quality_table` output.
        e1_df: E1 rows (nominal, seed 0), used to pick ``pdr_global``/
            ``pdr_por_familia``'s winning rule names and each instance's family.
        pdr_candidates: The PDR names eligible as comparators (nine, or the
            arrival-safe subset under an arrivals regime).
        regime_label: Carried into the output ``regime`` column.

    Returns:
        Up to three rows: ``comparator`` in
        ``{"pdr_global", "pdr_por_familia", "oraculo_best_per_instance"}``,
        ``comparator_method`` (rule name(s), or "(oraculo, varia)"),
        ``statistic``, ``p_value``, ``effect_size_r``, ``median_diff``, ``n``.
        A comparator is skipped (not silently zero-filled) if RL or the comparator has
        no matching rows in ``regime_realized_df``.
    """
    rl_rows = regime_realized_df[regime_realized_df["method"] == "rl"]
    if rl_rows.empty:
        return pd.DataFrame(
            columns=["regime", "comparator", "comparator_method", "statistic", "p_value", "effect_size_r", "median_diff", "n"]
        )

    out_rows = []

    glob = pdr_global(e1_df, candidates=pdr_candidates)
    glob_rows = regime_realized_df[regime_realized_df["method"] == glob.method]
    if not glob_rows.empty:
        merged = pd.concat([glob_rows.assign(_arm="glob"), rl_rows.assign(_arm="rl")])
        is_glob = merged["_arm"] == "glob"
        x, y, _ = align_paired(
            merged, key_cols=["instance"], x_col="mean_rpd_realized_pct", y_col="mean_rpd_realized_pct",
            x_filter=is_glob, y_filter=~is_glob,
        )
        result = wilcoxon_signed_rank(x, y)
        out_rows.append(
            {
                "regime": regime_label, "comparator": "pdr_global", "comparator_method": glob.method,
                "statistic": result.statistic, "p_value": result.p_value, "effect_size_r": result.effect_size_r,
                "median_diff": result.median_diff, "n": result.n_effective,
            }
        )

    familias = pdr_por_familia(e1_df, candidates=pdr_candidates)
    familia_method_by_instance = {}
    family_of_instance = e1_df.drop_duplicates("instance").set_index("instance")["family"].to_dict()
    familia_winner = {f.family: f.method for f in familias}
    for instance in regime_realized_df["instance"].unique():
        fam = family_of_instance.get(instance)
        if fam in familia_winner:
            familia_method_by_instance[instance] = familia_winner[fam]
    if familia_method_by_instance:
        familia_rows = regime_realized_df[
            regime_realized_df.apply(lambda r: familia_method_by_instance.get(r["instance"]) == r["method"], axis=1)
        ]
        if not familia_rows.empty:
            merged = pd.concat([familia_rows.assign(_arm="familia"), rl_rows.assign(_arm="rl")])
            is_familia = merged["_arm"] == "familia"
            x, y, _ = align_paired(
                merged, key_cols=["instance"], x_col="mean_rpd_realized_pct", y_col="mean_rpd_realized_pct",
                x_filter=is_familia, y_filter=~is_familia,
            )
            result = wilcoxon_signed_rank(x, y)
            out_rows.append(
                {
                    "regime": regime_label, "comparator": "pdr_por_familia",
                    "comparator_method": ", ".join(sorted(set(familia_method_by_instance.values()))),
                    "statistic": result.statistic, "p_value": result.p_value, "effect_size_r": result.effect_size_r,
                    "median_diff": result.median_diff, "n": result.n_effective,
                }
            )

    oracle_df = oracle_best_per_instance(e1_df, candidates=pdr_candidates)
    oracle_realized_rows = []
    for _, orow in oracle_df.iterrows():
        cell = regime_realized_df[
            (regime_realized_df["instance"] == orow["instance"]) & (regime_realized_df["method"] == orow["oracle_winning_pdr"])
        ]
        if not cell.empty:
            oracle_realized_rows.append(cell.iloc[0])
    if oracle_realized_rows:
        oracle_realized_df = pd.DataFrame(oracle_realized_rows)
        merged = pd.concat([oracle_realized_df.assign(_arm="oracle"), rl_rows.assign(_arm="rl")])
        is_oracle = merged["_arm"] == "oracle"
        x, y, _ = align_paired(
            merged, key_cols=["instance"], x_col="mean_rpd_realized_pct", y_col="mean_rpd_realized_pct",
            x_filter=is_oracle, y_filter=~is_oracle,
        )
        result = wilcoxon_signed_rank(x, y)
        out_rows.append(
            {
                "regime": regime_label, "comparator": "oraculo_best_per_instance",
                "comparator_method": "(oraculo, varia por instancia)",
                "statistic": result.statistic, "p_value": result.p_value, "effect_size_r": result.effect_size_r,
                "median_diff": result.median_diff, "n": result.n_effective,
            }
        )

    return pd.DataFrame(out_rows)


def to_markdown(realized_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    lines = ["## RPD realizada por metodo y regimen", "", "| Regimen | Instancia | Metodo | RPD realizada media (%) | DE | CVaR90 |", "|---|---|---|---|---|---|"]
    for _, row in realized_df.iterrows():
        lines.append(
            f"| {row['regime']} | {row['instance']} | {row['method']} | {row['mean_rpd_realized_pct']:.3f} | "
            f"{row['std_rpd_realized_pct']:.3f} | {row['cvar_90_rpd_realized_pct']:.3f} |"
        )
    lines += ["", "## RL vs. los tres comparadores (T4.1), separados", "", "| Regimen | Comparador | Regla(s) | W | p | r=Z/sqrt(N) | Mediana diff | N |", "|---|---|---|---|---|---|---|---|"]
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['regime']} | {row['comparator']} | {row['comparator_method']} | {row['statistic']:.2f} | "
            f"{row['p_value']:.2e} | {row['effect_size_r']:+.4f} | {row['median_diff']:+.3f} | {int(row['n'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: build Tabla_calidad_realizada_disrupcion across every H3/H4 regime that has
    landed so far (``--skip-missing``-style: a missing regime CSV is skipped, not fatal).

        python -m experiments.stats.realized_quality \\
            --results-dir experiments/results \\
            --out results/tablas/Tabla_calidad_realizada_disrupcion
    """
    import argparse

    from experiments.stats.cli import default_experiment_csvs
    from experiments.stats.io import load_results
    from experiments.stats.report import H3_H4_INSTANCES, REGIMES
    from experiments.stats.tables import to_latex_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "results" / "tablas" / "Tabla_calidad_realizada_disrupcion")
    args = parser.parse_args(argv)

    csvs = default_experiment_csvs(args.results_dir)
    combined, _provenance = load_results(csvs, skip_missing=True)

    e1 = combined[(combined.get("experiment_id") == "E1") & (combined.get("disruption") == "none")] if not combined.empty else combined
    if e1.empty:
        print("[realized_quality] no E1 data available; nothing to compute")
        return 0
    certified_optimum = e1.dropna(subset=["known_optimum"]).drop_duplicates("instance").set_index("instance")["known_optimum"].astype(float).to_dict()

    realized_frames, comparison_frames = [], []
    for regime_label, _disruption, _level, candidates in REGIMES:
        regime_df = combined[(combined["experiment_id"] == regime_label) & (combined["instance"].isin(H3_H4_INSTANCES))]
        if regime_df.empty:
            continue
        realized = realized_quality_table(regime_df, certified_optimum=certified_optimum, regime_label=regime_label)
        if realized.empty:
            continue
        realized_frames.append(realized)
        comparison_frames.append(
            compare_rl_vs_three_comparators(realized, e1, pdr_candidates=candidates, regime_label=regime_label)
        )

    if not realized_frames:
        print("[realized_quality] no regime data available yet; nothing to compute")
        return 0

    realized_all = pd.concat(realized_frames, ignore_index=True)
    comparison_all = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    md = to_markdown(realized_all, comparison_all)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".csv").write_text(realized_all.to_csv(index=False), encoding="utf-8")
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".tex").write_text(
        to_latex_table(md, caption="Calidad realizada bajo disrupcion (RPD vs. optimo certificado) y comparacion RL vs. tres comparadores", label="calidad-realizada-disrupcion"),
        encoding="utf-8",
    )
    print(f"wrote {args.out}.{{csv,md,tex}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
