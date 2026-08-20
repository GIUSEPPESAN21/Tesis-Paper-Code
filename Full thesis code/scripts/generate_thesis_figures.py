"""Camera-ready thesis figures for Chapter 6, in Spanish, colorblind-safe.

Regenerates the 9 figures already cited by file path in ``Chapter6_FINAL.md``
(Figuras 6.3.1, 6.3.2, 6.4.1, 6.5.1), at thesis print quality (300dpi PNG + vector
PDF), with Spanish axis/legend labels and a validated colorblind-safe palette
(``dataviz`` skill's reference categorical palette, checked with
``scripts/validate_palette.js`` -- see the commit/session notes for the exact
validator invocations and results).

Data is NOT recomputed independently of the already-tested statistics pipeline:
every number plotted here comes from the same ``experiments.stats`` functions
(``io.load_results``, ``report.nominal_makespan_lookup``/``add_row_ri``/
``best_pdr_for_regime``, ``omnibus.friedman_test``/``critical_difference``) that
``experiments/stats/cli.py`` already ran and that ``Chapter6_FINAL.md``'s numbers
are drawn from -- this script only changes how the numbers are rendered, not what
they are. Re-running this script must reproduce numerically identical figures to
the ones already cited, just with thesis-appropriate styling instead of the
pipeline's internal verification styling.

Usage
-----
    python scripts/generate_thesis_figures.py [--out-dir ../figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.cost_accounting import amortization_curve, inference_cost_per_instance
from experiments.slack_vs_robustness import paired_points
from experiments.stats.cli import default_experiment_csvs
from experiments.stats.io import load_results
from experiments.stats.omnibus import critical_difference, friedman_test
from experiments.stats.report import (
    ALL_PDR_NAMES,
    H2_TIERS,
    H3_H4_INSTANCES,
    REGIMES,
    add_row_ri,
    best_pdr_for_regime,
    nominal_makespan_lookup,
)

# --------------------------------------------------------------------------- style
#
# Validated (node scripts/validate_palette.js, dataviz skill, --pairs all):
#   4-series set {azul, violeta, verde, rojo}: ALL CHECKS PASS (CVD worst-pair
#   WARN band 7.2, legal with secondary encoding -- distinct markers/linestyles
#   below satisfy that).
#   2-series {azul, naranja}: ALL CHECKS PASS cleanly (worst-pair CVD ΔE 24.7).
# Color assignments match the captions already written in Chapter6_FINAL.md
# verbatim ("la mejor PDR (azul)", "la serie kappa=0.5 (verde)", "RL (azul)",
# "CP-SAT (naranja)") -- these are not free choices, they were fixed by the prose
# this script's output must agree with.

AZUL = "#2a78d6"
VIOLETA = "#4a3aa7"
VERDE = "#008300"
ROJO = "#e34948"
NARANJA = "#eb6834"
GRIS = "#8a8a86"
NEGRO = "#1a1a19"

PDR_SPANISH = {
    "spt": "SPT", "lpt": "LPT", "mwkr": "MWKR", "lwkr": "LWKR", "mor": "MOR",
    "lor": "LOR", "fifo": "FIFO", "edd": "EDD", "cr": "CR",
}


def _set_thesis_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dedcd4",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#1a1a19",
        "text.color": "#1a1a19",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        # Embed real (searchable, LaTeX/Word-safe) fonts in vector PDFs, not
        # Type 3 bitmap-ish fonts (matplotlib's default) which some LaTeX/Word
        # PDF pipelines render poorly.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def load_combined() -> pd.DataFrame:
    csvs = default_experiment_csvs(REPO_ROOT / "experiments" / "results")
    combined, _provenance = load_results(csvs, skip_missing=True)
    return combined


# --------------------------------------------------------------------------- Fig 6.3.1


def plot_cd_diagram_es(average_ranks: dict, cd: float, *, title: str, out_dir: Path, stem: str) -> list[Path]:
    """Demsar-style CD diagram, Spanish labels, thesis styling. Monochrome by
    convention (Demsar 2006 diagrams are traditionally black-and-white; there is
    no categorical-identity color job here, only rank position), so the
    colorblind-palette check does not apply to this figure.
    """
    methods = sorted(average_ranks, key=lambda m: average_ranks[m])
    ranks = [average_ranks[m] for m in methods]
    k = len(methods)
    labels = [PDR_SPANISH.get(m, m.upper()) for m in methods]

    lo, hi = min(ranks) - 0.6, max(ranks) + 0.6

    # Cluster-based label staggering: chain consecutive ranks within
    # `collision_gap` into clusters, then give each member of a cluster its own
    # row (0, 1, 2, ...) so no two mutually-close labels ever share a row --
    # simple immediate-neighbor alternation isn't enough once 3+ methods bunch
    # together (checked by eye per the dataviz skill's render-and-look step;
    # tiers with mor/mwkr and edd/lor/lwkr/cr clusters overlapped badly under
    # 2-row alternation).
    collision_gap = (hi - lo) * 0.075
    clusters: list[list[int]] = [[0]]
    for i in range(1, k):
        if ranks[i] - ranks[i - 1] <= collision_gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    row_of = [0] * k
    for cluster in clusters:
        for row, idx in enumerate(cluster):
            row_of[idx] = row
    max_row = max(row_of)
    row_step = 0.34  # vertical room per stagger row; must clear a 2-line label block

    fig, ax = plt.subplots(figsize=(max(8.5, k * 1.15), 2.6 + max_row * 0.75))
    ax.grid(False)
    ax.set_xlim(lo, hi)
    top = 0.85 + row_step * (max_row + 1) + 0.15
    ax.set_ylim(0, top)
    ax.axhline(0.85, color=NEGRO, lw=1)
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "left", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    for rank, label, row in zip(ranks, labels, row_of):
        tick_top = 0.90 + row * row_step
        text_y = tick_top + 0.02
        ax.plot([rank, rank], [0.85, tick_top], color=NEGRO, lw=1)
        ax.text(rank, text_y, f"{label}\n({rank:.2f})", ha="center", va="bottom", fontsize=9)

    cd_y = 0.50
    ax.plot([lo + 0.15, lo + 0.15 + cd], [cd_y, cd_y], color=NEGRO, lw=2)
    ax.text(lo + 0.15 + cd / 2, cd_y + 0.04, f"DC = {cd:.3f}", ha="center", fontsize=9)

    order = np.argsort(ranks)
    sorted_ranks = np.array(ranks)[order]
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] - sorted_ranks[start] > cd:
            if i - 1 > start:
                groups.append((start, i - 1))
            start = i
    if len(sorted_ranks) - 1 > start:
        groups.append((start, len(sorted_ranks) - 1))

    bracket_y = 0.72
    for lo_i, hi_i in groups:
        ax.plot([sorted_ranks[lo_i], sorted_ranks[hi_i]], [bracket_y, bracket_y],
                 color=NEGRO, lw=3.5, solid_capstyle="butt")
        bracket_y -= 0.08

    ax.set_xlabel("Rango medio (menor = mejor)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def generate_fig_6_3_1(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e3 = combined[combined["experiment_id"] == "E3"]
    h2_methods = ("rl", *ALL_PDR_NAMES)
    written: list[Path] = []
    tier_stems = {
        "1.5-2x (la21-la25)": ("Fig_6.3.1a_dc_tier_1.5x", "E3 -- tier 1.5x (la21-la25), H2"),
        "2.0x (la26-la30)": ("Fig_6.3.1b_dc_tier_2.0x", "E3 -- tier 2.0x (la26-la30), H2"),
        "3.0x (la31-la35)": ("Fig_6.3.1d_dc_tier_3.0x", "E3 -- tier 3.0x (la31-la35), H2"),
        "2.25x (la36-la40)": ("Fig_6.3.1c_dc_tier_2.25x", "E3 -- tier 2.25x (la36-la40), H2"),
    }
    for tier_label, instances in H2_TIERS.items():
        tier_df = e3[e3["instance"].isin(instances) & e3["method"].isin(h2_methods)]
        pivot = tier_df.pivot_table(index="instance", columns="method", values="rpd_pct", aggfunc="first").dropna()
        friedman = friedman_test(pivot)
        cd = critical_difference(friedman.k_methods, friedman.n_blocks)
        stem, title = tier_stems[tier_label]
        written.extend(plot_cd_diagram_es(friedman.average_ranks, cd, title=title, out_dir=out_dir, stem=stem))
    return written


# --------------------------------------------------------------------------- Fig 6.3.2


def generate_fig_6_3_2(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e3 = combined[combined["experiment_id"] == "E3"]
    h2_methods = ("rl", *ALL_PDR_NAMES)
    e3_scoped = e3[e3["method"].isin(h2_methods)].copy()
    tier_of = {inst: tier for tier, insts in H2_TIERS.items() for inst in insts}
    e3_scoped["tier"] = e3_scoped["instance"].map(tier_of)
    e3_scoped = e3_scoped.dropna(subset=["tier"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pdr_df = e3_scoped[e3_scoped["method"] != "rl"]
    ax.scatter(pdr_df["num_operations"], pdr_df["rpd_pct"], marker="x", s=22,
               color=GRIS, alpha=0.55, linewidths=1.1, label="9 PDR (contexto)", zorder=1)
    rl_df = e3_scoped[e3_scoped["method"] == "rl"].sort_values("num_operations")
    ax.scatter(rl_df["num_operations"], rl_df["rpd_pct"], marker="o", s=55,
               color=AZUL, edgecolors="white", linewidths=0.6, label="RL", zorder=3)

    ax.set_xlabel("Número de operaciones (tamaño de instancia)")
    ax.set_ylabel("RPD (%)")
    ax.set_title("E3 -- RPD de RL en función del tamaño de instancia (H2)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_6.3.2_rpd_vs_tamano")


# --------------------------------------------------------------------------- Fig 6.4.1


def generate_fig_6_4_1(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e1_all = combined[combined["experiment_id"] == "E1"]
    nominal = nominal_makespan_lookup(e1_all)

    axes_spec = (
        ("Averías", "Disponibilidad objetivo ($A_k$)",
         [("E4_breakdowns_A0.95", 0.95), ("E4_breakdowns_A0.90", 0.90), ("E4_breakdowns_A0.80", 0.80)],
         "Fig_6.4.1a_ri_averias"),
        ("Tiempos estocásticos", "$\\sigma$ (dispersión log-normal)",
         [("E5_stochastic_s0.10", 0.10), ("E5_stochastic_s0.25", 0.25), ("E5_stochastic_s0.50", 0.50)],
         "Fig_6.4.1b_ri_tiempos_estocasticos"),
        ("Llegadas", "Utilización objetivo ($\\rho$)",
         [("E6_arrivals_u0.70", 0.70), ("E6_arrivals_u0.85", 0.85), ("E6_arrivals_u0.95", 0.95)],
         "Fig_6.4.1c_ri_llegadas"),
    )
    series_style = {
        "best_pdr": dict(color=AZUL, marker="o", linestyle="-", label="Mejor PDR"),
        "rl_kappa0": dict(color=VIOLETA, marker="s", linestyle="--", label="RL, $\\kappa=0$"),
        "rl_kappa0.5": dict(color=VERDE, marker="^", linestyle="-", label="RL, $\\kappa=0.5$"),
        "rl_kappa1.0": dict(color=ROJO, marker="D", linestyle="-.", label="RL, $\\kappa=1.0$"),
    }

    written: list[Path] = []
    for axis_name, x_label, regime_levels, stem in axes_spec:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        series = {k: ([], []) for k in series_style}
        for regime_label, level in regime_levels:
            regime_df = combined[(combined["experiment_id"] == regime_label) & (combined["instance"].isin(H3_H4_INSTANCES))]
            if not regime_df.empty:
                regime_ri = add_row_ri(regime_df, nominal)
                candidates = next(c for rl, _, _, c in REGIMES if rl == regime_label)
                pdr_ri = regime_ri[regime_ri["method"].isin(candidates)]
                if not pdr_ri.empty:
                    _best_pdr, best_ri = best_pdr_for_regime(pdr_ri, candidates)
                    series["best_pdr"][0].append(level)
                    series["best_pdr"][1].append(best_ri)
                rl_ri = regime_ri[regime_ri["method"] == "rl"]["ri"]
                if not rl_ri.empty:
                    series["rl_kappa0"][0].append(level)
                    series["rl_kappa0"][1].append(float(rl_ri.mean()))
            for kappa in (0.5, 1.0):
                k_df = combined[combined["experiment_id"] == f"E7_kappa{kappa}_{regime_label}"]
                if k_df.empty:
                    continue
                k_ri = add_row_ri(k_df, nominal)["ri"]
                if not k_ri.empty:
                    series[f"rl_kappa{kappa}"][0].append(level)
                    series[f"rl_kappa{kappa}"][1].append(float(k_ri.mean()))

        for name, (xs, ys) in series.items():
            if xs:
                order = np.argsort(xs)
                style = series_style[name]
                ax.plot(np.array(xs)[order], np.array(ys)[order], markersize=7, linewidth=2, **style)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Índice de Robustez ($RI$, media)")
        ax.set_title(f"{axis_name} -- $RI$ vs. nivel de perturbación (H3/H4)")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        written.extend(_save(fig, out_dir, stem))
    return written


# --------------------------------------------------------------------------- Fig 6.5.1


def generate_fig_6_5_1(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e8 = combined[combined["experiment_id"] == "E8"]
    if e8.empty:
        e8 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    rl_rows = e8[(e8["method"] == "rl") & (e8["seed"] == 0) & e8["ms_per_decision"].notna()]
    cpsat_rows = e8[(e8["method"] == "cpsat") & e8["ms_per_decision"].notna()]
    rl_med = rl_rows.groupby("instance").agg(ms_rl=("ms_per_decision", "median"), num_operations=("num_operations", "first"))
    cpsat_med = cpsat_rows.groupby("instance").agg(ms_cpsat=("ms_per_decision", "median"))
    merged = rl_med.join(cpsat_med, how="inner").sort_values("num_operations")
    merged["ratio_cpsat_over_rl"] = merged["ms_cpsat"] / merged["ms_rl"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.scatter(merged["num_operations"], merged["ms_rl"], marker="o", s=32, color=AZUL, label="RL (mediana ms/decisión)", zorder=3)
    ax1.scatter(merged["num_operations"], merged["ms_cpsat"], marker="x", s=42, color=NARANJA, linewidths=1.4, label="CP-SAT (mediana ms/decisión)", zorder=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("ms / decisión (escala log)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_title("E8 -- tiempo de decisión vs. tamaño de instancia (H5)")

    ax2.scatter(merged["num_operations"], merged["ratio_cpsat_over_rl"], marker="o", s=32, color=VERDE, zorder=3)
    ax2.axhline(100, color="#52514e", linestyle="--", linewidth=1.2, label="Umbral 1% (razón=100)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Número de operaciones (tamaño de instancia)")
    ax2.set_ylabel("Razón CP-SAT/RL\n(escala log)")
    ax2.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_6.5.1_latencia_vs_tamano")


# --------------------------------------------------------------------------- Fig holgura (T1.4)


def generate_fig_holgura_vs_ri(
    slack_df: pd.DataFrame, combined: pd.DataFrame, out_dir: Path, *, metric: str = "slack_ratio"
) -> list[Path]:
    """Fig_holgura_vs_RI (T1.4, Recomendacion 2): dispersion de una metrica de holgura
    (por defecto ``slack_ratio``) vs. RI, un panel por eje de disrupcion (averias,
    tiempos estocasticos, llegadas), con ajuste lineal por panel.

    Design constraint carried over from ``experiments.slack_vs_robustness`` (T1.3):
    this figure reports whatever sign/shape the data actually has -- it does not
    force the "more slack -> more robust" narrative if the fit line comes out flat or
    positive on a given axis.

    Args:
        slack_df: ``run_slack_study.py``'s nominal slack metrics (one row per
            instance/method).
        combined: The pipeline's usual E1-E8 long-format frame (``load_combined()``),
            used to build each axis's RI-by-regime target via
            ``experiments.stats.report.add_row_ri``/``nominal_makespan_lookup``.
        metric: Which of the six slack metrics to plot (T1.3's full correlation table
            covers all six; this figure highlights one at a time -- call again with a
            different ``metric`` for additional panels/figures if the committee asks).

    Returns:
        Written file paths, or ``[]`` if neither ``slack_df`` nor the regime data is
        available yet (printed note, consistent with this script's other figures).
    """
    if slack_df.empty:
        print("[generate_thesis_figures] Fig_holgura_vs_RI: skipped -- no slack_df (run experiments.run_slack_study first)")
        return []

    e1_all = combined[combined["experiment_id"] == "E1"]
    if e1_all.empty:
        print("[generate_thesis_figures] Fig_holgura_vs_RI: skipped -- no E1 data for nominal makespans")
        return []
    nominal = nominal_makespan_lookup(e1_all)

    axes_spec = (
        ("Averías", ["E4_breakdowns_A0.95", "E4_breakdowns_A0.90", "E4_breakdowns_A0.80"]),
        ("Tiempos estocásticos", ["E5_stochastic_s0.10", "E5_stochastic_s0.25", "E5_stochastic_s0.50"]),
        ("Llegadas", ["E6_arrivals_u0.70", "E6_arrivals_u0.85", "E6_arrivals_u0.95"]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    any_panel_has_data = False
    for ax, (axis_name, regime_labels) in zip(axes, axes_spec):
        regime_df = combined[combined["experiment_id"].isin(regime_labels) & combined["instance"].isin(H3_H4_INSTANCES)]
        if regime_df.empty:
            ax.set_title(f"{axis_name} (sin datos)")
            continue
        regime_ri = add_row_ri(regime_df, nominal)
        ri_by_instance_method = regime_ri.groupby(["instance", "method"], as_index=False)["ri"].mean()

        points = paired_points(slack_df, ri_by_instance_method, metric=metric, target_col="ri")
        if points.empty:
            ax.set_title(f"{axis_name} (sin pares instancia/metodo)")
            continue
        any_panel_has_data = True

        ax.scatter(points["metric_value"], points["target_value"], color=AZUL, edgecolors="white", linewidths=0.5, s=45, zorder=3)
        if len(points) >= 2 and points["metric_value"].std() > 0:
            slope, intercept = np.polyfit(points["metric_value"], points["target_value"], 1)
            xs = np.linspace(points["metric_value"].min(), points["metric_value"].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=NARANJA, linewidth=2, linestyle="--", zorder=2)
        ax.set_title(axis_name)
        ax.set_xlabel(metric)

    axes[0].set_ylabel("Índice de Robustez ($RI$, media)")
    fig.suptitle(f"Holgura ({metric}) vs. $RI$, por eje de disrupción (T1.4)")
    fig.tight_layout()
    if not any_panel_has_data:
        plt.close(fig)
        print("[generate_thesis_figures] Fig_holgura_vs_RI: skipped -- no regime had matching data")
        return []
    return _save(fig, out_dir, "Fig_holgura_vs_RI")


# --------------------------------------------------------------------------- Fig 6.5.2 (T5.2)


def generate_fig_6_5_2(combined: pd.DataFrame, out_dir: Path, *, e8_timing_csv: Path, training_wall_clock_s: float | None) -> list[Path]:
    """Fig_costo_amortizado (T5.2, Recomendacion 8): cumulative cost, GNN-DRL
    (train-once + infer-per-instance) vs. CP-SAT (reoptimize-from-scratch every
    instance), as a function of N instances solved -- marks the crossover point where
    the offline (CP-SAT) approach stops amortizing.

    Args:
        e8_timing_csv: ``experiments/results/week24/e8_cpu_timing.csv`` (T5.3: full
            43-instance coverage). Figure is skipped (returns ``[]``, with a printed
            note) if this file does not exist yet -- consistent with this script's
            existing behaviour of skipping a figure when its source data is absent
            (``generate_fig_6_3_1``/etc. all guard on ``combined``/``e3`` being empty).
        training_wall_clock_s: GNN-DRL's one-time training cost (mean over seeds, from
            a ``summary_*.json``). ``None`` skips the figure the same way.
    """
    if not e8_timing_csv.exists() or training_wall_clock_s is None:
        print(
            f"[generate_thesis_figures] Fig_costo_amortizado: skipped -- "
            f"{'missing ' + str(e8_timing_csv) if not e8_timing_csv.exists() else ''}"
            f"{'no training_wall_clock_s given' if training_wall_clock_s is None else ''}"
        )
        return []

    e8_df = pd.read_csv(e8_timing_csv)
    inference_df = inference_cost_per_instance(e8_df).sort_values("num_operations")

    e1 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    cpsat_rows = e1[e1["method"] == "cpsat"]
    if cpsat_rows.empty:
        print("[generate_thesis_figures] Fig_costo_amortizado: skipped -- no E1 cpsat rows for the offline comparator")
        return []
    offline_per_instance_s = float(cpsat_rows["wall_clock_s"].mean())

    curve = amortization_curve(
        training_wall_clock_s, inference_df["total_inference_s"], offline_per_instance_s=offline_per_instance_s
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(curve["n_instances"], curve["gnn_drl_cumulative_s"], color=AZUL, linewidth=2, label="GNN-DRL (entrenar una vez + inferir)")
    ax.plot(curve["n_instances"], curve["offline_cumulative_s"], color=NARANJA, linewidth=2, linestyle="--", label="CP-SAT (reoptimizar cada instancia)")

    crossover = curve.index[curve["gnn_drl_cheaper"]].min() if curve["gnn_drl_cheaper"].any() else None
    if crossover is not None:
        n_star = int(curve.loc[crossover, "n_instances"])
        ax.axvline(n_star, color=GRIS, linestyle=":", linewidth=1.4)
        ax.text(n_star, ax.get_ylim()[1] * 0.05, f"  cruce en N={n_star}", fontsize=9, color=GRIS)

    ax.set_xlabel("Número de instancias resueltas (N)")
    ax.set_ylabel("Costo acumulado (s)")
    ax.set_title("Costo amortizado: GNN-DRL vs. CP-SAT en función de N (T5.2)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_6.5.2_costo_amortizado")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT.parent / "figures")
    parser.add_argument(
        "--e8-timing-csv", type=Path,
        default=REPO_ROOT / "experiments" / "results" / "week24" / "e8_cpu_timing.csv",
        help="T5.3: full-coverage timing CSV, input to Fig_costo_amortizado",
    )
    parser.add_argument(
        "--slack-csv", type=Path,
        default=REPO_ROOT.parent / "results" / "datos_crudos" / "slack" / "slack_metrics.csv",
        help="T1.4: experiments.run_slack_study's output, input to Fig_holgura_vs_RI",
    )
    parser.add_argument(
        "--training-summary-glob", default=None,
        help="glob for one run's summary_*_seed*.json files, to compute "
        "Fig_costo_amortizado's training-cost term (T5.2). Omit to skip that figure.",
    )
    args = parser.parse_args(argv)

    _set_thesis_style()
    combined = load_combined()

    written: list[Path] = []
    written += generate_fig_6_3_1(combined, args.out_dir)
    written += generate_fig_6_3_2(combined, args.out_dir)
    written += generate_fig_6_4_1(combined, args.out_dir)
    written += generate_fig_6_5_1(combined, args.out_dir)

    if args.slack_csv.exists():
        slack_df = pd.read_csv(args.slack_csv)
        written += generate_fig_holgura_vs_ri(slack_df, combined, args.out_dir)
    else:
        print(f"[generate_thesis_figures] Fig_holgura_vs_RI: skipped -- {args.slack_csv} not found")

    training_wall_clock_s = None
    if args.training_summary_glob:
        import glob as _glob
        import json as _json

        paths = _glob.glob(args.training_summary_glob)
        if paths:
            values = [_json.loads(Path(p).read_text(encoding="utf-8"))["train_wall_clock_s"] for p in paths]
            training_wall_clock_s = float(np.mean(values))
    written += generate_fig_6_5_2(combined, args.out_dir, e8_timing_csv=args.e8_timing_csv, training_wall_clock_s=training_wall_clock_s)

    print(f"[generate_thesis_figures] wrote {len(written)} files to {args.out_dir}")
    for p in sorted(written):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
