"""Compact, English-language figures for the IEEE conference paper
(``papers/ConferencePaper_DRAFT.md``), page-budget-aware (target: 6-8 pages,
two-column).

This does NOT recompute any statistic independently of the already-tested
pipeline: every number plotted comes from the same ``experiments.stats``
functions (``io.load_results``, ``report.*``, ``omnibus.friedman_test``/
``critical_difference``) that ``generate_thesis_figures.py`` already uses for
the thesis figures, and that Chapter 6/7 cite. This script only combines and
restyles them -- multi-panel/compact layouts and English labels, since the
thesis figures (one figure per tier/regime, Spanish labels) would blow the
paper's page budget if reused as-is.

The one genuinely new computation is the FJSP "gap to best{GA,CP-SAT}"
overview bar (Fig_P1) and the H5 la40-sensitivity table (see
``export_paper_supplement_tables.py``) -- both are direct re-expressions of
numbers already derived and cited in Chapter 6/7 (Section 6.2.1's Tabla
6.2.3 recomputation and Section 6.5.1's sensitivity reading), not new
experiments.

Usage
-----
    python scripts/generate_paper_figures.py [--out-dir ../figures/paper]
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

AZUL = "#2a78d6"
VIOLETA = "#4a3aa7"
VERDE = "#008300"
ROJO = "#e34948"
NARANJA = "#eb6834"
GRIS = "#8a8a86"
NEGRO = "#1a1a19"


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dedcd4",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#1a1a19",
        "text.color": "#1a1a19",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
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


# --------------------------------------------------------------------------- Fig P1: H1 overview


def generate_fig_p1_h1_overview(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Grouped bar: mean RPD by method, one group per instance family.

    JSSP (E1): RPD is already computed against the published optimum
    (``rpd_pct``), for RL, GA, CP-SAT, and "best PDR per instance" (the same
    per-instance-min aggregate ``report.run_h1`` uses for the H1 Wilcoxon test).

    FJSP arms (E2a synthetic, E2b Brandimarte): no PDR has an FJSP
    machine-selection rule (Section III-C), so there is no "best PDR" bar;
    RPD is expressed relative to best{GA, CP-SAT} per instance -- the exact
    comparator ``pairwise.h1_fjsp_test`` uses -- for all three methods
    (RL, GA, CP-SAT), so GA's and CP-SAT's own bars show their gap to
    whichever of the two is smaller on a given instance.
    """
    groups: dict[str, dict[str, float]] = {}

    e1 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    pdr_rows = e1[e1["method"].isin(ALL_PDR_NAMES)]
    best_pdr = pdr_rows.loc[pdr_rows.groupby("instance")["rpd_pct"].idxmin()]
    groups["JSSP\n(N=43)"] = {
        "RL": float(e1[(e1["method"] == "rl") & (e1["seed"] == 0)]["rpd_pct"].mean()),
        "Best PDR": float(best_pdr["rpd_pct"].mean()),
        "GA": float(e1[e1["method"] == "ga"]["rpd_pct"].mean()),
        "CP-SAT": float(e1[e1["method"] == "cpsat"]["rpd_pct"].mean()),
    }

    for experiment_id, label in (("E2a", "FJSP synth.\n(N=43)"), ("E2b", "FJSP Brandim.\n(N=7)")):
        e2 = combined[(combined["experiment_id"] == experiment_id) & (combined["disruption"] == "none")]
        if e2.empty:
            continue
        wide = e2.pivot_table(index="instance", columns="method", values="makespan", aggfunc="first")
        best_comparator = wide[["ga", "cpsat_fjsp"]].min(axis=1)
        vals = {}
        for method, out_name in (("rl", "RL"), ("ga", "GA"), ("cpsat_fjsp", "CP-SAT")):
            rpd = (wide[method] - best_comparator) / best_comparator * 100.0
            vals[out_name] = float(rpd.mean())
        groups[label] = vals

    method_order = ["RL", "Best PDR", "GA", "CP-SAT"]
    method_color = {"RL": AZUL, "Best PDR": GRIS, "GA": VERDE, "CP-SAT": NARANJA}
    family_labels = list(groups.keys())

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    n_families = len(family_labels)
    bar_w = 0.19
    x = np.arange(n_families)
    for i, method in enumerate(method_order):
        ys = [groups[fam].get(method, np.nan) for fam in family_labels]
        offset = (i - 1.5) * bar_w
        bars = ax.bar(x + offset, ys, width=bar_w, color=method_color[method], label=method,
                       edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.1f", fontsize=5.6, padding=1.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels)
    ax.set_ylabel("Mean RPD (%)")
    ax.set_title("H1 -- static competitiveness by instance family")
    ax.legend(loc="upper left", frameon=False, ncols=2, columnspacing=0.8, handlelength=1.2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_P1_h1_rpd_overview")


# --------------------------------------------------------------------------- Fig P2: H2 CD grid


def _draw_cd_diagram(ax: plt.Axes, average_ranks: dict, cd: float, *, title: str) -> None:
    methods = sorted(average_ranks, key=lambda m: average_ranks[m])
    ranks = [average_ranks[m] for m in methods]
    labels = [m.upper() for m in methods]
    k = len(methods)

    lo, hi = min(ranks) - 0.6, max(ranks) + 0.6
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
    row_step = 0.40

    ax.grid(False)
    ax.set_xlim(lo, hi)
    top = 0.85 + row_step * (max_row + 1) + 0.15
    ax.set_ylim(0, top)
    ax.axhline(0.85, color=NEGRO, lw=0.9)
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "left", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    for rank, label, row in zip(ranks, labels, row_of):
        tick_top = 0.90 + row * row_step
        text_y = tick_top + 0.02
        ax.plot([rank, rank], [0.85, tick_top], color=NEGRO, lw=0.8)
        ax.text(rank, text_y, f"{label}\n({rank:.2f})", ha="center", va="bottom", fontsize=6.2)

    cd_y = 0.50
    ax.plot([lo + 0.15, lo + 0.15 + cd], [cd_y, cd_y], color=NEGRO, lw=1.6)
    ax.text(lo + 0.15 + cd / 2, cd_y + 0.04, f"CD={cd:.2f}", ha="center", fontsize=6.2)

    order = np.argsort(ranks)
    sorted_ranks = np.array(ranks)[order]
    bracket_groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] - sorted_ranks[start] > cd:
            if i - 1 > start:
                bracket_groups.append((start, i - 1))
            start = i
    if len(sorted_ranks) - 1 > start:
        bracket_groups.append((start, len(sorted_ranks) - 1))

    bracket_y = 0.72
    for lo_i, hi_i in bracket_groups:
        ax.plot([sorted_ranks[lo_i], sorted_ranks[hi_i]], [bracket_y, bracket_y],
                 color=NEGRO, lw=3.0, solid_capstyle="butt")
        bracket_y -= 0.08

    ax.set_xlabel("Mean rank (lower = better)", fontsize=7.5)
    ax.set_title(title, fontsize=8)


def generate_fig_p2_h2_cd_grid(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e3 = combined[combined["experiment_id"] == "E3"]
    h2_methods = ("rl", *ALL_PDR_NAMES)
    tier_titles = {
        "1.5-2x (la21-la25)": "1.5x (la21-la25)",
        "2.0x (la26-la30)": "2.0x (la26-la30)",
        "2.25x (la36-la40)": "2.25x (la36-la40)",
        "3.0x (la31-la35)": "3.0x (la31-la35)",
    }
    order = ["1.5-2x (la21-la25)", "2.0x (la26-la30)", "2.25x (la36-la40)", "3.0x (la31-la35)"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for ax, tier_label in zip(axes.flat, order):
        instances = H2_TIERS[tier_label]
        tier_df = e3[e3["instance"].isin(instances) & e3["method"].isin(h2_methods)]
        pivot = tier_df.pivot_table(index="instance", columns="method", values="rpd_pct", aggfunc="first").dropna()
        friedman = friedman_test(pivot)
        cd = critical_difference(friedman.k_methods, friedman.n_blocks)
        _draw_cd_diagram(ax, friedman.average_ranks, cd, title=tier_titles[tier_label])

    fig.suptitle("H2 -- critical-difference diagrams, all four size tiers", fontsize=9.5, y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_P2_h2_cd_grid")


# --------------------------------------------------------------------------- Fig P3: H3/H4 combined


def generate_fig_p3_h3h4_combined(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e1_all = combined[combined["experiment_id"] == "E1"]
    nominal = nominal_makespan_lookup(e1_all)

    axes_spec = (
        ("Breakdowns", "Target availability ($A_k$)",
         [("E4_breakdowns_A0.95", 0.95), ("E4_breakdowns_A0.90", 0.90), ("E4_breakdowns_A0.80", 0.80)]),
        ("Stochastic times", "$\\sigma$ (log-normal spread)",
         [("E5_stochastic_s0.10", 0.10), ("E5_stochastic_s0.25", 0.25), ("E5_stochastic_s0.50", 0.50)]),
        ("Arrivals", "Target utilization ($\\rho$)",
         [("E6_arrivals_u0.70", 0.70), ("E6_arrivals_u0.85", 0.85), ("E6_arrivals_u0.95", 0.95)]),
    )
    series_style = {
        "best_pdr": dict(color=AZUL, marker="o", linestyle="-", label="Best PDR"),
        "rl_kappa0": dict(color=VIOLETA, marker="s", linestyle="--", label="RL, $\\kappa=0$"),
        "rl_kappa0.5": dict(color=VERDE, marker="^", linestyle="-", label="RL, $\\kappa=0.5$"),
        "rl_kappa1.0": dict(color=ROJO, marker="D", linestyle="-.", label="RL, $\\kappa=1.0$"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.15), sharey=True)
    for ax, (axis_name, x_label, regime_levels) in zip(axes, axes_spec):
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
                ax.plot(np.array(xs)[order], np.array(ys)[order], markersize=4.5, linewidth=1.4, **style)
        ax.set_xlabel(x_label, fontsize=7.5)
        ax.set_title(axis_name, fontsize=8.5)

    axes[0].set_ylabel("Robustness Index (RI, mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("H3/H4 -- robustness index vs. disruption level, all three axes", fontsize=9)
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out_dir, "Fig_P3_h3h4_ri_combined")


# --------------------------------------------------------------------------- Fig P4: H5 latency


def generate_fig_p4_h5_latency(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    e8 = combined[combined["experiment_id"] == "E8"]
    if e8.empty:
        e8 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    rl_rows = e8[(e8["method"] == "rl") & (e8["seed"] == 0) & e8["ms_per_decision"].notna()]
    cpsat_rows = e8[(e8["method"] == "cpsat") & e8["ms_per_decision"].notna()]
    rl_med = rl_rows.groupby("instance").agg(ms_rl=("ms_per_decision", "median"), num_operations=("num_operations", "first"))
    cpsat_med = cpsat_rows.groupby("instance").agg(ms_cpsat=("ms_per_decision", "median"))
    merged = rl_med.join(cpsat_med, how="inner").sort_values("num_operations")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(merged["num_operations"], merged["ms_rl"], marker="o", s=18, color=AZUL,
               label="RL (median ms/decision)", zorder=3)
    ax.scatter(merged["num_operations"], merged["ms_cpsat"], marker="x", s=24, color=NARANJA,
               linewidths=1.1, label="CP-SAT (median ms/decision)", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of operations (instance size)")
    ax.set_ylabel("ms / decision (log scale)")
    ax.set_title("H5 -- decision latency vs. instance size")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, "Fig_P4_h5_latency")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT.parent / "figures" / "paper")
    args = parser.parse_args(argv)

    _set_paper_style()
    combined = load_combined()

    written: list[Path] = []
    written += generate_fig_p1_h1_overview(combined, args.out_dir)
    written += generate_fig_p2_h2_cd_grid(combined, args.out_dir)
    written += generate_fig_p3_h3h4_combined(combined, args.out_dir)
    written += generate_fig_p4_h5_latency(combined, args.out_dir)

    print(f"[generate_paper_figures] wrote {len(written)} files to {args.out_dir}")
    for p in sorted(written):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
