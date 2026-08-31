"""Publication figures. Vector PDF + 600-dpi PNG, colourblind-safe, grayscale-legible."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

import os
LANG = os.environ.get("FIGLANG", "en")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/out"
FIG = ROOT / ("figures" if LANG == "en" else f"figures_{LANG}")
FIG.mkdir(parents=True, exist_ok=True)

def T(en, es):
    return en if LANG == "en" else es

# Validated categorical palette (six checks PASS, light surface)
C = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#b7791f", "#0987a0"]
INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#767676", "#d8d8d8"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.8,
    "axes.axisbelow": True, "figure.dpi": 130, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

panel = pd.read_csv(OUT / "metric_panel.csv")
axis_tbl = pd.read_csv(OUT / "axis_table.csv")
artifact = pd.read_csv(OUT / "ri_artifact.csv")
reversal = pd.read_csv(OUT / "rank_reversal.csv")
ranks = pd.read_csv(OUT / "avg_ranks.csv")

LAB = {"rl": "GNN-PPO", "ga": "GA", "cpsat": "CP-SAT", "spt": "SPT", "lpt": "LPT",
       "mwkr": "MWKR", "lwkr": "LWKR", "mor": "MOR", "lor": "LOR", "fifo": "FIFO",
       "edd": "EDD", "cr": "CR"}
AXES = ["Breakdowns", "Stochastic times", "Dynamic arrivals"]
AXTITLE = {"Breakdowns": T("Breakdowns", "Averías"),
           "Stochastic times": T("Stochastic times", "Tiempos estocásticos"),
           "Dynamic arrivals": T("Dynamic arrivals", "Llegadas dinámicas")}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ===========================================================================
# Fig 1 — the artifact: RI is a monotone decreasing function of nominal error
# ===========================================================================
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.55))
for ax, axis_name in zip(axs, AXES):
    sub = axis_tbl[axis_tbl.axis == axis_name]
    art = artifact[artifact.axis == axis_name]
    x, y = sub.rpd_nom.to_numpy(), sub.RI.to_numpy()
    # trend
    sl, ic = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() * 0.95, x.max() * 1.05, 50)
    ax.plot(xs, sl * xs + ic, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    for _, r in sub.iterrows():
        is_rl = r.method == "rl"
        ax.scatter(r.rpd_nom, r.RI, s=42 if is_rl else 26,
                   color=C[0] if is_rl else C[1], zorder=3,
                   edgecolor="white", linewidth=1.0,
                   marker="D" if is_rl else "o")
    # de-cluttered labels: cluster on the x axis, then fan the members out
    srt = sub.sort_values("rpd_nom").reset_index(drop=True)
    span = srt.rpd_nom.max() - srt.rpd_nom.min()
    cluster, clusters = [0], []
    for i in range(1, len(srt)):
        if srt.rpd_nom[i] - srt.rpd_nom[cluster[-1]] < 0.055 * span:
            cluster.append(i)
        else:
            clusters.append(cluster); cluster = [i]
    clusters.append(cluster)
    for cl in clusters:
        members = srt.loc[cl].sort_values("RI", ascending=False)
        for j, (_, r) in enumerate(members.iterrows()):
            is_rl = r.method == "rl"
            if len(cl) == 1:
                dx, dy, ha = 0, 8 if is_rl else 6, "center"
            else:
                dx, dy, ha = 11, 11 - 10.0 * j, "left"
                ax.plot([r.rpd_nom, r.rpd_nom + 0.030 * span],
                        [r.RI, r.RI], color=GRID, lw=0.5, zorder=2)
            ax.annotate(LAB[r.method], (r.rpd_nom, r.RI), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center",
                        fontsize=6.2, color=INK if is_rl else INK2,
                        fontweight="bold" if is_rl else "normal")
    rho = art.rho_RI_vs_nominalRPD.median()
    ax.set_title(AXTITLE[axis_name], color=INK, pad=6)
    ax.text(0.97, 0.93, r"$\rho_{s}=%.2f$" % rho, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK)
    ax.set_xlabel(T("Nominal RPD (%)", "RPD nominal (%)"))
    ax.margins(x=0.14, y=0.20)
axs[0].set_ylabel(T("Robustness index $RI$", "Índice de robustez $RI$"))
save(fig, "Fig1_ri_artifact")


# ===========================================================================
# Fig 2 — rank reversal: ordering by RI vs ordering by realized quality
# ===========================================================================
fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.15))
for ax, axis_name in zip(axs, AXES):
    sub = axis_tbl[axis_tbl.axis == axis_name].copy()
    sub["r_ri"] = sub.RI.rank()
    sub["r_q"] = sub.Q.rank()
    n = len(sub)
    for _, r in sub.iterrows():
        is_rl = r.method == "rl"
        col = C[0] if is_rl else (C[1] if r.r_q > r.r_ri else C[2])
        ax.plot([0, 1], [n - r.r_ri + 1, n - r.r_q + 1], color=col,
                lw=1.9 if is_rl else 1.0, alpha=1.0 if is_rl else 0.72,
                zorder=3 if is_rl else 2, solid_capstyle="round")
        ax.scatter([0, 1], [n - r.r_ri + 1, n - r.r_q + 1],
                   s=30 if is_rl else 16, color=col, zorder=4,
                   edgecolor="white", linewidth=0.9)
        ax.annotate(LAB[r.method], (0, n - r.r_ri + 1), textcoords="offset points",
                    xytext=(-7, 0), ha="right", va="center", fontsize=6.3,
                    color=INK if is_rl else INK2,
                    fontweight="bold" if is_rl else "normal")
        ax.annotate(LAB[r.method], (1, n - r.r_q + 1), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=6.3,
                    color=INK if is_rl else INK2,
                    fontweight="bold" if is_rl else "normal")
    tau = reversal[reversal.axis == axis_name].kendall_tau.median()
    ax.set_title(AXTITLE[axis_name], color=INK, pad=6)
    ax.text(0.5, -0.255, r"Kendall $\tau=%.2f$" % tau, transform=ax.transAxes,
            ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([T("by $RI$", "por $RI$"), T("by realized quality", "por calidad realizada")], fontsize=7.2)
    ax.set_yticks([]); ax.set_xlim(-0.55, 1.55); ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.08)
axs[0].text(-0.03, 1.04, T("best", "mejor"), transform=axs[0].transAxes, fontsize=6.8,
            color=MUTED, ha="right")
save(fig, "Fig2_rank_reversal")


# ===========================================================================
# Fig 3 — critical-difference (Nemenyi) diagrams, one per axis
# ===========================================================================
fig, axs = plt.subplots(3, 1, figsize=(7.1, 4.9))
for ax, axis_name in zip(axs, AXES):
    g = ranks[ranks.axis == axis_name].groupby("method", as_index=False).agg(
        avg_rank=("avg_rank", "mean"), CD=("CD", "first"))
    g = g.sort_values("avg_rank").reset_index(drop=True)
    cd = float(g.CD.iloc[0])
    lo, hi = 0.7, g.avg_rank.max() + 0.5
    pad = 0.34 * (hi - lo)                     # gutters that hold the labels
    n = len(g)
    n_left = int(np.ceil(n / 2))
    step = 1.0
    depth = step * max(n_left, n - n_left)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-depth - 0.25, 2.2)
    # rank axis (Demsar convention: rank 1 at the left, worse ranks to the right)
    ax.plot([lo, hi], [0, 0], color=INK2, lw=0.9)
    for t in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.plot([t, t], [0, 0.30], color=INK2, lw=0.8)
        ax.text(t, 0.52, f"{int(t)}", ha="center", fontsize=7, color=INK2)
    # clique bar: everything not distinguishable from the best rank
    best = g.avg_rank.min()
    clique = g[g.avg_rank <= best + cd]
    ax.plot([clique.avg_rank.min(), clique.avg_rank.max()], [-0.42, -0.42],
            color=C[0], lw=4.0, alpha=0.30, solid_capstyle="butt")
    for i, r in g.iterrows():
        is_rl = r.method == "rl"
        left = i < n_left
        y = -1.05 - step * (i if left else n - 1 - i)
        xe = lo - pad * 0.92 if left else hi + pad * 0.92
        col = C[0] if is_rl else INK2
        lw = 1.6 if is_rl else 0.8
        ax.plot([r.avg_rank, r.avg_rank], [0, y], color=col, lw=lw)
        ax.plot([r.avg_rank, xe], [y, y], color=col, lw=lw)
        ax.text(xe + (-0.05 if left else 0.05), y,
                f"{LAB[r.method]}  ({r.avg_rank:.2f})" if is_rl else LAB[r.method],
                ha="right" if left else "left", va="center",
                fontsize=7, color=INK if is_rl else INK2,
                fontweight="bold" if is_rl else "normal")
    ax.text(lo - pad, 1.75, AXTITLE[axis_name], fontsize=8.5, color=INK, ha="left",
            fontweight="bold")
    ax.text(hi + pad, 1.75, f"CD $=$ {cd:.2f}", fontsize=7.2, color=INK2, ha="right")
    ax.axis("off")
fig.subplots_adjust(hspace=0.42)
save(fig, "Fig3_critical_difference")


# ===========================================================================
# Fig 4 — realized quality: mean and CVaR90 per method and axis
# ===========================================================================
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.9))
for ax, axis_name in zip(axs, AXES):
    sub = axis_tbl[axis_tbl.axis == axis_name].sort_values("Q")
    y = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        is_rl = r.method == "rl"
        col = C[0] if is_rl else INK2
        ax.plot([r.Q, r.CVaR90], [i, i], color=col, lw=1.7 if is_rl else 1.0,
                alpha=0.85 if is_rl else 0.45, solid_capstyle="round", zorder=2)
        ax.scatter(r.Q, i, s=34 if is_rl else 22, color=col, zorder=4,
                   edgecolor="white", linewidth=0.9)
        ax.scatter(r.CVaR90, i, s=30 if is_rl else 20, color=col, zorder=4,
                   marker="|", linewidth=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[m] for m in sub.method], fontsize=7)
    for t, m in zip(ax.get_yticklabels(), sub.method):
        if m == "rl":
            t.set_color(INK); t.set_fontweight("bold")
    ax.set_title(AXTITLE[axis_name], color=INK, pad=6)
    ax.set_xlabel(T("Realized quality (% over optimum)", "Calidad realizada (% sobre el óptimo)"))
    ax.margins(x=0.10, y=0.06)
    ax.grid(axis="y", visible=False)
handles = [Line2D([], [], marker="o", color=INK2, lw=0, markersize=5, label=T("mean", "media")),
           Line2D([], [], marker="|", color=INK2, lw=0, markersize=7, label="CVaR$_{90}$")]
axs[-1].legend(handles=handles, loc="lower right", fontsize=7)
save(fig, "Fig4_realized_quality")


# ===========================================================================
# Fig 5 — research programme Gantt
# ===========================================================================
PHASES_EN = [
    ("Problem formulation and scoping",            1,  3, 0),
    ("Systematic review (PRISMA) and gap analysis", 2,  6, 0),
    ("Simulation environment: static JSSP/FJSP",    4,  8, 1),
    ("Disruption models (breakdowns/times/arrivals)", 7, 11, 1),
    ("Exact baseline: CP-SAT formulation",          6,  9, 2),
    ("Metaheuristic baseline: genetic algorithm",   8, 11, 2),
    ("Dispatching-rule library (nine rules)",       9, 11, 2),
    ("Heterogeneous GNN encoder and PPO policy",   10, 16, 3),
    ("Hyper-parameter optimisation (Optuna)",      15, 18, 3),
    ("Pre-registration frozen",                    17, 17, 4),
    ("Curriculum and domain-randomised training",  17, 21, 3),
    ("Confirmatory campaign (nine regimes)",       21, 24, 4),
    ("Verification and validation protocol",       22, 25, 4),
    ("Measurement audit and metric re-analysis",   25, 28, 5),
    ("Manuscript preparation and submission",      27, 30, 5),
]
PHASES_ES = [
    ("Formulación del problema y alcance",             1,  3, 0),
    ("Revisión sistemática (PRISMA) y análisis de brechas", 2, 6, 0),
    ("Entorno de simulación: JSSP/FJSP estático",      4,  8, 1),
    ("Modelos de perturbación (averías/tiempos/llegadas)", 7, 11, 1),
    ("Línea base exacta: formulación CP-SAT",          6,  9, 2),
    ("Línea base metaheurística: algoritmo genético",  8, 11, 2),
    ("Biblioteca de reglas de despacho (nueve reglas)", 9, 11, 2),
    ("Codificador GNN heterogéneo y política PPO",    10, 16, 3),
    ("Optimización de hiperparámetros (Optuna)",      15, 18, 3),
    ("Congelación del preregistro",                   17, 17, 4),
    ("Entrenamiento con currículo y aleatorización",  17, 21, 3),
    ("Campaña confirmatoria (nueve regímenes)",       21, 24, 4),
    ("Protocolo de verificación y validación",        22, 25, 4),
    ("Auditoría de medición y reanálisis métrico",    25, 28, 5),
    ("Preparación del manuscrito y envío",            27, 30, 5),
]
PHASES = PHASES_EN if LANG == "en" else PHASES_ES
GROUPS = (["Formulation", "Environment", "Baselines", "Learning", "Evaluation", "Audit"]
          if LANG == "en" else
          ["Formulación", "Entorno", "Líneas base", "Aprendizaje", "Evaluación", "Auditoría"])

fig, ax = plt.subplots(figsize=(7.1, 3.45))
n = len(PHASES)
for i, (name, s, e, gi) in enumerate(PHASES):
    y = n - i - 1
    dur = max(e - s + 1, 1)
    if s == e:                                              # milestone
        ax.scatter(s + 0.5, y, marker="D", s=46, color=C[gi], zorder=5,
                   edgecolor="white", linewidth=1.1)
    else:
        ax.add_patch(Rectangle((s, y - 0.30), dur, 0.60, facecolor=C[gi],
                               edgecolor="white", linewidth=1.0, alpha=0.92,
                               zorder=3))
        ax.text(s + dur + 0.35, y, T(f"w{s}–{e}", f"s{s}–{e}"), va="center", ha="left",
                fontsize=5.8, color=MUTED, zorder=4)
ax.set_yticks(range(n))
ax.set_yticklabels([p[0] for p in reversed(PHASES)], fontsize=7)
ax.tick_params(axis="y", length=0, pad=3)
for t in ax.get_yticklabels():
    t.set_color(INK)
ax.set_ylim(-0.75, n - 0.25)
ax.set_xlim(0, 33)
ax.set_xticks(range(0, 34, 3))
ax.set_xlabel(T("Week of the research programme", "Semana del programa de investigación"))
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
for wk, lbl, yy in [(17, T("pre-registration frozen", "preregistro congelado"), n - 0.62),
                    (24, T("confirmatory campaign closed", "campaña confirmatoria cerrada"), -0.66)]:
    ax.axvline(wk, color=INK2, lw=0.7, ls=":", zorder=1)
    ax.text(wk - 0.3, yy, lbl, fontsize=6.1, color=INK2, ha="right",
            va="bottom", style="italic")
ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=C[i], alpha=0.92, label=g)
                   for i, g in enumerate(GROUPS)],
          loc="lower left", bbox_to_anchor=(-0.30, -0.30), ncol=6, fontsize=7,
          handlelength=1.0, handleheight=0.7, columnspacing=1.1)
save(fig, "Fig5_gantt")


# ===========================================================================
# Fig 6 — comparator sensitivity: the same result, three legitimate baselines
# ===========================================================================
comp = pd.read_csv(OUT / "comparators_e1.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.1, 2.5))
bases = ["pdr_global", "pdr_per_family", "oracle_best_per_instance"]
names = ([ "Mean of the nine rules\n(PDR$_{global}$)",
           "Best of each rule family\n(PDR$_{family}$)",
           "Best rule per instance\n(oracle — not deployable)"]
         if LANG == "en" else
         [ "Media de las nueve reglas\n(PDR$_{global}$)",
           "Mejor regla de cada familia\n(PDR$_{family}$)",
           "Mejor regla por instancia\n(oráculo — no desplegable)"])
x = np.arange(3)
d = [comp[b] - comp["rl"] for b in bases]
bp = ax.boxplot(d, positions=x, widths=0.42, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=1.4),
                whiskerprops=dict(color=INK2, lw=0.8),
                capprops=dict(color=INK2, lw=0.8))
for i, box in enumerate(bp["boxes"]):
    box.set(facecolor=C[0] if np.median(d[i]) > 0 else C[1],
            edgecolor="white", linewidth=1.0, alpha=0.88)
for i, dd in enumerate(d):
    ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(dd)), dd, s=5,
               color=INK2, alpha=0.35, zorder=4)
    ax.text(i + 0.27, np.median(dd), f"{np.median(dd):+.1f} pp", va="center", ha="left",
            fontsize=7.5, color=INK, fontweight="bold")
ax.axhline(0, color=INK, lw=0.9, zorder=1)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.2)
ax.set_ylabel(T("RPD advantage of GNN-PPO (pp)", "Ventaja en RPD de GNN-PPO (pp)"))
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(-30, 700)
ax.set_xlim(-0.55, 2.85)
ax.grid(axis="x", visible=False)
save(fig, "Fig6_comparator_sensitivity")

print("\nAll figures written to", FIG)
