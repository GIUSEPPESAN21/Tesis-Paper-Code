"""Emit complete LaTeX table floats (booktabs) for both language versions.

Table bodies are NOT \\input inside a tabular -- the CAS class breaks on that --
so each file here contains a full float, \\input at document level.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/out"
TEX = ROOT / "analysis/tex"
TEX.mkdir(parents=True, exist_ok=True)

axis_tbl = pd.read_csv(OUT / "axis_table.csv")
artifact = pd.read_csv(OUT / "ri_artifact.csv")
reversal = pd.read_csv(OUT / "rank_reversal.csv")
friedman = pd.read_csv(OUT / "friedman.csv")
comp_tests = pd.read_csv(OUT / "comparator_tests.csv")
head = json.loads((OUT / "headline.json").read_text())

LAB = {"rl": r"\textbf{GNN--PPO}", "ga": "GA", "cpsat": "CP-SAT", "spt": "SPT",
       "lpt": "LPT", "mwkr": "MWKR", "lwkr": "LWKR", "mor": "MOR", "lor": "LOR",
       "fifo": "FIFO", "edd": "EDD", "cr": "CR"}
PLAIN = {k: v.replace(r"\textbf{", "").replace("}", "") for k, v in LAB.items()}
PLAIN["rl"] = r"\textbf{GNN--PPO}"

AXES = ["Breakdowns", "Stochastic times", "Dynamic arrivals"]
ES = {"Breakdowns": "Averías", "Stochastic times": "Tiempos estocásticos",
      "Dynamic arrivals": "Llegadas dinámicas"}


def fmt_p(p: float) -> str:
    if p < 1e-8:
        return r"$<10^{-8}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def float_wrap(label: str, caption: str, colspec: str, header: str,
               body: str, star: bool = True) -> str:
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[t]\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular*}}{{\\tblwidth}}{{{colspec}}}\n"
            f"\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular*}}\n"
            f"\\end{{{env}}}\n")


def w(name: str, body: str) -> None:
    (TEX / name).write_text(body)


# ---------------------------------------------------------------- T1 --------
def t1(es: bool) -> str:
    rows, prev = [], None
    for _, r in artifact.iterrows():
        ax = ES[r.axis] if es else r.axis
        show = ax if ax != prev else ""
        prev = ax
        rows.append(f"{show} & {r.level} & {int(r.k)} & "
                    f"${r.rho_RI_vs_nominalRPD:.3f}$ & {fmt_p(r.p_value)} \\\\")
    cap = ("Correlación de rangos entre la desviación nominal y el índice de "
           "robustez convencional, por régimen. $k$ es el número de métodos "
           "comparados. Un $\\rho_s$ fuertemente negativo significa que los "
           "métodos con peores planificaciones nominales reciben mejores "
           "puntuaciones de robustez."
           if es else
           "Rank correlation between nominal deviation and the conventional "
           "robustness index, by regime. $k$ is the number of methods compared. "
           "A strongly negative $\\rho_s$ means that methods with worse nominal "
           "schedules receive better robustness scores.")
    hdr = (("Clase de perturbación & Severidad & $k$ & $\\rho_s$ & $p$ \\\\")
           if es else
           ("Disruption class & Severity & $k$ & $\\rho_s$ & $p$ \\\\"))
    return float_wrap("tbl:artifact", cap, "@{}LLcrl@{}", hdr,
                      "\n".join(rows), star=False)


# ---------------------------------------------------------------- T2 --------
def t2(es: bool) -> str:
    rows = []
    for ax in AXES:
        sub = axis_tbl[axis_tbl.axis == ax].sort_values("Q")
        rows.append(r"\multicolumn{5}{@{}l}{\textit{" + (ES[ax] if es else ax)
                    + r"}} \\[2pt]")
        for _, r in sub.iterrows():
            st = r"$^{\star}$" if r.method == "rl" else ""
            rows.append(f"\\quad {PLAIN[r.method]}{st} & {r.rpd_nom:.1f} & "
                        f"{r.RI:+.3f} & {r.Q:.1f} & {r.CVaR90:.1f} \\\\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    cap = ("Panel de métricas por clase de perturbación, métodos ordenados por "
           "calidad realizada. $r$ es la desviación nominal respecto del óptimo "
           "certificado; $\\RI$ el índice convencional; $\\RQ$ la calidad "
           "realizada; $\\CVaR_{90}$ la media de su peor decil. Todos en "
           "porcentaje. $^{\\star}$ marca la política aprendida."
           if es else
           "Metric panel by disruption class, methods ordered by realised "
           "quality. $r$ is nominal deviation from the certified optimum; "
           "$\\RI$ the conventional index; $\\RQ$ realised quality; "
           "$\\CVaR_{90}$ its worst-decile mean. All percentages. "
           "$^{\\star}$ marks the learned policy.")
    hdr = (r"Método & $r$ (\%) & $\RI$ & $\RQ$ (\%) & $\CVaR_{90}$ (\%) \\"
           if es else
           r"Method & $r$ (\%) & $\RI$ & $\RQ$ (\%) & $\CVaR_{90}$ (\%) \\")
    return float_wrap("tbl:axis", cap, "@{}Lrrrr@{}", hdr, "\n".join(rows),
                      star=False)


# ---------------------------------------------------------------- T3 --------
def t3(es: bool) -> str:
    rows, prev = [], None
    for _, r in reversal.iterrows():
        ax = ES[r.axis] if es else r.axis
        show = ax if ax != prev else ""
        prev = ax
        rows.append(f"{show} & {r.level} & ${r.kendall_tau:+.3f}$ & "
                    f"{PLAIN[r.winner_by_RI]} & {PLAIN[r.winner_by_Q]} & "
                    f"{r.Q_of_RI_winner:.1f} & {r.Q_of_Q_winner:.1f} & "
                    f"$+{r.penalty_pp:.1f}$ \\\\")
    cap = ("Desacuerdo entre los dos ordenamientos. $\\tau$ es la correlación "
           "de rangos de Kendall entre el orden por $\\RI$ y el orden por "
           "calidad realizada. Las tres últimas columnas dan la calidad "
           "realizada del método que nomina el índice, la del método realmente "
           "mejor, y la penalización de seguir al índice."
           if es else
           "Disagreement between the two orderings. $\\tau$ is Kendall's rank "
           "correlation between the ordering by $\\RI$ and the ordering by "
           "realised quality. The last three columns give the realised quality "
           "of the method the index nominates, that of the method that is "
           "actually best, and the penalty incurred by following the index.")
    if es:
        hdr = (r"\multirow{2}{*}{Clase} & \multirow{2}{*}{Severidad} & "
               r"\multirow{2}{*}{$\tau$} & \multicolumn{2}{c}{Método elegido} & "
               r"\multicolumn{3}{c}{Calidad realizada (\%)} \\" "\n"
               r"\cmidrule(lr){4-5}\cmidrule(l){6-8}" "\n"
               r" & & & por $\RI$ & por $\RQ$ & del 1.\textordmasculine{} & "
               r"del 2.\textordmasculine{} & penaliz. \\")
    else:
        hdr = (r"\multirow{2}{*}{Disruption class} & \multirow{2}{*}{Severity} & "
               r"\multirow{2}{*}{$\tau$} & \multicolumn{2}{c}{Selected method} & "
               r"\multicolumn{3}{c}{Realised quality (\%)} \\" "\n"
               r"\cmidrule(lr){4-5}\cmidrule(l){6-8}" "\n"
               r" & & & by $\RI$ & by $\RQ$ & of former & of latter & penalty \\")
    return float_wrap("tbl:reversal", cap, "@{}LLrllrrr@{}", hdr, "\n".join(rows))


# ---------------------------------------------------------------- T4 --------
def t4(es: bool) -> str:
    rows, prev = [], None
    for _, r in friedman.iterrows():
        ax = ES[r.axis] if es else r.axis
        show = ax if ax != prev else ""
        prev = ax
        gap = float(r.rl_rank) - float(r.best_rank)
        flag = r"\checkmark" if gap <= r.CD else r"$\times$"
        rows.append(f"{show} & {r.level} & {int(r.k)} & {int(r.n_blocks)} & "
                    f"{r.chi2:.1f} & {fmt_p(r.p_value)} & {r.CD:.2f} & "
                    f"{PLAIN[r.best_rank_method]} & {r.rl_rank:.2f} & {flag} \\\\")
    cap = ("Pruebas de Friedman sobre calidad realizada con post-hoc de "
           "Nemenyi, por régimen. Los bloques son pares (instancia, semilla). "
           "La última columna indica si la política aprendida es "
           "estadísticamente indistinguible del método mejor clasificado a la "
           "diferencia crítica."
           if es else
           "Friedman tests on realised quality with Nemenyi post-hoc, by "
           "regime. Blocks are (instance, seed) pairs. The last column records "
           "whether the learned policy is statistically indistinguishable from "
           "the best-ranked method at the critical difference.")
    hdr = ((r"Clase & Severidad & $k$ & $N$ & $\chi^2$ & $p$ & CD & "
            r"Mejor rango & Rango GNN--PPO & Empata \\")
           if es else
           (r"Disruption class & Severity & $k$ & $N$ & $\chi^2$ & $p$ & CD & "
            r"Best-ranked & GNN--PPO rank & Tied \\"))
    return float_wrap("tbl:friedman", cap, "@{}LLrrrlrlrc@{}", hdr, "\n".join(rows))


# ---------------------------------------------------------------- T5 --------
CMP = {"pdr_global": r"$\mathrm{PDR}_{\mathrm{global}}$",
       "pdr_per_family": r"$\mathrm{PDR}_{\mathrm{family}}$",
       "oracle_best_per_instance": r"$\mathrm{PDR}_{\mathrm{oracle}}$"}


def t5(es: bool) -> str:
    rows, prev = [], None
    for _, r in comp_tests.iterrows():
        c = CMP[r.comparator]
        show = c if c != prev else ""
        prev = c
        v = (r"\textsc{gana}" if r.median_diff > 0 else r"\textsc{pierde}") if es \
            else (r"\textsc{wins}" if r.median_diff > 0 else r"\textsc{loses}")
        rows.append(f"{show} & {r.mean_comparator:.1f} & {PLAIN[r.method]} & "
                    f"{r.mean_method:.1f} & ${r.median_diff:+.1f}$ & {int(r.n)} & "
                    f"{r.z:.2f} & {fmt_p(r.p_value)} & "
                    f"${r.rank_biserial:+.3f}$ & {v} \\\\")
    cap = ("La misma política frente a tres definiciones de «mejor regla de "
           "despacho». Las desviaciones son RPD respecto del óptimo "
           "certificado; la columna de diferencia es la mediana de la "
           "diferencia pareada en puntos porcentuales, con $z$, $p$ de Wilcoxon "
           "y tamaño de efecto biserial de rangos sobre las instancias pareadas; "
           "$N=43$ en las filas de la política aprendida y $N=40$ donde se "
           "descartan empates exactos, como registra la columna $n$."
           if es else
           "The same policy against three definitions of ``best dispatching "
           "rule''. Deviations are relative percentage deviation from the "
           "certified optimum; the difference column is the median paired "
           "difference in percentage points, with Wilcoxon $z$, $p$ and "
           "rank-biserial effect size over the paired instances; $N=43$ for the "
           "learned-policy rows and $N=40$ where exact ties are dropped, as the "
           "$n$ column records.")
    hdr = ((r"Comparador & Su RPD & Método & Su RPD & Dif.\ mediana & $n$ & $z$ & "
            r"$p$ & $r_{rb}$ & Veredicto \\")
           if es else
           (r"Comparator & Its RPD & Method & Its RPD & Median diff. & $n$ & $z$ & "
            r"$p$ & $r_{rb}$ & Verdict \\"))
    return float_wrap("tbl:comparators", cap, "@{}LrLrrrrlrl@{}", hdr,
                      "\n".join(rows))


for name, fn in [("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5)]:
    w(f"{name}.tex", fn(False))
    w(f"{name}_es.tex", fn(True))
    print("wrote", name)


# --------------------------------------------------------------- macros -----
sup = json.loads((OUT / "supplementary.json").read_text())


def p_ceil(p: float, d: int = 1) -> str:
    """Round a p-value UP to d significant decimals so a `p <= x` claim is true."""
    e = math.floor(math.log10(p))
    m = math.ceil(p / 10 ** e * 10 ** d) / 10 ** d
    if m >= 10:
        m, e = m / 10, e + 1
    return r"\ensuremath{" + f"{m:g}" + r"\times10^{" + f"{e}" + "}}"


def num(x, d=3):
    """Math-mode-safe number: works inside and outside $...$ alike."""
    return r"\ensuremath{" + f"{x:.{d}f}" + "}"


macros = "\n".join([
    rf"\newcommand{{\NRegimes}}{{{head['n_regimes']}}}",
    r"\newcommand{\NRunsDisrupted}{" + f"{head['n_runs_disrupted']:,}".replace(",", r"{,}") + "}",
    rf"\newcommand{{\NInstances}}{{{head['n_instances_nominal']}}}",
    # confirmatory correlation set (six k=10 regimes)
    rf"\newcommand{{\NConfRegimes}}{{{head['n_conf_regimes']}}}",
    rf"\newcommand{{\NDescRegimes}}{{{head['n_desc_regimes']}}}",
    rf"\newcommand{{\RhoConfMin}}{{{num(head['rho_conf_min'])}}}",
    rf"\newcommand{{\RhoConfMax}}{{{num(head['rho_conf_max'])}}}",
    rf"\newcommand{{\RhoConfMedian}}{{{num(head['rho_conf_median'])}}}",
    rf"\newcommand{{\PConfMax}}{{{p_ceil(head['p_conf_max'])}}}",
    rf"\newcommand{{\RhoDesc}}{{{num(head['rho_desc_min'])}}}",
    rf"\newcommand{{\NRhoNegative}}{{{head['n_regimes_rho_negative']}}}",
    # rank reversal
    rf"\newcommand{{\NDisagree}}{{{head['n_regimes_disagree']}}}",
    rf"\newcommand{{\PenaltyMin}}{{{reversal.penalty_pp.min():.1f}}}",
    rf"\newcommand{{\PenaltyMax}}{{{head['max_penalty_pp']:.1f}}}",
    rf"\newcommand{{\PenaltyMedian}}{{{head['median_penalty_pp']:.1f}}}",
    rf"\newcommand{{\TauMin}}{{{num(head['tau_min'], 2)}}}",
    rf"\newcommand{{\TauMax}}{{{num(head['tau_max'], 2)}}}",
    # Friedman / Nemenyi
    rf"\newcommand{{\NRLNotWorse}}{{{head['n_regimes_rl_not_sig_worse']}}}",
    rf"\newcommand{{\NRLBest}}{{{head['n_regimes_rl_best']}}}",
    rf"\newcommand{{\FriedmanPMax}}{{{p_ceil(head['friedman_p_max'])}}}",
    rf"\newcommand{{\ArrCD}}{{{sup['arrivals_CD']:.2f}}}",
    rf"\newcommand{{\ArrRankMin}}{{{sup['arrivals_rl_rank_min']:.2f}}}",
    rf"\newcommand{{\ArrRankMax}}{{{sup['arrivals_rl_rank_max']:.2f}}}",
    rf"\newcommand{{\ArrGapMin}}{{{sup['arrivals_runnerup_gap_min']:.2f}}}",
    rf"\newcommand{{\ArrGapMax}}{{{sup['arrivals_runnerup_gap_max']:.2f}}}",
    # index winners
    rf"\newcommand{{\CRWins}}{{{head['cr_wins_by_RI']}}}",
    rf"\newcommand{{\LWKRWins}}{{{head['lwkr_wins_by_RI']}}}",
    # comparators
    rf"\newcommand{{\CompGlobal}}{{{head['comp_rl_vs_global']:.1f}}}",
    rf"\newcommand{{\CompFamily}}{{{head['comp_rl_vs_family']:.1f}}}",
    rf"\newcommand{{\CompOracle}}{{{abs(head['comp_rl_vs_oracle']):.1f}}}",
    rf"\newcommand{{\CompPMax}}{{{p_ceil(sup['comparator_p_max_rl'])}}}",
    # Proposition 1 audit
    rf"\newcommand{{\PropCells}}{{{sup['prop1_n_cells']}}}",
    rf"\newcommand{{\PropErr}}{{{p_ceil(sup['prop1_max_abs_err'])}}}",
    rf"\newcommand{{\PctRINeg}}{{{100*sup['frac_cells_RI_negative']:.0f}}}",
    rf"\newcommand{{\NRINeg}}{{{sup['n_cells_RI_negative']}}}",
    # Section 6.6
    rf"\newcommand{{\QCpsat}}{{{sup['Q_cpsat_ft06_A090']:.1f}}}",
    rf"\newcommand{{\QGa}}{{{sup['Q_ga_ft06_A090']:.1f}}}",
    rf"\newcommand{{\QMor}}{{{sup['Q_mor_ft06_A090']:.1f}}}",
    rf"\newcommand{{\QMwkr}}{{{sup['Q_mwkr_ft06_A090']:.1f}}}",
    rf"\newcommand{{\QRl}}{{{sup['Q_rl_ft06_A090']:.1f}}}",
    rf"\newcommand{{\WallGa}}{{{sup['wall_ga_ft06_A090']:.0f}}}",
    rf"\newcommand{{\WallCpsat}}{{{sup['wall_cpsat_ft06_A090']:.3f}}}",
    rf"\newcommand{{\WallRl}}{{{sup['wall_rl_ft06_A090']:.3f}}}",
    rf"\newcommand{{\PRlCpsat}}{{{sup['rl_vs_cpsat_p']:.3f}}}",
    rf"\newcommand{{\PRlGa}}{{{sup['rl_vs_ga_p']:.3f}}}",
    rf"\newcommand{{\NEffOffline}}{{{sup['rl_vs_cpsat_n_eff']}}}",
    # arrival-regime exclusions
    rf"\newcommand{{\NExcludedArr}}{{{sup['n_pdr_excluded_under_arrivals']}}}",
    # tail size, clique sizes, matched nominal figures
    rf"\newcommand{{\CVaRTail}}{{{sup['cvar_tail_size_max']}}}",
    rf"\newcommand{{\SeedsPerCell}}{{{sup['seeds_per_cell']}}}",
    rf"\newcommand{{\CliqueArr}}{{{sup['clique_sizes']['Dynamic arrivals'] - 1}}}",
    rf"\newcommand{{\CliqueOther}}{{{sup['clique_sizes']['Breakdowns'] - 1}}}",
    rf"\newcommand{{\NomRlTwo}}{{{sup['nominal_two_instances']['rl']:.1f}}}",
    rf"\newcommand{{\NomCrTwo}}{{{sup['nominal_two_instances']['cr']:.1f}}}",
    rf"\newcommand{{\NomRlAll}}{{{sup['nominal_all_instances']['rl']:.1f}}}",
    rf"\newcommand{{\NomCrAll}}{{{sup['nominal_all_instances']['cr']:.1f}}}",
    rf"\newcommand{{\NomMwkrTwo}}{{{sup['nominal_two_instances']['mwkr']:.1f}}}",
])
w("macros.tex", macros)
print("wrote macros")
