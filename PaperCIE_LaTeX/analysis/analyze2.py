"""Second pass: corrected effect sizes, exact p-values, per-method detail tables."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/out"
RAW = ROOT / "results/datos_crudos/E1_E2_E3_E4_E5_E6_semana24"

panel = pd.read_csv(OUT / "metric_panel.csv")
comp = pd.read_csv(OUT / "comparators_e1.csv", index_col=0)

METHOD_LABEL = {
    "rl": "GNN--PPO", "ga": "GA", "cpsat": "CP-SAT",
    "spt": "SPT", "lpt": "LPT", "mwkr": "MWKR", "lwkr": "LWKR",
    "mor": "MOR", "lor": "LOR", "fifo": "FIFO", "edd": "EDD", "cr": "CR",
}


# --- corrected Wilcoxon with rank-biserial effect size ----------------------
def wilcoxon_full(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    d = d[d != 0]
    n = len(d)
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                         method="approx")
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = n * (n + 1) / 2
    rb = (w_plus - w_minus) / total          # rank-biserial correlation in [-1, 1]
    return dict(n=n, W=float(min(w_plus, w_minus)), p_value=float(res.pvalue),
                z=float(res.zstatistic) if hasattr(res, "zstatistic") else np.nan,
                rank_biserial=float(rb),
                median_diff=float(np.median(a - b)),
                mean_diff=float(np.mean(a - b)))


rows = []
for base in ["pdr_global", "pdr_per_family", "oracle_best_per_instance"]:
    for m in ["rl", "ga", "cpsat"]:
        r = wilcoxon_full(comp[base].to_numpy(), comp[m].to_numpy())
        r.update(comparator=base, method=m,
                 mean_comparator=float(comp[base].mean()),
                 mean_method=float(comp[m].mean()))
        rows.append(r)
comp_tests = pd.DataFrame(rows)[
    ["comparator", "method", "n", "mean_comparator", "mean_method",
     "median_diff", "mean_diff", "W", "z", "p_value", "rank_biserial"]]
comp_tests.to_csv(OUT / "comparator_tests.csv", index=False)


# --- per-method detail for each disruption axis -----------------------------
detail = (panel.groupby(["axis", "level", "method"])
          .agg(RI=("RI", "mean"), Q=("Q_mean", "mean"),
               Q_sd=("Q_sd", "mean"), CVaR90=("Q_cvar90", "mean"),
               rpd_nom=("rpd_nom", "mean"))
          .reset_index())
detail["label"] = detail.method.map(METHOD_LABEL)
detail.to_csv(OUT / "per_method_detail.csv", index=False)


# --- axis-level aggregate (the main results table) --------------------------
axis_tbl = (panel.groupby(["axis", "method"])
            .agg(RI=("RI", "mean"), Q=("Q_mean", "mean"),
                 CVaR90=("Q_cvar90", "mean"), rpd_nom=("rpd_nom", "mean"))
            .reset_index())
axis_tbl["label"] = axis_tbl.method.map(METHOD_LABEL)
axis_tbl.to_csv(OUT / "axis_table.csv", index=False)


# --- Nemenyi pairwise vs. the best-ranked method, per regime ----------------
ranks = pd.read_csv(OUT / "avg_ranks.csv")
pw = []
for reg, g in ranks.groupby("regime"):
    cd = float(g.CD.iloc[0])
    best = g.loc[g.avg_rank.idxmin()]
    for _, r in g.iterrows():
        diff = r.avg_rank - best.avg_rank
        pw.append(dict(regime=reg, axis=r.axis, level=r.level, method=r.method,
                       avg_rank=r.avg_rank, best_method=best.method,
                       rank_gap=diff, CD=cd,
                       sig_worse_than_best=bool(diff > cd)))
pd.DataFrame(pw).to_csv(OUT / "nemenyi_vs_best.csv", index=False)


# --- refreshed headline numbers --------------------------------------------
artifact = pd.read_csv(OUT / "ri_artifact.csv")
reversal = pd.read_csv(OUT / "rank_reversal.csv")
friedman = pd.read_csv(OUT / "friedman.csv")
nem = pd.read_csv(OUT / "nemenyi_vs_best.csv")

rl_not_worse = nem[(nem.method == "rl") & (~nem.sig_worse_than_best)]

# The six k=10 regimes (breakdowns + stochastic times) form the confirmatory set
# for the correlation claim; the three k=5 arrival regimes have too few methods
# to sustain a Spearman test and are reported descriptively.
conf = artifact[artifact.k == 10]
desc = artifact[artifact.k == 5]

head = json.loads((OUT / "headline.json").read_text())
head.update(
    n_conf_regimes=int(len(conf)),
    n_desc_regimes=int(len(desc)),
    rho_conf_min=float(conf.rho_RI_vs_nominalRPD.min()),
    rho_conf_max=float(conf.rho_RI_vs_nominalRPD.max()),
    rho_conf_median=float(conf.rho_RI_vs_nominalRPD.median()),
    p_conf_max=float(conf.p_value.max()),
    rho_desc_min=float(desc.rho_RI_vs_nominalRPD.min()),
    rho_desc_max=float(desc.rho_RI_vs_nominalRPD.max()),
    n_regimes_rho_negative=int((artifact.rho_RI_vs_nominalRPD < 0).sum()),
)
head.update(
    rho_median=float(artifact.rho_RI_vs_nominalRPD.median()),
    tau_min=float(reversal.kendall_tau.min()),
    tau_max=float(reversal.kendall_tau.max()),
    n_regimes_tau_negative=int((reversal.kendall_tau < 0).sum()),
    friedman_p_max=float(friedman.p_value.max()),
    n_regimes_rl_not_sig_worse=int(len(rl_not_worse)),
    n_regimes_rl_best=int((friedman.best_rank_method == "rl").sum()),
    cr_wins_by_RI=int((reversal.winner_by_RI == "cr").sum()),
    lwkr_wins_by_RI=int((reversal.winner_by_RI == "lwkr").sum()),
    comp_rl_vs_global=float(comp_tests.query("comparator=='pdr_global' and method=='rl'").median_diff.iloc[0]),
    comp_rl_vs_family=float(comp_tests.query("comparator=='pdr_per_family' and method=='rl'").median_diff.iloc[0]),
    comp_rl_vs_oracle=float(comp_tests.query("comparator=='oracle_best_per_instance' and method=='rl'").median_diff.iloc[0]),
)
(OUT / "headline.json").write_text(json.dumps(head, indent=2))

print(json.dumps(head, indent=2))
print("\n=== COMPARATOR TESTS (corrected) ===")
print(comp_tests.round(4).to_string(index=False))
print("\n=== AXIS TABLE ===")
print(axis_tbl.round(3).to_string(index=False))
print("\n=== RL vs BEST (Nemenyi) ===")
print(nem[nem.method == "rl"].round(3).to_string(index=False))
