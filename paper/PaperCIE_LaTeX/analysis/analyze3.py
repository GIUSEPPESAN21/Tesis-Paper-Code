"""Third pass: the offline-method comparison of Section 6.6, the Proposition-1
identity check, and the arrival-regime rank-tie check -- all of which the
manuscript quotes in prose and must therefore be script-generated."""
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

res = {}

# --- Proposition 1 identity, over every cell of the panel --------------------
panel = pd.read_csv(OUT / "metric_panel.csv")
r = panel.rpd_nom / 100.0
q = panel.Q_mean / 100.0
pred = (q - r) / (1 + r)
res["prop1_n_cells"] = int(len(panel))
res["prop1_max_abs_err"] = float((pred - panel.RI).abs().max())
res["prop1_sign_agreement_pct"] = float(
    100.0 * (np.sign(panel.RI) == np.sign(q - r)).mean())
res["frac_cells_RI_negative"] = float((panel.RI < 0).mean())
res["n_cells_RI_negative"] = int((panel.RI < 0).sum())

# --- Section 6.6: offline methods vs online methods under one regime ---------
d = pd.read_csv(RAW / "e4e5e6_pdr/descriptive_ga_cpsat_breakdowns90_ft06.csv")
full = pd.read_csv(RAW / "e4e5e6_pdr/e4_breakdowns_avail0.90_with_rl.csv")
full = full[(full.instance == "ft06") & (full.seed < 5)]

rows = []
for src in (d, full):
    for m, g in src.groupby("method"):
        rows.append(dict(method=m, Q_mean=g.rpd_pct.mean(),
                         Q_sd=g.rpd_pct.std(ddof=1),
                         wall_s=g.wall_clock_s.mean(), n=len(g)))
offline = pd.DataFrame(rows).sort_values("Q_mean").reset_index(drop=True)
offline.to_csv(OUT / "offline_vs_online_ft06_A090.csv", index=False)

rl = full[full.method == "rl"].sort_values("seed").rpd_pct.to_numpy()
for m in ("cpsat", "ga"):
    a = d[d.method == m].sort_values("seed").rpd_pct.to_numpy()
    diff = rl - a
    n_eff = int((diff != 0).sum())
    p = float(stats.wilcoxon(a, rl).pvalue)
    res[f"rl_vs_{m}_mean_diff_pp"] = float(diff.mean())
    res[f"rl_vs_{m}_p"] = p
    res[f"rl_vs_{m}_n_eff"] = n_eff
for m in ("cpsat", "ga", "mor", "mwkr", "rl"):
    row = offline[offline.method == m].iloc[0]
    res[f"Q_{m}_ft06_A090"] = float(row.Q_mean)
    res[f"wall_{m}_ft06_A090"] = float(row.wall_s)

# --- how many realisations the CVaR90 tail actually averages -----------------
import glob as _glob
tail_sizes = set()
nom_ref = pd.read_csv(OUT / "nominal_reference.csv").set_index(["instance", "method"])
for f in sorted(_glob.glob(str(RAW / "e4e5e6_pdr/*_with_rl.csv"))):
    dd = pd.read_csv(f)
    dd = dd.join(nom_ref[["opt"]], on=["instance", "method"])
    dd["Q"] = 100.0 * (dd.makespan - dd.opt) / dd.opt
    for _, g in dd.groupby(["instance", "method"]):
        v = g.Q.to_numpy()
        tail_sizes.add(int((v >= np.quantile(v, 0.90)).sum()))
res["cvar_tail_sizes"] = sorted(tail_sizes)
res["cvar_tail_size_max"] = max(tail_sizes)
res["seeds_per_cell"] = 30

# --- Nemenyi clique membership, per axis ------------------------------------
nem_all = pd.read_csv(OUT / "nemenyi_vs_best.csv")
cliques = {}
for ax, g in nem_all.groupby("axis"):
    members = sorted(set(g[~g.sig_worse_than_best].method))
    cliques[ax] = members
res["nemenyi_cliques"] = cliques
res["clique_sizes"] = {k: len(v) for k, v in cliques.items()}

# --- nominal deviation on the two disruption instances, for prose -----------
two = nom_ref.reset_index()
two = two[two.instance.isin(["ft06", "la01"])]
res["nominal_two_instances"] = {
    m: float(g.rpd_nom.mean()) for m, g in two.groupby("method")}
allinst = pd.read_csv(OUT / "nominal_reference.csv")
res["nominal_all_instances"] = {
    m: float(g.rpd_nom.mean()) for m, g in allinst.groupby("method")}

# --- arrival regimes: who is inside the critical difference of the best ------
nem = pd.read_csv(OUT / "nemenyi_vs_best.csv")
arr = nem[nem.axis == "Dynamic arrivals"]
tied = arr[(~arr.sig_worse_than_best) & (arr.method != arr.best_method)]
res["arrivals_CD"] = float(arr.CD.iloc[0])
res["arrivals_rl_rank_min"] = float(arr[arr.method == "rl"].avg_rank.min())
res["arrivals_rl_rank_max"] = float(arr[arr.method == "rl"].avg_rank.max())
res["arrivals_tied_methods"] = sorted(tied.method.unique().tolist())
res["arrivals_runnerup_gap_min"] = float(tied.rank_gap.min()) if len(tied) else None
res["arrivals_runnerup_gap_max"] = float(tied.rank_gap.max()) if len(tied) else None
res["arrivals_n_methods_tied_with_best"] = int(
    len(tied.method.unique())) if len(tied) else 0

# --- methods actually present in each regime (exclusion accounting) ----------
import glob
present = {}
for f in sorted(glob.glob(str(RAW / "e4e5e6_pdr/*_with_rl.csv"))):
    tag = Path(f).stem.replace("_with_rl", "")
    present[tag] = sorted(pd.read_csv(f).method.unique().tolist())
res["methods_per_regime"] = present
all9 = {"spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"}
arr_present = set(present["e6_arrivals_util0.7"]) - {"rl"}
res["pdr_excluded_under_arrivals"] = sorted(all9 - arr_present)
res["n_pdr_excluded_under_arrivals"] = len(all9 - arr_present)

# --- exact p-values of the comparator tests ---------------------------------
ct = pd.read_csv(OUT / "comparator_tests.csv")
rl_ct = ct[ct.method == "rl"]
res["comparator_p_max_rl"] = float(rl_ct.p_value.max())
res["comparator_p_min_rl"] = float(rl_ct.p_value.min())

# --- worst-case p across the audit tables -----------------------------------
res["friedman_p_max"] = float(pd.read_csv(OUT / "friedman.csv").p_value.max())
res["artifact_p_max"] = float(pd.read_csv(OUT / "ri_artifact.csv").p_value.max())

(OUT / "supplementary.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
print()
print(offline.round(3).to_string(index=False))
