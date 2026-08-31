"""
Re-analysis of the dynamic-FJSP disruption corpus under the corrected
measurement protocol (advisor recommendations 1-10).

Every number this script emits is computed from the raw per-run CSVs in
results/datos_crudos/. Nothing is hard-coded, nothing is imputed.

Outputs land in analysis/out/ as CSV + LaTeX fragments.
"""
from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results/datos_crudos/E1_E2_E3_E4_E5_E6_semana24"
OUT = ROOT / "analysis/out"
OUT.mkdir(parents=True, exist_ok=True)

PDR9 = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]
PDR_FAMILY = {
    "spt": "duration", "lpt": "duration",
    "mwkr": "work-remaining", "lwkr": "work-remaining",
    "mor": "ops-remaining", "lor": "ops-remaining",
    "fifo": "arrival", "edd": "due-date", "cr": "due-date",
}
REGIMES = [
    ("e4_breakdowns_avail0.95", "Breakdowns", r"$A_k=0.95$"),
    ("e4_breakdowns_avail0.90", "Breakdowns", r"$A_k=0.90$"),
    ("e4_breakdowns_avail0.80", "Breakdowns", r"$A_k=0.80$"),
    ("e5_stochastic_sigma0.1", "Stochastic times", r"$\sigma=0.10$"),
    ("e5_stochastic_sigma0.25", "Stochastic times", r"$\sigma=0.25$"),
    ("e5_stochastic_sigma0.5", "Stochastic times", r"$\sigma=0.50$"),
    ("e6_arrivals_util0.7", "Dynamic arrivals", r"$\rho=0.70$"),
    ("e6_arrivals_util0.85", "Dynamic arrivals", r"$\rho=0.85$"),
    ("e6_arrivals_util0.95", "Dynamic arrivals", r"$\rho=0.95$"),
]


def load_nominal() -> pd.DataFrame:
    df = pd.read_csv(RAW / "e1_merged.csv")
    return df[df.disruption == "none"].copy()


def load_regime(tag: str) -> pd.DataFrame:
    f = RAW / "e4e5e6_pdr" / f"{tag}_with_rl.csv"
    return pd.read_csv(f)


# ---------------------------------------------------------------------------
# 1. Nominal reference: per (instance, method) nominal makespan and RPD
# ---------------------------------------------------------------------------
nom_all = load_nominal()
# Pre-registration convention: exactly ONE deterministic run per (instance, method)
# enters the reference. Every baseline has a single seed-0 row; the learned policy
# additionally has four stochastic rollouts (seeds 1-4) that the pre-registration
# designates descriptive. Using seed 0 throughout keeps the unit of analysis
# identical across methods.
nom = nom_all[nom_all.seed == 0].copy()
nom_ref = (
    nom.groupby(["instance", "method"])
    .agg(cmax_nom=("makespan", "mean"),
         rpd_nom=("rpd_pct", "mean"),
         opt=("known_optimum", "first"))
    .reset_index()
)
nom_ref.to_csv(OUT / "nominal_reference.csv", index=False)

# the four stochastic rollouts, kept separately for descriptive reporting only
rl_stoch = nom_all[(nom_all.method == "rl") & (nom_all.seed > 0)]
rl_stoch.groupby("instance").rpd_pct.agg(["mean", "std"]).to_csv(
    OUT / "rl_stochastic_rollouts.csv")


# ---------------------------------------------------------------------------
# 2. Per-regime metric panel
#    RI      : conventional robustness index (the metric under audit)
#    Q       : realized quality  = RPD of the *disrupted* makespan vs. certified optimum
#    CVaR90  : upper-tail mean of realized quality
# ---------------------------------------------------------------------------
panel_rows = []
for tag, axis, level in REGIMES:
    d = load_regime(tag)
    for (inst, meth), g in d.groupby(["instance", "method"]):
        ref = nom_ref[(nom_ref.instance == inst) & (nom_ref.method == meth)]
        if ref.empty:
            continue
        cmax_nom = float(ref.cmax_nom.iloc[0])
        rpd_nom = float(ref.rpd_nom.iloc[0])
        opt = float(ref.opt.iloc[0])
        cmax_pert = g.makespan.to_numpy(dtype=float)
        q = 100.0 * (cmax_pert - opt) / opt                      # realized quality, %
        ri = (cmax_pert.mean() - cmax_nom) / cmax_nom            # conventional RI
        cvar = q[q >= np.quantile(q, 0.90)].mean()
        panel_rows.append(dict(
            regime=tag, axis=axis, level=level, instance=inst, method=meth,
            n=len(g), cmax_nom=cmax_nom, rpd_nom=rpd_nom,
            cmax_pert_mean=cmax_pert.mean(), cmax_pert_sd=cmax_pert.std(ddof=1),
            RI=ri, Q_mean=q.mean(), Q_sd=q.std(ddof=1), Q_cvar90=cvar,
        ))
panel = pd.DataFrame(panel_rows)
panel.to_csv(OUT / "metric_panel.csv", index=False)


# ---------------------------------------------------------------------------
# 3. THE ARTIFACT: rank-correlation between RI and nominal schedule quality
#    A strongly negative rho means RI systematically *rewards* methods whose
#    nominal schedule was already bad.
# ---------------------------------------------------------------------------
art_rows = []
for tag, axis, level in REGIMES:
    sub = panel[panel.regime == tag]
    agg = sub.groupby("method").agg(RI=("RI", "mean"), rpd_nom=("rpd_nom", "mean"),
                                    Q=("Q_mean", "mean")).reset_index()
    rho, p = stats.spearmanr(agg.rpd_nom, agg.RI)
    rho_q, p_q = stats.spearmanr(agg.Q, agg.RI)
    art_rows.append(dict(regime=tag, axis=axis, level=level, k=len(agg),
                         rho_RI_vs_nominalRPD=rho, p_value=p,
                         rho_RI_vs_realizedQ=rho_q, p_value_Q=p_q))
artifact = pd.DataFrame(art_rows)
artifact.to_csv(OUT / "ri_artifact.csv", index=False)


# ---------------------------------------------------------------------------
# 4. RANK REVERSAL: does ranking by RI agree with ranking by realized quality?
# ---------------------------------------------------------------------------
rev_rows = []
for tag, axis, level in REGIMES:
    sub = panel[panel.regime == tag]
    agg = sub.groupby("method").agg(RI=("RI", "mean"), Q=("Q_mean", "mean")).reset_index()
    agg["rank_RI"] = agg.RI.rank()
    agg["rank_Q"] = agg.Q.rank()
    tau, p_tau = stats.kendalltau(agg.rank_RI, agg.rank_Q)
    best_ri = agg.loc[agg.RI.idxmin(), "method"]
    best_q = agg.loc[agg.Q.idxmin(), "method"]
    q_of_best_ri = float(agg.loc[agg.RI.idxmin(), "Q"])
    q_of_best_q = float(agg.loc[agg.Q.idxmin(), "Q"])
    rev_rows.append(dict(regime=tag, axis=axis, level=level,
                         kendall_tau=tau, p_value=p_tau,
                         winner_by_RI=best_ri, winner_by_Q=best_q,
                         agree=best_ri == best_q,
                         Q_of_RI_winner=q_of_best_ri, Q_of_Q_winner=q_of_best_q,
                         penalty_pp=q_of_best_ri - q_of_best_q))
reversal = pd.DataFrame(rev_rows)
reversal.to_csv(OUT / "rank_reversal.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Friedman + Nemenyi on realized quality, per regime
#    Blocks = (instance, seed); treatments = methods.
# ---------------------------------------------------------------------------
def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    q = q05.get(k, 3.164)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


fr_rows, rank_rows = [], []
for tag, axis, level in REGIMES:
    d = load_regime(tag)
    ref = nom_ref.set_index(["instance", "method"])
    d = d.join(ref[["opt"]], on=["instance", "method"])
    d["Q"] = 100.0 * (d.makespan - d.opt) / d.opt
    wide = d.pivot_table(index=["instance", "seed"], columns="method", values="Q")
    wide = wide.dropna(axis=0, how="any")
    methods = list(wide.columns)
    k, n = len(methods), len(wide)
    chi2, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    ranks = wide.rank(axis=1).mean(axis=0)
    cd = nemenyi_cd(k, n)
    fr_rows.append(dict(regime=tag, axis=axis, level=level, k=k, n_blocks=n,
                        chi2=chi2, p_value=p, CD=cd,
                        best_rank_method=ranks.idxmin(), best_rank=ranks.min(),
                        rl_rank=float(ranks.get("rl", np.nan))))
    for m in methods:
        rank_rows.append(dict(regime=tag, axis=axis, level=level,
                              method=m, avg_rank=float(ranks[m]), CD=cd, n_blocks=n, k=k))
friedman = pd.DataFrame(fr_rows)
ranks_df = pd.DataFrame(rank_rows)
friedman.to_csv(OUT / "friedman.csv", index=False)
ranks_df.to_csv(OUT / "avg_ranks.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Three explicitly separated PDR comparators on the nominal corpus (E1)
# ---------------------------------------------------------------------------
piv = nom.pivot_table(index="instance", columns="method", values="rpd_pct")
present = [p for p in PDR9 if p in piv.columns]
comp = pd.DataFrame(index=piv.index)
comp["pdr_global"] = piv[present].mean(axis=1)          # mean over all 9 rules
fam_best = {}
for fam in sorted(set(PDR_FAMILY.values())):
    cols = [c for c in present if PDR_FAMILY[c] == fam]
    fam_best[fam] = piv[cols].min(axis=1)
comp["pdr_per_family"] = pd.DataFrame(fam_best).mean(axis=1)
comp["oracle_best_per_instance"] = piv[present].min(axis=1)   # ORACLE - not deployable
for m in ["rl", "ga", "cpsat"]:
    if m in piv.columns:
        comp[m] = piv[m]
comp = comp.dropna()
comp.to_csv(OUT / "comparators_e1.csv")

comp_tests = []
for base in ["pdr_global", "pdr_per_family", "oracle_best_per_instance"]:
    for m in ["rl", "ga", "cpsat"]:
        if m not in comp.columns:
            continue
        diff = comp[base] - comp[m]
        if np.allclose(diff, 0):
            continue
        w, p = stats.wilcoxon(comp[base], comp[m])
        z = stats.norm.isf(p / 2) * np.sign(np.median(diff))
        comp_tests.append(dict(comparator=base, method=m, n=len(comp),
                               median_diff_pp=float(np.median(diff)),
                               mean_diff_pp=float(diff.mean()),
                               W=float(w), p_value=float(p),
                               effect_r=float(z / np.sqrt(len(comp)))))
pd.DataFrame(comp_tests).to_csv(OUT / "comparator_tests.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Headline aggregate numbers for the abstract
# ---------------------------------------------------------------------------
head = dict(
    n_regimes=len(REGIMES),
    n_runs_disrupted=int(sum(len(load_regime(t)) for t, _, _ in REGIMES)),
    n_instances_nominal=int(nom.instance.nunique()),
    n_methods_nominal=int(nom.method.nunique()),
    rho_min=float(artifact.rho_RI_vs_nominalRPD.min()),
    rho_max=float(artifact.rho_RI_vs_nominalRPD.max()),
    p_max=float(artifact.p_value.max()),
    n_regimes_significant=int((artifact.p_value < 0.05).sum()),
    n_regimes_disagree=int((~reversal.agree).sum()),
    max_penalty_pp=float(reversal.penalty_pp.max()),
    median_penalty_pp=float(reversal.penalty_pp.median()),
)
(OUT / "headline.json").write_text(json.dumps(head, indent=2))

print(json.dumps(head, indent=2))
print("\n=== RI ARTIFACT ===")
print(artifact.to_string(index=False))
print("\n=== RANK REVERSAL ===")
print(reversal.to_string(index=False))
print("\n=== FRIEDMAN ===")
print(friedman.to_string(index=False))
