"""E4-E6 H3 verdict: RI computation + paired Wilcoxon signed-rank test per regime,
exactly per PreRegistration_FINAL.md Sec 4's frozen decision rule.

RI = (E[Cmax_pert] - Cmax_nom) / Cmax_nom, Cmax_nom is method-specific: the E1
deterministic (seed==0) rollout makespan for that (instance, method) --
StatsAnalysis_TechSpec.md Sec 5.2's exact pinned source, reused here rather than
recomputed.

Decision rule (verbatim, PreRegistration_FINAL.md Sec 4):
  Accept H3 for a given regime iff p<0.05 AND median(RI_PDR - RI_RL) > 0 in that
  regime, using the best-performing PDR (lowest RI) as the comparator.
  Accept H3 overall only if the regime-level rule holds in all 9 regimes; otherwise
  report "partial support for H3", not rounded up to a full accept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent
E1_CSV = RESULTS_DIR / "e1_merged.csv"
PDR_DIR = RESULTS_DIR / "e4e5e6"
RL_DIR = RESULTS_DIR / "e4e5e6_rl"

INSTANCES = ["ft06", "la01"]
ALL_PDR_NAMES = ["spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr"]
ARRIVAL_SAFE_PDR_NAMES = ["lwkr", "mor", "lor", "fifo"]

#: (regime_id, pdr_csv, rl_csv, axis_label, level, candidate_pdrs)
REGIMES = [
    ("E4_breakdowns_A0.95", "e4_breakdowns_avail0.95.csv", "e4_breakdowns_avail0.95_rl.csv", "breakdowns", 0.95, ALL_PDR_NAMES),
    ("E4_breakdowns_A0.90", "e4_breakdowns_avail0.90.csv", "e4_breakdowns_avail0.90_rl.csv", "breakdowns", 0.90, ALL_PDR_NAMES),
    ("E4_breakdowns_A0.80", "e4_breakdowns_avail0.80.csv", "e4_breakdowns_avail0.80_rl.csv", "breakdowns", 0.80, ALL_PDR_NAMES),
    ("E5_stochastic_s0.10", "e5_stochastic_sigma0.1.csv", "e5_stochastic_sigma0.1_rl.csv", "stochastic_times", 0.10, ALL_PDR_NAMES),
    ("E5_stochastic_s0.25", "e5_stochastic_sigma0.25.csv", "e5_stochastic_sigma0.25_rl.csv", "stochastic_times", 0.25, ALL_PDR_NAMES),
    ("E5_stochastic_s0.50", "e5_stochastic_sigma0.5.csv", "e5_stochastic_sigma0.5_rl.csv", "stochastic_times", 0.50, ALL_PDR_NAMES),
    ("E6_arrivals_u0.70", "e6_arrivals_util0.7.csv", "e6_arrivals_util0.7_rl.csv", "arrivals", 0.70, ARRIVAL_SAFE_PDR_NAMES),
    ("E6_arrivals_u0.85", "e6_arrivals_util0.85.csv", "e6_arrivals_util0.85_rl.csv", "arrivals", 0.85, ARRIVAL_SAFE_PDR_NAMES),
    ("E6_arrivals_u0.95", "e6_arrivals_util0.95.csv", "e6_arrivals_util0.95_rl.csv", "arrivals", 0.95, ARRIVAL_SAFE_PDR_NAMES),
]


def load_nominal_makespans() -> dict[tuple[str, str], float]:
    """(instance, method) -> nominal (--disruption none, seed==0) makespan, from E1."""
    df = pd.read_csv(E1_CSV)
    nominal: dict[tuple[str, str], float] = {}
    for instance in INSTANCES:
        for method in ALL_PDR_NAMES + ["rl"]:
            rows = df[(df["instance"] == instance) & (df["method"] == method) & (df["seed"] == 0)]
            if rows.empty:
                raise ValueError(f"no E1 seed==0 row for ({instance}, {method})")
            nominal[(instance, method)] = float(rows.iloc[0]["makespan"])
    return nominal


def compute_ri_per_row(df: pd.DataFrame, nominal: dict[tuple[str, str], float]) -> pd.DataFrame:
    df = df.copy()
    df["nominal_makespan"] = df.apply(lambda r: nominal[(r["instance"], r["method"])], axis=1)
    df["ri"] = (df["makespan"] - df["nominal_makespan"]) / df["nominal_makespan"]
    return df


def best_pdr_for_regime(pdr_df: pd.DataFrame, candidates: list[str]) -> tuple[str, float]:
    """Lowest-median-RI PDR among ``candidates``, across both instances x 30 seeds."""
    medians = {
        name: pdr_df[pdr_df["method"] == name]["ri"].median()
        for name in candidates
    }
    best_name = min(medians, key=medians.get)
    return best_name, medians[best_name]


def main() -> None:
    nominal = load_nominal_makespans()
    print("Nominal (E1, seed==0) makespans used as RI denominators:")
    for (instance, method), val in sorted(nominal.items()):
        print(f"  {instance:>6} {method:>6}: {val:.1f}")
    print()

    results = []
    for regime_id, pdr_csv, rl_csv, axis, level, candidates in REGIMES:
        pdr_df = compute_ri_per_row(pd.read_csv(PDR_DIR / pdr_csv), nominal)
        rl_df = compute_ri_per_row(pd.read_csv(RL_DIR / rl_csv), nominal)

        best_pdr, best_pdr_median_ri = best_pdr_for_regime(pdr_df, candidates)
        pdr_best_df = pdr_df[pdr_df["method"] == best_pdr]

        # Pair by (instance, seed) -- both files must agree on the same 60 keys.
        merged = pdr_best_df.merge(
            rl_df, on=["instance", "seed"], suffixes=("_pdr", "_rl"), how="inner"
        )
        assert len(merged) == len(INSTANCES) * 30, (
            f"{regime_id}: expected {len(INSTANCES) * 30} paired rows, got {len(merged)}"
        )

        diff = merged["ri_pdr"] - merged["ri_rl"]
        median_diff = float(diff.median())
        rl_median_ri = float(rl_df["ri"].median())

        # Wilcoxon requires at least one non-zero difference.
        if (diff != 0).any():
            stat, p_value = wilcoxon(diff)
        else:
            stat, p_value = float("nan"), 1.0

        accept = bool(p_value < 0.05 and median_diff > 0)
        results.append(
            {
                "regime_id": regime_id,
                "axis": axis,
                "level": level,
                "best_pdr": best_pdr,
                "best_pdr_median_ri": round(best_pdr_median_ri, 4),
                "rl_median_ri": round(rl_median_ri, 4),
                "median_ri_pdr_minus_rl": round(median_diff, 4),
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "accept_h3_this_regime": accept,
                "n_pairs": len(merged),
            }
        )
        print(
            f"{regime_id:<22} best_pdr={best_pdr:<5} RI_pdr={best_pdr_median_ri:+.4f} "
            f"RI_rl={rl_median_ri:+.4f} median_diff={median_diff:+.4f} p={p_value:.4g} "
            f"accept={accept}"
        )

    n_accept = sum(r["accept_h3_this_regime"] for r in results)
    overall = "ACCEPT H3" if n_accept == 9 else ("PARTIAL SUPPORT FOR H3" if n_accept > 0 else "REJECT H3")
    print()
    print(f"H3 regimes accepted: {n_accept}/9")
    print(f"H3 OVERALL VERDICT: {overall}")

    out = {
        "per_regime": results,
        "n_regimes_accepted": n_accept,
        "n_regimes_total": 9,
        "overall_verdict": overall,
    }
    out_path = RESULTS_DIR / "h3_verdict.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}")

    pd.DataFrame(results).to_csv(RESULTS_DIR / "h3_verdict.csv", index=False)


if __name__ == "__main__":
    main()
