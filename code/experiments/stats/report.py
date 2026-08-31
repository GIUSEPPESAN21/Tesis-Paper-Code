"""Orchestrates io/metrics/pairwise/omnibus/cd_diagram/tables into the H1-H5 verdict
table and every Chapter 6 table/figure. ``StatsAnalysis_TechSpec.md`` Section 11.

``generate_report()`` iterates H1 -> H5 in ``PreRegistration_FINAL.md``'s own order and
records the run in ``stats_manifest.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stats.cd_diagram import plot_cd_diagram
from experiments.stats.comparators import (
    h2_absolute_competitive_table,
    h2_dual_reading_markdown,
    pdr_global,
    tabla_comparadores_pdr_markdown,
)
from experiments.stats.io import LoadedExperiment
from experiments.stats.metrics import scan_for_negative_ri
from experiments.stats.omnibus import critical_difference, friedman_test
from experiments.stats.pairwise import (
    HypothesisVerdict,
    align_paired,
    h1_fjsp_test,
    h1_jssp_test,
    h3_overall_verdict,
    h3_regime_test,
    h4_overall_verdict,
    h4_test,
    h5_check,
)
from experiments.stats.tables import hypothesis_verdict_table, to_latex_table

ALL_PDR_NAMES: tuple[str, ...] = ("spt", "lpt", "mwkr", "lwkr", "mor", "lor", "fifo", "edd", "cr")
ARRIVAL_SAFE_PDR_NAMES: tuple[str, ...] = ("lwkr", "mor", "lor", "fifo")
H3_H4_INSTANCES: tuple[str, ...] = ("ft06", "la01")

#: (regime_label, experiment_id_disruption, axis, level, candidate_pdrs). Matches
#: PreRegistration_FINAL.md Section 4's frozen 9-regime grid and
#: experiments/results/week24/compute_h3_verdict.py's already-executed precedent.
REGIMES: tuple[tuple[str, str, float, tuple[str, ...]], ...] = (
    ("E4_breakdowns_A0.95", "breakdowns", 0.95, ALL_PDR_NAMES),
    ("E4_breakdowns_A0.90", "breakdowns", 0.90, ALL_PDR_NAMES),
    ("E4_breakdowns_A0.80", "breakdowns", 0.80, ALL_PDR_NAMES),
    ("E5_stochastic_s0.10", "stochastic_times", 0.10, ALL_PDR_NAMES),
    ("E5_stochastic_s0.25", "stochastic_times", 0.25, ALL_PDR_NAMES),
    ("E5_stochastic_s0.50", "stochastic_times", 0.50, ALL_PDR_NAMES),
    ("E6_arrivals_u0.70", "arrivals", 0.70, ARRIVAL_SAFE_PDR_NAMES),
    ("E6_arrivals_u0.85", "arrivals", 0.85, ARRIVAL_SAFE_PDR_NAMES),
    ("E6_arrivals_u0.95", "arrivals", 0.95, ARRIVAL_SAFE_PDR_NAMES),
)

#: PreRegistration_FINAL.md Section 3's four generalization tiers, as actually executed
#: (EXPERIMENT_LOG_WEEK24.md's own descriptive table -- 4 tiers of 5 instances, not the
#: pre-registration table's more ambiguous 3-row summary).
H2_TIERS: dict[str, tuple[str, ...]] = {
    "1.5-2x (la21-la25)": tuple(f"la{i:02d}" for i in range(21, 26)),
    "2.0x (la26-la30)": tuple(f"la{i:02d}" for i in range(26, 31)),
    "3.0x (la31-la35)": tuple(f"la{i:02d}" for i in range(31, 36)),
    "2.25x (la36-la40)": tuple(f"la{i:02d}" for i in range(36, 41)),
}


def nominal_makespan_lookup(e1_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``(instance, method) -> nominal (--disruption none, seed==0) makespan``, from E1.

    ``StatsAnalysis_TechSpec.md`` Section 5.2's pinned RI denominator source -- reused
    here for every regime, never recomputed per-regime.
    """
    lookup: dict[tuple[str, str], float] = {}
    nominal_rows = e1_df[(e1_df["disruption"] == "none") & (e1_df["seed"] == 0)]
    for (instance, method), rows in nominal_rows.groupby(["instance", "method"]):
        lookup[(instance, method)] = float(rows.iloc[0]["makespan"])
    return lookup


def add_row_ri(df: pd.DataFrame, nominal: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Per-ROW pseudo-RI: ``(row.makespan - nominal[instance,method]) / nominal[...]``.

    This is the quantity H3/H4's confirmatory Wilcoxon test actually pairs on (one value
    per (instance, seed) observation), NOT the aggregated, mean-based
    ``metrics.compute_ri`` (which is for the descriptive RI summary table, with its
    std/CVaR_90 companions). Matches
    ``experiments/results/week24/compute_h3_verdict.py::compute_ri_per_row`` exactly --
    that script is this thesis's own already-executed, already-reported H3 precedent.

    Raises:
        KeyError: If a row's ``(instance, method)`` has no E1 nominal makespan.
    """
    out = df.copy()
    out["nominal_makespan"] = [nominal[(row.instance, row.method)] for row in out.itertuples()]
    out["ri"] = (out["makespan"] - out["nominal_makespan"]) / out["nominal_makespan"]
    return out


def best_pdr_for_regime(pdr_ri_df: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, float]:
    """Lowest-median-RI PDR among ``candidates`` -- H1/H3's "best PDR" convention."""
    medians = {name: pdr_ri_df[pdr_ri_df["method"] == name]["ri"].median() for name in candidates}
    best_name = min(medians, key=lambda k: medians[k])
    return best_name, float(medians[best_name])


# ------------------------------------------------------------------------------------- H1


def run_h1(combined: pd.DataFrame) -> list[HypothesisVerdict]:
    """H1: static competitiveness. JSSP arm (E1) + FJSP arms (E2a synthetic, E2b
    Brandimarte), reported separately, never pooled (PreRegistration_FINAL.md Sec 2).
    """
    verdicts: list[HypothesisVerdict] = []

    e1 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    if not e1.empty:
        rl_det = e1[(e1["method"] == "rl") & (e1["seed"] == 0)]
        pdr_rows = e1[e1["method"].isin(ALL_PDR_NAMES)]
        best_pdr_rpd = pdr_rows.loc[pdr_rows.groupby("instance")["rpd_pct"].idxmin()]
        frame = pd.concat([best_pdr_rpd, rl_det])
        is_pdr = frame["method"].isin(ALL_PDR_NAMES)
        rpd_best, rpd_rl, keys = align_paired(
            frame, key_cols=["instance"], x_col="rpd_pct", y_col="rpd_pct", x_filter=is_pdr, y_filter=~is_pdr
        )
        verdicts.append(h1_jssp_test(rpd_best, rpd_rl))
    else:
        verdicts.append(
            HypothesisVerdict("H1 (JSSP)", "blocked", None, "E1 data not available.", raw={})
        )

    for experiment_id, arm in (("E2a", "synthetic"), ("E2b", "brandimarte")):
        e2 = combined[(combined["experiment_id"] == experiment_id) & (combined["disruption"] == "none")]
        if e2.empty:
            verdicts.append(
                HypothesisVerdict(f"H1 (FJSP, {arm})", "blocked", None, f"{experiment_id} data not available.", raw={})
            )
            continue
        rl_det = e2[(e2["method"] == "rl") & (e2["seed"] == 0)]
        comparator_rows = e2[e2["method"].isin(("ga", "cpsat"))]
        best_comparator = comparator_rows.loc[comparator_rows.groupby("instance")["makespan"].idxmin()]
        frame = pd.concat([best_comparator, rl_det])
        is_comparator = frame["method"].isin(("ga", "cpsat"))
        makespan_best, makespan_rl, keys = align_paired(
            frame, key_cols=["instance"], x_col="makespan", y_col="makespan", x_filter=is_comparator, y_filter=~is_comparator
        )
        verdicts.append(h1_fjsp_test(makespan_best, makespan_rl, arm=arm))

    return verdicts


# ------------------------------------------------------------------------------------- H2


def run_h2(combined: pd.DataFrame) -> tuple[list[HypothesisVerdict], dict[str, Path]]:
    """H2: size generalization. Friedman + Nemenyi per tier (PreRegistration_FINAL.md
    Sec 3). CD diagrams are only produced for Friedman-significant tiers (Sec 8: a CD
    diagram implies a rejected omnibus null).
    """
    e3 = combined[combined["experiment_id"] == "E3"]
    verdicts: list[HypothesisVerdict] = []
    figures: dict[str, Path] = {}
    if e3.empty:
        return [HypothesisVerdict("H2 (overall)", "blocked", None, "E3 data not available.", raw={})], figures

    any_reject_significant_worse = False
    h2_methods = ("rl", *ALL_PDR_NAMES)  # PreRegistration_FINAL.md Sec 3: k=10 (RL + nine PDRs), ga/cpsat excluded
    for tier_label, instances in H2_TIERS.items():
        tier_df = e3[e3["instance"].isin(instances) & e3["method"].isin(h2_methods)]
        pivot = tier_df.pivot_table(index="instance", columns="method", values="rpd_pct", aggfunc="first")
        if pivot.isna().any().any():
            dropped = pivot.index[pivot.isna().any(axis=1)].tolist()
            pivot = pivot.dropna()
            note_drop = f" ({len(dropped)} instance(s) dropped for missing method cell(s): {dropped})"
        else:
            note_drop = ""

        friedman = friedman_test(pivot)
        if friedman.p_value >= 0.05:
            verdicts.append(
                HypothesisVerdict(
                    f"H2 ({tier_label})",
                    "inconclusive",
                    None,
                    f"Friedman not significant (p={friedman.p_value:.4f}){note_drop} -- "
                    "methods not distinguishable in this tier, neither accept nor reject "
                    "(PreRegistration_FINAL.md Sec 3).",
                    raw={"friedman": asdict(friedman)},
                )
            )
            continue

        cd = critical_difference(friedman.k_methods, friedman.n_blocks)
        best_pdr = min((m for m in friedman.average_ranks if m in ALL_PDR_NAMES), key=lambda m: friedman.average_ranks[m])
        rl_rank = friedman.average_ranks.get("rl")
        pdr_rank = friedman.average_ranks[best_pdr]
        rl_significantly_worse = rl_rank is not None and (rl_rank - pdr_rank) > cd

        fig_path = plot_cd_diagram(
            friedman.average_ranks, cd, title=f"E3 -- {tier_label} (H2)",
            output_path=Path("experiments/results/stats/figures") / f"cd_diagram_e3_{tier_label.split()[0]}.png",
        )
        figures[tier_label] = fig_path

        verdict = "reject" if rl_significantly_worse else "accept"
        if rl_significantly_worse:
            any_reject_significant_worse = True
        verdicts.append(
            HypothesisVerdict(
                f"H2 ({tier_label})",
                verdict,
                None,
                f"Friedman p={friedman.p_value:.4g}{note_drop}, CD={cd:.3f}, "
                f"rl_rank={rl_rank:.2f}, best_pdr={best_pdr}({pdr_rank:.2f}). "
                "Reject iff RL significantly worse than best PDR by Nemenyi CD "
                "(PreRegistration_FINAL.md Sec 3).",
                raw={"friedman": asdict(friedman), "cd": cd, "best_pdr": best_pdr},
            )
        )

    non_inconclusive = [v for v in verdicts if v.verdict != "inconclusive"]
    overall = "reject" if any_reject_significant_worse else ("accept" if non_inconclusive else "inconclusive")
    verdicts.append(
        HypothesisVerdict(
            "H2 (overall)",
            overall,
            None,
            "Accept iff RL not significantly worse than best PDR in every (non-inconclusive) "
            "tier; reject if significantly worse in any tier (PreRegistration_FINAL.md Sec 3).",
            raw={"n_tiers": len(H2_TIERS)},
        )
    )
    return verdicts, figures


# ------------------------------------------------------------------------------------- H3


def run_h3(combined: pd.DataFrame) -> list[HypothesisVerdict]:
    """H3: robustness under disruption, per PreRegistration_FINAL.md Sec 4.

    Requires each of the 9 regimes to be loaded under its own ``experiment_id`` tag
    (e.g. ``"E4_breakdowns_A0.95"``, matching :data:`REGIMES`'s labels exactly -- the
    convention ``cli.py::default_experiment_csvs`` and the golden regression test both
    use). Regime *level* has no dedicated column (one CSV per level), so a coarser tag
    (just ``"E4"``) cannot be disambiguated by level and is deliberately not supported
    as a silent fallback here -- a "blocked" verdict for that regime is safer than
    guessing which level's rows a coarsely-tagged slice contains.
    """
    e1 = combined[combined["experiment_id"] == "E1"]
    if e1.empty:
        return [HypothesisVerdict("H3 (overall)", "blocked", None, "E1 (nominal makespans) not available.", raw={})]
    nominal = nominal_makespan_lookup(e1)

    regime_ids_present = set(combined["experiment_id"]) & {label for label, *_ in REGIMES}
    if not regime_ids_present:
        return [HypothesisVerdict("H3 (overall)", "blocked", None, "E4/E5/E6 data not available.", raw={})]

    per_regime: list[HypothesisVerdict] = []
    for regime_label, disruption, level, candidates in REGIMES:
        regime_df = combined[
            (combined["experiment_id"] == regime_label) & (combined["instance"].isin(H3_H4_INSTANCES))
        ]

        if regime_df.empty:
            per_regime.append(HypothesisVerdict(f"H3 ({regime_label})", "blocked", None, "no data", raw={"regime": regime_label}))
            continue

        regime_ri = add_row_ri(regime_df, nominal)
        pdr_ri = regime_ri[regime_ri["method"].isin(candidates)]
        rl_ri = regime_ri[regime_ri["method"] == "rl"]

        best_pdr, _ = best_pdr_for_regime(pdr_ri, candidates)
        pdr_best_df = pdr_ri[pdr_ri["method"] == best_pdr]

        frame = pd.concat([pdr_best_df, rl_ri])
        is_pdr = frame["method"] == best_pdr
        ri_pdr, ri_rl, keys = align_paired(
            frame, key_cols=["instance", "seed"], x_col="ri", y_col="ri", x_filter=is_pdr, y_filter=~is_pdr
        )
        verdict = h3_regime_test(ri_pdr, ri_rl, regime_label)
        per_regime.append(
            HypothesisVerdict(
                verdict.hypothesis, verdict.verdict, verdict.test_result, verdict.notes,
                raw={**verdict.raw, "best_pdr": best_pdr},
            )
        )

    return [*per_regime, h3_overall_verdict(per_regime)]


# ------------------------------------------------------------------------------------- H4


def run_h4_full(combined: pd.DataFrame, kappa_values: tuple[float, ...] = (0.5, 1.0)) -> list[HypothesisVerdict]:
    """H4: disruption-aware reward, per PreRegistration_FINAL.md Sec 5.

    kappa=0's arm is E1/E4-E6's own RL rows (the SAME policy, reused per Sec 5, not a
    separate E7 CSV). kappa in {0.5, 1.0} come from E7's evaluation grid, tagged
    ``experiment_id in {"E7_kappa0.5", "E7_kappa1.0"}`` (disruption grid) and the same
    ids under ``disruption == "none"`` for the nominal-RPD side. Returns "blocked"
    verdicts if E7/E4-E6 data has not landed yet -- this pipeline is designed to run
    incrementally as results arrive (Section 11's ``--skip-missing``).
    """
    e1 = combined[combined["experiment_id"] == "E1"]
    if e1.empty:
        return [HypothesisVerdict("H4 (overall)", "blocked", None, "E1 (nominal makespans) not available.", raw={})]
    nominal = nominal_makespan_lookup(e1)

    # kappa=0's disruption arm: the SAME per-regime-tagged E4/E5/E6 rows H3 uses (e.g.
    # "E4_breakdowns_A0.95"), not a coarse "E4"/"E5"/"E6" tag -- see run_h3's docstring
    # for why coarse tags can't be disambiguated by level.
    kappa0_disruption = combined[
        combined["experiment_id"].isin({label for label, *_ in REGIMES}) & (combined["method"] == "rl")
        & (combined["instance"].isin(H3_H4_INSTANCES))
    ]
    kappa0_nominal_rpd = e1[(e1["disruption"] == "none") & (e1["method"] == "rl") & (e1["seed"] == 0)]
    kappa0_nominal_rpd = kappa0_nominal_rpd[kappa0_nominal_rpd["instance"].isin(H3_H4_INSTANCES)]["rpd_pct"].to_numpy()

    all_verdicts: list[HypothesisVerdict] = []
    for kappa in kappa_values:
        exp_prefix = f"E7_kappa{kappa}"
        kappa_disruption = combined[
            combined["experiment_id"].str.startswith(exp_prefix, na=False) & (combined["disruption"] != "none")
        ]
        kappa_nominal = combined[
            combined["experiment_id"].str.startswith(exp_prefix, na=False) & (combined["disruption"] == "none")
        ]
        if kappa_disruption.empty or kappa0_disruption.empty:
            all_verdicts.append(
                HypothesisVerdict(f"H4 (kappa={kappa}, overall)", "blocked", None, f"{exp_prefix} disruption data not available.", raw={"kappa": kappa})
            )
            continue
        rpd_nominal_kappa = kappa_nominal["rpd_pct"].to_numpy()
        if rpd_nominal_kappa.size == 0:
            rpd_nominal_kappa = np.array([np.nan])

        per_regime: list[HypothesisVerdict] = []
        for regime_label, disruption, level, _candidates in REGIMES:
            k0_regime = kappa0_disruption[kappa0_disruption["experiment_id"] == regime_label]
            k_regime = kappa_disruption[kappa_disruption["experiment_id"] == f"{exp_prefix}_{regime_label}"]
            if k0_regime.empty or k_regime.empty:
                continue

            k0_ri = add_row_ri(k0_regime, nominal)
            k_ri = add_row_ri(k_regime, nominal)
            frame = pd.concat([k0_ri.assign(_arm="k0"), k_ri.assign(_arm="k")])
            is_k0 = frame["_arm"] == "k0"
            ri_k0, ri_k, keys = align_paired(
                frame, key_cols=["instance", "seed"], x_col="ri", y_col="ri", x_filter=is_k0, y_filter=~is_k0
            )
            per_regime.append(
                h4_test(ri_k, ri_k0, rpd_nominal_kappa, kappa0_nominal_rpd, kappa, regime_label=regime_label)
            )

        if not per_regime:
            all_verdicts.append(
                HypothesisVerdict(f"H4 (kappa={kappa}, overall)", "blocked", None, "no matching regime data.", raw={"kappa": kappa})
            )
            continue
        all_verdicts.extend(per_regime)
        all_verdicts.append(h4_overall_verdict(per_regime, kappa))

    return all_verdicts


# ------------------------------------------------------------------------------------- H5


def run_h5(combined: pd.DataFrame) -> HypothesisVerdict:
    """H5: decision latency (PreRegistration_FINAL.md Sec 6, no significance test).

    RL rows are restricted to ``seed == 0`` (the deterministic/argmax rollout), matching
    the ``rollout_mode`` convention Section 2.2 establishes everywhere else in this
    pipeline (H1's primary-vs-descriptive split, etc.): seeds 1-4 are ``--rl-sample``
    stochastic rollouts, which sample the masked action distribution instead of taking
    the argmax and are measurably slower per decision for reasons unrelated to the
    policy's actual inference cost (categorical-sampling overhead, not a deployment-
    representative number). Pooling them into a latency claim would inflate RL's
    reported ms/decision without a defensible reason -- H5 is about inference speed,
    not about which rollout mode happens to be slower to sample from.
    """
    e8 = combined[combined["experiment_id"] == "E8"]
    if e8.empty:
        e8 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    if e8.empty:
        return HypothesisVerdict("H5", "blocked", None, "E1/E8 timing data not available.", raw={})

    rl_rows = e8[(e8["method"] == "rl") & (e8["seed"] == 0) & e8["ms_per_decision"].notna()]
    cpsat_rows = e8[(e8["method"] == "cpsat") & e8["ms_per_decision"].notna()]

    ms_rl = {inst: grp["ms_per_decision"].to_numpy() for inst, grp in rl_rows.groupby("instance")}
    ms_cpsat = {inst: grp["ms_per_decision"].to_numpy() for inst, grp in cpsat_rows.groupby("instance")}
    return h5_check(ms_rl, ms_cpsat)


# --------------------------------------------------------------------------- Chapter 6 artifacts
#
# StatsAnalysis_TechSpec.md Sections 9.1/9.3 ask for RPD/RI/latency summary tables and
# the E3 scatter / E4-E6 RI-vs-level / E8 latency-vs-size figures, in addition to the
# H1-H5 verdict table above -- these are the tables/figures Chapter 6's prose actually
# cites (the verdict table is the summary a reader checks first, not the whole chapter).


def _write_table(tables_dir: Path, name: str, markdown: str, *, caption: str) -> Path:
    path = tables_dir / f"{name}.md"
    path.write_text(markdown, encoding="utf-8")
    tex_path = tables_dir / f"{name}.tex"
    try:
        tex_path.write_text(to_latex_table(markdown, caption=caption, label=name.replace("_", "-")), encoding="utf-8")
    except ValueError:
        pass  # a table with zero data rows has nothing to convert; the .md still records that
    return path


def generate_chapter6_artifacts(
    combined: pd.DataFrame, out_dir: Path, *, h2_verdicts: list[HypothesisVerdict] | None = None
) -> dict[str, Path]:
    """6.1 (E1), 6.2 (E3), 6.3/6.4 (E4-E7), 6.5 (E8) summary tables and figures, plus
    the Recomendaciones 6/7 comparator tables (T4.1/T4.2, ``experiments.stats.comparators``).

    Args:
        h2_verdicts: H2's per-tier verdicts (``run_h2``'s first return value), used
            only to pull each tier's rank-based reading text into T4.2's dual-reading
            table. If ``None``, the dual-reading table is skipped (still descriptive,
            not required for H1-H5's own computation, which never depends on this).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.stats.tables import rpd_summary_table

    tables_dir = Path(out_dir) / "tables"
    figures_dir = Path(out_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # 6.1 -- E1 static competitiveness: one deterministic (seed==0) row per (instance,
    # method), so PDR/GA/CP-SAT/RL are on the same footing (no stochastic-seed averaging
    # mixed in, matching H1's own primary-rollout convention).
    e1 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none") & (combined["seed"] == 0)]
    if not e1.empty:
        md = rpd_summary_table(e1, ["instance"])
        written["e1_rpd_summary"] = _write_table(tables_dir, "e1_rpd_summary", md, caption="E1: RPD (\\%) by instance and method, deterministic seed 0")

        # T4.1 (Recomendacion 6): pdr_global / pdr_por_familia / oraculo_best_per_instance,
        # always reported separately -- never fused into a single "best PDR" figure.
        try:
            comparadores_md = tabla_comparadores_pdr_markdown(e1, candidates=ALL_PDR_NAMES)
            written["comparadores_pdr_e1"] = _write_table(
                tables_dir, "comparadores_pdr_e1", comparadores_md,
                caption="E1: pdr\\_global, pdr\\_por\\_familia y oraculo\\_best\\_per\\_instance (RPD medio, \\%)",
            )
        except ValueError:
            pass  # no PDR rows in this slice yet (e.g. mid-Semana-24 partial run)

    # 6.2 -- E3 size generalization: per-tier RPD summary + RPD-vs-instance-size scatter.
    e3 = combined[combined["experiment_id"] == "E3"]
    if not e3.empty:
        h2_methods = ("rl", *ALL_PDR_NAMES)
        e3_scoped = e3[e3["method"].isin(h2_methods)].copy()
        tier_of = {inst: tier for tier, insts in H2_TIERS.items() for inst in insts}
        e3_scoped["tier"] = e3_scoped["instance"].map(tier_of)
        md = rpd_summary_table(e3_scoped.dropna(subset=["tier"]), ["tier"])
        written["e3_rpd_summary"] = _write_table(tables_dir, "e3_rpd_summary", md, caption="E3: mean RPD (\\%) by generalization tier and method")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, marker in (("rl", "o"), *((m, "x") for m in ALL_PDR_NAMES)):
            sub = e3_scoped[e3_scoped["method"] == method]
            if sub.empty:
                continue
            style = dict(marker=marker, linestyle="none", alpha=0.85 if method == "rl" else 0.35,
                         label="RL" if method == "rl" else None,
                         color="tab:red" if method == "rl" else "tab:gray",
                         markersize=8 if method == "rl" else 4, zorder=3 if method == "rl" else 1)
            ax.plot(sub["num_operations"], sub["rpd_pct"], **style)
        ax.set_xlabel("num_operations (instance size)")
        ax.set_ylabel("RPD (%)")
        ax.set_title("E3 -- RPD vs. instance size (H2)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig_path = figures_dir / "e3_rpd_vs_size.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        written["e3_rpd_vs_size"] = fig_path

        # T4.2 (Recomendacion 7): dual reading of H2 -- rank-based (already computed by
        # run_h2) alongside RL's absolute-RPD gap vs. pdr_global, in the same table, so
        # H2's acceptance can never be quoted as absolute competitiveness on its own.
        if not e1.empty:
            try:
                pdr_glob = pdr_global(e1, candidates=ALL_PDR_NAMES)
                abs_df = h2_absolute_competitive_table(e3_scoped, H2_TIERS, pdr_global_method=pdr_glob.method)
                if not abs_df.empty:
                    rank_notes = {
                        v.hypothesis.replace("H2 (", "").rstrip(")"): v.notes
                        for v in (h2_verdicts or [])
                        if v.hypothesis.startswith("H2 (") and not v.hypothesis.startswith("H2 (overall")
                    }
                    dual_md = h2_dual_reading_markdown(abs_df, rank_notes, pdr_global_method=pdr_glob.method)
                    written["e3_h2_dual_reading"] = _write_table(
                        tables_dir, "e3_h2_dual_reading", dual_md,
                        caption="E3: H2 -- lectura de rango vs. lectura competitiva absoluta, por tier",
                    )
            except ValueError:
                pass  # no E1 PDR rows to compute pdr_global from yet

    # 6.3/6.4 -- E4-E6 robustness (H3) and E7 disruption-aware reward (H4): one RI
    # summary table + one RI-vs-level line plot per axis, each line an available kappa
    # (kappa=0 is the reused E4-E6 RL arm; kappa in {0.5, 1.0} come from E7's grid).
    e1_all = combined[combined["experiment_id"] == "E1"]
    if not e1_all.empty and any(rid in set(combined["experiment_id"]) for rid, *_ in REGIMES):
        nominal = nominal_makespan_lookup(e1_all)
        axes_spec = (
            ("breakdowns", "availability", [("E4_breakdowns_A0.95", 0.95), ("E4_breakdowns_A0.90", 0.90), ("E4_breakdowns_A0.80", 0.80)]),
            ("stochastic_times", "sigma", [("E5_stochastic_s0.10", 0.10), ("E5_stochastic_s0.25", 0.25), ("E5_stochastic_s0.50", 0.50)]),
            ("arrivals", "utilization", [("E6_arrivals_u0.70", 0.70), ("E6_arrivals_u0.85", 0.85), ("E6_arrivals_u0.95", 0.95)]),
        )
        ri_rows = []
        for axis_name, x_label, regime_levels in axes_spec:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            series = {"best_pdr": ([], []), "rl_kappa0": ([], []), "rl_kappa0.5": ([], []), "rl_kappa1.0": ([], [])}
            for regime_label, level in regime_levels:
                regime_df = combined[(combined["experiment_id"] == regime_label) & (combined["instance"].isin(H3_H4_INSTANCES))]
                if not regime_df.empty:
                    regime_ri = add_row_ri(regime_df, nominal)
                    candidates = next(c for rl, _, _, c in REGIMES if rl == regime_label)
                    pdr_ri = regime_ri[regime_ri["method"].isin(candidates)]
                    if not pdr_ri.empty:
                        best_pdr, best_ri = best_pdr_for_regime(pdr_ri, candidates)
                        series["best_pdr"][0].append(level)
                        series["best_pdr"][1].append(best_ri)
                        ri_rows.append({"axis": axis_name, "level": level, "series": f"best_pdr ({best_pdr})", "ri": best_ri})
                    rl_ri = regime_ri[regime_ri["method"] == "rl"]["ri"]
                    if not rl_ri.empty:
                        series["rl_kappa0"][0].append(level)
                        series["rl_kappa0"][1].append(float(rl_ri.mean()))
                        ri_rows.append({"axis": axis_name, "level": level, "series": "rl_kappa0", "ri": float(rl_ri.mean())})
                for kappa in (0.5, 1.0):
                    k_df = combined[combined["experiment_id"] == f"E7_kappa{kappa}_{regime_label}"]
                    if k_df.empty:
                        continue
                    k_ri = add_row_ri(k_df, nominal)["ri"]
                    if not k_ri.empty:
                        series[f"rl_kappa{kappa}"][0].append(level)
                        series[f"rl_kappa{kappa}"][1].append(float(k_ri.mean()))
                        ri_rows.append({"axis": axis_name, "level": level, "series": f"rl_kappa{kappa}", "ri": float(k_ri.mean())})
            for name, (xs, ys) in series.items():
                if xs:
                    order = np.argsort(xs)
                    ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=name)
            ax.set_xlabel(x_label)
            ax.set_ylabel("RI (mean)")
            ax.set_title(f"{axis_name} -- RI vs. {x_label} (H3/H4)")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig_path = figures_dir / f"ri_vs_{x_label}_{axis_name}.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            written[f"ri_vs_{x_label}"] = fig_path

        if ri_rows:
            ri_df = pd.DataFrame(ri_rows)
            lines = ["| Axis | Level | Series | RI (mean) |", "|---|---|---|---|"]
            for _, row in ri_df.iterrows():
                lines.append(f"| {row['axis']} | {row['level']} | {row['series']} | {row['ri']:.4f} |")
            written["e4e5e6e7_ri_summary"] = _write_table(
                tables_dir, "e4e5e6e7_ri_summary", "\n".join(lines),
                caption="E4-E6 (kappa=0, reused) and E7 (kappa in 0.5, 1.0): mean RI by regime and series",
            )

            # T8.3 (Recomendacion del tutor): a negative RI is not interpretable as
            # robustness (AUDIT_PREFLIGHT.md Fase 0a/0b) -- warn loudly and record which
            # (axis, level, series) cells trigger it in the manifest, never silently.
            negative_ri = scan_for_negative_ri(ri_rows)
            if negative_ri:
                import warnings

                warnings.warn(
                    f"{len(negative_ri)} (axis, level, series) cell(s) have RI < 0 -- see "
                    "ri_negative_warnings.json for the full list. Not interpretable as "
                    "robustness (AUDIT_PREFLIGHT.md Fase 0a/0b).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warnings_path = tables_dir / "ri_negative_warnings.json"
                warnings_path.write_text(json.dumps(negative_ri, indent=2), encoding="utf-8")
                written["ri_negative_warnings"] = warnings_path

    # 6.5 -- E8 decision latency: ms/decision-vs-size log-log plot with the CP-SAT/RL
    # ratio overlaid (PreRegistration_FINAL.md Sec 6's "does the ratio widen" figure),
    # plus the per-instance summary table. RL restricted to seed==0 (see run_h5's
    # docstring for why stochastic-sampled seeds don't belong in a latency claim).
    e8 = combined[combined["experiment_id"] == "E8"]
    if e8.empty:
        e8 = combined[(combined["experiment_id"] == "E1") & (combined["disruption"] == "none")]
    if not e8.empty:
        rl_rows = e8[(e8["method"] == "rl") & (e8["seed"] == 0) & e8["ms_per_decision"].notna()]
        cpsat_rows = e8[(e8["method"] == "cpsat") & e8["ms_per_decision"].notna()]
        rl_med = rl_rows.groupby("instance").agg(ms_rl=("ms_per_decision", "median"), num_operations=("num_operations", "first"))
        cpsat_med = cpsat_rows.groupby("instance").agg(ms_cpsat=("ms_per_decision", "median"))
        merged = rl_med.join(cpsat_med, how="inner").sort_values("num_operations")
        if not merged.empty:
            merged["ratio_cpsat_over_rl"] = merged["ms_cpsat"] / merged["ms_rl"]
            merged["passes_1pct_rule"] = merged["ms_rl"] < 0.01 * merged["ms_cpsat"]

            lines = ["| Instance | num_operations | ms/decision (RL, median) | ms/decision (CP-SAT, median) | ratio (CP-SAT/RL) | passes 1% rule |", "|---|---|---|---|---|---|"]
            for inst, row in merged.iterrows():
                lines.append(
                    f"| {inst} | {int(row['num_operations'])} | {row['ms_rl']:.4f} | {row['ms_cpsat']:.4f} | "
                    f"{row['ratio_cpsat_over_rl']:.4f} | {row['passes_1pct_rule']} |"
                )
            written["e8_latency_summary"] = _write_table(
                tables_dir, "e8_latency_summary", "\n".join(lines),
                caption="E8: median ms/decision, RL vs. CP-SAT, by instance (ordered by size)",
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
            ax1.plot(merged["num_operations"], merged["ms_rl"], marker="o", label="RL (median ms/decision)")
            ax1.plot(merged["num_operations"], merged["ms_cpsat"], marker="x", label="CP-SAT (median ms/decision)")
            ax1.set_yscale("log")
            ax1.set_ylabel("ms / decision (log scale)")
            ax1.legend(loc="best")
            ax1.set_title("E8 -- decision latency vs. instance size (H5)")

            ax2.plot(merged["num_operations"], merged["ratio_cpsat_over_rl"], marker="o", color="tab:green")
            ax2.axhline(100, color="black", linestyle="--", linewidth=1, label="1% threshold (ratio=100)")
            ax2.set_yscale("log")
            ax2.set_xlabel("num_operations (instance size)")
            ax2.set_ylabel("ratio CP-SAT/RL (log scale)")
            ax2.legend(loc="best")
            fig.tight_layout()
            fig_path = figures_dir / "e8_latency_vs_size.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            written["e8_latency_vs_size"] = fig_path

    return written


# --------------------------------------------------------------------------- orchestration


def generate_report(
    combined: pd.DataFrame,
    provenance: list[LoadedExperiment],
    out_dir: Path,
    *,
    kappa_values: tuple[float, ...] = (0.5, 1.0),
) -> dict:
    """Run H1 -> H5, write every table/figure, record ``stats_manifest.json``.

    Returns:
        A dict with ``verdicts`` (flat list of every HypothesisVerdict produced) and
        ``manifest`` (what was written and why).
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_verdicts: list[HypothesisVerdict] = []
    all_verdicts.extend(run_h1(combined))
    h2_verdicts, h2_figures = run_h2(combined)
    all_verdicts.extend(h2_verdicts)
    all_verdicts.extend(run_h3(combined))
    all_verdicts.extend(run_h4_full(combined, kappa_values))
    all_verdicts.append(run_h5(combined))

    verdict_md = hypothesis_verdict_table(all_verdicts)
    (tables_dir / "h1_h5_verdict.md").write_text(verdict_md, encoding="utf-8")
    (tables_dir / "h1_h5_verdict.tex").write_text(
        to_latex_table(_verdicts_to_df(all_verdicts), caption="H1-H5 hypothesis verdicts", label="h1-h5-verdict"),
        encoding="utf-8",
    )

    chapter6_artifacts = generate_chapter6_artifacts(combined, out_dir, h2_verdicts=h2_verdicts)
    chapter6_figures = {k: v for k, v in chapter6_artifacts.items() if v.suffix == ".png"}
    chapter6_tables = {k: v for k, v in chapter6_artifacts.items() if v.suffix == ".md"}

    manifest = {
        "provenance": [asdict(p) | {"paths": [str(x) for x in p.paths]} for p in provenance],
        "n_verdicts": len(all_verdicts),
        "figures": {**{k: str(v) for k, v in h2_figures.items()}, **{k: str(v) for k, v in chapter6_figures.items()}},
        "tables": ["h1_h5_verdict.md", "h1_h5_verdict.tex", *[f"{v.name}" for v in chapter6_tables.values()]],
    }
    (out_dir / "stats_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {"verdicts": all_verdicts, "manifest": manifest}


def _verdicts_to_df(verdicts: list[HypothesisVerdict]) -> pd.DataFrame:
    rows = []
    for v in verdicts:
        rows.append(
            {
                "Hypothesis": v.hypothesis,
                "Statistic": v.test_result.statistic if v.test_result else None,
                "p_value": v.test_result.p_value if v.test_result else None,
                "Effect_size_r": v.test_result.effect_size_r if v.test_result else None,
                "Verdict": v.verdict.upper(),
            }
        )
    return pd.DataFrame(rows)
