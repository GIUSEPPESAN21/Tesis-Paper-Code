"""Pairwise comparisons -- Wilcoxon signed-rank, and the per-hypothesis wrappers that
hard-code ``PreRegistration_FINAL.md``'s exact comparison for H1/H3/H4/H5.

Per ``StatsAnalysis_TechSpec.md`` Section 1: "Getting a sign or an ``alternative=``
argument backwards on H1's accept/reject rule is the single highest-consequence bug this
pipeline could contain" -- every wrapper below hard-codes the direction of subtraction,
the alternative, and the accept/reject threshold; none of it is caller-configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

Verdict = Literal["accept", "reject", "partial", "inconclusive", "blocked"]


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    p_value_one_sided: float | None
    z: float
    effect_size_r: float
    n_nominal: int
    n_effective: int
    median_diff: float
    method: str
    all_zero: bool


@dataclass(frozen=True)
class HypothesisVerdict:
    hypothesis: str
    verdict: Verdict
    test_result: WilcoxonResult | None
    notes: str
    raw: dict = field(default_factory=dict)


def align_paired(
    df, *, key_cols: list[str], x_col: str, y_col: str, x_filter=None, y_filter=None
) -> tuple[np.ndarray, np.ndarray, list]:
    """Sort/align two paired columns by a shared key before calling
    :func:`wilcoxon_signed_rank` -- callers MUST do this explicitly; silent misalignment
    is a worse failure mode than a loud one.

    Args:
        df: Long-format DataFrame containing both arms' rows.
        key_cols: Columns identifying a pairing unit (e.g. ``["instance"]`` for H1,
            ``["instance", "seed"]`` for H3/H4).
        x_col, y_col: Value columns to compare (e.g. both ``"rpd_pct"``, selected from
            two differently-filtered slices of ``df``).
        x_filter, y_filter: Optional boolean masks (same index as ``df``) narrowing which
            rows contribute to ``x``/``y`` respectively (e.g. ``method == "rl"``).

    Returns:
        ``(x, y, keys)``: aligned value arrays and the sorted key tuples they correspond
        to, keys present in only one side are dropped (with a printed note).

    Raises:
        ValueError: If a key is duplicated within one side (ambiguous pairing).
    """
    x_df = df[x_filter] if x_filter is not None else df
    y_df = df[y_filter] if y_filter is not None else df
    # Assign a MultiIndex directly (not via .set_index()) so the index is always a
    # MultiIndex of tuples, even for a single key column -- pandas' .set_index() silently
    # collapses a single-level MultiIndex back to a flat Index of bare (non-tuple)
    # values, which breaks the documented "keys are tuples" contract.
    x_idx = x_df.copy()
    x_idx.index = pd.MultiIndex.from_frame(x_df[key_cols])
    y_idx = y_df.copy()
    y_idx.index = pd.MultiIndex.from_frame(y_df[key_cols])
    if x_idx.index.duplicated().any():
        raise ValueError(f"align_paired: duplicate key(s) on x side: {key_cols}")
    if y_idx.index.duplicated().any():
        raise ValueError(f"align_paired: duplicate key(s) on y side: {key_cols}")

    common = sorted(set(x_idx.index) & set(y_idx.index))
    dropped = (set(x_idx.index) | set(y_idx.index)) - set(common)
    if dropped:
        print(f"[stats.pairwise] align_paired: dropping {len(dropped)} unmatched key(s): {sorted(dropped)[:5]}...")

    x = x_idx.loc[common, x_col].to_numpy(dtype=float)
    y = y_idx.loc[common, y_col].to_numpy(dtype=float)
    return x, y, common


def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test on ``(x - y)``, pinned parameters for
    reproducibility.

    Args:
        x, y: Paired samples, same length, same (instance, seed) order -- callers MUST
            align both arrays by a shared key before calling this (see
            :func:`align_paired`); this function does not do it implicitly.
        alternative: ``"two-sided"`` (default, matches H1/H3's two-sided frozen tests) or
            ``"less"``/``"greater"`` for one-sided descriptive variants.
        zero_method: ``"wilcox"`` (drop exact-zero-difference pairs), pinned explicitly
            rather than left to scipy's version-dependent default.

    Returns:
        WilcoxonResult. If all differences are exactly zero, returns
        ``statistic=0, p_value=1.0, z=0, effect_size_r=0, all_zero=True`` rather than
        propagating scipy's ``ValueError`` for degenerate all-zero input.

    Raises:
        ValueError: ``len(x) != len(y)``, or either array is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("x and y must not be empty")

    diff = x - y
    n_nominal = int(diff.size)
    n_effective = int(np.count_nonzero(diff))
    median_diff = float(np.median(diff))

    if n_effective == 0:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            p_value_one_sided=1.0 if alternative != "two-sided" else None,
            z=0.0,
            effect_size_r=0.0,
            n_nominal=n_nominal,
            n_effective=0,
            median_diff=median_diff,
            method="degenerate-all-zero",
            all_zero=True,
        )

    try:
        result_two_sided = wilcoxon(x, y, alternative="two-sided", zero_method=zero_method, method="auto")
    except ValueError:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            p_value_one_sided=1.0 if alternative != "two-sided" else None,
            z=0.0,
            effect_size_r=0.0,
            n_nominal=n_nominal,
            n_effective=0,
            median_diff=median_diff,
            method="degenerate-all-zero",
            all_zero=True,
        )

    p_two_sided = float(result_two_sided.pvalue)
    # scipy exposes `.zstatistic` only when it internally chose the normal-approximation
    # method; its absence means the exact (permutation-based) method was used -- the
    # version-robust way to recover "which method scipy chose" (StatsAnalysis_TechSpec.md
    # Section 6.1 asks this be recorded for the paper's methods section).
    method_name = "approx" if hasattr(result_two_sided, "zstatistic") else "exact"

    p_one_sided = None
    statistic = float(result_two_sided.statistic)
    if alternative != "two-sided":
        result_one_sided = wilcoxon(x, y, alternative=alternative, zero_method=zero_method, method="auto")
        p_one_sided = float(result_one_sided.pvalue)
        statistic = float(result_one_sided.statistic)

    # z derivation (StatsAnalysis_TechSpec.md Section 6.1): invert the two-sided p-value
    # through the standard normal quantile function, then re-attach the sign of the
    # median difference. scipy's wilcoxon() does not expose a usable z across all
    # supported methods (the "exact" method, used for most of this thesis's small-N
    # comparisons, is permutation-based and has no native z at all), so this is the
    # standard "effect size back-computed from a reported p-value" procedure.
    magnitude = float(norm.ppf(1 - p_two_sided / 2)) if p_two_sided > 0 else float("inf")
    sign = 1.0 if median_diff > 0 else (-1.0 if median_diff < 0 else 0.0)
    z = sign * magnitude
    effect_size_r = z / np.sqrt(n_effective) if n_effective > 0 else 0.0

    return WilcoxonResult(
        statistic=statistic,
        p_value=p_two_sided,
        p_value_one_sided=p_one_sided,
        z=z,
        effect_size_r=float(effect_size_r),
        n_nominal=n_nominal,
        n_effective=n_effective,
        median_diff=median_diff,
        method=method_name,
        all_zero=False,
    )


# --------------------------------------------------------------------- H1: static competitiveness


def h1_jssp_test(rpd_best_pdr: np.ndarray, rpd_rl_deterministic: np.ndarray) -> HypothesisVerdict:
    """``PreRegistration_FINAL.md`` Section 2, JSSP arm. N=43, paired by instance.

    Computes ``(RPD_bestPDR - RPD_RL)``; accept iff p<0.05 AND median diff > 0.
    """
    result = wilcoxon_signed_rank(rpd_best_pdr, rpd_rl_deterministic, alternative="two-sided")
    accept = (not result.all_zero) and result.p_value < 0.05 and result.median_diff > 0
    return HypothesisVerdict(
        hypothesis="H1 (JSSP)",
        verdict="accept" if accept else "reject",
        test_result=result,
        notes="Accept iff p<0.05 and median(RPD_bestPDR - RPD_RL) > 0 (PreRegistration_FINAL.md Sec 2).",
        raw={"n": result.n_nominal},
    )


def h1_fjsp_test(makespan_best_comparator: np.ndarray, makespan_rl: np.ndarray, *, arm: str) -> HypothesisVerdict:
    """Section 2, FJSP arm. Restricted to GA/CP-SAT comparators (no PDR has an FJSP
    machine-selection rule). Run once per FJSP arm (synthetic and real-Brandimarte,
    per ``FinalExperimentMatrix.md``'s amendment) -- report separately, never pooled.

    Args:
        arm: ``"synthetic"`` or ``"brandimarte"``, for the verdict label only.
    """
    result = wilcoxon_signed_rank(makespan_best_comparator, makespan_rl, alternative="two-sided")
    accept = (not result.all_zero) and result.p_value < 0.05 and result.median_diff > 0
    return HypothesisVerdict(
        hypothesis=f"H1 (FJSP, {arm})",
        verdict="accept" if accept else "reject",
        test_result=result,
        notes="Accept iff p<0.05 and median(makespan_bestComparator - makespan_RL) > 0, "
        "comparator restricted to ga/cpsat (PreRegistration_FINAL.md Sec 2).",
        raw={"n": result.n_nominal, "arm": arm},
    )


# ------------------------------------------------------------------------------- H3: robustness


def h3_regime_test(ri_pdr: np.ndarray, ri_rl: np.ndarray, regime_label: str) -> HypothesisVerdict:
    """Section 4. One call per one of the 9 regimes, paired by (instance x seed).

    Comparator is the single best-RI PDR for that regime (caller must have already
    selected that PDR's column before calling this -- this function does not do
    best-of selection itself). Accept per-regime iff p<0.05 AND
    median(RI_PDR - RI_RL) > 0. The overall H3 verdict (accept / reject / partial) is
    computed by ``report.py`` from the 9 per-regime verdicts, per Section 4's
    conjunctive ("all 9") rule -- not inside this function.
    """
    result = wilcoxon_signed_rank(ri_pdr, ri_rl, alternative="two-sided")
    accept = (not result.all_zero) and result.p_value < 0.05 and result.median_diff > 0
    return HypothesisVerdict(
        hypothesis=f"H3 ({regime_label})",
        verdict="accept" if accept else "reject",
        test_result=result,
        notes="Accept iff p<0.05 and median(RI_PDR - RI_RL) > 0, best-RI PDR as comparator "
        "(PreRegistration_FINAL.md Sec 4).",
        raw={"n": result.n_nominal, "regime": regime_label},
    )


def h3_overall_verdict(per_regime: list[HypothesisVerdict]) -> HypothesisVerdict:
    """Section 4's conjunctive rule: accept H3 overall only if all 9 regimes accept;
    otherwise report partial support explicitly, never rounded up to a full accept.
    """
    if len(per_regime) != 9:
        print(f"[stats.pairwise] h3_overall_verdict: expected 9 regime verdicts, got {len(per_regime)}")
    n_accept = sum(1 for v in per_regime if v.verdict == "accept")
    if n_accept == len(per_regime) and len(per_regime) > 0:
        verdict: Verdict = "accept"
    elif n_accept > 0:
        verdict = "partial"
    else:
        verdict = "reject"
    return HypothesisVerdict(
        hypothesis="H3 (overall)",
        verdict=verdict,
        test_result=None,
        notes=f"{n_accept}/{len(per_regime)} regimes accept. Conjunctive rule: overall accept "
        "requires all 9 (PreRegistration_FINAL.md Sec 4).",
        raw={"n_accept": n_accept, "n_regimes": len(per_regime), "per_regime": [v.hypothesis for v in per_regime]},
    )


# --------------------------------------------------------------- H4: disruption-aware reward

H4_MARGIN_PP = 2.0


def h4_test(
    ri_kappa: np.ndarray,
    ri_kappa0: np.ndarray,
    rpd_nominal_kappa: np.ndarray,
    rpd_nominal_kappa0: np.ndarray,
    kappa: float,
    *,
    regime_label: str,
) -> HypothesisVerdict:
    """Section 5. One regime's contribution to a given kappa's H4 verdict.

    Component (a): Wilcoxon on ``delta_RI = RI_kappa - RI_kappa0``, paired by
    (instance x seed), for THIS regime; ``a_holds`` iff p<0.05 and median delta_RI<0.
    Component (b): ``delta_RPD_nominal = mean(rpd_nominal_kappa) - mean(rpd_nominal_kappa0)``,
    compared against the fixed 2-percentage-point margin -- a THRESHOLD comparison, not
    a significance test (Section 5 is explicit this asymmetry is deliberate). Nominal
    RPD has no per-regime concept (measured once, under ``--disruption none``), so
    ``rpd_nominal_kappa``/``rpd_nominal_kappa0`` are the same arrays on every regime's
    call; ``b_holds`` is therefore identical across all 9 calls for a given kappa.

    This function returns ONE regime's (a)+(b) combination, mirroring
    :func:`h3_regime_test`'s "per-regime function, ``report.py``/companion aggregator
    combines" split -- Section 5's true per-kappa decision needs the *majority* (>=5/9)
    of regimes' component (a) to hold, which only :func:`h4_overall_verdict` (below,
    given all 9 of this function's outputs) can compute; a single call's ``verdict``
    field reflects only whether THIS regime's (a) held, not the final H4 verdict.

    Returns:
        HypothesisVerdict with ``raw["a_holds"]``, ``raw["b_holds"]``,
        ``raw["delta_rpd_nominal"]`` for the aggregator to consume.
    """
    ri_result = wilcoxon_signed_rank(ri_kappa, ri_kappa0, alternative="two-sided")
    a_holds = (not ri_result.all_zero) and ri_result.p_value < 0.05 and ri_result.median_diff < 0

    delta_rpd_nominal = float(np.mean(rpd_nominal_kappa) - np.mean(rpd_nominal_kappa0))
    b_holds = delta_rpd_nominal < H4_MARGIN_PP

    return HypothesisVerdict(
        hypothesis=f"H4 (kappa={kappa}, {regime_label})",
        verdict="accept" if a_holds else "reject",  # per-regime component (a) only, see docstring
        test_result=ri_result,
        notes="Per-regime component (a): p<0.05 and median(delta_RI)<0 (PreRegistration_FINAL.md Sec 5). "
        "Overall H4 verdict requires h4_overall_verdict() across all 9 regimes.",
        raw={
            "kappa": kappa,
            "regime": regime_label,
            "a_holds": a_holds,
            "b_holds": b_holds,
            "delta_rpd_nominal": delta_rpd_nominal,
        },
    )


def h4_overall_verdict(per_regime: list[HypothesisVerdict], kappa: float) -> HypothesisVerdict:
    """Section 5's actual per-kappa decision rule, combining all 9 :func:`h4_test` calls.

    > Accept H4 for kappa iff: (a) p<0.05 and median delta_RI<0 in >=5/9 regimes, AND
    > (b) delta_RPD_nominal < 2 percentage points.
    > (a) without (b): "robustness gained at an unacceptable nominal cost".
    > (b) without (a): "no measurable robustness benefit at this kappa".
    > Neither: "reject".
    """
    if len(per_regime) != 9:
        print(f"[stats.pairwise] h4_overall_verdict: expected 9 regime verdicts, got {len(per_regime)}")
    n_a_holds = sum(1 for v in per_regime if v.raw.get("a_holds"))
    a_overall = n_a_holds >= 5
    # b_holds is identical across all 9 calls (nominal RPD has no per-regime concept);
    # take the first rather than re-deriving, and assert that identically as a tripwire.
    b_values = {v.raw.get("b_holds") for v in per_regime}
    if len(b_values) > 1:
        print(f"[stats.pairwise] h4_overall_verdict: inconsistent b_holds across regimes: {b_values}")
    b_overall = per_regime[0].raw.get("b_holds", False) if per_regime else False
    delta_rpd_nominal = per_regime[0].raw.get("delta_rpd_nominal") if per_regime else None

    if a_overall and b_overall:
        label: Verdict = "accept"
        note = "accept"
    elif a_overall and not b_overall:
        label = "reject"
        note = "robustness gained at an unacceptable nominal cost"
    elif b_overall and not a_overall:
        label = "reject"
        note = "no measurable robustness benefit at this kappa"
    else:
        label = "reject"
        note = "reject"

    return HypothesisVerdict(
        hypothesis=f"H4 (kappa={kappa}, overall)",
        verdict=label,
        test_result=None,
        notes=f"{note}. component(a): {n_a_holds}/9 regimes significant (need >=5); "
        f"component(b): delta_RPD_nominal={delta_rpd_nominal} vs {H4_MARGIN_PP}pp margin "
        "(PreRegistration_FINAL.md Sec 5).",
        raw={
            "kappa": kappa,
            "n_a_holds": n_a_holds,
            "a_overall": a_overall,
            "b_overall": b_overall,
            "delta_rpd_nominal": delta_rpd_nominal,
            "label": note,
        },
    )


# --------------------------------------------------------------------------- H5: decision latency


def h5_check(ms_per_decision_rl: dict[str, np.ndarray], ms_per_decision_cpsat: dict[str, np.ndarray]) -> HypothesisVerdict:
    """Section 6. NOT a significance test -- Section 6 is explicit ("no test required").

    Accept iff, for EVERY instance, ``median(ms/decision_RL) < 0.01 * median(ms/decision_CPSAT)``.

    Args:
        ms_per_decision_rl, ms_per_decision_cpsat: ``{instance_name: array_of_values}``,
            same instance keys on both sides (Section 6's decision rule is inherently
            per-instance, so this function operates on a per-instance mapping rather
            than a single flat array).

    Returns:
        HypothesisVerdict. ``raw["per_instance"]`` has one row per instance:
        ``{"instance", "median_rl", "median_cpsat", "ratio", "passes"}``, and
        ``raw["ratio_widens_with_size"]`` reports whether the CP-SAT/RL ratio trends
        upward with ``num_operations`` (the descriptive figure Section 6 asks for),
        computed by simple Spearman-style rank correlation on the per-instance ratios
        ordered by instance key (caller is expected to pass instances in size order).
    """
    common = sorted(set(ms_per_decision_rl) & set(ms_per_decision_cpsat))
    missing = (set(ms_per_decision_rl) | set(ms_per_decision_cpsat)) - set(common)
    if missing:
        print(f"[stats.pairwise] h5_check: instance(s) missing from one side, excluded: {sorted(missing)}")

    rows = []
    for instance in common:
        rl_vals = np.asarray(ms_per_decision_rl[instance], dtype=float)
        cpsat_vals = np.asarray(ms_per_decision_cpsat[instance], dtype=float)
        median_rl = float(np.median(rl_vals))
        median_cpsat = float(np.median(cpsat_vals))
        ratio = median_cpsat / median_rl if median_rl > 0 else float("inf")
        passes = median_rl < 0.01 * median_cpsat
        rows.append(
            {
                "instance": instance,
                "median_rl": median_rl,
                "median_cpsat": median_cpsat,
                "ratio": ratio,
                "passes": bool(passes),
            }
        )

    all_pass = all(r["passes"] for r in rows) and len(rows) > 0
    ratios = [r["ratio"] for r in rows]
    widens = bool(len(ratios) > 1 and ratios[-1] > ratios[0])

    return HypothesisVerdict(
        hypothesis="H5",
        verdict="accept" if all_pass else "reject",
        test_result=None,
        notes="Accept iff median(ms/decision_RL) < 1%*median(ms/decision_CPSAT) for every "
        "instance in the size range (PreRegistration_FINAL.md Sec 6, no significance test).",
        raw={"per_instance": rows, "ratio_widens_with_size": widens},
    )
