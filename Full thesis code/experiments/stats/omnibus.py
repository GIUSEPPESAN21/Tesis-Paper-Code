"""Omnibus comparisons -- Friedman test and Nemenyi post-hoc / critical difference.

Per ``StatsAnalysis_TechSpec.md`` Section 7. Used for H2 (size generalization) and any
other k>2-method comparison ranked over a block design of instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2_dist
from statsmodels.stats.libqsturng import qsturng

#: Demsar (2006) Table 5(b), alpha=0.05 -- cross-check only, not the production path
#: (Section 7.2: "a transcription error in a hardcoded critical-value table is exactly
#: the class of silent, hard-to-catch bug this pipeline exists to avoid").
_DEMSAR_Q_ALPHA_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


@dataclass(frozen=True)
class FriedmanResult:
    chi_square: float
    df: int
    p_value: float
    k_methods: int
    n_blocks: int
    average_ranks: dict[str, float]


def friedman_test(pivoted: pd.DataFrame) -> FriedmanResult:
    """Friedman test over a (n_blocks x k_methods) matrix.

    Args:
        pivoted: Rows = blocks (instances), columns = methods, values = the metric
            being ranked (RPD for E1/E3, RI for a multi-method robustness comparison).
            Every cell must be present -- see the missing-data policy in this module's
            top docstring / ``report.py``.

    Returns:
        FriedmanResult. Ranks are computed method-averaged across blocks, ascending
        (rank 1 = best/lowest RPD), matching D6's convention.

    Raises:
        ValueError: If ``pivoted`` has any missing (NaN) cell -- Friedman requires a
            complete block design. This function does not impute or drop rows; the
            caller (``report.py``) must decide whether to drop the offending
            instance-block entirely and must say which choice it made in the resulting
            report, not silently drop rows here.
    """
    if pivoted.isna().any().any():
        bad = pivoted.index[pivoted.isna().any(axis=1)].tolist()
        raise ValueError(
            f"friedman_test: pivoted has missing cell(s) in block(s) {bad}; Friedman "
            "requires a complete block design. Drop or impute before calling, and "
            "report the choice (StatsAnalysis_TechSpec.md Sec 7.1)."
        )
    if pivoted.shape[0] < 1 or pivoted.shape[1] < 2:
        raise ValueError(f"friedman_test: need >=1 block and >=2 methods, got shape {pivoted.shape}")

    k = pivoted.shape[1]
    n = pivoted.shape[0]

    # scipy.stats.friedmanchisquare hard-rejects k<3 samples even though the Friedman
    # statistic is well-defined for k=2 -- reimplement its exact formula (tie-corrected
    # chi-square, Iman & Davenport / Demsar 2006) directly so k=2 comparisons (which this
    # module's own >=2-methods contract promises) work too. Verified to reproduce
    # scipy's k>=3 output bit-for-bit (see TestFriedmanGoldenWeek21).
    ranks = pivoted.rank(axis=1, method="average", ascending=True)
    average_ranks = {str(col): float(ranks[col].mean()) for col in pivoted.columns}

    rank_matrix = ranks.to_numpy(dtype=float)
    ties = 0
    for row in rank_matrix:
        _, counts = np.unique(row, return_counts=True)
        for t in counts[counts > 1]:
            ties += int(t) * (int(t) * int(t) - 1)
    correction = 1 - ties / (k * (k * k - 1) * n)

    col_rank_sums = rank_matrix.sum(axis=0)
    ssbn = float(np.sum(col_rank_sums**2))
    statistic = (12.0 / (k * n * (k + 1)) * ssbn - 3 * n * (k + 1)) / correction
    p_value = float(_chi2_dist.sf(statistic, k - 1))

    return FriedmanResult(
        chi_square=float(statistic),
        df=k - 1,
        p_value=p_value,
        k_methods=k,
        n_blocks=n,
        average_ranks=average_ranks,
    )


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """``CD = q_alpha * sqrt(k*(k+1) / (6*n))``, Demsar (2006) / D6's exact formula.

    ``q_alpha`` is the studentized range statistic (infinite df) divided by
    ``sqrt(2)``, computed via ``qsturng(1 - alpha, k, np.inf) / sqrt(2)``.

    Args:
        k: Number of methods compared.
        n: Number of blocks (instances).
        alpha: Family-wise significance level, 0.05 throughout this thesis.

    Returns:
        The critical difference in average-rank units.

    Raises:
        ValueError: If the computed ``q_alpha`` disagrees with the Demsar (2006)
            reference table (Section 7.2) by more than 0.01 for ``k<=10`` -- a
            dependency/version bug to fix before trusting any CD diagram, not a
            situation to silently proceed through.
    """
    if k < 2:
        raise ValueError(f"critical_difference: need k>=2 methods, got {k}")
    if n < 1:
        raise ValueError(f"critical_difference: need n>=1 blocks, got {n}")

    q_alpha = float(qsturng(1 - alpha, k, np.inf)) / math.sqrt(2)

    if k <= 10 and abs(alpha - 0.05) < 1e-12:
        reference = _DEMSAR_Q_ALPHA_005[k]
        if abs(q_alpha - reference) > 0.01:
            raise ValueError(
                f"critical_difference: qsturng-computed q_alpha={q_alpha:.4f} for k={k} "
                f"disagrees with Demsar (2006) reference {reference} by more than 0.01 "
                "-- statsmodels/qsturng dependency issue, not a data issue."
            )

    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_posthoc(friedman_result: FriedmanResult, cd: float) -> pd.DataFrame:
    """Pairwise significant-difference flags: ``|rank_i - rank_j| > cd``.

    Returns:
        A k x k boolean DataFrame (method names as both index and columns), True where
        the pair is significantly different at the family-wise alpha.

    Raises:
        ValueError: If ``friedman_result.p_value >= 0.05`` -- only call this when the
            Friedman null was rejected; ``report.py`` must check this and, if not
            rejected, report the comparison as "not significant, no post-hoc run"
            rather than running Nemenyi anyway.
    """
    if friedman_result.p_value >= 0.05:
        raise ValueError(
            f"nemenyi_posthoc: Friedman p={friedman_result.p_value:.4f} >= 0.05 (not "
            "rejected) -- do not run Nemenyi on a non-significant omnibus result "
            "(StatsAnalysis_TechSpec.md Sec 7.2)."
        )
    methods = list(friedman_result.average_ranks.keys())
    ranks = np.array([friedman_result.average_ranks[m] for m in methods])
    diff = np.abs(ranks[:, None] - ranks[None, :])
    significant = diff > cd
    return pd.DataFrame(significant, index=methods, columns=methods)
