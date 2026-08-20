"""Metric functions -- RPD, Robustness Index, decision latency, summary stats.

Formulas copied verbatim from ``Step 4 Thesis.md`` D6, per
``StatsAnalysis_TechSpec.md`` Section 5. Do not re-derive.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def compute_rpd(makespan: float, optimum: float) -> float:
    """RPD (%) of a makespan against a reference optimum/best-known value.

    Args:
        makespan: Achieved makespan.
        optimum: Reference C* (proven optimum or best-known upper bound).

    Returns:
        RPD as a percentage. 0.0 means the reference was matched exactly.

    Raises:
        ValueError: If ``optimum <= 0``.
    """
    if optimum <= 0:
        raise ValueError(f"optimum must be > 0, got {optimum}")
    return 100.0 * (makespan - optimum) / optimum


@dataclass(frozen=True)
class RobustnessResult:
    """RI plus the dispersion measures ``PreRegistration_FINAL.md`` Section 4 requires
    alongside it.

    Attributes:
        ri_negative: T8.3 (Recomendacion del tutor, AUDIT_PREFLIGHT.md Fase 0a/0b):
            ``True`` iff ``ri < 0`` -- the perturbed mean makespan came in *below* the
            nominal one. This is very often an artifact of a poor nominal plan having
            more room to improve by chance under disruption than a near-optimal one
            (regression to the mean), not evidence of robustness. Callers must not read
            a negative RI as "this method is robust under this regime."
    """

    ri: float
    std: float
    cvar_90: float
    n_replicates: int
    nominal_makespan: float
    ri_negative: bool = False


def compute_ri(perturbed_makespans: np.ndarray, nominal_makespan: float) -> RobustnessResult:
    """Robustness index for one (instance, method, regime) cell.

    Args:
        perturbed_makespans: One value per replicate seed (expect ``len == 30`` per the
            frozen D4 grid, but does not hard-fail on a different count -- a warning is
            printed instead, since a partially-completed sweep should still be
            inspectable).
        nominal_makespan: ``C_max`` under ``--disruption none`` for the same
            (instance, method) -- for the RL arm this is E1's deterministic-rollout
            (``seed==0``) makespan; for PDR arms, the E1 CSV's PDR row for that
            instance. Reused, never recomputed here (Section 5.2's pinned source).

    Returns:
        RobustnessResult with ``ri``, ``std``, ``cvar_90`` (mean of the worst 10% of
        ``perturbed_makespans``, i.e. worst 3 of 30 at full replication), ``n_replicates``.

    Raises:
        ValueError: If ``nominal_makespan <= 0`` or ``perturbed_makespans`` is empty.
    """
    values = np.asarray(perturbed_makespans, dtype=float)
    if nominal_makespan <= 0:
        raise ValueError(f"nominal_makespan must be > 0, got {nominal_makespan}")
    if values.size == 0:
        raise ValueError("perturbed_makespans must not be empty")
    if values.size != 30:
        print(
            f"[stats.metrics] compute_ri: expected 30 replicates (D4's frozen grid), "
            f"got {values.size} -- reporting anyway (partial sweep)."
        )

    mean_perturbed = float(np.mean(values))
    ri = (mean_perturbed - nominal_makespan) / nominal_makespan
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    n_worst = max(1, int(np.ceil(0.1 * values.size)))
    worst = np.sort(values)[-n_worst:]
    cvar_90 = float(np.mean(worst))

    ri_negative = ri < 0.0
    if ri_negative:
        warnings.warn(
            f"RI={ri:.4f} < 0 (mean perturbed makespan {mean_perturbed:.2f} < nominal "
            f"{nominal_makespan:.2f}): RI is NOT interpretable as robustness for this "
            "(instance, method, regime) cell -- likely an artifact of a poor nominal "
            "plan having more room to improve by chance under disruption than a "
            "near-optimal one (AUDIT_PREFLIGHT.md Fase 0a/0b). Report descriptively; "
            "do not read this as evidence the method is robust.",
            RuntimeWarning,
            stacklevel=2,
        )

    return RobustnessResult(
        ri=ri, std=std, cvar_90=cvar_90, n_replicates=int(values.size),
        nominal_makespan=float(nominal_makespan), ri_negative=ri_negative,
    )


def scan_for_negative_ri(records: list[dict]) -> list[dict]:
    """T8.3: given a list of RI records (dicts with at least an ``"ri"`` key, plus
    whatever identifying fields the caller included -- e.g. ``instance``/``method``/
    ``regime``), return the subset with ``ri < 0``, for logging into
    ``stats_manifest.json`` ("registrar en el manifiesto que metodos y regimenes la
    disparan"). Pure filtering, no warning emitted here -- :func:`compute_ri` is the
    single place that warns; this is for manifest bookkeeping over records that may
    not have gone through :func:`compute_ri` directly (e.g. ``report.py``'s per-row
    pseudo-RI descriptive tables).
    """
    return [r for r in records if r.get("ri", 0.0) < 0.0]


@dataclass(frozen=True)
class RealizedQualityResult:
    """T8.2 (Recomendacion del tutor): RPD realizada bajo disrupcion frente al OPTIMO
    CERTIFICADO -- comparable ACROSS methods (unlike RI, which is self-relative to
    each method's own nominal makespan and therefore not a fair cross-method
    comparison, exactly the artifact ``AUDIT_PREFLIGHT.md`` Fase 0a/0b documents).
    Reported as a primary metric alongside RI, not instead of it -- the manuscript's
    own discussion already recommends this but never applies it to H3 (T8.2's
    motivating gap)."""

    mean_rpd_realized_pct: float
    std_rpd_realized_pct: float
    cvar_90_rpd_realized_pct: float
    n_replicates: int
    certified_optimum: float


def compute_realized_quality(
    perturbed_makespans: np.ndarray, certified_optimum: float
) -> RealizedQualityResult:
    """RPD of every perturbed replicate against the CERTIFIED optimum (T7.1's
    ``certified_optimal``/``mip_gap`` distinction -- only a CP-SAT row with
    ``status == "OPTIMAL"`` may serve as ``certified_optimum`` for an instance), plus
    CVaR_0.9 of that realized-RPD distribution.

    Unlike :func:`compute_ri` (denominator = this method's OWN nominal makespan, so a
    bad nominal plan mechanically produces a more forgiving RI, per
    ``AUDIT_PREFLIGHT.md`` Fase 0a/0b), every method's realized RPD here shares the
    SAME denominator (the instance's certified optimum) -- so realized RPD is directly
    comparable across methods, which RI by construction is not.

    Args:
        perturbed_makespans: One value per replicate seed.
        certified_optimum: The instance's CP-SAT-certified optimum (``status ==
            "OPTIMAL"``), never a ``FEASIBLE``/time-limited incumbent.

    Returns:
        A :class:`RealizedQualityResult`.

    Raises:
        ValueError: If ``certified_optimum <= 0`` or ``perturbed_makespans`` is empty.
    """
    values = np.asarray(perturbed_makespans, dtype=float)
    if certified_optimum <= 0:
        raise ValueError(f"certified_optimum must be > 0, got {certified_optimum}")
    if values.size == 0:
        raise ValueError("perturbed_makespans must not be empty")

    rpd_values = 100.0 * (values - certified_optimum) / certified_optimum
    n_worst = max(1, int(np.ceil(0.1 * rpd_values.size)))
    worst = np.sort(rpd_values)[-n_worst:]

    return RealizedQualityResult(
        mean_rpd_realized_pct=float(np.mean(rpd_values)),
        std_rpd_realized_pct=float(np.std(rpd_values, ddof=1)) if rpd_values.size > 1 else 0.0,
        cvar_90_rpd_realized_pct=float(np.mean(worst)),
        n_replicates=int(rpd_values.size),
        certified_optimum=float(certified_optimum),
    )


def compute_decision_latency(wall_clock_s: float, num_operations: int) -> float:
    """Milliseconds per decision epoch.

    Args:
        wall_clock_s: Total episode wall-clock time (the harness's ``wall_clock_s``
            column).
        num_operations: Number of decision epochs in one complete schedule (see
            Section 2.4's derivation rule -- ``num_jobs * num_machines`` for JSSP, a
            static per-instance lookup for the seven bundled FJSP instances).

    Returns:
        Milliseconds per decision, for direct comparison against CP-SAT's per-decision
        figure (``BASELINES_RESULTS_WEEK12.md``'s ``wall_clock_s`` divided the same way).

    Raises:
        ValueError: If ``num_operations <= 0``.
    """
    if num_operations <= 0:
        raise ValueError(f"num_operations must be > 0, got {num_operations}")
    return wall_clock_s / num_operations * 1000.0


def summary_stats(values: np.ndarray) -> dict:
    """Mean, median, std, min, max, n.

    Median is D6's preferred summary for latency (avoids one slow-outlier episode
    dominating the number); mean +/- std is the convention used elsewhere (RPD, RI).

    Raises:
        ValueError: If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }
