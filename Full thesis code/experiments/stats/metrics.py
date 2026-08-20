"""Metric functions -- RPD, Robustness Index, decision latency, summary stats.

Formulas copied verbatim from ``Step 4 Thesis.md`` D6, per
``StatsAnalysis_TechSpec.md`` Section 5. Do not re-derive.
"""

from __future__ import annotations

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
    alongside it."""

    ri: float
    std: float
    cvar_90: float
    n_replicates: int
    nominal_makespan: float


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

    return RobustnessResult(
        ri=ri, std=std, cvar_90=cvar_90, n_replicates=int(values.size), nominal_makespan=float(nominal_makespan)
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
