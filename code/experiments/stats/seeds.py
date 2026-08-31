"""Fase 2 (T2.3): agregacion multi-semilla -- media, desviacion estandar, intervalo de
confianza al 95% (t de Student), y rango.

Hoy la mayoria de metricas de la politica (best_training_makespan, RI medio por
regimen, RPD nominal, etc.) se reportan como un unico valor puntual de una sola
semilla de entrenamiento. Este modulo es la unica pieza de agregacion multi-semilla
que ``report.py``/``generate_thesis_figures.py`` deben usar de aqui en adelante para
convertir un valor puntual en media +/- DE con IC95, una vez T2.1 produzca mas de una
semilla por corrida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats


@dataclass(frozen=True)
class SeedAggregate:
    """Multi-seed summary of one scalar metric."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    range_low: float
    range_high: float
    n_seeds: int


def aggregate_over_seeds(values: np.ndarray | list[float]) -> SeedAggregate:
    """Mean, std, 95% CI (t-Student), and range over one metric's per-seed values.

    Args:
        values: One value per seed. ``n=1`` degenerates gracefully (std=0, CI collapses
            to the point value, range is that single value repeated) rather than
            raising -- a single-seed run is still a valid (if uninformative) input,
            since T2.1's multi-seed sweep is opt-in and this function must not force a
            caller to special-case the pre-T2.1 single-seed case.

    Returns:
        A :class:`SeedAggregate`.

    Raises:
        ValueError: If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")

    mean = float(np.mean(arr))
    n = int(arr.size)
    if n == 1:
        return SeedAggregate(
            mean=mean, std=0.0, ci95_low=mean, ci95_high=mean,
            range_low=mean, range_high=mean, n_seeds=1,
        )

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    # Two-sided 95% CI via the t distribution (n-1 df) -- the standard, correct choice
    # for small n (this thesis's seed counts are typically 3-5), not a normal-quantile
    # approximation that understates the interval at small n.
    t_crit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * sem

    return SeedAggregate(
        mean=mean, std=std, ci95_low=mean - margin, ci95_high=mean + margin,
        range_low=float(np.min(arr)), range_high=float(np.max(arr)), n_seeds=n,
    )


def format_mean_ci(agg: SeedAggregate, *, decimals: int = 3) -> str:
    """``"12.345 +/- 1.234 (IC95 [11.000, 13.500], n=5)"``-style display string, for
    dropping directly into a Markdown table cell."""
    if agg.n_seeds == 1:
        return f"{agg.mean:.{decimals}f} (n=1, sin IC)"
    return (
        f"{agg.mean:.{decimals}f} +/- {agg.std:.{decimals}f} "
        f"(IC95 [{agg.ci95_low:.{decimals}f}, {agg.ci95_high:.{decimals}f}], n={agg.n_seeds})"
    )
