"""Due-date synthesis for the EDD/CR dispatching rules.

Part B6's EDD (Earliest Due Date) and CR (Critical Ratio) rules are defined in terms of
a per-job due date ``d_j``, but the benchmark families used in this thesis (``ft``/``la``/
``ta``) are pure-makespan instances that carry no published due dates --- ``JSSPInstance``
has no due-date field at all. This module fills that gap explicitly with the Total Work
Content (TWK) rule, a standard device for synthesizing due dates on due-date-free
benchmarks (see ``Baselines_TechSpec.md`` §4 and §9.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DueDateConfig:
    """Configuration for :func:`synthesize_due_dates`.

    Attributes:
        tightness: The TWK multiplier ``k`` in ``d_j = r_j + k * job_totals[j]``.
            ``1.5`` (the default) is a reasonable, literature-standard convention, not a
            value derived from Part B6 itself --- flagged explicitly as a modeling
            assumption, in the same spirit as ``envs/disruptions.py``'s
            ``ttr_shape_sigma=0.5``. Exposed as configuration (not a hardcoded constant)
            so a sensitivity sweep over it is a one-line change.
    """

    tightness: float = 1.5


def synthesize_due_dates(
    job_totals: np.ndarray,
    release_times: np.ndarray,
    tightness: float = 1.5,
) -> np.ndarray:
    """Total Work Content (TWK) due-date synthesis: ``d_j = r_j + tightness * job_totals[j]``.

    Args:
        job_totals: Per-job total processing time, e.g. ``instance.job_totals``. Shape
            ``(num_jobs,)``.
        release_times: Per-job release date ``r_j``. All-zeros for a static instance
            (every job is available at ``t=0``); for a dynamically arrived job this
            should be its actual arrival time, so the synthesized due date is relative to
            when the job actually became available rather than to episode start. Shape
            ``(num_jobs,)``, matching ``job_totals``.
        tightness: The TWK multiplier ``k``. See :class:`DueDateConfig`.

    Returns:
        Per-job synthesized due date. Shape ``(num_jobs,)``.

    Raises:
        ValueError: If ``job_totals`` and ``release_times`` disagree in shape.
    """
    job_totals = np.asarray(job_totals, dtype=np.float64)
    release_times = np.asarray(release_times, dtype=np.float64)
    if job_totals.shape != release_times.shape:
        raise ValueError(
            f"job_totals{job_totals.shape} and release_times{release_times.shape} "
            "must have the same shape"
        )
    return release_times + tightness * job_totals


def ensure_due_dates(env, config: DueDateConfig | None = None) -> np.ndarray:
    """Lazily compute and cache ``env.due_dates``, idempotent across repeated calls.

    ``edd_rule``/``cr_rule`` call this at the top of every invocation so they work as a
    drop-in ``(env, rng) -> op_id`` callable with zero setup --- exactly like the other
    seven rules, and in particular so they are picked up unmodified by the existing
    ``RULES``-keyed parametrized tests in ``tests/test_env.py``/``tests/test_dynamic_env.py``,
    which call a rule directly (``RULES[name](env, rng)``) with no due-date setup step of
    their own.

    Every job present at ``reset()`` time is assumed released at ``t=0`` (release times
    are all-zero). This covers ``JSSPEnv`` and a ``DynamicJSSPEnv`` with arrivals
    disabled; due dates for a dynamically *arrived* job (release time = actual arrival
    time, per ``Baselines_TechSpec.md`` §4.2) are out of scope this sprint, since it is
    not exercised by any evaluation in ``STEP3``/``STEP4`` (static ``ft``/``la``
    instances only) --- see ``BASELINES_RESULTS_WEEK9.md``.

    Args:
        env: A :class:`~envs.jssp_env.JSSPEnv` (or subclass) whose ``instance`` is the
            base :class:`~envs.instance.JSSPInstance`.
        config: Tightness configuration. Defaults to ``DueDateConfig()``. Ignored if
            ``env.due_dates`` is already populated (e.g. a caller pre-set it with a
            custom tightness before running the rule).

    Returns:
        ``env.due_dates``, an array of shape ``(instance.num_jobs,)``.
    """
    cached = getattr(env, "due_dates", None)
    if cached is not None:
        return cached
    cfg = config if config is not None else DueDateConfig()
    release_times = np.zeros(env.instance.num_jobs, dtype=np.float64)
    env.due_dates = synthesize_due_dates(env.instance.job_totals, release_times, cfg.tightness)
    return env.due_dates
