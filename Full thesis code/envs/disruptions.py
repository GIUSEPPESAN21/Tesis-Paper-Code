"""Independently testable disruption generators for the dynamic JSSP extension.

Three disruption axes, each a standalone class with its own ``numpy.random.Generator``,
usable and unit-testable without instantiating a full
:class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`:

:class:`BreakdownGenerator`
    Machine time-to-failure (Exponential) and time-to-repair (Lognormal), calibrated to
    a target per-machine availability.
:class:`ProcessingTimeSampler`
    A mean-1 Lognormal multiplier applied to an operation's nominal duration.
:class:`ArrivalProcess`
    A Poisson arrival process for new jobs, calibrated to a target shop utilization, with
    a mandatory cutoff so the process is guaranteed to terminate (see
    :func:`ArrivalProcess.next_arrival_time`).

Independent RNG streams
------------------------
:func:`spawn_disruption_rngs` derives three child generators from one top-level seed via
``numpy.random.default_rng(seed).spawn(3)`` in the fixed order (breakdown, processing,
arrival). Because ``spawn`` always produces the same three independent streams regardless
of which disruption axes are actually enabled, toggling one axis on or off never perturbs
the random draws consumed by the others --- required for reproducible ablations across the
experimental matrix (see ``DynamicEnv_TechSpec.md`` §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from envs.instance import JSSPInstance


@dataclass(frozen=True)
class DisruptionConfig:
    """Configuration for all three disruption axes plus the preemption policy.

    Every field has an "off" default, so ``DynamicJSSPEnv(instance, DisruptionConfig())``
    is numerically identical to ``JSSPEnv(instance)`` --- the regression contract checked
    by ``tests/test_dynamic_env.py::test_default_config_matches_static_env_exactly``.

    Attributes:
        target_availability: Per-machine availability ``A_k``. A single float applies the
            same target to every machine; a ``dict`` keys availability by machine index
            (machines absent from the dict default to ``1.0``). ``1.0`` disables
            breakdowns for that machine.
        mttr_multiplier: ``MTTR = mttr_multiplier * mean_processing_time``.
        ttr_shape_sigma: Shape parameter ``sigma_r`` of the Lognormal TTR distribution.
        processing_time_sigma: Shape parameter ``s`` of the Lognormal duration
            multiplier. ``0.0`` disables stochastic processing times (deterministic,
            static-env-equivalent).
        target_utilization: Target utilization ``rho`` driving the arrival rate. ``None``
            disables dynamic arrivals.
        arrival_cutoff_time: Simulated time after which no new arrivals are generated.
            ``None`` -> ``0.8 * total_processing_time`` of the base instance.
        max_dynamic_jobs: Hard cap on the number of dynamically arriving jobs; also sizes
            the pre-allocated growable arrays in ``DynamicJSSPEnv``.
        preemption_policy: ``"resume"`` (default, required) or ``"restart"`` (retained as
            a documented, config-gated ablation --- see ``DynamicEnv_TechSpec.md`` §5).
        seed: Top-level seed for :func:`spawn_disruption_rngs`.
    """

    target_availability: float | dict[int, float] = 1.0
    mttr_multiplier: float = 2.0
    ttr_shape_sigma: float = 0.5
    processing_time_sigma: float = 0.0
    target_utilization: float | None = None
    arrival_cutoff_time: float | None = None
    max_dynamic_jobs: int = 50
    preemption_policy: Literal["resume", "restart"] = "resume"
    seed: int | None = None


def spawn_disruption_rngs(
    seed: int | None,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Derive three independent child RNGs (breakdown, processing, arrival) from ``seed``.

    Args:
        seed: Top-level seed. ``None`` draws fresh entropy (non-reproducible).

    Returns:
        ``(breakdown_rng, processing_rng, arrival_rng)``.
    """
    parent = np.random.default_rng(seed)
    breakdown_rng, processing_rng, arrival_rng = parent.spawn(3)
    return breakdown_rng, processing_rng, arrival_rng


class BreakdownGenerator:
    """TTF ~ Exponential(lambda_k), TTR ~ Lognormal(mu_r, sigma_r^2).

    Calibrated so that the renewal process ``TTF, TTR, TTF, TTR, ...`` has long-run
    availability ``A_k = MTBF_k / (MTBF_k + MTTR_k)`` matching
    ``config.target_availability`` (see ``DynamicEnv_TechSpec.md`` §4.1 for the full
    derivation)::

        p_bar   = instance.durations.mean()
        MTTR    = config.mttr_multiplier * p_bar
        sigma_r = config.ttr_shape_sigma
        mu_r    = ln(MTTR) - sigma_r**2 / 2         # so E[Lognormal] == MTTR exactly
        MTBF_k  = A_k * MTTR / (1 - A_k)
        lambda_k = 1 / MTBF_k

    ``A_k = 1.0`` disables breakdowns for machine ``k`` entirely (no BREAKDOWN event is
    ever scheduled for it) rather than being treated as a literal zero rate.
    """

    def __init__(
        self,
        instance: JSSPInstance,
        config: DisruptionConfig,
        rng: np.random.Generator,
    ) -> None:
        self.rng = rng
        self.num_machines = instance.num_machines

        p_bar = float(instance.flat_durations.mean())
        self.mttr = config.mttr_multiplier * p_bar
        self.sigma_r = float(config.ttr_shape_sigma)
        self.mu_r = float(np.log(self.mttr) - self.sigma_r**2 / 2.0)

        target = config.target_availability
        if isinstance(target, dict):
            availability = np.array(
                [float(target.get(k, 1.0)) for k in range(self.num_machines)]
            )
        else:
            availability = np.full(self.num_machines, float(target))
        self.availability = availability
        self.enabled = availability < 1.0

        with np.errstate(divide="ignore"):
            mtbf = np.where(
                self.enabled, availability * self.mttr / (1.0 - availability), np.inf
            )
        self.mtbf = mtbf
        self.lambda_k = np.where(self.enabled, 1.0 / mtbf, 0.0)

    def is_enabled(self, machine: int) -> bool:
        """Whether breakdowns are active for ``machine`` (``False`` iff ``A_k == 1.0``)."""
        return bool(self.enabled[machine])

    def draw_ttf(self, machine: int) -> float:
        """Sample a time-to-failure for ``machine``. ``inf`` if breakdowns are disabled."""
        if not self.is_enabled(machine):
            return float("inf")
        return float(self.rng.exponential(scale=1.0 / self.lambda_k[machine]))

    def draw_ttr(self) -> float:
        """Sample a time-to-repair. Independent of which machine broke down."""
        return float(self.rng.lognormal(mean=self.mu_r, sigma=self.sigma_r))


class ProcessingTimeSampler:
    """Lognormal(-s^2/2, s^2) duration multiplier, mean 1 by construction.

    **Design decision** (flagged, ``DynamicEnv_TechSpec.md`` §10.5): the multiplier is
    drawn once per operation, at the moment it is dispatched, fixing
    ``effective_duration`` for the rest of that operation's life (including across a
    preempt-resume pause). This mirrors how the static environment already treats
    duration as fixed-at-dispatch-time; only the value being fixed changes.
    """

    def __init__(self, sigma: float, rng: np.random.Generator) -> None:
        self.sigma = float(sigma)
        self.rng = rng

    @property
    def enabled(self) -> bool:
        """Whether stochastic processing times are active (``sigma > 0``)."""
        return self.sigma > 0.0

    def draw_multiplier(self) -> float:
        """Sample one multiplier. Exactly ``1.0`` when disabled."""
        if not self.enabled:
            return 1.0
        return float(self.rng.lognormal(mean=-(self.sigma**2) / 2.0, sigma=self.sigma))

    def effective_duration(self, base_duration: float) -> float:
        """``base_duration`` scaled by one freshly-drawn multiplier."""
        return float(base_duration) * self.draw_multiplier()


class ArrivalProcess:
    """Poisson arrival process for new jobs, calibrated to target utilization ``rho``.

    Calibration (``DynamicEnv_TechSpec.md`` §4.3)::

        E_load_per_job = instance.job_totals.mean()
        lambda_a = rho * num_machines / E_load_per_job

    Inter-arrival times are drawn incrementally (``Exp(lambda_a)``) rather than
    pre-generated, and the FEL never holds more than one pending ``ARRIVAL`` event: call
    :meth:`next_arrival_time` only after the previous arrival has actually fired.

    **Termination safety** (mandatory: a Poisson process alone never stops on its own):
    :meth:`next_arrival_time` returns ``None`` once ``max_dynamic_jobs`` jobs have already
    been generated (via :meth:`generate_job`) or the candidate time would exceed
    ``arrival_cutoff_time``, whichever comes first.

    **Content model for arriving jobs** (design decision, flagged --- §10.2, not
    specified by the framework document): each new job reuses ``instance.ops_per_job``
    (rectangularity preserved); its machine order is a uniformly random permutation of
    ``range(num_machines)`` truncated to ``ops_per_job`` (preserving the "no repeated
    machine within a job" invariant), and its durations are drawn i.i.d. by bootstrap
    resampling from the base instance's empirical duration distribution
    (``instance.flat_durations``) rather than a parametric fit.
    """

    def __init__(
        self,
        instance: JSSPInstance,
        config: DisruptionConfig,
        rng: np.random.Generator,
    ) -> None:
        self.instance = instance
        self.rng = rng
        self.ops_per_job = instance.ops_per_job
        self.num_machines = instance.num_machines
        self.enabled = config.target_utilization is not None
        self.max_dynamic_jobs = config.max_dynamic_jobs

        horizon = max(instance.total_processing_time, 1)
        self.cutoff_time = (
            float(config.arrival_cutoff_time)
            if config.arrival_cutoff_time is not None
            else 0.8 * horizon
        )

        if self.enabled:
            e_load_per_job = float(instance.job_totals.mean())
            self.lambda_a = config.target_utilization * self.num_machines / e_load_per_job
        else:
            self.lambda_a = 0.0

        self.num_generated = 0

    def next_arrival_time(self, current_time: float) -> float | None:
        """Draw the next absolute arrival time, or ``None`` if arrivals have stopped.

        Args:
            current_time: Simulated time the previous arrival (or the episode start)
                occurred at; the next inter-arrival gap is measured from here.

        Returns:
            The next arrival's absolute time, or ``None`` if disabled, the
            ``max_dynamic_jobs`` cap has been reached, or the drawn time would exceed
            ``arrival_cutoff_time``.
        """
        if not self.enabled or self.num_generated >= self.max_dynamic_jobs:
            return None
        candidate = float(current_time) + float(self.rng.exponential(scale=1.0 / self.lambda_a))
        if candidate > self.cutoff_time:
            return None
        return candidate

    def generate_job(self) -> dict[str, np.ndarray]:
        """Materialize one arriving job's machine order and durations.

        Increments the internal arrival counter that :meth:`next_arrival_time` checks
        against ``max_dynamic_jobs`` --- call this exactly once per fired ``ARRIVAL``
        event, never speculatively.

        Returns:
            ``{"machines": int array, "durations": int array}``, each of length
            ``ops_per_job``.
        """
        machines = self.rng.permutation(self.num_machines)[: self.ops_per_job].astype(np.int64)
        durations = self.rng.choice(
            self.instance.flat_durations, size=self.ops_per_job, replace=True
        ).astype(np.int64)
        self.num_generated += 1
        return {"machines": machines, "durations": durations}
