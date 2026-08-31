"""Flexible job-shop scheduling (FJSP) instance representation.

:class:`~envs.instance.JSSPInstance` models the *classic* JSSP: exactly one machine
per operation. The FJSP relaxes that to an *eligible set* of machines per operation,
each with its own (possibly different) processing time. :class:`FJSPInstance` is that
representation, promoted here from what was previously a solver-local fixture living
inside ``baselines/cpsat.py`` -- machine eligibility is instance data, not a detail of
any one solver, so it belongs next to :class:`~envs.instance.JSSPInstance` where every
other consumer (CP-SAT's ``solve_cpsat_fjsp``, the GA's assignment-vector chromosome)
can import it from one canonical place instead of each other.

No real FJSP benchmark loader (Brandimarte ``MK*``, Hurink) is bundled this sprint --
fetching and parsing a new benchmark family is a research-scope decision for a later
week, not a code-health one. :func:`generate_fjsp_instance` fills the gap for testing
and evaluation in the meantime: it derives a reproducible synthetic FJSP instance from
a real bundled :class:`~envs.instance.JSSPInstance` by adding alternative eligible
machines on top of that instance's real routing, so every downstream FJSP code path
(this module's validation, the GA's assignment gene, CP-SAT's ``OptionalIntervalVar``
model) can be exercised against real ``ft``/``la`` processing-time data rather than only
a hand-built toy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.instance import JSSPInstance


@dataclass(frozen=True)
class FJSPInstance:
    """A flexible JSSP instance: a ragged eligible-machine set per operation.

    Args:
        name: Instance identifier.
        num_jobs: Number of jobs.
        ops_per_job: Operations per job (rectangular across jobs, like
            :class:`~envs.instance.JSSPInstance`; only the *machine* dimension is
            ragged here).
        num_machines: Number of distinct machines in the shop.
        eligible_machines: ``eligible_machines[job][pos]`` is the non-empty list of
            machine indices operation ``(job, pos)`` may run on.
        durations: ``durations[job][pos]`` is the list of processing times, parallel to
            ``eligible_machines[job][pos]`` (duration may depend on the chosen machine,
            as in general FJSP).

    Raises:
        ValueError: If the ragged structure is inconsistent (wrong job/position count,
            an eligible-machine list and its duration list disagree in length), an
            operation has no eligible machine, a machine index is out of range, a
            duration is negative, or the same machine appears twice for one operation.
    """

    name: str
    num_jobs: int
    ops_per_job: int
    num_machines: int
    eligible_machines: list[list[list[int]]]
    durations: list[list[list[int]]]

    def __post_init__(self) -> None:
        if self.num_jobs <= 0 or self.ops_per_job <= 0 or self.num_machines <= 0:
            raise ValueError(
                f"{self.name}: num_jobs/ops_per_job/num_machines must all be positive, "
                f"got ({self.num_jobs}, {self.ops_per_job}, {self.num_machines})"
            )
        if len(self.eligible_machines) != self.num_jobs or len(self.durations) != self.num_jobs:
            raise ValueError(
                f"{self.name}: expected {self.num_jobs} job rows, got "
                f"{len(self.eligible_machines)} (eligible_machines) / {len(self.durations)} (durations)"
            )
        for job in range(self.num_jobs):
            machines_row = self.eligible_machines[job]
            durations_row = self.durations[job]
            if len(machines_row) != self.ops_per_job or len(durations_row) != self.ops_per_job:
                raise ValueError(
                    f"{self.name}: job {job} expected {self.ops_per_job} operations, got "
                    f"{len(machines_row)} (eligible_machines) / {len(durations_row)} (durations)"
                )
            for pos in range(self.ops_per_job):
                machines = machines_row[pos]
                durs = durations_row[pos]
                if len(machines) != len(durs):
                    raise ValueError(
                        f"{self.name}: operation ({job}, {pos}) has {len(machines)} eligible "
                        f"machines but {len(durs)} durations"
                    )
                if not machines:
                    raise ValueError(f"{self.name}: operation ({job}, {pos}) has no eligible machine")
                if len(set(machines)) != len(machines):
                    raise ValueError(
                        f"{self.name}: operation ({job}, {pos}) lists a machine more than once"
                    )
                for m in machines:
                    if not 0 <= m < self.num_machines:
                        raise ValueError(
                            f"{self.name}: operation ({job}, {pos}) references machine {m} "
                            f"outside [0, {self.num_machines})"
                        )
                if any(d < 0 for d in durs):
                    raise ValueError(f"{self.name}: operation ({job}, {pos}) has a negative duration")

    @property
    def num_operations(self) -> int:
        """Total number of operations."""
        return self.num_jobs * self.ops_per_job

    def op_id(self, job: int, position: int) -> int:
        """Global id of operation ``(job, position)``, same convention as :class:`~envs.instance.JSSPInstance`."""
        return job * self.ops_per_job + position

    def job_of(self, op_id: int) -> int:
        """Job owning the given global operation id. Inverse of :meth:`op_id`."""
        return op_id // self.ops_per_job

    def position_of(self, op_id: int) -> int:
        """Position within its job of the given global operation id. Inverse of :meth:`op_id`."""
        return op_id % self.ops_per_job

    def __str__(self) -> str:
        n_alt = sum(
            len(self.eligible_machines[job][pos])
            for job in range(self.num_jobs)
            for pos in range(self.ops_per_job)
        )
        avg_alt = n_alt / max(self.num_operations, 1)
        return (
            f"FJSPInstance({self.name}: {self.num_jobs} jobs x {self.ops_per_job} ops, "
            f"{self.num_machines} machines, avg {avg_alt:.2f} eligible machines/op)"
        )


def generate_fjsp_instance(
    base: JSSPInstance,
    *,
    max_extra_alternatives: int = 2,
    duration_variation: float = 0.2,
    seed: int | None = None,
) -> FJSPInstance:
    """Derive a reproducible synthetic :class:`FJSPInstance` from a real bundled JSSP instance.

    Every operation's original ``(machine, duration)`` is kept as its first eligible
    alternative -- so the generated instance is never *more* constrained than ``base``,
    and a degenerate CP-SAT/GA run against it (every op forced to its sole original
    alternative) reproduces ``base``'s own optimum exactly, which is what
    ``tests/test_fjsp_instance.py`` checks. On top of that, each operation gets
    ``Uniform{0, ..., max_extra_alternatives}`` further alternative machines drawn
    without replacement from the machines it does *not* already use, each with a
    duration perturbed by ``+/- duration_variation`` (relative) around the original.

    Args:
        base: The real JSSP instance to derive machine flexibility from.
        max_extra_alternatives: Upper bound (exclusive) on how many alternative
            machines beyond the original are added per operation.
        duration_variation: Relative perturbation applied to an alternative machine's
            duration, e.g. ``0.2`` draws a multiplier in ``[0.8, 1.2]``.
        seed: Seeds the alternative-machine/duration sampling. ``None`` draws fresh
            entropy (non-reproducible).

    Returns:
        A synthetic :class:`FJSPInstance` named ``f"{base.name}_fjsp"``.

    Raises:
        ValueError: If ``base`` has fewer than 2 machines (no alternative to add).
    """
    if base.num_machines < 2:
        raise ValueError(
            f"{base.name}: need at least 2 machines to synthesize routing flexibility, "
            f"got {base.num_machines}"
        )
    rng = np.random.default_rng(seed)

    eligible_machines: list[list[list[int]]] = []
    durations: list[list[list[int]]] = []
    for job in range(base.num_jobs):
        job_machines: list[list[int]] = []
        job_durations: list[list[int]] = []
        for pos in range(base.ops_per_job):
            primary_machine = int(base.machines[job, pos])
            primary_duration = int(base.durations[job, pos])

            other_machines = [m for m in range(base.num_machines) if m != primary_machine]
            rng.shuffle(other_machines)
            n_extra = min(int(rng.integers(0, max_extra_alternatives + 1)), len(other_machines))
            chosen = other_machines[:n_extra]

            machines_here = [primary_machine] + chosen
            durations_here = [primary_duration]
            for _ in chosen:
                multiplier = 1.0 + rng.uniform(-duration_variation, duration_variation)
                durations_here.append(max(1, int(round(primary_duration * multiplier))))

            job_machines.append(machines_here)
            job_durations.append(durations_here)
        eligible_machines.append(job_machines)
        durations.append(job_durations)

    return FJSPInstance(
        name=f"{base.name}_fjsp",
        num_jobs=base.num_jobs,
        ops_per_job=base.ops_per_job,
        num_machines=base.num_machines,
        eligible_machines=eligible_machines,
        durations=durations,
    )
