"""Internal representation of a static job-shop scheduling (JSSP) instance.

A classic JSSP instance consists of ``num_jobs`` jobs, each of which is a *fixed
chain* of ``ops_per_job`` operations.  Operation ``(j, k)`` must run on a prescribed
machine for a prescribed, deterministic duration, and cannot start before operation
``(j, k - 1)`` has finished.  Each machine processes at most one operation at a time
and operations are non-preemptive.  The objective is minimum makespan (``C_max``).

Everything downstream (the Gymnasium environment, the dispatching-rule baselines, the
CP-SAT wrapper) consumes :class:`JSSPInstance` so that adding a new file format only
requires a new parser, never a change to the solvers.

Indexing convention
-------------------
Operations carry a **global** integer id used as the node id in the disjunctive graph
and as the action id in the environment::

    op_id = job * ops_per_job + position

:meth:`JSSPInstance.job_of` and :meth:`JSSPInstance.position_of` invert this mapping.

Extension note (FJSP)
---------------------
The thesis eventually targets the *flexible* JSSP, where an operation may be assigned
to any machine from an eligible set.  This class models the classic (non-flexible)
case only: ``machines`` holds exactly one machine per operation.  The FJSP extension
planned for later weeks will subclass or generalise this into a ragged
``eligible_machines`` structure; the ``ops_per_job`` rectangularity assumption is the
main thing that will have to relax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Operation:
    """A single operation, returned by :meth:`JSSPInstance.operations`.

    Attributes:
        op_id: Global operation id (``job * ops_per_job + position``).
        job: Index of the owning job.
        position: 0-based position of this operation within its job's chain.
        machine: Index of the machine that must process this operation.
        duration: Deterministic processing time.
    """

    op_id: int
    job: int
    position: int
    machine: int
    duration: int


@dataclass(frozen=True)
class JSSPInstance:
    """An immutable static JSSP instance.

    Args:
        name: Instance identifier, e.g. ``"ft06"``.
        machines: Integer array of shape ``(num_jobs, ops_per_job)``; machine index
            required by operation ``(j, k)``.
        durations: Integer array of shape ``(num_jobs, ops_per_job)``; processing time
            of operation ``(j, k)``.
        num_machines: Number of distinct machines in the shop.
        source_file: Path the instance was parsed from, for provenance.

    Raises:
        ValueError: If the arrays disagree in shape, machine indices fall outside
            ``[0, num_machines)``, durations are negative, or a job visits the same
            machine twice (which is legal in general JSSP but never occurs in the
            ``ft``/``la``/``abz`` families and almost always signals a parse error).
    """

    name: str
    machines: np.ndarray
    durations: np.ndarray
    num_machines: int
    source_file: str | None = None
    # Cached derived quantities, populated in __post_init__ (frozen -> object.__setattr__).
    _job_totals: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _machine_totals: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        machines = np.asarray(self.machines, dtype=np.int64)
        durations = np.asarray(self.durations, dtype=np.int64)
        if machines.ndim != 2 or durations.ndim != 2:
            raise ValueError(f"{self.name}: machines/durations must be 2-D, got {machines.ndim}-D")
        if machines.shape != durations.shape:
            raise ValueError(
                f"{self.name}: shape mismatch machines{machines.shape} vs durations{durations.shape}"
            )
        if machines.size == 0:
            raise ValueError(f"{self.name}: instance is empty")
        if machines.min() < 0 or machines.max() >= self.num_machines:
            raise ValueError(
                f"{self.name}: machine indices out of range [0, {self.num_machines}); "
                f"observed [{machines.min()}, {machines.max()}]"
            )
        if durations.min() < 0:
            raise ValueError(f"{self.name}: negative processing time {durations.min()}")
        for job in range(machines.shape[0]):
            if len(np.unique(machines[job])) != machines.shape[1]:
                raise ValueError(
                    f"{self.name}: job {job} visits a machine more than once "
                    "(unexpected for this benchmark family; likely a parse error)"
                )

        object.__setattr__(self, "machines", machines)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "_job_totals", durations.sum(axis=1))
        machine_totals = np.zeros(self.num_machines, dtype=np.int64)
        np.add.at(machine_totals, machines.ravel(), durations.ravel())
        object.__setattr__(self, "_machine_totals", machine_totals)

    # ------------------------------------------------------------------ shape

    @property
    def num_jobs(self) -> int:
        """Number of jobs."""
        return int(self.machines.shape[0])

    @property
    def ops_per_job(self) -> int:
        """Number of operations in every job (instances are rectangular)."""
        return int(self.machines.shape[1])

    @property
    def num_operations(self) -> int:
        """Total number of operations, i.e. the number of nodes / decision steps."""
        return self.num_jobs * self.ops_per_job

    # ---------------------------------------------------------------- indexing

    def op_id(self, job: int, position: int) -> int:
        """Global id of operation ``(job, position)``."""
        return job * self.ops_per_job + position

    def job_of(self, op_id: int) -> int:
        """Job owning the given global operation id."""
        return op_id // self.ops_per_job

    def position_of(self, op_id: int) -> int:
        """Position within its job of the given global operation id."""
        return op_id % self.ops_per_job

    def operations(self) -> Iterator[Operation]:
        """Yield every operation in global-id order."""
        for job in range(self.num_jobs):
            for pos in range(self.ops_per_job):
                yield Operation(
                    op_id=self.op_id(job, pos),
                    job=job,
                    position=pos,
                    machine=int(self.machines[job, pos]),
                    duration=int(self.durations[job, pos]),
                )

    # ------------------------------------------------------------ flat views

    @property
    def flat_machines(self) -> np.ndarray:
        """Machine of each operation, indexed by global op id. Shape ``(num_operations,)``."""
        return self.machines.reshape(-1)

    @property
    def flat_durations(self) -> np.ndarray:
        """Duration of each operation, indexed by global op id. Shape ``(num_operations,)``."""
        return self.durations.reshape(-1)

    # ------------------------------------------------------------- aggregates

    @property
    def job_totals(self) -> np.ndarray:
        """Total processing time per job. Shape ``(num_jobs,)``."""
        return self._job_totals

    @property
    def machine_totals(self) -> np.ndarray:
        """Total processing time per machine. Shape ``(num_machines,)``."""
        return self._machine_totals

    @property
    def total_processing_time(self) -> int:
        """Sum of all processing times; a trivial upper bound on any active schedule."""
        return int(self.durations.sum())

    @property
    def max_duration(self) -> int:
        """Largest single processing time, used to normalise node features."""
        return int(self.durations.max())

    def remaining_work(self) -> np.ndarray:
        """Remaining work per operation: ``sum`` of its own and all later durations in its job.

        Returns:
            Array of shape ``(num_jobs, ops_per_job)``.
        """
        return np.cumsum(self.durations[:, ::-1], axis=1)[:, ::-1].copy()

    def head_times(self) -> np.ndarray:
        """Job-chain *head* of each operation: total duration of its predecessors.

        This is the earliest possible start time of the operation when machine
        capacity is ignored, i.e. a valid release-date relaxation.

        Returns:
            Array of shape ``(num_jobs, ops_per_job)``.
        """
        heads = np.zeros_like(self.durations)
        heads[:, 1:] = np.cumsum(self.durations[:, :-1], axis=1)
        return heads

    def tail_times(self) -> np.ndarray:
        """Job-chain *tail* of each operation: total duration of its successors.

        Returns:
            Array of shape ``(num_jobs, ops_per_job)``.
        """
        tails = np.zeros_like(self.durations)
        tails[:, :-1] = np.cumsum(self.durations[:, :0:-1], axis=1)[:, ::-1]
        return tails

    def __str__(self) -> str:
        return (
            f"JSSPInstance({self.name}: {self.num_jobs} jobs x {self.ops_per_job} ops, "
            f"{self.num_machines} machines, total work {self.total_processing_time})"
        )
