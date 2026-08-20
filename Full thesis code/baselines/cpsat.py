"""Exact / near-exact JSSP solver built on Google OR-Tools CP-SAT.

This is the *ground truth* for the thesis: CP-SAT proves optimality on the small and
medium classic instances within seconds, which lets us (a) validate that the instance
parser produced the intended data and (b) compute the true relative-percentage-deviation
of every heuristic and learned policy rather than comparing against a literature table.

Model
-----
Standard interval formulation.  One ``IntervalVar`` per operation, ``AddNoOverlap`` per
machine to enforce capacity, and a chain of ``start(next) >= end(prev)`` constraints per
job to enforce the technological order.  The objective minimises the maximum end time.

``ortools`` is imported lazily so that the rest of the codebase --- and the whole test
suite --- keeps working in an environment where OR-Tools is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs.disjunctive_graph import is_schedule_feasible
from envs.fjsp_instance import FJSPInstance
from envs.instance import JSSPInstance


@dataclass
class CPSATResult:
    """Outcome of a CP-SAT solve.

    Three distinct things a caller must not conflate (Recomendacion 1/10 del tutor,
    ``MEJORAS_TUTOR_IMPLEMENTADAS.md``): the *solver* is always ``"CP-SAT"`` (not a
    field here -- callers already know which function they called); ``status`` is the
    *solution status* CP-SAT itself reports; ``is_optimal``/``mip_gap`` together are
    the *certification* -- a schedule is "exacto-optimo" only when
    ``status == "OPTIMAL"`` (equivalently ``is_optimal`` and ``mip_gap == 0.0``), never
    merely because a solution was found (``status == "FEASIBLE"`` means CP-SAT hit its
    time limit with a feasible-but-unproven-optimal incumbent).

    Attributes:
        instance: Instance name.
        status: Solver status name (``"OPTIMAL"``, ``"FEASIBLE"``, ``"INFEASIBLE"``,
            ``"UNKNOWN"``) -- the *solution status*, not a claim of optimality by
            itself.
        makespan: Best makespan found, or ``None`` if no solution was found.
        best_bound: Best proven lower bound on the makespan.
        is_optimal: ``True`` iff ``status == "OPTIMAL"`` -- the only condition under
            which this result may be reported as "exacto-optimo" anywhere downstream.
        mip_gap: Relative optimality gap, ``|makespan - best_bound| / max(|best_bound|, 1)``.
            Exactly ``0.0`` when ``is_optimal`` is ``True`` (CP-SAT only reports
            ``OPTIMAL`` when the gap has closed); ``NaN`` when no solution was found
            at all (``status in {"INFEASIBLE", "UNKNOWN"}`` with no incumbent). A
            positive gap on a ``FEASIBLE`` row is the quantitative version of "this is
            not proven optimal" -- report it alongside ``status``, not instead of it.
        wall_time: Solver wall-clock seconds.
        start_times: Start time per global operation id, or ``None`` if no solution.
    """

    instance: str
    status: str
    makespan: int | None
    best_bound: int
    is_optimal: bool
    wall_time: float
    mip_gap: float = field(default=float("nan"))
    start_times: np.ndarray | None = field(default=None, repr=False)


def compute_mip_gap(makespan: int | None, best_bound: int, is_optimal: bool) -> float:
    """Relative CP-SAT optimality gap for one solve.

    Args:
        makespan: Best incumbent objective found, or ``None`` if no solution exists.
        best_bound: Best proven lower bound (``CpSolver.best_objective_bound``).
        is_optimal: Whether the solver's status was ``OPTIMAL``.

    Returns:
        ``0.0`` when ``is_optimal`` (CP-SAT never reports ``OPTIMAL`` with a residual
        gap); ``NaN`` when ``makespan is None`` (no incumbent to compare against a
        bound); otherwise ``|makespan - best_bound| / max(|best_bound|, 1)``, the
        standard relative-gap definition (denominator floored at 1 so a ``best_bound``
        of 0 -- possible only for a degenerate zero-length instance -- does not divide
        by zero).
    """
    if is_optimal:
        return 0.0
    if makespan is None:
        return float("nan")
    return abs(makespan - best_bound) / max(abs(best_bound), 1)


def solve_cpsat(
    instance: JSSPInstance,
    time_limit_s: float = 60.0,
    num_workers: int = 8,
    log_progress: bool = False,
    random_seed: int = 1,
) -> CPSATResult:
    """Solve a JSSP instance to (proven) minimum makespan with CP-SAT.

    Args:
        instance: The instance to solve.
        time_limit_s: Wall-clock budget.  Small classic instances (``ft06``, ``la01``-
            ``la15``) prove optimality well under a second; ``ft10``/``la16``-``la40``
            may need seconds to minutes.
        num_workers: CP-SAT search worker threads.
        log_progress: Stream the solver log to stdout.
        random_seed: Seeds CP-SAT's internal search (``solver.parameters.random_seed``).
            Defaults to **1**, OR-Tools' own default value for this parameter --
            explicit here (this call previously left it unset, which is
            behaviorally identical to passing 1, so this default reproduces every
            already-published result exactly; it does NOT default to 0, which would
            silently change behavior relative to every prior run). Fixes one source
            of run-to-run variance -- a proven-``OPTIMAL`` result's objective is
            deterministic regardless (branch-and-bound correctness doesn't depend on
            the seed), but a time-limited ``FEASIBLE`` result's specific solution can
            otherwise vary between runs. Does NOT make a ``FEASIBLE`` result fully
            reproducible when ``num_workers > 1``: CP-SAT's parallel portfolio search
            has residual thread-scheduling non-determinism that a fixed seed alone
            doesn't eliminate (see `README.md`'s reproducibility notes). Pass
            ``num_workers=1`` for bit-for-bit-reproducible ``FEASIBLE`` results.

    Returns:
        A :class:`CPSATResult`.

    Raises:
        ImportError: If ``ortools`` is not installed.
        AssertionError: If the returned schedule violates a JSSP constraint --- a
            paranoid cross-check of the model against
            :func:`~envs.disjunctive_graph.is_schedule_feasible`.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "solve_cpsat requires OR-Tools. Install it with `pip install ortools`."
        ) from exc

    horizon = instance.total_processing_time
    model = cp_model.CpModel()

    starts: dict[int, "cp_model.IntVar"] = {}
    ends: dict[int, "cp_model.IntVar"] = {}
    intervals_by_machine: dict[int, list] = {m: [] for m in range(instance.num_machines)}

    for op in instance.operations():
        start = model.NewIntVar(0, horizon, f"s_{op.op_id}")
        end = model.NewIntVar(0, horizon, f"e_{op.op_id}")
        interval = model.NewIntervalVar(start, op.duration, end, f"i_{op.op_id}")
        starts[op.op_id] = start
        ends[op.op_id] = end
        intervals_by_machine[op.machine].append(interval)

    # Machine capacity: one operation at a time.
    for machine, intervals in intervals_by_machine.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # Technological order within each job.
    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job - 1):
            model.Add(starts[instance.op_id(job, pos + 1)] >= ends[instance.op_id(job, pos)])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(ends.values()))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_progress)
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    start_array: np.ndarray | None = None
    best_makespan: int | None = None
    if has_solution:
        start_array = np.array(
            [solver.Value(starts[op_id]) for op_id in range(instance.num_operations)],
            dtype=np.float64,
        )
        best_makespan = int(solver.Value(makespan))
        feasible, violations = is_schedule_feasible(instance, start_array)
        assert feasible, f"CP-SAT returned an infeasible schedule: {violations[:3]}"

    best_bound = int(solver.best_objective_bound)
    is_optimal = status == cp_model.OPTIMAL
    return CPSATResult(
        instance=instance.name,
        status=status_name,
        makespan=best_makespan,
        best_bound=best_bound,
        is_optimal=is_optimal,
        wall_time=float(solver.wall_time),
        mip_gap=compute_mip_gap(best_makespan, best_bound, is_optimal),
        start_times=start_array,
    )


def solve_cpsat_fjsp(
    instance: FJSPInstance,
    time_limit_s: float = 60.0,
    num_workers: int = 8,
    log_progress: bool = False,
    random_seed: int = 1,
) -> CPSATResult:
    """Solve an FJSP instance (routing + sequencing) with CP-SAT.

    For each operation and each eligible machine, an ``OptionalIntervalVar`` is bound to
    a fresh boolean presence literal; ``AddExactlyOne`` over an operation's alternatives
    is CP-SAT's translation of the FJSP assignment constraint (exactly one machine chosen
    per operation), without needing a MILP's explicit big-M disjunctive pair. Optional
    intervals are grouped by machine and passed to ``AddNoOverlap`` exactly as in the JSSP
    path (:func:`solve_cpsat`) --- CP-SAT automatically ignores intervals whose presence
    literal is false, so the same constraint call handles routing flexibility with no
    extra bookkeeping. Precedence uses each operation's actual start/end variables
    (defined regardless of which alternative is chosen), unchanged from the JSSP path.

    ``AddCumulative`` note (documented extension point, not implemented this sprint): for
    the capacity-1-per-machine case used throughout this thesis, ``AddNoOverlap`` and
    ``AddCumulative(intervals, demands=[1]*n, capacity=1)`` are equivalent. This function
    uses ``AddNoOverlap`` (simpler, and consistent with the already-validated JSSP path);
    ``AddCumulative`` would be the natural generalisation if a future extension groups
    identical machines into a capacity->1 pooled resource.

    Args:
        instance: The FJSP instance to solve (see :class:`FJSPInstance`, which itself
            guarantees every operation has at least one eligible machine --- see its
            own ``__post_init__`` --- so this function does not re-check that).
        time_limit_s: Wall-clock budget, same default as :func:`solve_cpsat`.
        num_workers: CP-SAT search worker threads.
        log_progress: Stream the solver log to stdout.

    Returns:
        A :class:`CPSATResult`. The chosen machine per operation is not part of the
        reported schedule (``start_times`` is the canonical cross-method representation,
        per ``Baselines_TechSpec.md`` §2.1/§6.2) and is only used internally, to
        self-verify the returned schedule against a routing-aware feasibility check.

    Raises:
        ImportError: If ``ortools`` is not installed.
        AssertionError: If the returned schedule violates a constraint --- a paranoid
            cross-check of the model, mirroring :func:`solve_cpsat`.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "solve_cpsat_fjsp requires OR-Tools. Install it with `pip install ortools`."
        ) from exc

    horizon = sum(
        max(instance.durations[job][pos])
        for job in range(instance.num_jobs)
        for pos in range(instance.ops_per_job)
    )
    model = cp_model.CpModel()

    starts: dict[int, "cp_model.IntVar"] = {}
    ends: dict[int, "cp_model.IntVar"] = {}
    presence: dict[tuple[int, int], "cp_model.IntVar"] = {}
    intervals_by_machine: dict[int, list] = {m: [] for m in range(instance.num_machines)}

    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job):
            op = instance.op_id(job, pos)
            machines = instance.eligible_machines[job][pos]
            durations = instance.durations[job][pos]

            start = model.NewIntVar(0, horizon, f"s_{op}")
            end = model.NewIntVar(0, horizon, f"e_{op}")
            starts[op] = start
            ends[op] = end

            alternatives = []
            for machine, duration in zip(machines, durations):
                lit = model.NewBoolVar(f"x_{op}_{machine}")
                interval = model.NewOptionalIntervalVar(start, duration, end, lit, f"i_{op}_{machine}")
                intervals_by_machine[machine].append(interval)
                presence[(op, machine)] = lit
                alternatives.append(lit)
            model.AddExactlyOne(alternatives)

    # Machine capacity: one operation at a time, over only the alternatives assigned to it.
    for machine, intervals in intervals_by_machine.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # Technological order within each job -- unchanged from the JSSP path.
    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job - 1):
            model.Add(starts[instance.op_id(job, pos + 1)] >= ends[instance.op_id(job, pos)])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(ends.values()))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_progress)
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    start_array: np.ndarray | None = None
    best_makespan: int | None = None
    if has_solution:
        start_array = np.array(
            [solver.Value(starts[op_id]) for op_id in range(instance.num_operations)],
            dtype=np.float64,
        )
        best_makespan = int(solver.Value(makespan))
        machine_assignment = {
            op: machine for (op, machine), lit in presence.items() if solver.Value(lit)
        }
        feasible, violations = is_fjsp_schedule_feasible(instance, start_array, machine_assignment)
        assert feasible, f"CP-SAT returned an infeasible FJSP schedule: {violations[:3]}"

    best_bound = int(solver.best_objective_bound)
    is_optimal = status == cp_model.OPTIMAL
    return CPSATResult(
        instance=instance.name,
        status=status_name,
        makespan=best_makespan,
        best_bound=best_bound,
        is_optimal=is_optimal,
        wall_time=float(solver.wall_time),
        mip_gap=compute_mip_gap(best_makespan, best_bound, is_optimal),
        start_times=start_array,
    )


def is_fjsp_schedule_feasible(
    instance: FJSPInstance,
    start_times: np.ndarray,
    machine_assignment: dict[int, int],
    tolerance: float = 1e-9,
) -> tuple[bool, list[str]]:
    """Routing-aware feasibility check for an :class:`FJSPInstance` schedule.

    ``envs.disjunctive_graph.is_schedule_feasible`` assumes one fixed machine per
    operation (``JSSPInstance.flat_machines``), which does not exist for FJSP until a
    machine has actually been chosen --- this is the FJSP-aware equivalent. Used both as
    :func:`solve_cpsat_fjsp`'s internal self-check and by
    :class:`~baselines.interface.GABaseline`/:class:`~baselines.interface.CPSATFJSPBaseline`
    to populate :class:`~baselines.interface.BaselineResult` uniformly for FJSP instances.

    Returns:
        ``(feasible, violations)``, mirroring :func:`~envs.disjunctive_graph.is_schedule_feasible`.
    """
    violations: list[str] = []
    starts = np.asarray(start_times, dtype=np.float64)

    if (starts < -tolerance).any():
        violations.append("an operation starts at a negative time")

    ends: dict[int, float] = {}
    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job):
            op = instance.op_id(job, pos)
            machine = machine_assignment.get(op)
            if machine is None:
                violations.append(f"operation {op} has no chosen machine")
                continue
            alt_index = instance.eligible_machines[job][pos].index(machine)
            duration = instance.durations[job][pos][alt_index]
            ends[op] = float(starts[op]) + duration

    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job - 1):
            u, v = instance.op_id(job, pos), instance.op_id(job, pos + 1)
            if starts[v] + tolerance < ends[u]:
                violations.append(
                    f"precedence violated in job {job}: op {v} starts {starts[v]:g} "
                    f"before op {u} ends {ends[u]:g}"
                )

    # Zero-duration operations occupy an *empty* interval [s, s) and therefore cannot
    # overlap anything, so they are excluded from the machine-capacity check rather than
    # sorted into a sequence where the ``starts[curr] < ends[prev]`` test would flag a
    # spurious violation.  This matters for Brandimarte instances, whose ragged job
    # lengths ``envs.fjsp_loader`` pads with exactly such operations, and matches
    # CP-SAT's own treatment of zero-size intervals under ``NoOverlap``.
    by_machine: dict[int, list[int]] = {m: [] for m in range(instance.num_machines)}
    for op, machine in machine_assignment.items():
        if ends.get(op, 0.0) - float(starts[op]) <= tolerance:
            continue
        by_machine[machine].append(op)
    for machine, ops in by_machine.items():
        order = sorted(ops, key=lambda o: starts[o])
        for prev, curr in zip(order[:-1], order[1:]):
            if starts[curr] + tolerance < ends[prev]:
                violations.append(
                    f"machine {machine} overlap: op {curr} starts {starts[curr]:g} "
                    f"before op {prev} ends {ends[prev]:g}"
                )

    return not violations, violations


def dispatch_order_from_starts(instance: JSSPInstance, start_times: np.ndarray) -> list[int]:
    """Convert a schedule's start times into a dispatch order for :class:`~envs.jssp_env.JSSPEnv`.

    Replaying the operations of a schedule in non-decreasing start-time order through the
    environment's earliest-feasible-placement transition reproduces that schedule's
    makespan or improves on it, because the environment places each operation no later
    than the source schedule did.  This is what lets an exact CP-SAT solution be turned
    into a *verified optimal action sequence* for the environment tests, instead of
    hand-transcribing one.

    Ties are broken by ``(start_time, job, position)`` --- the ``position`` component
    guarantees the result respects each job's technological order, so every action in
    the returned sequence is legal when replayed.

    Args:
        instance: The instance the schedule belongs to.
        start_times: Start time per global operation id.

    Returns:
        Global operation ids in a legal dispatch order.
    """
    order = sorted(
        range(instance.num_operations),
        key=lambda op: (
            float(start_times[op]),
            instance.job_of(op),
            instance.position_of(op),
        ),
    )
    return order
