"""Fase 6 (Recomendacion 9 del tutor), T6.1: estrategias de reprogramacion mas
fuertes para los metodos fuera de linea (CP-SAT, GA), como CAPA NUEVA -- no como
modificacion de ``dynamic_jssp_env.py`` (Regla explicita del tutor).

Hoy CP-SAT y GA se ejecutan solo con reparacion por desplazamiento a la derecha
(``run_evaluation.run_predictive_reactive``: resolver una vez fuera de linea, ejecutar
el plan, caer a una PDR de respaldo para lo no anticipado). Eso favorece
estructuralmente a las PDR, que reaccionan en cada evento, y deja abierto si la
perdida de los metodos fuera de linea viene de la optimalidad del solver o del
compromiso con una secuencia fija. Este modulo anade dos politicas de ejecucion
adicionales que SI reoptimizan durante el episodio:

- :func:`periodic_reopt` -- reoptimiza el subproblema residual cada ``period``
  unidades de tiempo simulado.
- :func:`event_triggered_reopt` -- reoptimiza al ocurrir una averia, una llegada, o
  cuando la desviacion acumulada respecto al plan supera un umbral.

Ambas resuelven el subproblema residual (lo que aun no se ha despachado, con las
operaciones en curso fijas) con el MISMO solucionador -- CP-SAT o una busqueda local
de reprogramacion (T6.1 nota: un sustituto simplificado de un GA poblacional completo,
documentado como tal en ``MEJORAS_TUTOR_IMPLEMENTADAS.md`` -- suficiente para ejercitar
la capa de politicas de ejecucion que es el entregable central de esta fase) -- con
presupuesto reducido, y respetan las operaciones ya en curso (nunca las reasignan).

**Hallazgo de la primera ejecucion (2026-08-21), no un defecto de este modulo:**
un plan residual optimo, reproducido via el mismo tipo de reejecucion en orden de
inicio que ``run_evaluation.run_predictive_reactive``/``baselines.cpsat.dispatch_order_from_starts``
ya usan con exito para ``JSSPEnv`` (estatico), NO reproduce exactamente el optimo
al reejecutarse contra ``DynamicJSSPEnv`` -- en ft06, sin ninguna disrupcion, un
unico reoptimize upfront realiza 60 en vez de 55 (~9%); con reoptimizaciones
frecuentes (period=5) baja a 59, pero no cierra la brecha del todo. Esto NO es un
bug de ``rescheduling.py`` -- es la MISMA discrepancia de legalidad
"completion-gated vs. dispatch-gated" que ``E7_E8_FinalPlan.md`` Sec 2.3 ya
cuantifica para el adaptador de la politica RL ("a few percent of makespan on
ft06/la01 with disruptions off"): ``DynamicJSSPEnv.action_masks()`` exige que la
maquina de una operacion este libre EN ESE INSTANTE para que la accion sea legal
(dispatch-gated), no solo que su predecesora de precedencia ya haya terminado
(completion-gated, el supuesto bajo el cual la garantia "reproduce el optimo o
mejora" de la tecnica de reejecucion fue probada para ``JSSPEnv``). Reportar esto
como un hallazgo explicito en cualquier tabla que compare ``right_shift`` contra
``periodic``/``event_triggered`` (T6.3), no como ruido -- es evidencia adicional,
medida independientemente, de que la brecha de legalidad dispatch-gated tiene un
costo real incluso sin disrupcion.

No se ejecuta nada mas de este modulo en esta sesion salvo la verificacion anterior
(Regla Cero para el resto del trabajo de esta sesion).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from envs.events import MachineState, OpState
from envs.instance import JSSPInstance


@dataclass(frozen=True)
class ResidualSubproblem:
    """A snapshot of a live episode's not-yet-fully-realised state, ready to hand to a
    residual solver.

    Attributes:
        unscheduled_ops: Global op ids not yet dispatched (``OpState.UNSCHEDULED``).
        fixed_intervals: ``{op_id: (start, end)}`` for operations already DONE or
            IN_PROCESS -- these NEVER move (T6.1's "no reasigna operaciones ya en
            curso" requirement).
        machine_free_time: Per machine, the earliest time a NEW (non-fixed) interval
            may start on it.
        job_ready_time: Per job, the earliest time its next UNSCHEDULED operation may
            start (its predecessor's fixed end time, or 0 if it is the job's first op).
        current_time: Simulated time this snapshot was taken at.
    """

    unscheduled_ops: tuple[int, ...]
    fixed_intervals: dict[int, tuple[float, float]]
    machine_free_time: np.ndarray
    job_ready_time: np.ndarray
    current_time: float


def snapshot_residual_subproblem(env: DynamicJSSPEnv) -> ResidualSubproblem:
    """Read a :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`'s current state into a
    :class:`ResidualSubproblem`, using only its already-public arrays (``op_state``,
    ``start_times``, ``end_times``, ``effective_duration``, ``machine_free_time``,
    ``job_ready_time``) -- read-only, does not mutate ``env``."""
    n_active = env.active_ops
    op_state = env.op_state[:n_active]
    unscheduled = tuple(int(op) for op in np.flatnonzero(op_state == OpState.UNSCHEDULED))

    fixed: dict[int, tuple[float, float]] = {}
    for state in (OpState.DONE, OpState.IN_PROCESS, OpState.PREEMPTED):
        for op in np.flatnonzero(op_state == state):
            op = int(op)
            start = float(env.start_times[op])
            if state == OpState.IN_PROCESS:
                end = start + float(env.effective_duration[op])
            elif state == OpState.PREEMPTED:
                # Not currently occupying a machine (see envs/dynamic_jssp_env.py's
                # preemption handling) -- excluded from fixed_intervals (it will be
                # re-dispatched like any unscheduled op once its machine repairs);
                # only DONE/IN_PROCESS ops are truly immovable.
                continue
            else:
                end = float(env.end_times[op])
            fixed[op] = (start, end)

    return ResidualSubproblem(
        unscheduled_ops=unscheduled,
        fixed_intervals=fixed,
        machine_free_time=env.machine_free_time.copy(),
        job_ready_time=env.job_ready_time[: env.num_jobs_arrived].copy(),
        current_time=float(env._t),
    )


def _precedence_pairs(env: DynamicJSSPEnv, ops: tuple[int, ...]) -> list[tuple[int, int]]:
    """``(pred, succ)`` pairs among ``ops`` implied by job chains, using the SAME
    adjacency builder ``envs/disjunctive_graph.py`` already provides for the static
    case, applied over the live env's growable arrays (not reimplemented)."""
    ops_per_job = env.ops_per_job
    pairs = []
    op_set = set(ops)
    for op in ops:
        job = int(env._job_index[op])
        pos = int(env._position_index[op])
        if pos + 1 < ops_per_job:
            succ = job * ops_per_job + pos + 1
            if succ in op_set:
                pairs.append((op, succ))
    return pairs


def _select_next_op(candidates: list[int], planned_starts: dict[int, float], env: DynamicJSSPEnv) -> int:
    """Pick the legal op to dispatch next, honoring the plan's intended order.

    Bug found by execution (2026-08-21, not caught by Regla-Cero-era review): sorting
    candidates by ``planned_starts[op]`` alone, with no tiebreak, let Python's ``min()``
    fall back to an effectively arbitrary tiebreak (set/dict iteration order) whenever
    two candidates shared the same planned start time -- common at t=0 when several
    operations are all scheduled to begin immediately on different machines. That
    silently permuted the machine sequencing relative to the residual solver's actual
    optimal plan, producing a worse realized makespan (60 instead of the true 55 on a
    disruption-free ft06 residual solve). Fixed by using the SAME tiebreak
    ``baselines.cpsat.dispatch_order_from_starts`` already uses and already relies on
    elsewhere in this codebase (``run_evaluation.run_predictive_reactive``):
    ``(start_time, job, position)`` -- the ``position`` component guarantees the
    result still respects each job's technological order.
    """
    def key(op: int) -> tuple[float, int, int]:
        return (planned_starts[op], int(env._job_index[op]), int(env._position_index[op]))

    return min(candidates, key=key)


def solve_residual_cpsat(
    env: DynamicJSSPEnv, subproblem: ResidualSubproblem, *, time_limit_s: float = 10.0, num_workers: int = 4
) -> dict[int, float]:
    """Residual CP-SAT solve: fixed intervals for DONE/IN_PROCESS ops (never
    reassigned), free intervals (lower-bounded by ready time) for UNSCHEDULED ops,
    minimise makespan. Self-contained model (not a call into
    ``baselines.cpsat.solve_cpsat``, which has no notion of partially-fixed
    intervals) -- same modelling pattern (one ``IntervalVar`` per operation,
    ``AddNoOverlap`` per machine, job-chain precedence), generalised to mixed
    fixed/free intervals.

    Args:
        env: The live episode (read-only: machine/job indices, not mutated).
        subproblem: From :func:`snapshot_residual_subproblem`.
        time_limit_s: CP-SAT wall-clock budget for the RESIDUAL problem -- deliberately
            smaller than a from-scratch solve's budget (T6.2 tracks this as the extra
            computational cost re-optimisation is not free).

    Returns:
        ``{op_id: start_time}`` for every operation in ``subproblem.unscheduled_ops``
        (fixed ops are not included -- callers already know their times).

    Raises:
        ImportError: If OR-Tools is not installed.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("solve_residual_cpsat requires OR-Tools.") from exc

    ops = subproblem.unscheduled_ops
    if not ops:
        return {}

    horizon = int(subproblem.current_time) + int(env._flat_durations[list(ops)].sum()) + 1
    horizon += int(max(subproblem.machine_free_time, default=0.0))

    model = cp_model.CpModel()
    starts: dict[int, "cp_model.IntVar"] = {}
    ends: dict[int, "cp_model.IntVar"] = {}
    intervals_by_machine: dict[int, list] = {}

    for op in ops:
        job = int(env._job_index[op])
        machine = int(env._flat_machines[op])
        duration = max(int(round(float(env._flat_durations[op]))), 0)
        ready = int(np.ceil(subproblem.job_ready_time[job])) if job < len(subproblem.job_ready_time) else 0
        lb = max(ready, 0)
        start = model.NewIntVar(lb, horizon, f"s_{op}")
        end = model.NewIntVar(lb, horizon, f"e_{op}")
        model.Add(end == start + duration)
        starts[op] = start
        ends[op] = end
        intervals_by_machine.setdefault(machine, []).append(
            model.NewIntervalVar(start, duration, end, f"i_{op}")
        )

    # Machine capacity: free intervals must not overlap each other, and must not start
    # before that machine's already-fixed occupancy clears.
    for machine, intervals in intervals_by_machine.items():
        if intervals:
            model.AddNoOverlap(intervals)
        free_at = int(np.ceil(subproblem.machine_free_time[machine])) if machine < len(subproblem.machine_free_time) else 0
        for op in ops:
            if int(env._flat_machines[op]) == machine:
                model.Add(starts[op] >= free_at)

    # Job-chain precedence among the unscheduled ops themselves.
    for pred, succ in _precedence_pairs(env, ops):
        model.Add(starts[succ] >= ends[pred])

    makespan = model.NewIntVar(0, horizon, "makespan")
    all_ends = list(ends.values())
    fixed_ends = [int(np.ceil(end)) for _, end in subproblem.fixed_intervals.values()]
    model.AddMaxEquality(makespan, all_ends + [model.NewConstant(e) for e in fixed_ends] if fixed_ends else all_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"solve_residual_cpsat: no feasible solution (status={solver.StatusName(status)})")

    return {op: float(solver.Value(starts[op])) for op in ops}


def solve_residual_priority_search(
    env: DynamicJSSPEnv, subproblem: ResidualSubproblem, *, budget_s: float = 5.0, rng: np.random.Generator | None = None
) -> dict[int, float]:
    """Residual reoptimization via randomized-restart list scheduling -- a simplified
    stand-in for a full population-based GA (documented limitation,
    ``MEJORAS_TUTOR_IMPLEMENTADAS.md``): repeatedly samples a random job-priority
    order, decodes it into a feasible list-scheduling assignment respecting
    ``subproblem``'s ready times and fixed machine occupancy, and keeps the
    lowest-makespan result found within ``budget_s``. Never reassigns a fixed
    (DONE/IN_PROCESS) operation -- those are not part of the search space at all.

    Args:
        budget_s: Wall-clock search budget (T6.2 tracks this as GA's residual cost,
            same convention as :func:`solve_residual_cpsat`'s ``time_limit_s``).
        rng: Optional generator for reproducibility; a fresh ``default_rng()`` if omitted.

    Returns:
        ``{op_id: start_time}`` for every operation in ``subproblem.unscheduled_ops``.
    """
    ops = subproblem.unscheduled_ops
    if not ops:
        return {}
    rng = rng if rng is not None else np.random.default_rng()

    jobs = sorted({int(env._job_index[op]) for op in ops})
    ops_by_job = {job: sorted(op for op in ops if int(env._job_index[op]) == job) for job in jobs}

    def decode(job_priority: list[int]) -> tuple[dict[int, float], float]:
        machine_free = {
            m: float(subproblem.machine_free_time[m]) if m < len(subproblem.machine_free_time) else 0.0
            for m in range(int(env.num_machines))
        }
        job_free = {
            j: float(subproblem.job_ready_time[j]) if j < len(subproblem.job_ready_time) else 0.0
            for j in jobs
        }
        remaining = {j: list(ops_by_job[j]) for j in jobs}
        starts: dict[int, float] = {}
        pending = set(ops)
        while pending:
            for job in job_priority:
                queue = remaining[job]
                if not queue:
                    continue
                op = queue[0]
                machine = int(env._flat_machines[op])
                duration = float(env._flat_durations[op])
                start = max(machine_free.get(machine, 0.0), job_free[job])
                starts[op] = start
                end = start + duration
                machine_free[machine] = end
                job_free[job] = end
                queue.pop(0)
                pending.discard(op)
        makespan = max(starts[op] + float(env._flat_durations[op]) for op in ops)
        return starts, makespan

    best_starts: dict[int, float] | None = None
    best_makespan = float("inf")
    t0 = time.perf_counter()
    # At least one iteration always runs, even if budget_s is tiny/zero -- a residual
    # subproblem must always return SOME feasible schedule.
    first = True
    while first or (time.perf_counter() - t0) < budget_s:
        first = False
        priority = list(jobs)
        rng.shuffle(priority)
        starts, makespan = decode(priority)
        if makespan < best_makespan:
            best_makespan = makespan
            best_starts = starts

    assert best_starts is not None
    return best_starts


ResidualSolver = Callable[[DynamicJSSPEnv, ResidualSubproblem], dict[int, float]]


def _make_solver(solver: Literal["cpsat", "ga"], *, budget_s: float) -> ResidualSolver:
    if solver == "cpsat":
        return lambda env, sub: solve_residual_cpsat(env, sub, time_limit_s=budget_s)
    if solver == "ga":
        return lambda env, sub: solve_residual_priority_search(env, sub, budget_s=budget_s)
    raise ValueError(f"unknown solver {solver!r}. Known: 'cpsat', 'ga'")


@dataclass(frozen=True)
class RescheduleResult:
    """Outcome of one execution-policy rollout, mirroring
    :class:`~baselines.interface.BaselineResult`'s shape closely enough to build a
    CSV row the same way ``run_evaluation.py`` already does, plus the reoptimization
    accounting T6.2 needs."""

    makespan: float
    start_times: np.ndarray
    feasible: bool
    violations: list[str]
    n_reoptimizations: int
    reoptimization_wall_clock_s: float


def _run_with_reopt(
    instance: JSSPInstance,
    disruption_config: DisruptionConfig,
    seed: int,
    *,
    solver: Literal["cpsat", "ga"],
    budget_s: float,
    should_reopt: Callable[[DynamicJSSPEnv, float, float], bool],
) -> RescheduleResult:
    """Shared execution loop for both policies: roll a plan out against a live
    :class:`DynamicJSSPEnv`, re-solving the residual subproblem whenever
    ``should_reopt(env, last_reopt_time, current_time)`` is ``True``.

    ``should_reopt`` is queried once per decision epoch (every ``env.step()``), never
    mid-operation -- so a reoptimization can only ever change UNSCHEDULED operations'
    order, never an operation already IN_PROCESS (T6.1's core safety requirement).
    """
    solve_fn = _make_solver(solver, budget_s=budget_s)
    env = DynamicJSSPEnv(instance, disruption_config)
    env.reset(seed=seed)

    n_reopt = 0
    reopt_seconds = 0.0
    last_reopt_time = -float("inf")

    def reoptimize() -> dict[int, float]:
        nonlocal n_reopt, reopt_seconds, last_reopt_time
        sub = snapshot_residual_subproblem(env)
        t0 = time.perf_counter()
        plan = solve_fn(env, sub)
        reopt_seconds += time.perf_counter() - t0
        n_reopt += 1
        last_reopt_time = float(env._t)
        return plan

    planned_starts = reoptimize()
    terminated = False
    info: dict = {}
    while not terminated:
        if should_reopt(env, last_reopt_time, float(env._t)):
            planned_starts = reoptimize()

        legal = set(np.flatnonzero(env.action_masks()).tolist())
        candidates = [op for op in legal if op in planned_starts]
        if candidates:
            action = _select_next_op(candidates, planned_starts, env)
        else:
            # Nothing in the current plan covers a legal op (e.g. a job arrived after
            # the last reopt) -- reoptimize immediately rather than guessing.
            planned_starts = reoptimize()
            candidates = [op for op in legal if op in planned_starts]
            action = _select_next_op(candidates, planned_starts, env) if candidates else min(legal)

        _, _, terminated, _, info = env.step(action)

    return RescheduleResult(
        makespan=float(info["makespan"]),
        start_times=env.start_times.copy(),
        feasible=bool(info["schedule_feasible"]),
        violations=list(info["violations"]),
        n_reoptimizations=n_reopt,
        reoptimization_wall_clock_s=reopt_seconds,
    )


def periodic_reopt(
    instance: JSSPInstance,
    disruption_config: DisruptionConfig,
    seed: int,
    *,
    period: float,
    solver: Literal["cpsat", "ga"] = "cpsat",
    budget_s: float = 10.0,
) -> RescheduleResult:
    """T6.1: reoptimize the residual subproblem every ``period`` units of simulated time.

    Args:
        period: Simulated-time interval between reoptimizations. Must be > 0.

    Raises:
        ValueError: If ``period <= 0``.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    def should_reopt(env: DynamicJSSPEnv, last_reopt_time: float, current_time: float) -> bool:
        return (current_time - last_reopt_time) >= period

    return _run_with_reopt(instance, disruption_config, seed, solver=solver, budget_s=budget_s, should_reopt=should_reopt)


@dataclass(frozen=True)
class EventTrigger:
    """Which events force a reoptimization, plus the accumulated-deviation threshold.

    Attributes:
        on_breakdown: Reoptimize whenever a machine breakdown has occurred since the
            last reoptimization.
        on_arrival: Reoptimize whenever a new job has arrived since the last
            reoptimization.
        deviation_threshold: Reoptimize when the sum of |actual_start - planned_start|
            over already-dispatched-since-last-reopt operations exceeds this. ``None``
            disables the deviation trigger.
    """

    on_breakdown: bool = True
    on_arrival: bool = True
    deviation_threshold: float | None = None


def event_triggered_reopt(
    instance: JSSPInstance,
    disruption_config: DisruptionConfig,
    seed: int,
    *,
    trigger: EventTrigger,
    solver: Literal["cpsat", "ga"] = "cpsat",
    budget_s: float = 10.0,
) -> RescheduleResult:
    """T6.1: reoptimize on a breakdown, an arrival, or accumulated planned-start
    deviation exceeding ``trigger.deviation_threshold`` -- whichever ``trigger``
    enables.

    Breakdown/arrival detection compares ``env.num_jobs_arrived``/a machine-repair
    counter against their values at the last reoptimization (both already tracked by
    ``DynamicJSSPEnv``, read here, not recomputed). Deviation uses
    ``info["planned_start_deviation"]`` (the same signal the kappa*Delta_inest reward
    term already reads, per ``envs/dynamic_jssp_env.py``) accumulated since the last
    reoptimization.
    """
    solve_fn = _make_solver(solver, budget_s=budget_s)
    env = DynamicJSSPEnv(instance, disruption_config)
    env.reset(seed=seed)

    n_reopt = 0
    reopt_seconds = 0.0
    last_reopt_time = -float("inf")
    last_num_jobs_arrived = env.num_jobs_arrived
    last_num_broken = int((env.machine_state == MachineState.BROKEN).sum())
    accumulated_deviation = 0.0

    def reoptimize() -> dict[int, float]:
        nonlocal n_reopt, reopt_seconds, last_reopt_time, last_num_jobs_arrived, last_num_broken, accumulated_deviation
        sub = snapshot_residual_subproblem(env)
        t0 = time.perf_counter()
        plan = solve_fn(env, sub)
        reopt_seconds += time.perf_counter() - t0
        n_reopt += 1
        last_reopt_time = float(env._t)
        last_num_jobs_arrived = env.num_jobs_arrived
        last_num_broken = int((env.machine_state == MachineState.BROKEN).sum())
        accumulated_deviation = 0.0
        return plan

    planned_starts = reoptimize()
    terminated = False
    info: dict = {}
    while not terminated:
        fires = False
        if trigger.on_arrival and env.num_jobs_arrived > last_num_jobs_arrived:
            fires = True
        if trigger.on_breakdown and int((env.machine_state == MachineState.BROKEN).sum()) > last_num_broken:
            fires = True
        if trigger.deviation_threshold is not None and accumulated_deviation > trigger.deviation_threshold:
            fires = True
        if fires:
            planned_starts = reoptimize()

        legal = set(np.flatnonzero(env.action_masks()).tolist())
        candidates = [op for op in legal if op in planned_starts]
        if candidates:
            action = _select_next_op(candidates, planned_starts, env)
        else:
            planned_starts = reoptimize()
            candidates = [op for op in legal if op in planned_starts]
            action = _select_next_op(candidates, planned_starts, env) if candidates else min(legal)

        _, _, terminated, _, info = env.step(action)
        if trigger.deviation_threshold is not None and "planned_start_deviation" in info:
            accumulated_deviation += abs(float(info["planned_start_deviation"]))

    return RescheduleResult(
        makespan=float(info["makespan"]),
        start_times=env.start_times.copy(),
        feasible=bool(info["schedule_feasible"]),
        violations=list(info["violations"]),
        n_reoptimizations=n_reopt,
        reoptimization_wall_clock_s=reopt_seconds,
    )
