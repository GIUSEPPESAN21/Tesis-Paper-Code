"""Disjunctive-graph structure and makespan lower bounds for the JSSP.

The disjunctive graph :math:`G = (V, C \\cup D)` of Roy & Sussmann (1964) represents a
JSSP instance as:

* **V** --- one node per operation (this codebase omits the artificial source/sink and
  handles them implicitly, which keeps node ids identical to action ids);
* **C** (conjunctive arcs) --- *directed* precedence arcs :math:`o_{j,k} \\to o_{j,k+1}`
  fixed by each job's technological order;
* **D** (disjunctive arcs) --- *undirected* arcs between every pair of operations that
  share a machine.  Solving the JSSP means orienting all of D acyclically; a scheduling
  agent orients them incrementally, one dispatch at a time.

Two adjacency matrices are exported to the learner:

``adj_precedence``
    Directed job-chain successor matrix, ``A[u, v] = 1`` iff ``v`` immediately follows
    ``u`` in its job.
``adj_machine``
    Symmetric machine-sharing matrix over *still-contending* operations, ``A[u, v] = 1``
    iff ``u != v``, both run on the same machine, and both are still unscheduled.  Once
    an operation is dispatched its disjunctions are resolved, so dropping its machine
    edges makes the observation shrink toward the pure job chain as the episode ends ---
    this is the encoding used by Zhang et al. (2020, "Learning to Dispatch", L2D) and it
    gives the GNN a direct signal about which conflicts remain open.

Lower bounds
------------
Two different bounds live here and they serve different purposes:

:func:`static_lower_bound`
    The classical job/machine bound on the *instance*.  Never changes during an
    episode.  Used to validate parsing (a correct parse must satisfy
    ``LB <= known optimum``) and to normalise features.

:func:`estimated_makespan`
    The *state-dependent* critical-path estimate ``H(s)`` used for the dense reward.
    It propagates realised completion times through the job chains and extends each
    unscheduled operation by its own duration, giving a quantity that is
    non-decreasing along an episode and equals the true makespan at termination.

Reference:
    Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020).
    *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.*
    NeurIPS 33.
"""

from __future__ import annotations

import numpy as np

from envs.events import OpState
from envs.instance import JSSPInstance


def build_precedence_adjacency(instance: JSSPInstance) -> np.ndarray:
    """Directed job-chain adjacency of the conjunctive arcs.

    Args:
        instance: The JSSP instance.

    Returns:
        Float array of shape ``(num_operations, num_operations)`` where entry
        ``[u, v]`` is 1.0 iff ``v`` is the immediate job successor of ``u``.
    """
    n = instance.num_operations
    adj = np.zeros((n, n), dtype=np.float32)
    ops = instance.ops_per_job
    for job in range(instance.num_jobs):
        base = job * ops
        for pos in range(ops - 1):
            adj[base + pos, base + pos + 1] = 1.0
    return adj


def build_machine_adjacency(
    instance: JSSPInstance, unscheduled_mask: np.ndarray | None = None
) -> np.ndarray:
    """Symmetric machine-sharing adjacency over operations that still contend.

    Args:
        instance: The JSSP instance.
        unscheduled_mask: Boolean array of shape ``(num_operations,)``; ``True`` marks
            an operation whose machine assignment is still unresolved.  ``None`` means
            "all operations", i.e. the initial fully-disjunctive graph.

    Returns:
        Float array of shape ``(num_operations, num_operations)``, symmetric with a
        zero diagonal.
    """
    n = instance.num_operations
    flat_machines = instance.flat_machines
    if unscheduled_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(unscheduled_mask, dtype=bool)

    same_machine = flat_machines[:, None] == flat_machines[None, :]
    both_active = active[:, None] & active[None, :]
    adj = (same_machine & both_active).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def static_lower_bound(instance: JSSPInstance) -> int:
    """Classical job-and-machine makespan lower bound for a JSSP instance.

    Combines two relaxations, each valid on its own:

    * **Job bound** --- relax machine capacity entirely.  Every job must run its whole
      chain sequentially, so ``C_max >= max_j sum_k p_{j,k}``.
    * **Machine bound (with heads and tails)** --- relax every machine except one.
      Machine ``m`` cannot start before the smallest head among its operations, must
      then process all of them, and after the last one the corresponding job still owes
      its tail.  So ``C_max >= min_head(m) + load(m) + min_tail(m)`` for every ``m``.

    The maximum of the two is returned.  This is the standard bound used to seed
    branch-and-bound for the JSSP and is always ``<=`` the optimal makespan --- which
    makes it a cheap and surprisingly sharp *parser* validator (see
    ``tests/test_loader.py``).

    Args:
        instance: The JSSP instance.

    Returns:
        A valid lower bound on the optimal makespan.
    """
    job_bound = int(instance.job_totals.max())

    heads = instance.head_times().ravel()
    tails = instance.tail_times().ravel()
    flat_machines = instance.flat_machines

    machine_bound = 0
    for machine in range(instance.num_machines):
        on_machine = flat_machines == machine
        if not on_machine.any():
            continue
        bound = (
            int(heads[on_machine].min())
            + int(instance.machine_totals[machine])
            + int(tails[on_machine].min())
        )
        machine_bound = max(machine_bound, bound)

    return max(job_bound, machine_bound)


def estimated_makespan(
    instance: JSSPInstance,
    completion_times: np.ndarray,
    scheduled: np.ndarray,
) -> float:
    """State-dependent makespan estimate ``H(s)`` driving the dense reward.

    For each job, walk its chain left to right.  A *scheduled* operation contributes its
    realised completion time.  An *unscheduled* operation is optimistically assumed to
    start the instant its predecessor finishes (machine contention ignored), so its
    estimated completion is ``prev_completion + duration``.  ``H(s)`` is the maximum
    estimated completion over all operations.

    Properties that make this a sound reward shaping potential:

    * ``H(s)`` is a valid lower bound on the makespan of any completion of ``s``.
    * ``H(s)`` is non-decreasing as operations get dispatched, so the reward
      ``r_t = H(s_t) - H(s_{t+1})`` is non-positive.
    * At termination ``H(s_T)`` equals the realised makespan, hence
      ``sum_t r_t = H(s_0) - C_max``.  With ``gamma = 1`` maximising return is exactly
      minimising makespan, up to the state-independent constant ``H(s_0)``.

    Args:
        instance: The JSSP instance.
        completion_times: Float array of shape ``(num_operations,)``; realised
            completion time for scheduled operations (ignored elsewhere).
        scheduled: Boolean array of shape ``(num_operations,)``.

    Returns:
        The estimate ``H(s)`` as a float.
    """
    durations = instance.durations
    ops = instance.ops_per_job
    completion = np.asarray(completion_times, dtype=np.float64).reshape(instance.num_jobs, ops)
    is_scheduled = np.asarray(scheduled, dtype=bool).reshape(instance.num_jobs, ops)

    estimate = np.zeros((instance.num_jobs, ops), dtype=np.float64)
    running = np.zeros(instance.num_jobs, dtype=np.float64)
    for pos in range(ops):
        optimistic = running + durations[:, pos]
        running = np.where(is_scheduled[:, pos], completion[:, pos], optimistic)
        estimate[:, pos] = running

    return float(estimate.max())


def is_schedule_feasible(
    instance: JSSPInstance,
    start_times: np.ndarray,
    tolerance: float = 1e-9,
) -> tuple[bool, list[str]]:
    """Verify a complete schedule against both JSSP constraint families.

    Checks, for the given start times:

    1. **Precedence** --- ``start(o_{j,k+1}) >= start(o_{j,k}) + p_{j,k}``.
    2. **Machine capacity** --- operations sharing a machine have disjoint
       ``[start, start + p)`` intervals.
    3. **Non-negativity** --- no operation starts before time zero.

    Args:
        instance: The JSSP instance.
        start_times: Float array of shape ``(num_operations,)`` indexed by global op id.
        tolerance: Numerical slack allowed before a violation is reported.

    Returns:
        ``(feasible, violations)`` where ``violations`` is a list of human-readable
        messages, empty iff ``feasible`` is ``True``.
    """
    starts = np.asarray(start_times, dtype=np.float64).reshape(-1)
    if starts.shape[0] != instance.num_operations:
        return False, [
            f"expected {instance.num_operations} start times, got {starts.shape[0]}"
        ]

    durations = instance.flat_durations.astype(np.float64)
    ends = starts + durations
    violations: list[str] = []

    if (starts < -tolerance).any():
        bad = int(np.argmin(starts))
        violations.append(f"operation {bad} starts at negative time {starts[bad]}")

    ops = instance.ops_per_job
    for job in range(instance.num_jobs):
        for pos in range(ops - 1):
            u, v = job * ops + pos, job * ops + pos + 1
            if starts[v] + tolerance < ends[u]:
                violations.append(
                    f"precedence violated in job {job}: op {v} starts {starts[v]:g} "
                    f"before op {u} ends {ends[u]:g}"
                )

    flat_machines = instance.flat_machines
    for machine in range(instance.num_machines):
        on_machine = np.flatnonzero(flat_machines == machine)
        order = on_machine[np.argsort(starts[on_machine], kind="stable")]
        for prev, curr in zip(order[:-1], order[1:]):
            if starts[curr] + tolerance < ends[prev]:
                violations.append(
                    f"machine {machine} overlap: op {curr} starts {starts[curr]:g} "
                    f"before op {prev} ends {ends[prev]:g}"
                )

    return not violations, violations


# ------------------------------------------------------------ dynamic extension (NEW)
#
# The two functions below are additive-only, per ``DynamicEnv_TechSpec.md`` §6.4 and §9.4:
# ``estimated_makespan`` and ``is_schedule_feasible`` above are unchanged, still operate on
# a fixed :class:`JSSPInstance`, and are exactly what the static ``JSSPEnv`` continues to
# use. :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv` cannot reuse them directly because a
# dynamically arrived job never belongs to the (frozen) base ``JSSPInstance`` and because
# neither function has any notion of a machine breakdown.


def estimated_makespan_dynamic(
    ops_per_job: int,
    num_active_jobs: int,
    flat_machines: np.ndarray,
    base_durations: np.ndarray,
    effective_durations: np.ndarray,
    op_state: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    remaining_duration: np.ndarray,
    machine_repair_complete_time: np.ndarray,
    machine_broken: np.ndarray,
) -> float:
    """Disruption-aware state-dependent makespan estimate ``H(s)`` for the dynamic env.

    Same job-chain optimistic-propagation principle as :func:`estimated_makespan`: walk
    each (already-arrived) job's chain left to right, accumulating an optimistic
    completion estimate. Four operation states are handled per position:

    * ``DONE`` --- contributes its realised completion time (``end_times``).
    * ``IN_PROCESS`` --- contributes its already-known projected completion,
      ``start_times + effective_durations`` (the duration was fixed the instant it was
      dispatched, see :class:`~envs.disruptions.ProcessingTimeSampler`).
    * ``PREEMPTED`` --- contributes ``machine_repair_complete_time[machine] +
      remaining_duration``, i.e. the machine's already-known repair completion time plus
      however much of the operation is left to run. This is exact, not a statistical
      expectation: the repair's TTR was already sampled at breakdown time.
    * ``UNSCHEDULED`` --- contributes ``max(running, broken_floor) + base_duration``,
      i.e. the static formula's optimistic "starts the instant its predecessor finishes"
      extended with a floor: if the operation's machine is currently broken, it cannot
      start before that machine's known repair completion, regardless of job-chain
      readiness. A machine that is merely busy (not broken) imposes no floor, exactly
      like the static estimate, which ignores ordinary machine contention throughout.

    ``H(s)`` is the maximum estimate over all active operations.

    Args:
        ops_per_job: Operations per job (constant across the base instance and every
            dynamically arrived job --- see ``ArrivalProcess``'s content model).
        num_active_jobs: Number of jobs instantiated so far (base instance + arrivals).
        flat_machines: Machine per operation, length >= ``num_active_jobs * ops_per_job``.
        base_durations: Nominal (pre-multiplier) duration per operation, same length.
        effective_durations: Realised duration once dispatched (undefined/unused before
            dispatch), same length.
        op_state: :class:`~envs.events.OpState` value per operation, same length.
        start_times: Realised dispatch start per operation (``-1`` if never dispatched).
        end_times: Realised completion per operation (``-1`` if not yet ``DONE``).
        remaining_duration: Remaining processing time for a ``PREEMPTED`` operation
            (``-1`` if never preempted).
        machine_repair_complete_time: Absolute time each machine's current repair (if
            any) completes; meaningless for a machine that is not currently broken.
        machine_broken: Boolean, ``True`` iff the machine is currently ``BROKEN``.

    Returns:
        The estimate ``H(s)`` as a float (``0.0`` if there are no active operations).
    """
    n_active = num_active_jobs * ops_per_job
    if n_active == 0:
        return 0.0

    shape = (num_active_jobs, ops_per_job)
    machines = flat_machines[:n_active].reshape(shape)
    base_dur = base_durations[:n_active].reshape(shape)
    eff_dur = effective_durations[:n_active].reshape(shape)
    state = op_state[:n_active].reshape(shape)
    starts = start_times[:n_active].reshape(shape)
    ends = end_times[:n_active].reshape(shape)
    remaining = remaining_duration[:n_active].reshape(shape)

    broken_floor_by_machine = np.where(
        machine_broken, machine_repair_complete_time, -np.inf
    )

    estimate = np.zeros(shape, dtype=np.float64)
    running = np.zeros(num_active_jobs, dtype=np.float64)
    for pos in range(ops_per_job):
        st = state[:, pos]
        m = machines[:, pos]

        is_done = st == OpState.DONE
        is_in_process = st == OpState.IN_PROCESS
        is_preempted = st == OpState.PREEMPTED

        done_val = ends[:, pos]
        in_process_val = starts[:, pos] + eff_dur[:, pos]
        preempted_val = machine_repair_complete_time[m] + remaining[:, pos]

        floor = broken_floor_by_machine[m]
        unscheduled_val = np.where(
            np.isfinite(floor),
            np.maximum(running, floor) + base_dur[:, pos],
            running + base_dur[:, pos],
        )

        running = np.select(
            [is_done, is_in_process, is_preempted],
            [done_val, in_process_val, preempted_val],
            default=unscheduled_val,
        )
        estimate[:, pos] = running

    return float(estimate.max())


def is_schedule_feasible_dynamic(
    ops_per_job: int,
    num_active_jobs: int,
    start_times: np.ndarray,
    end_times: np.ndarray,
    machine_busy_intervals: list[tuple[int, float, float]],
    machine_broken_intervals: list[tuple[int, float, float]],
    tolerance: float = 1e-9,
) -> tuple[bool, list[str]]:
    """Disruption-aware feasibility checker for a dynamic-environment rollout.

    :func:`is_schedule_feasible` has no notion of a broken interval and assumes every
    operation occupies a single contiguous ``[start, start + duration)`` window on its
    machine --- both assumptions break under preempt-resume, where one operation can
    occupy two disjoint segments on its machine (before and after the breakdown). This
    function checks, instead, against the environment's own record of what actually ran
    when:

    1. **Precedence** --- for every completed pair of consecutive operations in a job,
       the successor's start is not before the predecessor's realised completion.
    2. **Machine capacity** --- no two busy segments on the same machine overlap.
    3. **No work during a breakdown** --- no busy segment overlaps a broken interval on
       that machine.

    Args:
        ops_per_job: Operations per job (constant, as in
            :func:`estimated_makespan_dynamic`).
        num_active_jobs: Number of jobs instantiated so far.
        start_times: Realised dispatch start per operation (``-1`` if never dispatched).
        end_times: Realised completion per operation (``-1`` if not yet ``DONE``).
        machine_busy_intervals: One ``(machine, segment_start, segment_end)`` tuple per
            contiguous span of real machine occupancy (a non-preempted operation
            contributes one; a preempt-resumed operation contributes two).
        machine_broken_intervals: One ``(machine, breakdown_time, repair_complete_time)``
            tuple per breakdown that occurred.
        tolerance: Numerical slack allowed before a violation is reported.

    Returns:
        ``(feasible, violations)``, mirroring :func:`is_schedule_feasible`.
    """
    n_active = num_active_jobs * ops_per_job
    starts = np.asarray(start_times, dtype=np.float64)[:n_active]
    ends = np.asarray(end_times, dtype=np.float64)[:n_active]
    violations: list[str] = []

    dispatched = starts > -1 + tolerance
    if (starts[dispatched] < -tolerance).any():
        violations.append("an operation starts at a negative time")

    for job in range(num_active_jobs):
        for pos in range(ops_per_job - 1):
            u, v = job * ops_per_job + pos, job * ops_per_job + pos + 1
            if ends[u] <= -1 + tolerance or starts[v] <= -1 + tolerance:
                continue  # predecessor not yet done, or successor not yet dispatched
            if starts[v] + tolerance < ends[u]:
                violations.append(
                    f"precedence violated in job {job}: op {v} starts {starts[v]:g} "
                    f"before op {u} ends {ends[u]:g}"
                )

    segments_by_machine: dict[int, list[tuple[float, float]]] = {}
    for machine, seg_start, seg_end in machine_busy_intervals:
        segments_by_machine.setdefault(int(machine), []).append((float(seg_start), float(seg_end)))

    for machine, segments in segments_by_machine.items():
        segments.sort()
        for (s0, e0), (s1, e1) in zip(segments, segments[1:]):
            if s1 + tolerance < e0:
                violations.append(
                    f"machine {machine} overlap: segment starting {s1:g} begins before "
                    f"prior segment ends {e0:g}"
                )

    broken_by_machine: dict[int, list[tuple[float, float]]] = {}
    for machine, b_start, b_end in machine_broken_intervals:
        broken_by_machine.setdefault(int(machine), []).append((float(b_start), float(b_end)))

    for machine, segments in segments_by_machine.items():
        for b_start, b_end in broken_by_machine.get(machine, []):
            for s0, e0 in segments:
                overlap = min(e0, b_end) - max(s0, b_start)
                if overlap > tolerance:
                    violations.append(
                        f"machine {machine}: segment [{s0:g}, {e0:g}) overlaps its "
                        f"broken interval [{b_start:g}, {b_end:g})"
                    )

    return not violations, violations
