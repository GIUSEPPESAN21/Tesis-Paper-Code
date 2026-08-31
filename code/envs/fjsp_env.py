"""Gymnasium environment for the *flexible* job-shop scheduling problem (FJSP).

Relationship to :class:`~envs.jssp_env.JSSPEnv`
-----------------------------------------------
Sibling, not subclass --- the same relationship
:class:`~envs.dynamic_jssp_env.DynamicJSSPEnv` has to ``JSSPEnv``, and for the same
reason ``envs/base_env.py`` gives: the two model the same MDP *family* but differ in the
one thing that shapes every internal array.  Here that difference is the **action
space**.  ``JSSPEnv``'s action is "dispatch operation ``o``"; its machine is fixed by the
instance.  An FJSP operation has an *eligible set* of machines with (possibly different)
processing times, so a complete decision is the pair ``(operation, machine)`` --- the
"machine-selection component the current architecture doesn't have" that
``baselines/interface.py`` and ``GNN_PPO_Architecture_Spec_v2.md`` §0 both name as the
reason FJSP was deferred.  This module supplies it.

**Action** --- ``Discrete(num_operations * num_machines)``, flattened as
``action = op_id * num_machines + machine``.  An action is legal iff ``op_id`` is the
frontier (first unscheduled) operation of its job *and* ``machine`` is in that
operation's eligible set.  The flattened-pair encoding (rather than a
``MultiDiscrete``) is deliberate: ``sb3_contrib``'s ``MaskableCategorical`` masks a
single flat categorical, so one flat action space keeps the *entire* masking mechanism
--- ``env.action_masks()`` auto-discovery included --- byte-identical to the JSSP path.
The policy head that scores these actions
(:class:`~models.policy.PerPairActorHead`) reads one logit per ``(operation, machine)``
pair off the heterogeneous encoder's two node-embedding tensors, so its parameter count
is independent of both ``num_operations`` and ``num_machines``.

**Transition** --- identical list-scheduling rule to ``JSSPEnv``, with the chosen
machine's duration::

    start = max(job_ready_time[j], machine_free_time[m])
    end   = start + duration[o][m]

with one documented exception: a **zero-duration operation occupies no machine
capacity**, so it starts as soon as its job predecessor finishes and does not advance
``machine_free_time``.  This is what makes ``envs.fjsp_loader``'s rectangularity padding
exactly makespan-neutral (see that module's docstring); it is also simply the correct
semantics for an empty interval.

**Reward** --- the same dense telescoping shaping as ``JSSPEnv``,
``r_t = (H(s_t) - H(s_{t+1})) * reward_scale``, with :func:`fjsp_estimated_makespan` as
``H``.  ``H`` relaxes machine capacity and optimistically charges every unscheduled
operation its *cheapest* eligible duration, which keeps the three properties the
shaping argument needs: ``H`` is a lower bound on any completion of the state, is
non-decreasing along an episode (a dispatched operation's realised completion is never
earlier than the optimistic estimate it replaces), and equals the realised makespan at
termination.  So ``sum_t r_t = H(s_0) - C_max`` and, with ``gamma = 1``, maximising
undiscounted return is exactly minimising makespan --- the same structural argument
``TrainingPlan_Week17.md`` §3.1 makes for the JSSP path, unchanged.

**Observation** --- the same heterogeneous ``Dict`` the Week 18 encoder consumes from
``JSSPEnv``, with the same keys and widths (``NUM_NODE_FEATURES``,
``NUM_MACHINE_FEATURES``), so one policy architecture reads both.  The only structural
difference is that ``adj_eligibility`` rows are multi-hot here and one-hot there, and
``action_mask`` is ``num_operations * num_machines`` long rather than
``num_operations``.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.base_env import SchedulingEnv
from envs.fjsp_instance import FJSPInstance
from envs.jssp_env import (
    MACHINE_FEATURE_NAMES,  # noqa: F401  (re-exported for symmetry with jssp_env)
    NUM_MACHINE_FEATURES,
    NUM_NODE_FEATURES,
)


def eligibility_matrices(instance: FJSPInstance) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(num_operations, num_machines)`` views of the ragged eligible-machine data.

    Args:
        instance: The FJSP instance.

    Returns:
        ``(eligibility, durations)``.  ``eligibility[o, m]`` is ``1.0`` iff operation
        ``o`` may run on machine ``m``; ``durations[o, m]`` is its processing time there,
        or ``np.inf`` where ineligible (so ``argmin``/``min`` over a row automatically
        ignores ineligible machines).
    """
    n_ops, n_mach = instance.num_operations, instance.num_machines
    eligibility = np.zeros((n_ops, n_mach), dtype=np.float32)
    durations = np.full((n_ops, n_mach), np.inf, dtype=np.float64)
    for job in range(instance.num_jobs):
        for pos in range(instance.ops_per_job):
            op = instance.op_id(job, pos)
            for machine, duration in zip(
                instance.eligible_machines[job][pos], instance.durations[job][pos]
            ):
                eligibility[op, machine] = 1.0
                durations[op, machine] = float(duration)
    return eligibility, durations


def fjsp_static_lower_bound(instance: FJSPInstance) -> int:
    """Instance-level makespan lower bound for an FJSP instance.

    The FJSP counterpart of :func:`~envs.disjunctive_graph.static_lower_bound`.  Three
    relaxations, each valid on its own; the maximum of the three is returned:

    * **Job bound** --- relax machine capacity entirely.  Every job still runs its chain
      sequentially, and no operation can take less than its cheapest eligible duration,
      so ``C_max >= max_j sum_k min_m p_{j,k,m}``.
    * **Forced-load bound** --- an operation with a single eligible machine *must* run
      there, so each machine's forced load is a lower bound on its own busy time and
      hence on ``C_max``.
    * **Average-load bound** --- total work is at least ``sum_o min_m p_{o,m}`` however
      it is routed, and it is spread over ``num_machines`` machines, so the busiest
      machine carries at least ``ceil(total_min_work / num_machines)``.

    Args:
        instance: The FJSP instance.

    Returns:
        A valid lower bound on the optimal makespan (at least ``1``, so it is safe as a
        reward-scale denominator).
    """
    _, durations = eligibility_matrices(instance)
    min_durations = durations.min(axis=1)

    per_job = min_durations.reshape(instance.num_jobs, instance.ops_per_job)
    job_bound = float(per_job.sum(axis=1).max())

    eligibility, _ = eligibility_matrices(instance)
    single_machine = eligibility.sum(axis=1) == 1.0
    forced_bound = 0.0
    if single_machine.any():
        forced_machine = np.argmin(durations[single_machine], axis=1)
        forced_load = np.bincount(
            forced_machine,
            weights=min_durations[single_machine],
            minlength=instance.num_machines,
        )
        forced_bound = float(forced_load.max())

    average_bound = math.ceil(float(min_durations.sum()) / instance.num_machines)

    return max(1, int(math.ceil(max(job_bound, forced_bound, average_bound))))


def fjsp_estimated_makespan(
    num_jobs: int,
    ops_per_job: int,
    min_durations: np.ndarray,
    job_ready_time: np.ndarray,
    job_next_pos: np.ndarray,
) -> float:
    """State-dependent makespan estimate ``H(s)`` driving the FJSP dense reward.

    Same optimistic job-chain propagation as
    :func:`~envs.disjunctive_graph.estimated_makespan`, with two FJSP adaptations: the
    already-dispatched prefix of a job is summarised by ``job_ready_time`` (its last
    realised completion) rather than re-read per operation, and each still-unscheduled
    operation is charged its *cheapest eligible* duration.

    Args:
        num_jobs: Number of jobs.
        ops_per_job: Operations per job (rectangular; see ``envs.fjsp_loader``).
        min_durations: Cheapest eligible duration per global operation id, shape
            ``(num_jobs * ops_per_job,)``.
        job_ready_time: Completion time of each job's last dispatched operation, shape
            ``(num_jobs,)``.
        job_next_pos: Position of each job's frontier operation, shape ``(num_jobs,)``.

    Returns:
        The estimate ``H(s)``.
    """
    per_job = min_durations.reshape(num_jobs, ops_per_job)
    positions = np.arange(ops_per_job)[None, :]
    remaining = np.where(positions >= job_next_pos[:, None], per_job, 0.0).sum(axis=1)
    return float((job_ready_time + remaining).max())


class FJSPEnv(SchedulingEnv):
    """Flexible job-shop environment with a joint ``(operation, machine)`` action space.

    Args:
        instance: An :class:`~envs.fjsp_instance.FJSPInstance`, or an instance *name*
            such as ``"mk01"`` to be loaded from ``data/fjsp_instances/``.
        reward_scale: Multiplier applied to the raw reward.  ``None`` (default) uses
            ``1 / fjsp_static_lower_bound(instance)`` so returns are size-normalised;
            pass ``1.0`` to get rewards in raw time units.
        include_machine_edges: If ``False``, ``adj_machine`` is returned as an all-zero
            matrix.  Only useful for the "no disjunctive edges" ablation, mirroring
            :class:`~envs.jssp_env.JSSPEnv`'s flag of the same name.

    Attributes:
        instance: The instance being scheduled.
        start_times: Start time of each operation, ``-1`` while unscheduled.
        end_times: Completion time of each operation, ``-1`` while unscheduled.
        machine_of_op: Chosen machine per operation, ``-1`` while unscheduled.  The FJSP
            half of the solution that :class:`~envs.instance.JSSPInstance`-based code
            never needs, and that every FJSP feasibility check does.
        machine_free_time: Time each machine becomes available.
        job_ready_time: Completion time of each job's last dispatched operation.

    Raises:
        FileNotFoundError: If ``instance`` names an instance that is not bundled.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        instance: FJSPInstance | str = "mk01",
        reward_scale: float | None = None,
        include_machine_edges: bool = True,
    ) -> None:
        gym.Env.__init__(self)
        # NOTE: SchedulingEnv._init_common is deliberately *not* called.  It resolves
        # names through envs.loader (JSSP-only) and computes reward_scale from
        # disjunctive_graph.static_lower_bound, which needs JSSPInstance.flat_machines
        # --- neither exists for an FJSPInstance.  The five attributes it would set are
        # set here instead, so every SchedulingEnv property (static_lb, dispatch_order)
        # and every documented method of the shared contract still work unchanged.
        if isinstance(instance, str):
            from envs.fjsp_loader import load_fjsp_by_name  # noqa: PLC0415 (cycle-free)

            instance = load_fjsp_by_name(instance)
        self.instance: FJSPInstance = instance
        self.include_machine_edges = include_machine_edges
        self._static_lb = fjsp_static_lower_bound(self.instance)
        self.reward_scale = (
            float(reward_scale) if reward_scale is not None else 1.0 / self._static_lb
        )

        n_ops = self.instance.num_operations
        n_mach = self.instance.num_machines
        self.action_space = spaces.Discrete(n_ops * n_mach)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0, high=np.inf, shape=(n_ops, NUM_NODE_FEATURES), dtype=np.float32
                ),
                "adj_precedence": spaces.Box(
                    low=0.0, high=1.0, shape=(n_ops, n_ops), dtype=np.float32
                ),
                "adj_machine": spaces.Box(
                    low=0.0, high=1.0, shape=(n_ops, n_ops), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(n_ops * n_mach,), dtype=np.int8
                ),
                "machine_features": spaces.Box(
                    low=0.0, high=np.inf, shape=(n_mach, NUM_MACHINE_FEATURES), dtype=np.float32
                ),
                "adj_eligibility": spaces.Box(
                    low=0.0, high=1.0, shape=(n_ops, n_mach), dtype=np.float32
                ),
            }
        )

        # ---------------------------------------------------- structural quantities
        self._eligibility, self._durations = eligibility_matrices(self.instance)
        self._min_durations = self._durations.min(axis=1)
        self._max_durations = np.where(
            np.isfinite(self._durations), self._durations, -np.inf
        ).max(axis=1)
        self._n_eligible = self._eligibility.sum(axis=1)
        self._max_duration = max(float(self._max_durations.max()), 1.0)
        # Trivial upper bound on any active schedule: run every operation on its most
        # expensive eligible machine, sequentially.  Normalises every time-like feature.
        self._horizon = float(max(self._max_durations.sum(), 1.0))

        self._adj_precedence = np.zeros((n_ops, n_ops), dtype=np.float32)
        for job in range(self.instance.num_jobs):
            base = job * self.instance.ops_per_job
            for pos in range(self.instance.ops_per_job - 1):
                self._adj_precedence[base + pos, base + pos + 1] = 1.0
        # Two operations contend iff their eligible sets intersect.  Static upper
        # envelope; masked down to still-unscheduled pairs per observation.
        self._could_share = (self._eligibility @ self._eligibility.T) > 0.0
        np.fill_diagonal(self._could_share, False)
        self._job_progress = (
            np.tile(np.arange(self.instance.ops_per_job), self.instance.num_jobs).astype(
                np.float64
            )
            / max(self.instance.ops_per_job, 1)
        )
        # Expected pending load contributed to each machine if routing were uniform over
        # an operation's eligible set: (num_operations, num_machines).
        self._expected_load = np.where(
            np.isfinite(self._durations), self._durations, 0.0
        ) / np.maximum(self._n_eligible, 1)[:, None]

        # ------------------------------------------------------------ episode state
        self.scheduled: np.ndarray = np.zeros(n_ops, dtype=bool)
        self.start_times: np.ndarray = np.full(n_ops, -1.0)
        self.end_times: np.ndarray = np.full(n_ops, -1.0)
        self.machine_of_op: np.ndarray = np.full(n_ops, -1, dtype=np.int64)
        self.machine_free_time: np.ndarray = np.zeros(n_mach)
        self.job_ready_time: np.ndarray = np.zeros(self.instance.num_jobs)
        self._job_next_pos: np.ndarray = np.zeros(self.instance.num_jobs, dtype=np.int64)
        self._dispatch_order: list[int] = []
        self._h_estimate: float = 0.0
        self._num_scheduled: int = 0

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset to the empty schedule.

        Args:
            seed: Seeds the environment RNG.  The FJSP transition is deterministic given
                the ``(operation, machine)`` action, so the seed only matters for
                reproducibility of any sampling a caller does through ``self.np_random``.
            options: Optional dict; ``options["instance"]`` swaps in a different instance
                (name or object), mirroring :meth:`~envs.jssp_env.JSSPEnv.reset`.

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)

        if options and "instance" in options:
            self.__init__(  # noqa: PLC2801 - deliberate re-init to rebuild spaces
                instance=options["instance"],
                reward_scale=None,
                include_machine_edges=self.include_machine_edges,
            )

        n_ops = self.instance.num_operations
        self.scheduled = np.zeros(n_ops, dtype=bool)
        self.start_times = np.full(n_ops, -1.0)
        self.end_times = np.full(n_ops, -1.0)
        self.machine_of_op = np.full(n_ops, -1, dtype=np.int64)
        self.machine_free_time = np.zeros(self.instance.num_machines)
        self.job_ready_time = np.zeros(self.instance.num_jobs)
        self._job_next_pos = np.zeros(self.instance.num_jobs, dtype=np.int64)
        self._dispatch_order = []
        self._num_scheduled = 0
        self._h_estimate = self._estimated_makespan()

        return self._observation(), self._info()

    def step(
        self, action: int | np.integer
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Dispatch one ``(operation, machine)`` pair.

        Args:
            action: Flattened pair id ``op_id * num_machines + machine``.  Must be
                currently legal --- see :meth:`action_masks`.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Raises:
            ValueError: If ``action`` is out of range or masked as infeasible.  We fail
                loudly rather than absorbing the action, for the same reason
                :meth:`~envs.jssp_env.JSSPEnv.step` does: silently ignoring an illegal
                action corrupts the on-policy data a PPO update is computed from.
        """
        action = int(action)
        n_actions = self.instance.num_operations * self.instance.num_machines
        if not 0 <= action < n_actions:
            raise ValueError(f"action {action} outside [0, {n_actions})")
        if not self.action_masks()[action]:
            op_id, machine = self.decode_action(action)
            job, pos = self.instance.job_of(op_id), self.instance.position_of(op_id)
            reason = (
                "operation already scheduled"
                if self.scheduled[op_id]
                else "operation not yet ready"
                if int(self._job_next_pos[job]) != pos
                else f"machine {machine} not eligible for this operation"
            )
            raise ValueError(
                f"illegal action {action} (op {op_id} = job {job} position {pos}, "
                f"machine {machine}): {reason}"
            )

        op_id, machine = self.decode_action(action)
        job = self.instance.job_of(op_id)
        duration = float(self._durations[op_id, machine])

        if duration > 0.0:
            start = max(float(self.job_ready_time[job]), float(self.machine_free_time[machine]))
            end = start + duration
            self.machine_free_time[machine] = end
        else:
            # A zero-duration operation occupies an empty interval and therefore no
            # machine capacity: it neither waits for nor advances the machine.
            start = float(self.job_ready_time[job])
            end = start

        self.start_times[op_id] = start
        self.end_times[op_id] = end
        self.machine_of_op[op_id] = machine
        self.scheduled[op_id] = True
        self.job_ready_time[job] = end
        self._job_next_pos[job] += 1
        self._dispatch_order.append(op_id)
        self._num_scheduled += 1

        previous_h = self._h_estimate
        self._h_estimate = self._estimated_makespan()
        reward = (previous_h - self._h_estimate) * self.reward_scale

        terminated = self._num_scheduled == self.instance.num_operations
        info = self._info()
        info["decision_time"] = start
        info["dispatched_op"] = op_id
        info["dispatched_job"] = job
        info["dispatched_machine"] = machine
        if terminated:
            info["makespan"] = self.makespan()
            info["schedule_feasible"], info["violations"] = self.check_feasible()

        return self._observation(), float(reward), terminated, False, info

    def render(self) -> None:
        """Print a compact textual view of the current partial schedule."""
        print(
            f"{self.instance.name}: {self._num_scheduled}/{self.instance.num_operations} "
            f"scheduled, H(s)={self._h_estimate:.1f}, LB={self._static_lb}"
        )
        for machine in range(self.instance.num_machines):
            ops = [op for op in self._dispatch_order if int(self.machine_of_op[op]) == machine]
            seq = " ".join(
                f"J{self.instance.job_of(o)}[{self.start_times[o]:.0f}-{self.end_times[o]:.0f}]"
                for o in ops
            )
            print(f"  M{machine}: {seq}")

    # ------------------------------------------------------------- action encoding

    def decode_action(self, action: int) -> tuple[int, int]:
        """Split a flat action id into ``(op_id, machine)``.

        Args:
            action: Flat action id.

        Returns:
            The ``(operation id, machine index)`` pair it encodes.
        """
        n_mach = self.instance.num_machines
        return int(action) // n_mach, int(action) % n_mach

    def encode_action(self, op_id: int, machine: int) -> int:
        """Inverse of :meth:`decode_action`.

        Args:
            op_id: Global operation id.
            machine: Machine index.

        Returns:
            The flat action id.
        """
        return int(op_id) * self.instance.num_machines + int(machine)

    # ------------------------------------------------------------- public state

    def operation_masks(self) -> np.ndarray:
        """Boolean legality mask over *operations only*, shape ``(num_operations,)``.

        The frontier of the job chains, ignoring machine eligibility.  Exposed
        separately because it is what the node features and several diagnostics want;
        :meth:`action_masks` is the one ``sb3_contrib`` consumes.

        Returns:
            Boolean array of shape ``(num_operations,)``.
        """
        mask = np.zeros(self.instance.num_operations, dtype=bool)
        ops_per_job = self.instance.ops_per_job
        for job in range(self.instance.num_jobs):
            pos = int(self._job_next_pos[job])
            if pos < ops_per_job:
                mask[job * ops_per_job + pos] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """Boolean legality mask over the flattened ``(operation, machine)`` space.

        A pair is legal iff the operation is its job's frontier operation and the
        machine is in that operation's eligible set.

        Returns:
            Boolean array of shape ``(num_operations * num_machines,)``.  Guaranteed to
            contain at least one ``True`` whenever the episode has not terminated, since
            :class:`~envs.fjsp_instance.FJSPInstance` forbids an empty eligible set.
        """
        legal_ops = self.operation_masks()
        return (legal_ops[:, None] & (self._eligibility > 0.0)).reshape(-1)

    def makespan(self) -> float:
        """Makespan of the schedule built so far (``0.0`` on an empty schedule)."""
        if self._num_scheduled == 0:
            return 0.0
        return float(self.end_times[self.scheduled].max())

    def remaining_work_per_op(self) -> np.ndarray:
        """Remaining work from each operation onward in its job, cheapest routing.

        The FJSP reading of :meth:`~envs.jssp_env.JSSPEnv.remaining_work_per_op`: an
        operation's own work is ambiguous until a machine is chosen, so the optimistic
        (cheapest eligible) duration is used, exactly as in :meth:`_estimated_makespan`.

        Returns:
            Float array of shape ``(num_operations,)``.
        """
        per_job = self._min_durations.reshape(self.instance.num_jobs, self.instance.ops_per_job)
        return np.cumsum(per_job[:, ::-1], axis=1)[:, ::-1].ravel().copy()

    def schedule_dataframe_records(self) -> list[dict[str, float]]:
        """Per-operation schedule records, convenient for Gantt plots and CSV export.

        Returns:
            One dict per *scheduled* operation with keys ``op_id``, ``job``,
            ``position``, ``machine``, ``start``, ``end``, ``duration``.
        """
        records: list[dict[str, float]] = []
        for op_id in np.flatnonzero(self.scheduled):
            op_id = int(op_id)
            machine = int(self.machine_of_op[op_id])
            records.append(
                {
                    "op_id": op_id,
                    "job": self.instance.job_of(op_id),
                    "position": self.instance.position_of(op_id),
                    "machine": machine,
                    "start": float(self.start_times[op_id]),
                    "end": float(self.end_times[op_id]),
                    "duration": float(self._durations[op_id, machine]),
                }
            )
        return sorted(records, key=lambda r: (r["machine"], r["start"]))

    def machine_assignment(self) -> dict[int, int]:
        """Chosen machine per *scheduled* operation, in the shape FJSP checkers want.

        Returns:
            Mapping ``op_id -> machine`` covering exactly the scheduled operations, the
            argument :func:`~baselines.cpsat.is_fjsp_schedule_feasible` takes.
        """
        return {int(op): int(self.machine_of_op[op]) for op in np.flatnonzero(self.scheduled)}

    def check_feasible(self) -> tuple[bool, list[str]]:
        """Independently verify the schedule built so far.

        Returns:
            ``(feasible, violations)`` from
            :func:`~baselines.cpsat.is_fjsp_schedule_feasible`, the same checker every
            FJSP :class:`~baselines.interface.BaselineResult` is validated with, so an
            RL rollout and a GA/CP-SAT solution are held to one standard.
        """
        from baselines.cpsat import is_fjsp_schedule_feasible  # noqa: PLC0415 (cycle-free)

        return is_fjsp_schedule_feasible(
            self.instance, self.start_times, self.machine_assignment()
        )

    def current_time(self) -> float:
        """Decision-epoch clock reading; see :meth:`~envs.jssp_env.JSSPEnv.current_time`."""
        legal = self.operation_masks()
        if not legal.any():
            return self.makespan()
        return float(self._earliest_start_estimates()[legal].min())

    # --------------------------------------------------------------- internals

    def _estimated_makespan(self) -> float:
        """``H(s)`` for the current state; see :func:`fjsp_estimated_makespan`."""
        return fjsp_estimated_makespan(
            self.instance.num_jobs,
            self.instance.ops_per_job,
            self._min_durations,
            self.job_ready_time,
            self._job_next_pos,
        )

    def _earliest_start_estimates(self) -> np.ndarray:
        """Earliest-start estimate per operation under the current partial schedule.

        Scheduled operations report their realised start.  An unscheduled operation
        reports ``max(job chain availability, earliest free time among its eligible
        machines)`` propagated forward along its job --- the FJSP reading of
        :meth:`~envs.jssp_env.JSSPEnv._earliest_start_estimates`, where "its machine's
        free time" becomes "the best its eligible set can offer".

        Returns:
            Float array of shape ``(num_operations,)``.
        """
        ops_per_job = self.instance.ops_per_job
        est = np.zeros(self.instance.num_operations, dtype=np.float64)
        eligible_free = np.where(
            self._eligibility > 0.0, self.machine_free_time[None, :], np.inf
        ).min(axis=1)
        for job in range(self.instance.num_jobs):
            running = float(self.job_ready_time[job])
            for pos in range(ops_per_job):
                op_id = job * ops_per_job + pos
                if self.scheduled[op_id]:
                    est[op_id] = self.start_times[op_id]
                    continue
                start = max(running, float(eligible_free[op_id]))
                est[op_id] = start
                running = start + self._min_durations[op_id]
        return est

    def _machine_features(self) -> np.ndarray:
        """Per-machine feature matrix; columns are :data:`MACHINE_FEATURE_NAMES`.

        Identical semantics to :meth:`~envs.jssp_env.JSSPEnv._machine_features`, with
        ``assigned_pending_load`` generalised: an unscheduled FJSP operation is not
        committed to any one machine, so it contributes its duration on machine ``m``
        divided by the size of its eligible set --- the load machine ``m`` would carry
        under uniform routing.  Under fixed routing (``|eligible| == 1``) this reduces
        exactly to the JSSP definition.  ``utilization_accum`` uses the realised schedule
        span as its denominator for the reason given there.

        Returns:
            Float32 array of shape ``(num_machines, NUM_MACHINE_FEATURES)``.
        """
        horizon = self._horizon
        now = self.current_time()
        span = self.makespan()
        unscheduled = ~self.scheduled

        pending_load = self._expected_load[unscheduled].sum(axis=0)
        busy_so_far = np.zeros(self.instance.num_machines, dtype=np.float64)
        for op in np.flatnonzero(self.scheduled):
            machine = int(self.machine_of_op[op])
            busy_so_far[machine] += self.end_times[op] - self.start_times[op]

        time_to_available = np.maximum(self.machine_free_time - now, 0.0)
        is_free = (time_to_available <= 0.0).astype(np.float64)

        features = np.empty((self.instance.num_machines, NUM_MACHINE_FEATURES), dtype=np.float32)
        features[:, 0] = is_free
        features[:, 1] = 1.0 - is_free
        features[:, 2] = 0.0                       # is_broken: static shop
        features[:, 3] = time_to_available / horizon
        features[:, 4] = pending_load / horizon
        features[:, 5] = busy_so_far / span if span > 0.0 else 0.0
        features[:, 6] = 1.0                       # estimated_availability: no breakdowns
        return features

    def _observation(self) -> dict[str, np.ndarray]:
        """Assemble the ``Dict`` observation for the current state."""
        n_ops = self.instance.num_operations
        unscheduled = ~self.scheduled
        horizon = self._horizon
        ops_per_job = self.instance.ops_per_job

        remaining_job_work = np.cumsum(
            (self._min_durations * unscheduled).reshape(self.instance.num_jobs, ops_per_job)[
                :, ::-1
            ],
            axis=1,
        )[:, ::-1].ravel()

        job_complete = np.repeat(
            self._job_next_pos >= ops_per_job, ops_per_job
        ).astype(np.float64)

        # Expected remaining load on the machines this operation could run on.
        pending_machine_load = self._expected_load[unscheduled].sum(axis=0)
        eligible_mean_load = (self._eligibility @ pending_machine_load) / np.maximum(
            self._n_eligible, 1
        )

        features = np.empty((n_ops, NUM_NODE_FEATURES), dtype=np.float32)
        features[:, 0] = self._min_durations / self._max_duration
        features[:, 1] = self.scheduled
        features[:, 2] = self.operation_masks()
        features[:, 3] = job_complete
        features[:, 4] = self._earliest_start_estimates() / horizon
        features[:, 5] = remaining_job_work / horizon
        features[:, 6] = self._job_progress
        features[:, 7] = eligible_mean_load / horizon

        if self.include_machine_edges:
            both_active = unscheduled[:, None] & unscheduled[None, :]
            adj_machine = (self._could_share & both_active).astype(np.float32)
        else:
            adj_machine = np.zeros((n_ops, n_ops), dtype=np.float32)

        return {
            "node_features": features,
            "adj_precedence": self._adj_precedence,
            "adj_machine": adj_machine,
            "action_mask": self.action_masks().astype(np.int8),
            "machine_features": self._machine_features(),
            "adj_eligibility": self._eligibility,
        }

    def _info(self) -> dict[str, Any]:
        """Diagnostics attached to every ``reset``/``step`` return."""
        return {
            "instance": self.instance.name,
            "num_scheduled": self._num_scheduled,
            "num_operations": self.instance.num_operations,
            "makespan_estimate": self._h_estimate,
            "current_makespan": self.makespan(),
            "static_lower_bound": self._static_lb,
            "action_mask": self.action_masks(),
        }
