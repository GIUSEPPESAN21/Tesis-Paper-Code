"""Gymnasium environment for the static job-shop scheduling problem.

MDP formulation
---------------
The environment casts JSSP construction as a sequential decision process over the
disjunctive graph (see :mod:`envs.disjunctive_graph`).  One episode builds one complete
schedule in exactly ``num_operations`` steps.

**State** --- the partially-oriented disjunctive graph, exposed as a ``Dict`` observation:

============================ =================================== ==========================
key                          shape                               meaning
============================ =================================== ==========================
``node_features``            ``(num_operations, NUM_NODE_FEATURES)``  per-operation features
``adj_precedence``           ``(num_operations, num_operations)``     directed job-chain arcs
``adj_machine``              ``(num_operations, num_operations)``     unresolved disjunctions
``action_mask``              ``(num_operations,)``                    1 = legal dispatch
``machine_features``         ``(num_machines, NUM_MACHINE_FEATURES)`` per-machine features
``adj_eligibility``          ``(num_operations, num_machines)``       operation <-> machine
============================ =================================== ==========================

The last two keys are additive (Week 18, ``GNN_PPO_Architecture_Spec_v2.md`` §1.2): they
turn the observation into the *heterogeneous* graph the Week 18 encoder consumes
(operation nodes + machine nodes), without changing or removing anything the Week 1
homogeneous path, the ``Baseline`` adapters, or ``run_evaluation.py`` already read.  Under
``JSSPEnv`` (fixed routing) every ``adj_eligibility`` row is one-hot; the flexible variant
:class:`~envs.fjsp_env.FJSPEnv` emits the same key multi-hot.

**Action** --- ``Discrete(num_operations)``: dispatch operation ``a``.  An action is legal
iff ``a`` is unscheduled *and* all of its job predecessors are already scheduled, i.e.
``a`` is the frontier operation of an unfinished job.  Because every job has at most one
frontier operation, this is in bijection with the "pick a job" action space used by L2D;
the per-operation formulation is used here because it lets the policy head read a logit
straight off each GNN node embedding, which keeps the network size-agnostic.

**Transition** --- dispatching ``a`` (job ``j``, machine ``m``, duration ``p``) appends it
to machine ``m``'s sequence at the earliest feasible time::

    start = max(job_ready_time[j], machine_free_time[m])
    end   = start + p

This *list-scheduling* transition never inserts an operation into an earlier machine idle
gap, so the reachable set is exactly the semi-active schedules obtainable by dispatching
in some order.  That set contains an optimal schedule for every JSSP instance (dispatch
the operations of an optimal schedule in non-decreasing start-time order and this rule
reproduces it or improves on it), so **optimality is not lost** --- see
``tests/test_env.py::test_optimal_sequence_reaches_known_optimum_ft06``.

**Decision epochs** --- there is no global simulation clock to advance.  Each step *is* a
decision epoch, and its simulated time is the start time assigned to the dispatched
operation, reported as ``info["decision_time"]``.  This event-driven formulation is
standard for static JSSP and is what the dynamic extension (Week 6 of the thesis plan:
stochastic breakdowns and mid-episode job arrivals) will replace with a real event queue.

**Reward** --- dense, ``r_t = H(s_t) - H(s_{t+1})``, the negative increment of the
critical-path makespan estimate :func:`~envs.disjunctive_graph.estimated_makespan`.
Rewards are non-positive and telescope to ``H(s_0) - C_max``, so with ``gamma = 1``
maximising undiscounted return is equivalent to minimising makespan.  Set
``reward_scale`` to normalise magnitudes across instance sizes (defaults to
``1 / static_lower_bound``, which puts returns on a comparable scale for ft06 and la40).

**Termination** --- ``terminated`` once all operations are scheduled.  ``truncated`` is
always ``False``: the horizon is deterministic and finite, so there is nothing to
truncate.  The key is kept for API compliance.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from envs.base_env import SchedulingEnv
from envs.disjunctive_graph import (
    build_machine_adjacency,
    build_precedence_adjacency,
    estimated_makespan,
    is_schedule_feasible,
)
from envs.instance import JSSPInstance

#: Number of per-operation features in ``observation["node_features"]``.
NUM_NODE_FEATURES = 8

#: Human-readable names of the node features, in column order.  Kept next to the
#: constant so the thesis's methodology chapter and the code cannot drift apart.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "processing_time",          # p_o / max_p                      -- static
    "is_scheduled",             # 1 if already dispatched
    "is_ready",                 # 1 if legal to dispatch right now
    "is_job_complete",          # 1 if the owning job is fully scheduled
    "earliest_start_estimate",  # est_o / horizon
    "remaining_job_work",       # unscheduled work in this job from o onward / horizon
    "job_progress",             # position / ops_per_job           -- static
    "machine_remaining_load",   # unscheduled work on o's machine / horizon
)

#: Number of per-machine features in ``observation["machine_features"]`` (Week 18,
#: ``GNN_PPO_Architecture_Spec_v2.md`` §1.2).  Additive: does **not** touch
#: :data:`NUM_NODE_FEATURES`, which still describes operation nodes only.
NUM_MACHINE_FEATURES = 7

#: Human-readable names of the machine features, in column order.
MACHINE_FEATURE_NAMES: tuple[str, ...] = (
    "is_free",                 # status one-hot 1/3: free at the current decision epoch
    "is_busy",                 # status one-hot 2/3: occupied past the current epoch
    "is_broken",               # status one-hot 3/3: always 0.0 here (static shop); the
                               #   slot exists so DynamicJSSPEnv can populate it later
                               #   without changing NUM_MACHINE_FEATURES
    "time_to_available",       # max(0, free_time - t) / horizon
    "assigned_pending_load",   # unscheduled work still routed to this machine / horizon
    "utilization_accum",       # busy time so far / realised schedule span
                               #   (0.0 by convention on the empty schedule)
    "estimated_availability",  # empirical A_k; 1.0 here (no breakdowns in a static shop)
)


class JSSPEnv(SchedulingEnv):
    """Disjunctive-graph JSSP environment with action masking and a dense reward.

    Args:
        instance: A :class:`~envs.instance.JSSPInstance`, or an instance *name* such as
            ``"ft06"`` to be loaded from ``data/instances/``.
        reward_scale: Multiplier applied to the raw reward.  ``None`` (default) uses
            ``1 / static_lower_bound(instance)`` so returns are size-normalised; pass
            ``1.0`` to get rewards in raw time units.
        include_machine_edges: If ``False``, ``adj_machine`` is returned as an all-zero
            matrix.  Only useful for the "no disjunctive edges" ablation.
        reward_mode: ``"dense"`` (default) is the ``H(s_t) - H(s_{t+1})`` shaping reward
            documented above. ``"sparse"`` (``AblationPlan.md`` Axis 2) is ``0`` at every
            non-terminal step and ``-makespan * reward_scale`` at the terminal step only
            -- the direct empirical test of the credit-assignment argument for why dense
            shaping was chosen; both telescope to the same total up to the constant
            ``H(s_0) * reward_scale`` dense's shaping adds and sparse does not.

    Attributes:
        instance: The instance being scheduled.
        start_times: Start time of each operation, ``-1`` while unscheduled.
        end_times: Completion time of each operation, ``-1`` while unscheduled.
        machine_free_time: Time each machine becomes available.
        job_ready_time: Completion time of each job's last dispatched operation.

    Raises:
        FileNotFoundError: If ``instance`` names an instance that is not bundled.
    """

    def __init__(
        self,
        instance: JSSPInstance | str = "ft06",
        reward_scale: float | None = None,
        include_machine_edges: bool = True,
        reward_mode: Literal["dense", "sparse"] = "dense",
    ) -> None:
        super().__init__()
        if reward_mode not in ("dense", "sparse"):
            raise ValueError(f"reward_mode must be 'dense' or 'sparse', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self._init_common(instance, reward_scale, include_machine_edges)

        n_ops = self.instance.num_operations
        self.action_space = spaces.Discrete(n_ops)
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
                "action_mask": spaces.Box(low=0, high=1, shape=(n_ops,), dtype=np.int8),
                "machine_features": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.instance.num_machines, NUM_MACHINE_FEATURES),
                    dtype=np.float32,
                ),
                "adj_eligibility": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_ops, self.instance.num_machines),
                    dtype=np.float32,
                ),
            }
        )

        # Structural quantities that never change during an episode.
        self._adj_precedence = build_precedence_adjacency(self.instance)
        self._flat_durations = self.instance.flat_durations.astype(np.float64)
        self._flat_machines = self.instance.flat_machines
        self._remaining_work = self.instance.remaining_work().ravel().astype(np.float64)
        self._job_progress = (
            np.tile(np.arange(self.instance.ops_per_job), self.instance.num_jobs).astype(np.float64)
            / max(self.instance.ops_per_job, 1)
        )
        # (num_machines, num_operations) selector used to recompute machine loads fast.
        self._machine_membership = (
            self._flat_machines[None, :] == np.arange(self.instance.num_machines)[:, None]
        )
        # (num_operations, num_machines) operation<->machine incidence.  Structural under
        # fixed routing (one-hot per row) so it is built once, not per observation.
        self._adj_eligibility = self._machine_membership.T.astype(np.float32)

        # Episode state; populated by reset().
        self.scheduled: np.ndarray = np.zeros(n_ops, dtype=bool)
        self.start_times: np.ndarray = np.full(n_ops, -1.0)
        self.end_times: np.ndarray = np.full(n_ops, -1.0)
        self.machine_free_time: np.ndarray = np.zeros(self.instance.num_machines)
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
            seed: Seeds the environment RNG.  The static JSSP transition is fully
                deterministic, so the seed only matters for reproducibility of any
                sampling a caller does through ``self.np_random``; it is honoured for
                API compliance and forward-compatibility with the dynamic variant.
            options: Optional dict; ``options["instance"]`` swaps in a different
                instance (name or object).  Changing instance size rebuilds the spaces,
                so a vectorised trainer must not do this mid-batch.

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)

        if options and "instance" in options:
            new_instance = options["instance"]
            self.__init__(  # noqa: PLC2801 - deliberate re-init to rebuild spaces
                instance=new_instance,
                reward_scale=None,
                include_machine_edges=self.include_machine_edges,
                reward_mode=self.reward_mode,
            )

        n_ops = self.instance.num_operations
        self.scheduled = np.zeros(n_ops, dtype=bool)
        self.start_times = np.full(n_ops, -1.0)
        self.end_times = np.full(n_ops, -1.0)
        self.machine_free_time = np.zeros(self.instance.num_machines)
        self.job_ready_time = np.zeros(self.instance.num_jobs)
        self._job_next_pos = np.zeros(self.instance.num_jobs, dtype=np.int64)
        self._dispatch_order = []
        self._num_scheduled = 0
        self._h_estimate = estimated_makespan(self.instance, self.end_times, self.scheduled)

        return self._observation(), self._info()

    def step(
        self, action: int | np.integer
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Dispatch one operation.

        Args:
            action: Global operation id to dispatch.  Must be currently legal --- see
                :meth:`action_masks`.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Raises:
            ValueError: If ``action`` is out of range or masked as infeasible.  We fail
                loudly rather than absorbing the action: silently ignoring illegal
                actions corrupts the on-policy data a PPO update is computed from.
        """
        action = int(action)
        if not 0 <= action < self.instance.num_operations:
            raise ValueError(
                f"action {action} outside [0, {self.instance.num_operations})"
            )
        if not self.action_masks()[action]:
            job, pos = self.instance.job_of(action), self.instance.position_of(action)
            raise ValueError(
                f"illegal action {action} (job {job}, position {pos}): operation is "
                f"{'already scheduled' if self.scheduled[action] else 'not yet ready'}"
            )

        job = self.instance.job_of(action)
        machine = int(self._flat_machines[action])
        duration = float(self._flat_durations[action])

        start = max(float(self.job_ready_time[job]), float(self.machine_free_time[machine]))
        end = start + duration

        self.start_times[action] = start
        self.end_times[action] = end
        self.scheduled[action] = True
        self.machine_free_time[machine] = end
        self.job_ready_time[job] = end
        self._job_next_pos[job] += 1
        self._dispatch_order.append(action)
        self._num_scheduled += 1

        previous_h = self._h_estimate
        self._h_estimate = estimated_makespan(self.instance, self.end_times, self.scheduled)
        terminated = self._num_scheduled == self.instance.num_operations

        if self.reward_mode == "sparse":
            reward = -self.makespan() * self.reward_scale if terminated else 0.0
        else:
            reward = (previous_h - self._h_estimate) * self.reward_scale

        info = self._info()
        info["decision_time"] = start
        info["dispatched_op"] = action
        info["dispatched_job"] = job
        info["dispatched_machine"] = machine
        if terminated:
            info["makespan"] = self.makespan()
            info["schedule_feasible"], info["violations"] = is_schedule_feasible(
                self.instance, self.start_times
            )

        return self._observation(), float(reward), terminated, False, info

    def render(self) -> None:
        """Print a compact textual view of the current partial schedule."""
        print(f"{self.instance.name}: {self._num_scheduled}/{self.instance.num_operations} "
              f"scheduled, H(s)={self._h_estimate:.1f}, LB={self._static_lb}")
        for machine in range(self.instance.num_machines):
            ops = [
                op for op in self._dispatch_order if int(self._flat_machines[op]) == machine
            ]
            seq = " ".join(
                f"J{self.instance.job_of(o)}[{self.start_times[o]:.0f}-{self.end_times[o]:.0f}]"
                for o in ops
            )
            print(f"  M{machine}: {seq}")

    # ------------------------------------------------------------- public state

    def action_masks(self) -> np.ndarray:
        """Boolean legality mask over the action space.

        An operation is legal iff it is unscheduled and is the frontier (first
        unscheduled) operation of its job.

        Returns:
            Boolean array of shape ``(num_operations,)``.  Guaranteed to contain at
            least one ``True`` whenever the episode has not terminated.
        """
        mask = np.zeros(self.instance.num_operations, dtype=bool)
        ops_per_job = self.instance.ops_per_job
        for job in range(self.instance.num_jobs):
            pos = int(self._job_next_pos[job])
            if pos < ops_per_job:
                mask[job * ops_per_job + pos] = True
        return mask

    def makespan(self) -> float:
        """Makespan of the schedule built so far (``0.0`` on an empty schedule)."""
        if self._num_scheduled == 0:
            return 0.0
        return float(self.end_times[self.scheduled].max())

    def remaining_work_per_op(self) -> np.ndarray:
        """Remaining work of every operation, indexed by global op id.

        Pure delegation to :meth:`~envs.instance.JSSPInstance.remaining_work`; exists so
        that baseline dispatching rules (``baselines/dispatching_rules.py``) can query
        remaining work through the environment instead of indexing the base instance
        directly, which lets the same rule code run unmodified against
        :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv` (see that class's own override).

        Returns:
            Float array of shape ``(num_operations,)``.
        """
        return self.instance.remaining_work().ravel()

    def duration_per_op(self) -> np.ndarray:
        """Processing time of every operation, indexed by global op id.

        Pure delegation to :attr:`~envs.instance.JSSPInstance.flat_durations` -- exists
        for the same reason as :meth:`remaining_work_per_op`: so ``spt_rule``/
        ``lpt_rule`` (T8.1, Recomendacion del tutor sobre cobertura de PDR bajo
        llegadas) can query durations through the environment instead of indexing the
        base instance directly, letting the same rule code run unmodified against
        :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv` (see that class's own override,
        which additionally covers dynamically arrived jobs).

        Returns:
            Float array of shape ``(num_operations,)``.
        """
        return self.instance.flat_durations

    def machine_per_op(self) -> np.ndarray:
        """Machine index of every operation, indexed by global op id.

        Pure delegation to :attr:`~envs.instance.JSSPInstance.flat_machines`. See
        :meth:`duration_per_op`'s docstring for why this exists.

        Returns:
            Integer array of shape ``(num_operations,)``.
        """
        return self.instance.flat_machines

    def schedule_dataframe_records(self) -> list[dict[str, float]]:
        """Per-operation schedule records, convenient for Gantt plots and CSV export.

        Returns:
            One dict per *scheduled* operation with keys ``op_id``, ``job``,
            ``position``, ``machine``, ``start``, ``end``, ``duration``.
        """
        records: list[dict[str, float]] = []
        for op_id in np.flatnonzero(self.scheduled):
            op_id = int(op_id)
            records.append(
                {
                    "op_id": op_id,
                    "job": self.instance.job_of(op_id),
                    "position": self.instance.position_of(op_id),
                    "machine": int(self._flat_machines[op_id]),
                    "start": float(self.start_times[op_id]),
                    "end": float(self.end_times[op_id]),
                    "duration": float(self._flat_durations[op_id]),
                }
            )
        return sorted(records, key=lambda r: (r["machine"], r["start"]))

    # --------------------------------------------------------------- internals

    def _earliest_start_estimates(self) -> np.ndarray:
        """Earliest-start estimate for every operation under the current partial schedule.

        Scheduled operations report their realised start.  An unscheduled operation
        reports ``max(job chain availability, machine availability)`` propagated
        forward along its job, which is the same optimistic relaxation the reward's
        ``H(s)`` uses, tightened with the machine's current free time.

        Returns:
            Float array of shape ``(num_operations,)``.
        """
        ops_per_job = self.instance.ops_per_job
        est = np.zeros(self.instance.num_operations, dtype=np.float64)
        durations = self._flat_durations
        for job in range(self.instance.num_jobs):
            running = float(self.job_ready_time[job])
            for pos in range(ops_per_job):
                op_id = job * ops_per_job + pos
                if self.scheduled[op_id]:
                    est[op_id] = self.start_times[op_id]
                    continue
                machine = int(self._flat_machines[op_id])
                start = max(running, float(self.machine_free_time[machine]))
                est[op_id] = start
                running = start + durations[op_id]
        return est

    def current_time(self) -> float:
        """Decision-epoch clock reading used to normalise the machine features.

        The static list-scheduling MDP has no global simulation clock (see this module's
        docstring): each step *is* a decision epoch, and its simulated time is the start
        time the transition assigns.  The machine features nevertheless need *some*
        reference "now" to say whether a machine is free or how long until it frees up,
        so this method reports the earliest time at which anything can happen next ---
        the minimum earliest-start estimate over the currently legal (frontier)
        operations, which is exactly the time the *next* dispatch would be assigned if
        the policy picked the earliest-available frontier operation.

        On the empty schedule this is ``0.0``.  Once every operation is scheduled there
        is no frontier left, and the clock reports the realised makespan instead.

        Returns:
            The current decision-epoch time.
        """
        legal = self.action_masks()
        if not legal.any():
            return self.makespan()
        return float(self._earliest_start_estimates()[legal].min())

    def _machine_features(self) -> np.ndarray:
        """Per-machine feature matrix, shape ``(num_machines, NUM_MACHINE_FEATURES)``.

        Column order and semantics are :data:`MACHINE_FEATURE_NAMES`.  Every quantity is
        normalised the same way the operation features are (by ``self._horizon``, the
        instance's total processing time) so both node types enter the encoder on a
        comparable scale.

        ``is_broken`` is identically ``0.0`` and ``estimated_availability`` identically
        ``1.0``: this environment models a static shop with no breakdowns.  The columns
        exist so a policy trained here has the right input width to be loaded, unchanged,
        into the dynamic environment once that wiring lands.

        ``utilization_accum`` divides busy time by the **realised schedule span**
        (:meth:`makespan`), not by :meth:`current_time`.  Under list scheduling an
        operation is placed at the earliest feasible moment on its machine, which can be
        *later* than the next decision epoch, so a machine's committed busy time routinely
        exceeds ``current_time()`` and the ratio would exceed 1 --- meaningless as a
        utilisation, and a badly scaled input to the encoder.  The schedule span bounds
        every committed interval by construction, so the ratio stays in ``[0, 1]``.

        Returns:
            Float32 array of shape ``(num_machines, NUM_MACHINE_FEATURES)``.
        """
        num_machines = self.instance.num_machines
        horizon = self._horizon
        now = self.current_time()
        span = self.makespan()

        unscheduled = ~self.scheduled
        pending_load = self._machine_membership @ (self._flat_durations * unscheduled)
        busy_so_far = self._machine_membership @ (self._flat_durations * self.scheduled)

        time_to_available = np.maximum(self.machine_free_time - now, 0.0)
        is_free = (time_to_available <= 0.0).astype(np.float64)

        features = np.empty((num_machines, NUM_MACHINE_FEATURES), dtype=np.float32)
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

        # Remaining unscheduled work per machine, broadcast to that machine's operations.
        unscheduled_work = self._flat_durations * unscheduled
        machine_remaining = self._machine_membership @ unscheduled_work
        machine_remaining_per_op = machine_remaining[self._flat_machines]

        # Remaining unscheduled work in the job from each operation onward.  The static
        # cumulative tail minus the (already scheduled) part of that tail.
        remaining_job_work = np.empty(n_ops, dtype=np.float64)
        ops_per_job = self.instance.ops_per_job
        tail_unscheduled = (self._flat_durations * unscheduled).reshape(
            self.instance.num_jobs, ops_per_job
        )
        tail_cumulative = np.cumsum(tail_unscheduled[:, ::-1], axis=1)[:, ::-1]
        remaining_job_work[:] = tail_cumulative.ravel()

        job_complete = np.repeat(
            self._job_next_pos >= ops_per_job, ops_per_job
        ).astype(np.float64)

        features = np.empty((n_ops, NUM_NODE_FEATURES), dtype=np.float32)
        features[:, 0] = self._flat_durations / max(self.instance.max_duration, 1)
        features[:, 1] = self.scheduled
        features[:, 2] = self.action_masks()
        features[:, 3] = job_complete
        features[:, 4] = self._earliest_start_estimates() / horizon
        features[:, 5] = remaining_job_work / horizon
        features[:, 6] = self._job_progress
        features[:, 7] = machine_remaining_per_op / horizon

        if self.include_machine_edges:
            adj_machine = build_machine_adjacency(self.instance, unscheduled)
        else:
            adj_machine = np.zeros((n_ops, n_ops), dtype=np.float32)

        return {
            "node_features": features,
            "adj_precedence": self._adj_precedence,
            "adj_machine": adj_machine,
            "action_mask": self.action_masks().astype(np.int8),
            "machine_features": self._machine_features(),
            "adj_eligibility": self._adj_eligibility,
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
