"""Dynamic/stochastic extension of the static JSSP environment.

:class:`DynamicJSSPEnv` extends :class:`~envs.jssp_env.JSSPEnv` --- composing the same
disjunctive-graph observation and dense-reward design --- with a real discrete-event
simulation core (:mod:`envs.events`) so that machine breakdowns, stochastic processing
times, and mid-episode job arrivals (:mod:`envs.disruptions`) can be interleaved on one
shared simulated clock. ``envs/jssp_env.py`` is untouched; with
``DisruptionConfig()`` (every axis at its "off" default) this class is numerically
identical to ``JSSPEnv`` --- see
``tests/test_dynamic_env.py::test_default_config_matches_static_env_exactly``.

Why a real event loop, not the static list-scheduling transition
------------------------------------------------------------------
The static environment has no simulated clock: every ``step()`` is an instantaneous
decision, and an operation's start time is computed analytically as
``max(job_ready_time[job], machine_free_time[machine])``. That is exactly right when
nothing happens *between* dispatches. Once machines can break down and jobs can arrive on
their own schedule, asynchronous events must be able to interleave with dispatch
decisions, so ``DynamicJSSPEnv`` runs a real future-event list (FEL) and only *asks the
agent for an action* at a genuine decision epoch --- see :meth:`_has_decision_epoch`.
Between two decision epochs at the same instant (e.g. two different jobs' frontier
operations both idle on two different free machines) nothing is drained from the FEL at
all, which is what makes the "all disruptions off" case reduce exactly to the static
env's algorithm (proved by construction, not just asserted --- see the regression test).

Two spec clarifications made explicit here (flagged, in the same spirit as
``DynamicEnv_TechSpec.md`` §10)
------------------------------------------------------------------------------------------
1. **Action mask.** §6.2 of the spec lists "unscheduled AND frontier AND not preempted AND
   machine not broken" as the mask condition. Read literally, that would let two
   different jobs' frontier operations requiring the *same currently-busy* machine both
   show as legal, which would double-book that machine the instant either was dispatched.
   §3.5's decision-epoch definition *does* require the machine to be free, so this
   implementation adds "machine is FREE (not busy, not broken)" as the mask's machine
   clause; the "not broken" example was one specific instance of the general "not
   currently occupied" requirement, not the whole of it.
2. **``baselines/dispatching_rules.py`` (frozen, unchanged this sprint).** Three of its
   six rules (`spt_rule`, `lpt_rule`, `mwkr_rule`) index the *base* ``JSSPInstance``'s
   fixed-shape arrays (``instance.flat_durations``, ``instance.remaining_work()``)
   directly by operation id. Those arrays cannot cover an operation id belonging to a
   dynamically arrived job, so those three rules are only safe to run against
   ``DynamicJSSPEnv`` when arrivals are disabled. ``tests/test_dynamic_env.py`` exercises
   all six rules with breakdowns and stochastic processing times enabled (arrivals off);
   dynamic arrivals are validated separately by dedicated tests that don't route through
   the frozen baseline module. See ``VALIDATION_WEEK6.md`` for the full note.
3. **What "byte-identical to the static env" can and can't mean.** §3.2 (E2) requires a
   job's next operation to enter the ready set only when its predecessor's
   ``OP_COMPLETE`` actually fires --- i.e. at *realised completion* time. The static env
   advances the same frontier pointer at *dispatch* time instead (its whole point is to
   decouple decision order from real time; see ``JSSPEnv``'s docstring). These are
   different, and correctly so: real completion-gating is exactly what makes preemption
   correct under a breakdown. The observable consequence, confirmed empirically and
   reproduced by ``test_default_config_matches_static_env_exactly``, is that with every
   disruption axis off:

   * ``reward``, ``terminated``, ``info["makespan"]``, ``adj_precedence``, and every node
     feature that depends only on "has this operation been *dispatched* yet" (columns
     ``processing_time``, ``is_scheduled``, ``remaining_job_work``, ``job_progress``,
     ``machine_remaining_load``) match the static env exactly, step for step, for *any*
     dispatch sequence replayed through both envs.
   * ``is_ready`` / the action mask, ``is_job_complete``, and ``earliest_start_estimate``
     --- the three features that depend on job-frontier *completion* rather than
     dispatch --- can legitimately differ mid-episode (the dynamic env lags the static
     env by exactly the in-flight processing time of whatever is currently running).
     They converge to an exact match again at the final step, once every operation is
     simultaneously dispatched and done in both envs, and the reward telescoping (§H(s)
     is a function of "dispatched or not", never of "done or not") is what guarantees the
     return and makespan are unaffected by the difference in between.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from envs.base_env import SchedulingEnv
from envs.disjunctive_graph import (
    build_precedence_adjacency,
    estimated_makespan_dynamic,
    is_schedule_feasible_dynamic,
)
from envs.disruptions import (
    ArrivalProcess,
    BreakdownGenerator,
    DisruptionConfig,
    ProcessingTimeSampler,
    spawn_disruption_rngs,
)
from envs.events import EventKind, Event, FutureEventList, MachineState, OpState
from envs.instance import JSSPInstance
from envs.jssp_env import NODE_FEATURE_NAMES, NUM_NODE_FEATURES

#: Node features: the static 8 (see ``envs.jssp_env.NODE_FEATURE_NAMES``) plus
#: ``is_preempted``.
NUM_NODE_FEATURES_DYNAMIC = NUM_NODE_FEATURES + 1
NODE_FEATURE_NAMES_DYNAMIC: tuple[str, ...] = NODE_FEATURE_NAMES + ("is_preempted",)

#: Per-machine feature block width. Matches ``envs.jssp_env.NUM_MACHINE_FEATURES``
#: column-for-column (see ``MACHINE_FEATURE_NAMES``) -- the static env's own docstring
#: states this width was chosen precisely so a policy trained there could be loaded
#: unchanged into this environment. ``utilization_accum`` (added alongside this
#: comment) was the one column missing from the original Week 6 implementation.
NUM_MACHINE_FEATURES = 7
MACHINE_FEATURE_NAMES: tuple[str, ...] = (
    "is_free",
    "is_busy",
    "is_broken",
    "time_to_available",
    "assigned_pending_load",
    "utilization_accum",
    "estimated_availability",
)


class DynamicJSSPEnv(SchedulingEnv):
    """Event-driven JSSP environment with breakdowns, stochastic durations, and arrivals.

    Shares :class:`~envs.base_env.SchedulingEnv` with :class:`~envs.jssp_env.JSSPEnv`
    (instance resolution, ``reward_scale``/``static_lb``/``dispatch_order``) rather than
    subclassing ``JSSPEnv`` itself: this env's state is growable and event-driven, not a
    fixed-size list-scheduling array, so there is nothing left of ``JSSPEnv``'s own
    ``__init__``/``reset``/``step``/``_observation`` to reuse once those genuinely
    different representations are accounted for.

    Args:
        instance: A :class:`~envs.instance.JSSPInstance`, or an instance name such as
            ``"ft06"``. This is the *seed* instance; dynamically arriving jobs are
            generated on top of it (see :class:`~envs.disruptions.ArrivalProcess`).
        disruption_config: A :class:`~envs.disruptions.DisruptionConfig`. Defaults to
            ``DisruptionConfig()`` (every disruption axis off), the regression contract.
        reward_scale: As in :class:`~envs.jssp_env.JSSPEnv`; defaults to
            ``1 / static_lower_bound(instance)`` computed on the *base* instance only
            (see ``DynamicEnv_TechSpec.md`` §7 --- a "dynamic lower bound" that accounts
            for arrivals/breakdowns before they occur is not well-defined).
        include_machine_edges: As in :class:`~envs.jssp_env.JSSPEnv`.
        kappa: Disruption-aware reward weight (``Step 4 Thesis.md`` Part D3,
            ``PreRegistration_FINAL.md`` §5). The per-step reward is
            ``(previous_h - h_estimate) * reward_scale - kappa * planned_start_deviation
            * reward_scale`` --- ``reward_scale`` is applied to the ``Delta_inest`` term
            too, per ``E7_E8_FinalPlan.md`` §2.1's scale-matching decision, so that a
            given kappa means the same thing in magnitude across instances of different
            size. Defaults to ``0.0``, which reproduces the dense reward above exactly
            (bit-for-bit: multiplying the raw, finite ``planned_start_deviation`` by
            ``0.0`` is exactly ``0.0`` in IEEE 754, so this is a true no-op, not an
            approximation).

    Attributes not present on the static env:
        op_state: :class:`~envs.events.OpState` per operation (see module layout).
        machine_state: :class:`~envs.events.MachineState` per machine.
        remaining_duration: Remaining processing time for a currently ``PREEMPTED``
            operation (``-1`` otherwise).
        active_ops / num_jobs_arrived: How much of the pre-allocated arrays is "live".
    """

    def __init__(
        self,
        instance: JSSPInstance | str = "ft06",
        disruption_config: DisruptionConfig | None = None,
        reward_scale: float | None = None,
        include_machine_edges: bool = True,
        kappa: float = 0.0,
    ) -> None:
        super().__init__()
        self._init_common(instance, reward_scale, include_machine_edges)
        self.config = disruption_config if disruption_config is not None else DisruptionConfig()
        self.kappa = float(kappa)

        self.ops_per_job = self.instance.ops_per_job
        self.num_machines = self.instance.num_machines
        self.base_num_jobs = self.instance.num_jobs
        self.base_num_ops = self.instance.num_operations
        self.max_dynamic_jobs = self.config.max_dynamic_jobs
        self.max_jobs_total = self.base_num_jobs + self.max_dynamic_jobs
        #: Fixed size every growable per-operation array is pre-allocated to (§6.1).
        self.max_possible_ops = self.base_num_ops + self.max_dynamic_jobs * self.ops_per_job

        self.action_space = spaces.Discrete(self.max_possible_ops)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.max_possible_ops, NUM_NODE_FEATURES_DYNAMIC),
                    dtype=np.float32,
                ),
                "adj_precedence": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_possible_ops, self.max_possible_ops),
                    dtype=np.float32,
                ),
                "adj_machine": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_possible_ops, self.max_possible_ops),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.max_possible_ops,), dtype=np.int8
                ),
                "machine_features": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.num_machines, NUM_MACHINE_FEATURES),
                    dtype=np.float32,
                ),
                "adj_eligibility": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_possible_ops, self.num_machines),
                    dtype=np.float32,
                ),
            }
        )

        # Structural index helpers over the max op-id space; constant for the env's life.
        op_ids = np.arange(self.max_possible_ops)
        self._job_index = op_ids // self.ops_per_job
        self._position_index = op_ids % self.ops_per_job
        self._job_progress_full = self._position_index.astype(np.float64) / max(self.ops_per_job, 1)

        self._allocate_episode_state(seed=None)

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset to a fresh episode, re-seeding the disruption generators.

        Args:
            seed: Re-seeds both the Gym RNG and (via
                :func:`~envs.disruptions.spawn_disruption_rngs`) the three independent
                disruption-generator RNGs, so that repeated ``reset(seed=k)`` calls with
                varying ``k`` genuinely vary the disruption realization --- required for
                the multi-seed robustness sweeps in the experimental matrix. ``None``
                falls back to ``disruption_config.seed``.
            options: Reserved for API compliance. Unlike the static env,
                ``options["instance"]`` is not supported (disruption generators and
                array sizes are tied to the instance at construction time).

        Returns:
            ``(observation, info)``.

        Raises:
            NotImplementedError: If ``options`` requests an instance swap.
        """
        super().reset(seed=seed)
        if options and "instance" in options:
            raise NotImplementedError(
                "DynamicJSSPEnv does not support swapping instances via "
                "reset(options=...); construct a new DynamicJSSPEnv instead."
            )
        self._allocate_episode_state(seed=seed)
        return self._observation(), self._info()

    def step(
        self, action: int | np.integer
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Dispatch operation ``action`` at the current decision epoch, then advance
        the simulated clock (processing any intervening breakdowns/repairs/arrivals)
        until the next decision epoch or episode end.

        Args:
            action: Global operation id to dispatch. Must be currently legal --- see
                :meth:`action_masks`.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Raises:
            ValueError: If ``action`` is out of range or illegal.
        """
        action = int(action)
        if not 0 <= action < self.max_possible_ops:
            raise ValueError(f"action {action} outside [0, {self.max_possible_ops})")
        mask = self.action_masks()
        if not mask[action]:
            job = int(self._job_index[action])
            pos = int(self._position_index[action])
            state = OpState(int(self.op_state[action])).name.lower()
            machine = int(self._flat_machines[action])
            machine_state = MachineState(int(self.machine_state[machine])).name.lower()
            raise ValueError(
                f"illegal action {action} (job {job}, position {pos}): operation is "
                f"{state}, its machine {machine} is {machine_state}"
            )

        job = int(self._job_index[action])
        machine = int(self._flat_machines[action])
        base_duration = float(self._flat_durations[action])
        eff_duration = self._proc_sampler.effective_duration(base_duration)

        t0 = self._t
        planned_start_deviation = t0 - float(self._planned_start_estimate[action])

        self.start_times[action] = t0
        self.segment_start_time[action] = t0
        self.effective_duration[action] = eff_duration
        self.op_state[action] = OpState.IN_PROCESS
        self.machine_state[machine] = MachineState.BUSY
        self.machine_free_time[machine] = t0 + eff_duration
        self.machine_busy_op[machine] = action
        self._dispatch_order.append(action)
        self._fel.push(
            t0 + eff_duration, EventKind.OP_COMPLETE, {"op_id": action, "machine": machine}
        )

        previous_h = self._h_estimate
        self._run_until_decision_epoch()
        self._h_estimate = self._compute_h_estimate()
        reward = (previous_h - self._h_estimate) * self.reward_scale
        if self.kappa != 0.0:
            reward -= self.kappa * planned_start_deviation * self.reward_scale

        terminated = self._check_termination()
        if not terminated:
            self._planned_start_estimate = self._earliest_start_estimates_dynamic()

        info = self._info()
        info["decision_time"] = t0
        info["dispatched_op"] = action
        info["dispatched_job"] = job
        info["dispatched_machine"] = machine
        info["planned_start_deviation"] = planned_start_deviation
        if terminated:
            info["makespan"] = self.makespan()
            info["schedule_feasible"], info["violations"] = self._feasibility_check()

        return self._observation(), float(reward), terminated, False, info

    def render(self) -> None:
        """Print a compact textual view of the current dynamic state."""
        arrived = self.num_jobs_arrived - self.base_num_jobs
        print(
            f"{self.instance.name} [dynamic]: {self._num_done}/{self.active_ops} done, "
            f"{arrived} job(s) arrived, t={self._t:.1f}, H(s)={self._h_estimate:.1f}, "
            f"LB={self._static_lb}"
        )
        for machine in range(self.num_machines):
            state = MachineState(int(self.machine_state[machine])).name
            segs = " ".join(
                f"[{s:.0f}-{e:.0f})" for (m, s, e) in self._machine_busy_intervals if m == machine
            )
            print(f"  M{machine} [{state}]: {segs}")

    # ------------------------------------------------------------- public state

    def action_masks(self) -> np.ndarray:
        """Boolean legality mask over ``Discrete(max_possible_ops)``.

        An operation is legal iff it is unscheduled, is the frontier operation of an
        already-arrived job (i.e. every predecessor in its job has actually *completed*,
        not merely been dispatched), and its required machine is currently free (neither
        busy nor broken --- see the module docstring's clarification of §6.2).

        Returns:
            Boolean array of shape ``(max_possible_ops,)``.
        """
        unscheduled = self.op_state == OpState.UNSCHEDULED
        frontier = self._position_index == self._job_next_pos[self._job_index]
        machine_free = self.machine_state[self._flat_machines] == MachineState.FREE
        return unscheduled & frontier & machine_free

    def makespan(self) -> float:
        """Makespan of the operations completed so far (``0.0`` if none have)."""
        if self._num_done == 0:
            return 0.0
        done_mask = self.op_state[: self.active_ops] == OpState.DONE
        return float(self.end_times[: self.active_ops][done_mask].max())

    def remaining_work_per_op(self) -> np.ndarray:
        """Same contract as :meth:`JSSPEnv.remaining_work_per_op`, but covers every
        job that has arrived so far, not just the base instance.

        Computed as a per-job reverse-cumsum over ``self._flat_durations`` grouped by
        ``self._job_index``, mirroring :meth:`~envs.instance.JSSPInstance.remaining_work`'s
        semantics exactly. This is what makes ``lwkr_rule``/``cr_rule`` arrival-safe:
        unlike ``self.instance.remaining_work()`` (fixed-shape, base instance only), this
        method's output always covers every currently-active operation id.

        Returns:
            Float array of shape ``(active_ops,)``.
        """
        shape = (self.num_jobs_arrived, self.ops_per_job)
        durations = self._flat_durations[: self.active_ops].reshape(shape)
        remaining = np.cumsum(durations[:, ::-1], axis=1)[:, ::-1]
        return remaining.ravel()

    def duration_per_op(self) -> np.ndarray:
        """Same contract as :meth:`JSSPEnv.duration_per_op`, but covers every job that
        has arrived so far (T8.1: makes ``spt_rule``/``lpt_rule`` arrival-safe, the
        same way :meth:`remaining_work_per_op` already makes ``lwkr_rule`` arrival-safe).

        ``self._flat_durations[:base_num_ops]`` is populated at construction directly
        from ``self.instance.flat_durations`` (bit-identical to the static env's
        values) and extended in :meth:`_on_arrival` for every job that arrives after
        that -- so this method's output for the base instance's operations never
        differs from ``JSSPEnv.duration_per_op()``'s, only its *coverage* is wider.

        Returns:
            Float array of shape ``(active_ops,)``.
        """
        return self._flat_durations[: self.active_ops]

    def machine_per_op(self) -> np.ndarray:
        """Same contract as :meth:`JSSPEnv.machine_per_op`, arrival-safe. See
        :meth:`duration_per_op`'s docstring for the bit-identical-on-the-base-instance
        guarantee.

        Returns:
            Integer array of shape ``(active_ops,)``.
        """
        return self._flat_machines[: self.active_ops]

    def schedule_dataframe_records(self) -> list[dict[str, float]]:
        """Per-*completed*-operation schedule records; see
        :meth:`~envs.jssp_env.JSSPEnv.schedule_dataframe_records`."""
        records: list[dict[str, float]] = []
        for op_id in np.flatnonzero(self.op_state[: self.active_ops] == OpState.DONE):
            op_id = int(op_id)
            records.append(
                {
                    "op_id": op_id,
                    "job": int(self._job_index[op_id]),
                    "position": int(self._position_index[op_id]),
                    "machine": int(self._flat_machines[op_id]),
                    "start": float(self.start_times[op_id]),
                    "end": float(self.end_times[op_id]),
                    "duration": float(self.effective_duration[op_id]),
                }
            )
        return sorted(records, key=lambda r: (r["machine"], r["start"]))

    # --------------------------------------------------------------- internals

    def _allocate_episode_state(self, seed: int | None) -> None:
        """(Re)build all episode-scoped state: arrays, generators, and the initial FEL.

        Shared by ``__init__`` (so a freshly constructed env is immediately usable, as
        with the static env) and ``reset()``.
        """
        max_ops = self.max_possible_ops
        max_jobs = self.max_jobs_total

        self.op_state = np.full(max_ops, OpState.INACTIVE, dtype=np.int64)
        self.op_state[: self.base_num_ops] = OpState.UNSCHEDULED

        self._flat_machines = np.zeros(max_ops, dtype=np.int64)
        self._flat_machines[: self.base_num_ops] = self.instance.flat_machines
        self._flat_durations = np.zeros(max_ops, dtype=np.float64)
        self._flat_durations[: self.base_num_ops] = self.instance.flat_durations

        self.start_times = np.full(max_ops, -1.0)
        self.end_times = np.full(max_ops, -1.0)
        self.effective_duration = np.full(max_ops, -1.0)
        self.remaining_duration = np.full(max_ops, -1.0)
        self.segment_start_time = np.full(max_ops, -1.0)

        self.machine_state = np.full(self.num_machines, MachineState.FREE, dtype=np.int64)
        self.machine_free_time = np.zeros(self.num_machines)
        self.machine_repair_complete_time = np.zeros(self.num_machines)
        self.machine_busy_op = np.full(self.num_machines, -1, dtype=np.int64)
        self.machine_uptime_accum = np.zeros(self.num_machines)
        self.machine_downtime_accum = np.zeros(self.num_machines)
        self.machine_last_transition_time = np.zeros(self.num_machines)

        #: Compatible with ``baselines/dispatching_rules.py::fifo_rule`` unchanged.
        self.job_ready_time = np.zeros(max_jobs)
        self._job_next_pos = np.zeros(max_jobs, dtype=np.int64)

        self._adj_precedence = np.zeros((max_ops, max_ops), dtype=np.float32)
        self._adj_precedence[: self.base_num_ops, : self.base_num_ops] = build_precedence_adjacency(
            self.instance
        )

        self.num_jobs_arrived = self.base_num_jobs
        self.active_ops = self.base_num_ops

        self._t = 0.0
        self._fel = FutureEventList()
        self._dispatch_order: list[int] = []
        self._event_log: list[tuple] = []
        self._machine_busy_intervals: list[tuple[int, float, float]] = []
        self._machine_broken_intervals: list[tuple[int, float, float]] = []
        self._num_done = 0
        self._arrival_pending = False

        effective_seed = seed if seed is not None else self.config.seed
        breakdown_rng, processing_rng, arrival_rng = spawn_disruption_rngs(effective_seed)
        self._breakdown_gen = BreakdownGenerator(self.instance, self.config, breakdown_rng)
        self._proc_sampler = ProcessingTimeSampler(self.config.processing_time_sigma, processing_rng)
        self._arrival_process = ArrivalProcess(self.instance, self.config, arrival_rng)

        for k in range(self.num_machines):
            if self._breakdown_gen.is_enabled(k):
                ttf = self._breakdown_gen.draw_ttf(k)
                self._fel.push(ttf, EventKind.BREAKDOWN, {"machine": k})

        if self._arrival_process.enabled:
            t0 = self._arrival_process.next_arrival_time(0.0)
            if t0 is not None:
                self._fel.push(t0, EventKind.ARRIVAL, {})
                self._arrival_pending = True

        self._run_until_decision_epoch()
        self._h_estimate = self._compute_h_estimate()
        self._planned_start_estimate = self._earliest_start_estimates_dynamic()

    def _has_decision_epoch(self) -> bool:
        """Whether some machine is free with at least one eligible ready operation."""
        return bool(self.action_masks().any())

    def _check_termination(self) -> bool:
        """§3.5's termination check: no pending arrivals AND every instantiated op is done."""
        if self._arrival_pending:
            return False
        active = self.op_state[: self.active_ops]
        return bool((active == OpState.DONE).all()) if active.size else True

    def _run_until_decision_epoch(self) -> None:
        """Drain the FEL, processing every event in fixed tie-break order, until either a
        decision epoch is exposed or the episode has terminated.

        Termination is checked alongside the decision-epoch condition (not only inferred
        from an empty FEL) because breakdowns are an unbounded renewal process: once
        every operation is done, ``BREAKDOWN``/``REPAIR_COMPLETE`` pairs would otherwise
        keep the FEL non-empty forever with no decision epoch ever appearing again.
        """
        while not self._has_decision_epoch() and not self._check_termination():
            if self._fel.is_empty:
                return
            t_next = self._fel.peek_min_time()
            self._t = t_next
            while (not self._fel.is_empty) and self._fel.peek_min_time() == t_next:
                event = self._fel.pop_min()
                self._process_event(event)

    def _process_event(self, event: Event) -> None:
        if event.kind == EventKind.OP_COMPLETE:
            self._on_op_complete(event.payload)
        elif event.kind == EventKind.BREAKDOWN:
            self._on_breakdown(event.payload)
        elif event.kind == EventKind.REPAIR_COMPLETE:
            self._on_repair_complete(event.payload)
        elif event.kind == EventKind.ARRIVAL:
            self._on_arrival(event.payload)

    # ----------------------------------------------------------- event handlers

    def _on_op_complete(self, payload: dict) -> None:
        op = int(payload["op_id"])
        machine = int(payload["machine"])
        t = self._t

        self.end_times[op] = t
        self.op_state[op] = OpState.DONE
        self.machine_state[machine] = MachineState.FREE
        self.machine_busy_op[machine] = -1
        self._machine_busy_intervals.append((machine, float(self.segment_start_time[op]), t))

        job = int(self._job_index[op])
        self._job_next_pos[job] += 1
        self.job_ready_time[job] = t
        self._num_done += 1
        self._log_event("op_complete", t, op, machine)

    def _on_breakdown(self, payload: dict) -> None:
        machine = int(payload["machine"])
        t = self._t
        self._accrue_availability(machine, t)

        preempted_op: int | None = None
        if self.machine_state[machine] == MachineState.BUSY:
            op = int(self.machine_busy_op[machine])
            removed = self._fel.cancel(
                lambda e: e.kind == EventKind.OP_COMPLETE and e.payload.get("op_id") == op
            )
            assert len(removed) == 1, "expected exactly one pending OP_COMPLETE to cancel"
            s_o = float(self.segment_start_time[op])
            elapsed = t - s_o
            remaining = float(self.effective_duration[op]) - elapsed
            assert remaining > 1e-9, (
                "a breakdown coinciding exactly with its own OP_COMPLETE should never "
                "reach here: the tie-break order fires OP_COMPLETE first"
            )
            self.remaining_duration[op] = remaining
            self.op_state[op] = OpState.PREEMPTED
            self._machine_busy_intervals.append((machine, s_o, t))
            self.machine_busy_op[machine] = -1
            preempted_op = op

        self.machine_state[machine] = MachineState.BROKEN
        ttr = self._breakdown_gen.draw_ttr()
        repair_time = t + ttr
        self.machine_repair_complete_time[machine] = repair_time
        self._fel.push(repair_time, EventKind.REPAIR_COMPLETE, {"machine": machine})
        self._machine_broken_intervals.append((machine, t, repair_time))
        self._log_event("breakdown", t, machine, preempted_op)

    def _on_repair_complete(self, payload: dict) -> None:
        machine = int(payload["machine"])
        t = self._t
        self._accrue_availability(machine, t)
        self.machine_state[machine] = MachineState.FREE

        preempted_op = self._find_preempted_op(machine)
        if preempted_op is not None:
            if self.config.preemption_policy == "resume":
                remaining = float(self.remaining_duration[preempted_op])
                self.op_state[preempted_op] = OpState.IN_PROCESS
                self.segment_start_time[preempted_op] = t
                self.machine_state[machine] = MachineState.BUSY
                self.machine_busy_op[machine] = preempted_op
                self.machine_free_time[machine] = t + remaining
                self._fel.push(
                    t + remaining, EventKind.OP_COMPLETE, {"op_id": preempted_op, "machine": machine}
                )
            else:  # "restart" -- documented ablation, not exercised by the required tests
                self.remaining_duration[preempted_op] = -1.0
                self.op_state[preempted_op] = OpState.UNSCHEDULED
                self.start_times[preempted_op] = -1.0
                self.segment_start_time[preempted_op] = -1.0

        if self._breakdown_gen.is_enabled(machine):
            ttf = self._breakdown_gen.draw_ttf(machine)
            self._fel.push(t + ttf, EventKind.BREAKDOWN, {"machine": machine})
        self._log_event("repair_complete", t, machine)

    def _on_arrival(self, payload: dict) -> None:
        t = self._t
        job_spec = self._arrival_process.generate_job()
        job_idx = self.num_jobs_arrived
        base = job_idx * self.ops_per_job
        end = base + self.ops_per_job

        self._flat_machines[base:end] = job_spec["machines"]
        self._flat_durations[base:end] = job_spec["durations"]
        self.op_state[base:end] = OpState.UNSCHEDULED
        for pos in range(self.ops_per_job - 1):
            self._adj_precedence[base + pos, base + pos + 1] = 1.0

        self.num_jobs_arrived += 1
        self.active_ops += self.ops_per_job
        self._log_event("arrival", t, job_idx)

        next_t = self._arrival_process.next_arrival_time(t)
        if next_t is not None:
            self._fel.push(next_t, EventKind.ARRIVAL, {})
            self._arrival_pending = True
        else:
            self._arrival_pending = False

    def _find_preempted_op(self, machine: int) -> int | None:
        """The (at most one) operation currently ``PREEMPTED`` on ``machine``."""
        candidates = np.flatnonzero(
            (self.op_state[: self.active_ops] == OpState.PREEMPTED)
            & (self._flat_machines[: self.active_ops] == machine)
        )
        return int(candidates[0]) if candidates.size else None

    def _accrue_availability(self, machine: int, t: float) -> None:
        """Accumulate up/down time for ``machine`` since its last state transition."""
        last = self.machine_last_transition_time[machine]
        elapsed = t - last
        if self.machine_state[machine] == MachineState.BROKEN:
            self.machine_downtime_accum[machine] += elapsed
        else:
            self.machine_uptime_accum[machine] += elapsed
        self.machine_last_transition_time[machine] = t

    def _log_event(self, kind: str, time: float, *fields: Any) -> None:
        self._event_log.append((kind, round(float(time), 6), *fields))

    # ------------------------------------------------------------- estimates

    def _compute_h_estimate(self) -> float:
        return estimated_makespan_dynamic(
            ops_per_job=self.ops_per_job,
            num_active_jobs=self.num_jobs_arrived,
            flat_machines=self._flat_machines,
            base_durations=self._flat_durations,
            effective_durations=self.effective_duration,
            op_state=self.op_state,
            start_times=self.start_times,
            end_times=self.end_times,
            remaining_duration=self.remaining_duration,
            machine_repair_complete_time=self.machine_repair_complete_time,
            machine_broken=(self.machine_state == MachineState.BROKEN),
        )

    def _feasibility_check(self) -> tuple[bool, list[str]]:
        return is_schedule_feasible_dynamic(
            ops_per_job=self.ops_per_job,
            num_active_jobs=self.num_jobs_arrived,
            start_times=self.start_times,
            end_times=self.end_times,
            machine_busy_intervals=self._machine_busy_intervals,
            machine_broken_intervals=self._machine_broken_intervals,
        )

    def _earliest_start_estimates_dynamic(self) -> np.ndarray:
        """Per-operation start estimate, used only as the §7 ``planned_start_deviation``
        plumbing signal -- it does not feed the reward.

        Scheduled/in-process/preempted operations report their realised (or
        already-known projected) start; an unscheduled operation reports the optimistic
        job-chain estimate, floored by its machine's repair completion if currently
        broken.
        """
        est = np.zeros(self.max_possible_ops, dtype=np.float64)
        for job in range(self.num_jobs_arrived):
            base = job * self.ops_per_job
            running = 0.0
            for pos in range(self.ops_per_job):
                op = base + pos
                state = self.op_state[op]
                machine = int(self._flat_machines[op])
                if state == OpState.DONE:
                    est[op] = float(self.start_times[op])
                    running = float(self.end_times[op])
                elif state == OpState.IN_PROCESS:
                    est[op] = float(self.start_times[op])
                    running = float(self.machine_free_time[machine])
                elif state == OpState.PREEMPTED:
                    est[op] = float(self.start_times[op])
                    running = float(self.machine_repair_complete_time[machine]) + float(
                        self.remaining_duration[op]
                    )
                else:  # UNSCHEDULED
                    floor = running
                    if self.machine_state[machine] == MachineState.BROKEN:
                        floor = max(floor, float(self.machine_repair_complete_time[machine]))
                    est[op] = floor
                    running = floor + float(self._flat_durations[op])
        return est

    # ------------------------------------------------------------- observation

    def _machine_feature_block(self) -> np.ndarray:
        t = self._t
        horizon = self._horizon
        features = np.zeros((self.num_machines, NUM_MACHINE_FEATURES), dtype=np.float32)

        is_free = self.machine_state == MachineState.FREE
        is_busy = self.machine_state == MachineState.BUSY
        is_broken = self.machine_state == MachineState.BROKEN

        time_to_available = np.zeros(self.num_machines, dtype=np.float64)
        time_to_available[is_busy] = np.maximum(self.machine_free_time[is_busy] - t, 0.0)
        time_to_available[is_broken] = np.maximum(
            self.machine_repair_complete_time[is_broken] - t, 0.0
        )

        n_active = self.num_jobs_arrived * self.ops_per_job
        assigned_pending_load = np.zeros(self.num_machines, dtype=np.float64)
        if n_active > 0:
            machines_active = self._flat_machines[:n_active]
            unscheduled_active = self.op_state[:n_active] == OpState.UNSCHEDULED
            np.add.at(
                assigned_pending_load,
                machines_active,
                self._flat_durations[:n_active] * unscheduled_active,
            )

        # utilization_accum: busy time so far / realised schedule span, mirroring
        # envs.jssp_env.JSSPEnv._machine_features exactly (same formula, same column
        # semantics -- see that method's docstring). Deliberately *dispatch*-gated (like
        # assigned_pending_load above), not completion-gated like self.makespan(): using
        # start_times/effective_duration (fixed at dispatch time) rather than end_times
        # (fixed only at realised completion) is what makes this column byte-identical
        # to the static env's at every decision epoch when disruptions are off, not just
        # convergent by the terminal step -- see
        # test_default_config_matches_static_env_exactly's own note on which features are
        # dispatch-gated (match throughout) vs completion-gated (converge at the end).
        utilization_accum = np.zeros(self.num_machines, dtype=np.float64)
        if n_active > 0:
            machines_active = self._flat_machines[:n_active]
            dispatched_active = self.op_state[:n_active] >= OpState.IN_PROCESS
            busy_so_far = np.zeros(self.num_machines, dtype=np.float64)
            np.add.at(
                busy_so_far,
                machines_active,
                self.effective_duration[:n_active] * dispatched_active,
            )
            if dispatched_active.any():
                committed_span = float(
                    (
                        self.start_times[:n_active] + self.effective_duration[:n_active]
                    )[dispatched_active].max()
                )
                if committed_span > 0.0:
                    utilization_accum = busy_so_far / committed_span

        elapsed_since_transition = t - self.machine_last_transition_time
        up = self.machine_uptime_accum + np.where(is_broken, 0.0, elapsed_since_transition)
        down = self.machine_downtime_accum + np.where(is_broken, elapsed_since_transition, 0.0)
        denom = up + down
        estimated_availability = np.where(denom > 0.0, up / np.maximum(denom, 1e-12), 1.0)

        features[:, 0] = is_free
        features[:, 1] = is_busy
        features[:, 2] = is_broken
        features[:, 3] = (time_to_available / horizon).astype(np.float32)
        features[:, 4] = (assigned_pending_load / horizon).astype(np.float32)
        features[:, 5] = utilization_accum.astype(np.float32)
        features[:, 6] = estimated_availability.astype(np.float32)
        return features

    def _observation(self) -> dict[str, np.ndarray]:
        max_ops = self.max_possible_ops
        horizon = self._horizon
        n_active = self.num_jobs_arrived * self.ops_per_job

        mask = self.action_masks()
        unscheduled = self.op_state == OpState.UNSCHEDULED
        is_scheduled = self.op_state >= OpState.IN_PROCESS
        is_preempted = self.op_state == OpState.PREEMPTED

        job_complete = np.zeros(max_ops, dtype=np.float64)
        if self.num_jobs_arrived > 0:
            arrived_job_mask = self._job_index < self.num_jobs_arrived
            job_done = self._job_next_pos[self._job_index] >= self.ops_per_job
            job_complete[arrived_job_mask] = job_done[arrived_job_mask].astype(np.float64)

        earliest_start = self._earliest_start_estimates_dynamic()

        remaining_job_work = np.zeros(max_ops, dtype=np.float64)
        machine_remaining_per_op = np.zeros(max_ops, dtype=np.float64)
        if n_active > 0:
            dur_active = self._flat_durations[:n_active].reshape(self.num_jobs_arrived, self.ops_per_job)
            unsched_active = unscheduled[:n_active].reshape(self.num_jobs_arrived, self.ops_per_job)
            tail_unsched = dur_active * unsched_active
            tail_cum = np.cumsum(tail_unsched[:, ::-1], axis=1)[:, ::-1]
            remaining_job_work[:n_active] = tail_cum.ravel()

            machines_active = self._flat_machines[:n_active]
            machine_load = np.zeros(self.num_machines, dtype=np.float64)
            np.add.at(
                machine_load,
                machines_active,
                self._flat_durations[:n_active] * unscheduled[:n_active],
            )
            machine_remaining_per_op[:n_active] = machine_load[machines_active]

        features = np.zeros((max_ops, NUM_NODE_FEATURES_DYNAMIC), dtype=np.float32)
        features[:, 0] = self._flat_durations / max(self.instance.max_duration, 1)
        features[:, 1] = is_scheduled
        features[:, 2] = mask
        features[:, 3] = job_complete
        features[:, 4] = earliest_start / horizon
        features[:, 5] = remaining_job_work / horizon
        features[:, 6] = self._job_progress_full
        features[:, 7] = machine_remaining_per_op / horizon
        features[:, 8] = is_preempted

        adj_machine = np.zeros((max_ops, max_ops), dtype=np.float32)
        if self.include_machine_edges and n_active > 0:
            machines_active = self._flat_machines[:n_active]
            active_mask = unscheduled[:n_active]
            same_machine = machines_active[:, None] == machines_active[None, :]
            both_active = active_mask[:, None] & active_mask[None, :]
            block = (same_machine & both_active).astype(np.float32)
            np.fill_diagonal(block, 0.0)
            adj_machine[:n_active, :n_active] = block

        # adj_eligibility: operation<->machine incidence, one-hot per row -- structural
        # under this environment's fixed routing (DynamicJSSPEnv never models FJSP), so
        # it is a plain fact about an operation ("which machine does it run on"), true
        # for its whole lifetime once it exists, not gated by dispatch state -- unlike
        # adj_machine above, which specifically excludes already-dispatched ops. Only
        # arrived (active) rows are filled; not-yet-arrived rows stay all-zero, matching
        # adj_precedence/adj_machine's own "doesn't exist yet" convention for padding.
        adj_eligibility = np.zeros((max_ops, self.num_machines), dtype=np.float32)
        if n_active > 0:
            adj_eligibility[np.arange(n_active), self._flat_machines[:n_active]] = 1.0

        return {
            "node_features": features,
            "adj_precedence": self._adj_precedence,
            "adj_machine": adj_machine,
            "action_mask": mask.astype(np.int8),
            "machine_features": self._machine_feature_block(),
            "adj_eligibility": adj_eligibility,
        }

    def _info(self) -> dict[str, Any]:
        return {
            "instance": self.instance.name,
            "num_scheduled": int(self._num_done),
            "num_dispatched": len(self._dispatch_order),
            "num_operations": int(self.active_ops),
            "num_jobs_arrived": int(self.num_jobs_arrived),
            "sim_time": float(self._t),
            "makespan_estimate": self._h_estimate,
            "current_makespan": self.makespan(),
            "static_lower_bound": self._static_lb,
            "action_mask": self.action_masks(),
            "event_log": list(self._event_log),
        }
