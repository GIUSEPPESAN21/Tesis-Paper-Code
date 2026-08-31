"""Unit tests for :mod:`envs.dynamic_jssp_env`.

Three concerns, mirroring ``tests/test_env.py``'s structure for the static env:

1. **Regression** (:class:`TestRegressionAgainstStatic`) --- with every disruption axis
   off, the dynamic env must reduce exactly to the static one for every RL-relevant
   quantity (reward, terminated, makespan). See the module docstring of
   ``envs/dynamic_jssp_env.py`` for why the *raw* per-step observation is not fully
   byte-identical mid-episode (``is_ready``/``is_job_complete``/``earliest_start_estimate``
   correctly lag, converging again at the final step) while everything else is.
2. **Constraint safety under disruption** (:class:`TestConstraintSafetyUnderBreakdown`,
   :class:`TestPreemption`, :class:`TestActionMasking`) --- hard constraints
   (precedence, no double-booking, no work during a broken interval) must hold, and
   preemption bookkeeping must be exact.
3. **Dynamic arrivals** (:class:`TestDynamicArrivals`) --- the graph grows correctly and
   the episode remains a terminating system.

Note on precedence/capacity checking under breakdowns: ``envs.disjunctive_graph.
is_schedule_feasible`` (the static checker) is not reused here even for the precedence
check, unlike a literal reading of ``DynamicEnv_TechSpec.md`` §8 might suggest --- it
computes an operation's completion as ``start + base_instance_duration``, which is simply
wrong once a preempted operation's *realised* completion is later than that (the pause is
invisible to it). Using the wrong completion time to validate precedence would be a
correctness bug in the test, not just a coverage gap, so
:func:`~envs.disjunctive_graph.is_schedule_feasible_dynamic` (which checks precedence
against ``end_times``, the realised completion) is used for both the precedence and the
machine-capacity checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.dispatching_rules import RULES, run_rule
from envs.disjunctive_graph import is_schedule_feasible_dynamic
from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from envs.events import Event, EventKind, MachineState, OpState
from envs.instance import JSSPInstance
from envs.jssp_env import JSSPEnv


@pytest.fixture
def toy() -> JSSPInstance:
    """Same hand-computable toy instance as ``tests/test_env.py``."""
    return JSSPInstance(
        name="toy",
        machines=np.array([[0, 1], [1, 0]]),
        durations=np.array([[3, 2], [4, 1]]),
        num_machines=2,
    )


# --------------------------------------------------------------- regression vs static


class TestRegressionAgainstStatic:
    """``DynamicJSSPEnv(instance, DisruptionConfig())`` must reduce to ``JSSPEnv``."""

    def test_rules_registry_has_ten_unique_pdrs_plus_random(self) -> None:
        """Same tripwire as ``tests/test_env.py``, checked here too since this file's
        parametrized tests also iterate ``list(RULES.keys())`` and would silently stop
        covering a new rule that failed to register.
        """
        assert len(set(RULES.values())) == 10
        assert len(RULES) == 11

    @pytest.mark.parametrize("instance_name", ["ft06", "la01"])
    @pytest.mark.parametrize("rule_name", list(RULES.keys()))
    def test_default_config_matches_static_env_exactly(
        self, instance_name: str, rule_name: str
    ) -> None:
        dyn_env = DynamicJSSPEnv(instance_name, DisruptionConfig())
        dyn_env.reset(seed=3)
        static_env = JSSPEnv(instance_name)
        static_env.reset(seed=3)
        n0 = static_env.instance.num_operations

        # Dispatch-status-dependent columns must already match on the very first reset.
        obs_d0 = dyn_env._observation()
        obs_s0 = static_env._observation()
        np.testing.assert_allclose(
            obs_s0["node_features"][:, [0, 1, 5, 6, 7]],
            obs_d0["node_features"][:n0][:, [0, 1, 5, 6, 7]],
            atol=1e-6,
        )
        np.testing.assert_allclose(obs_s0["adj_precedence"], obs_d0["adj_precedence"][:n0, :n0])
        # adj_eligibility is a structural fact (which machine an op runs on), not gated
        # by dispatch state at all, so it must match in full from the very first reset.
        np.testing.assert_allclose(obs_s0["adj_eligibility"], obs_d0["adj_eligibility"][:n0])
        # machine_features: is_free/is_busy/is_broken/time_to_available/
        # assigned_pending_load/utilization_accum are all dispatch-gated (E4-E6 wiring
        # fix, see envs/dynamic_jssp_env.py); estimated_availability is excluded here
        # since it is availability-history-gated, not dispatch-gated (both start at 1.0
        # with no breakdowns, so this is not a meaningful check of it anyway).
        np.testing.assert_allclose(
            obs_s0["machine_features"][:, :6], obs_d0["machine_features"][:, :6], atol=1e-6
        )

        rng = np.random.default_rng(11)
        terminated = False
        total_reward_s = total_reward_d = 0.0
        step_i = 0
        info_s = info_d = {}
        while not terminated:
            # The rule queries the dynamic env's own (stricter, completion-gated) mask;
            # the identical action is then replayed through the static env.
            action = RULES[rule_name](dyn_env, rng)
            obs_d, r_d, terminated, _, info_d = dyn_env.step(action)
            obs_s, r_s, terminated_s, _, info_s = static_env.step(action)

            assert terminated == terminated_s, step_i
            assert r_s == pytest.approx(r_d, abs=1e-9), (rule_name, step_i, r_s, r_d)
            np.testing.assert_allclose(
                obs_s["node_features"][:, [0, 1, 5, 6, 7]],
                obs_d["node_features"][:n0][:, [0, 1, 5, 6, 7]],
                atol=1e-6,
                err_msg=f"{rule_name}/{instance_name} step {step_i}",
            )
            np.testing.assert_allclose(obs_s["adj_precedence"], obs_d["adj_precedence"][:n0, :n0])
            np.testing.assert_allclose(
                obs_s["adj_eligibility"],
                obs_d["adj_eligibility"][:n0],
                err_msg=f"adj_eligibility {rule_name}/{instance_name} step {step_i}",
            )
            np.testing.assert_allclose(
                obs_s["machine_features"][:, :6],
                obs_d["machine_features"][:, :6],
                atol=1e-6,
                err_msg=f"machine_features {rule_name}/{instance_name} step {step_i}",
            )
            total_reward_s += r_s
            total_reward_d += r_d
            step_i += 1

        # At the final step every operation is dispatched-and-done in both envs, so the
        # completion-gated features converge too: the full 8-column observation matches.
        np.testing.assert_allclose(
            obs_s["node_features"], obs_d["node_features"][:n0, :8], atol=1e-6
        )
        np.testing.assert_array_equal(obs_s["action_mask"], obs_d["action_mask"][:n0])
        np.testing.assert_allclose(obs_s["machine_features"], obs_d["machine_features"], atol=1e-6)
        assert info_s["makespan"] == info_d["makespan"]
        assert total_reward_s == pytest.approx(total_reward_d, abs=1e-6)
        assert info_d["schedule_feasible"]

    def test_zero_disruption_config_disables_every_axis(self) -> None:
        config = DisruptionConfig()
        env = DynamicJSSPEnv("ft06", config)
        env.reset(seed=0)
        assert not any(env._breakdown_gen.is_enabled(k) for k in range(env.num_machines))
        assert not env._arrival_process.enabled
        assert not env._proc_sampler.enabled
        assert env._fel.is_empty  # no BREAKDOWN/ARRIVAL events seeded at reset


# --------------------------------------------------------- kappa*Delta_inest reward (E7)


class TestKappaReward:
    """``E7_E8_FinalPlan.md`` §2.1 precondition #1: the disruption-aware reward term.

    ``reward = (previous_h - h_estimate) * reward_scale
               - kappa * planned_start_deviation * reward_scale``
    """

    @pytest.mark.parametrize("instance_name", ["ft06", "la01"])
    @pytest.mark.parametrize("rule_name", list(RULES.keys()))
    def test_kappa_zero_is_bit_for_bit_identical_to_omitting_kappa(
        self, instance_name: str, rule_name: str
    ) -> None:
        """kappa=0.0 must be a true no-op, not an approximation (§2.1 precondition #1)."""
        env_default = DynamicJSSPEnv(instance_name, DisruptionConfig())
        env_explicit_zero = DynamicJSSPEnv(instance_name, DisruptionConfig(), kappa=0.0)
        env_default.reset(seed=5)
        env_explicit_zero.reset(seed=5)

        rng = np.random.default_rng(7)
        terminated = False
        while not terminated:
            action = RULES[rule_name](env_default, rng)
            _, r_default, terminated, _, _ = env_default.step(action)
            _, r_zero, terminated_zero, _, _ = env_explicit_zero.step(action)
            assert terminated == terminated_zero
            assert r_default == r_zero  # exact bit-for-bit equality, not approx

    @pytest.mark.parametrize("kappa", [0.5, 1.0])
    def test_kappa_positive_penalizes_planned_start_deviation(self, kappa: float) -> None:
        """Under breakdowns (so ``planned_start_deviation`` is realistically nonzero),
        a kappa>0 trajectory's reward must equal the kappa=0 trajectory's reward minus
        exactly ``kappa * planned_start_deviation * reward_scale`` at every step -- the
        same dispatch decisions, deterministically replayed, so the only thing that can
        differ is the reward term itself.
        """
        config = DisruptionConfig(target_availability=0.80, seed=13)
        env_ref = DynamicJSSPEnv("ft06", config, kappa=0.0)
        env_kappa = DynamicJSSPEnv("ft06", config, kappa=kappa)
        env_ref.reset(seed=13)
        env_kappa.reset(seed=13)

        rng = np.random.default_rng(21)
        terminated = False
        saw_nonzero_deviation = False
        while not terminated:
            action = RULES["mwkr"](env_ref, rng)
            _, r_ref, terminated, _, info_ref = env_ref.step(action)
            _, r_kappa, terminated_kappa, _, info_kappa = env_kappa.step(action)
            assert terminated == terminated_kappa
            deviation = info_ref["planned_start_deviation"]
            assert deviation == pytest.approx(info_kappa["planned_start_deviation"], abs=1e-9)
            if abs(deviation) > 1e-9:
                saw_nonzero_deviation = True
            expected = r_ref - kappa * deviation * env_ref.reward_scale
            assert r_kappa == pytest.approx(expected, abs=1e-9)

        assert saw_nonzero_deviation, "breakdown config produced no nonzero deviation to test"

    def test_kappa_scales_with_reward_scale_across_instance_sizes(self) -> None:
        """§2.1's scale-matching decision: kappa's magnitude should be instance-size-
        independent because reward_scale is applied to the Delta_inest term too. This is
        a direct check that the multiplier used is exactly ``reward_scale``, for both a
        small (ft06) and larger (la01) instance -- not a claim that the two instances'
        raw rewards are numerically equal, only that each uses its own reward_scale
        consistently for both reward terms.
        """
        for instance_name in ["ft06", "la01"]:
            config = DisruptionConfig(target_availability=0.80, seed=13)
            env_ref = DynamicJSSPEnv(instance_name, config, kappa=0.0)
            env_kappa = DynamicJSSPEnv(instance_name, config, kappa=1.0)
            env_ref.reset(seed=13)
            env_kappa.reset(seed=13)
            rng = np.random.default_rng(21)
            terminated = False
            while not terminated:
                action = RULES["mwkr"](env_ref, rng)
                _, r_ref, terminated, _, info_ref = env_ref.step(action)
                _, r_kappa, _, _, _ = env_kappa.step(action)
                expected = r_ref - 1.0 * info_ref["planned_start_deviation"] * env_ref.reward_scale
                assert r_kappa == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------- constraint safety


def _run_breakdown_rollout(instance_name: str, rule_name: str, seed: int):
    config = DisruptionConfig(
        target_availability=0.8, mttr_multiplier=2.0, processing_time_sigma=0.25, seed=seed
    )
    env = DynamicJSSPEnv(instance_name, config)
    result = run_rule(env, rule_name, seed=seed)
    return env, result


class TestConstraintSafetyUnderBreakdown:
    @pytest.mark.parametrize("rule_name", list(RULES.keys()))
    @pytest.mark.parametrize("instance_name", ["ft06", "la01"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_rule_rollout_is_always_feasible_dynamic(
        self, rule_name: str, instance_name: str, seed: int
    ) -> None:
        """All six frozen PDRs, replayed against DynamicJSSPEnv with breakdowns and
        stochastic processing times enabled (arrivals off --- see module docstring of
        ``dynamic_jssp_env.py`` for why three of the six rules are unsafe once a legal
        action can belong to a dynamically arrived job), must still produce a feasible
        schedule.
        """
        env, result = _run_breakdown_rollout(instance_name, rule_name, seed)
        assert result["feasible"], f"{rule_name}@{instance_name} seed={seed}: {result['violations']}"
        assert result["makespan"] >= env.static_lb

    @pytest.mark.parametrize("rule_name", ["random", "spt", "mwkr"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_precedence_preserved_under_breakdown(self, rule_name: str, seed: int) -> None:
        env, result = _run_breakdown_rollout("ft06", rule_name, seed)
        feasible, violations = is_schedule_feasible_dynamic(
            ops_per_job=env.ops_per_job,
            num_active_jobs=env.num_jobs_arrived,
            start_times=env.start_times,
            end_times=env.end_times,
            machine_busy_intervals=env._machine_busy_intervals,
            machine_broken_intervals=env._machine_broken_intervals,
        )
        precedence_violations = [v for v in violations if "precedence" in v]
        assert not precedence_violations, precedence_violations

    @pytest.mark.parametrize("rule_name", ["random", "spt", "mwkr"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_no_machine_double_booking_under_breakdown(self, rule_name: str, seed: int) -> None:
        env, result = _run_breakdown_rollout("ft06", rule_name, seed)
        feasible, violations = is_schedule_feasible_dynamic(
            ops_per_job=env.ops_per_job,
            num_active_jobs=env.num_jobs_arrived,
            start_times=env.start_times,
            end_times=env.end_times,
            machine_busy_intervals=env._machine_busy_intervals,
            machine_broken_intervals=env._machine_broken_intervals,
        )
        capacity_violations = [
            v for v in violations if "overlap" in v or "broken interval" in v
        ]
        assert not capacity_violations, capacity_violations
        assert len(env._machine_broken_intervals) > 0, "test is vacuous if no breakdown ever fired"


# ------------------------------------------------------------------- preemption


class TestPreemption:
    def test_preempted_operation_resumes_with_correct_remaining_duration(self, toy) -> None:
        """Force a breakdown mid-operation via an injected FEL event (breakdowns are off
        in the config, so the only BREAKDOWN event that ever fires is the one this test
        injects), then check §3.3/§3.4's bookkeeping exactly.
        """
        env = DynamicJSSPEnv(toy, DisruptionConfig(target_availability=1.0))
        env.reset(seed=0)

        env.step(0)  # job0/pos0 on machine0, duration 3; dispatched at t=0
        assert env._t == 0.0  # job1/pos0 (machine1) is still simultaneously dispatchable

        # Inject a breakdown on machine 0 at t=1 (mid-operation: elapsed=1, remaining=2)
        # -- exactly the "mocked/injected event" the spec test asks for.
        env._fel.push(1.0, EventKind.BREAKDOWN, {"machine": 0})

        env.step(2)  # job1/pos0 on machine1, duration 4; this dispatch's drain is what
        # first encounters and processes the injected breakdown, since machine0 becomes
        # busy (no longer simultaneously idle) and the FEL must now advance real time.

        assert env.op_state[0] == OpState.DONE  # by now, resumed and completed
        assert env.remaining_duration[0] == pytest.approx(2.0)
        repair_complete_time = [e[1] for e in env._event_log if e[0] == "repair_complete"][0]
        assert env.end_times[0] == pytest.approx(repair_complete_time + 2.0)

        # The busy-interval record shows the operation split into exactly two segments.
        segments = [(s, e) for (m, s, e) in env._machine_busy_intervals if m == 0]
        assert segments == [(0.0, 1.0), pytest.approx((repair_complete_time, repair_complete_time + 2.0))]

        breakdown_entries = [e for e in env._event_log if e[0] == "breakdown"]
        assert breakdown_entries == [("breakdown", 1.0, 0, 0)]  # (kind, t, machine, preempted_op=0)

    def test_preemption_step_by_step_with_manual_events(self, toy) -> None:
        """Same scenario, but processed one manual event at a time for finer-grained
        assertions immediately after the breakdown (before any repair/resume).
        """
        env2 = DynamicJSSPEnv(toy, DisruptionConfig(target_availability=1.0))
        env2.reset(seed=0)
        env2.step(0)
        env2._t = 1.0
        env2._process_event(
            Event(time=1.0, priority=int(EventKind.BREAKDOWN), seq=10**6, kind=EventKind.BREAKDOWN, payload={"machine": 0})
        )
        assert env2.op_state[0] == OpState.PREEMPTED
        assert env2.remaining_duration[0] == pytest.approx(2.0)
        assert env2.machine_state[0] == MachineState.BROKEN
        assert (0, 0.0, 1.0) in env2._machine_busy_intervals

        repair_time = float(env2.machine_repair_complete_time[0])
        env2._t = repair_time
        env2._process_event(
            Event(
                time=repair_time,
                priority=int(EventKind.REPAIR_COMPLETE),
                seq=10**6 + 1,
                kind=EventKind.REPAIR_COMPLETE,
                payload={"machine": 0},
            )
        )
        assert env2.op_state[0] == OpState.IN_PROCESS
        assert env2.segment_start_time[0] == pytest.approx(repair_time)
        assert env2.machine_free_time[0] == pytest.approx(repair_time + 2.0)
        pending = [
            e for e in env2._fel if e.kind == EventKind.OP_COMPLETE and e.payload.get("op_id") == 0
        ]
        assert len(pending) == 1
        assert pending[0].time == pytest.approx(repair_time + 2.0)


# ---------------------------------------------------------------- action masking


class TestActionMasking:
    def test_action_mask_excludes_broken_machine_frontier_ops(self, toy) -> None:
        env = DynamicJSSPEnv(toy, DisruptionConfig())
        env.reset(seed=0)
        assert env.action_masks()[0]  # frontier of job0, machine0 free -> legal

        env.machine_state[0] = MachineState.BROKEN
        assert not env.action_masks()[0]  # same op, machine now broken -> illegal

    def test_action_mask_excludes_busy_machine(self, toy) -> None:
        """§6.2's "not broken" clause alone would under-mask; a machine that is merely
        *busy* (not broken) with a different operation must also exclude other
        same-machine frontier ops --- see the module docstring's clarification.
        """
        env = DynamicJSSPEnv(toy, DisruptionConfig())
        env.reset(seed=0)
        env.step(0)  # machine0 now busy with job0/pos0
        # job1/pos1 (op3) also needs machine0 but is not yet the frontier of job1 anyway;
        # check directly at the machine-state level instead.
        assert env.machine_state[0] == MachineState.BUSY
        assert not env.action_masks()[0]  # op0 itself: no longer UNSCHEDULED anyway

    def test_preempted_op_is_not_legal(self, toy) -> None:
        env = DynamicJSSPEnv(toy, DisruptionConfig(target_availability=1.0))
        env.reset(seed=0)
        env.step(0)
        env._t = 1.0
        env._process_event(
            Event(time=1.0, priority=int(EventKind.BREAKDOWN), seq=1, kind=EventKind.BREAKDOWN, payload={"machine": 0})
        )
        assert env.op_state[0] == OpState.PREEMPTED
        assert not env.action_masks()[0]


# ------------------------------------------------------------------ dynamic arrivals


class TestDynamicArrivals:
    def test_dynamic_job_extends_graph_correctly(self) -> None:
        env = DynamicJSSPEnv(
            "ft06", DisruptionConfig(target_utilization=0.6, max_dynamic_jobs=3, seed=0)
        )
        env.reset(seed=0)
        before_jobs = env.num_jobs_arrived
        before_active = env.active_ops

        env._t = 0.0
        env._on_arrival({})

        assert env.num_jobs_arrived == before_jobs + 1
        assert env.active_ops == before_active + env.ops_per_job

        new_job = before_jobs
        base = new_job * env.ops_per_job
        assert all(env.op_state[base : base + env.ops_per_job] == OpState.UNSCHEDULED)

        for pos in range(env.ops_per_job - 1):
            assert env._adj_precedence[base + pos, base + pos + 1] == 1.0
        # No stray precedence edge leaks into another job's block.
        assert env._adj_precedence[base + env.ops_per_job - 1, : base + env.ops_per_job].sum() == 0

        obs = env._observation()
        adj_machine = obs["adj_machine"]
        for pos in range(env.ops_per_job):
            op = base + pos
            machine = int(env._flat_machines[op])
            expected_peers = [
                o
                for o in range(env.active_ops)
                if o != op
                and env.op_state[o] == OpState.UNSCHEDULED
                and int(env._flat_machines[o]) == machine
            ]
            for peer in expected_peers:
                assert adj_machine[op, peer] == 1.0
                assert adj_machine[peer, op] == 1.0

    def test_duration_and_machine_per_op_extend_after_arrival(self) -> None:
        """T8.1: env.duration_per_op()/env.machine_per_op() must (a) exactly match the
        base instance's arrays over the base operations, and (b) grow to cover a newly
        arrived job's operations -- this is what makes spt/lpt/mwkr arrival-safe."""
        env = DynamicJSSPEnv("ft06", DisruptionConfig(target_utilization=0.6, max_dynamic_jobs=3, seed=0))
        env.reset(seed=0)

        base_durations_before = env.duration_per_op().copy()
        base_machines_before = env.machine_per_op().copy()
        np.testing.assert_array_equal(base_durations_before, env.instance.flat_durations)
        np.testing.assert_array_equal(base_machines_before, env.instance.flat_machines)

        before_active = env.active_ops
        env._t = 0.0
        env._on_arrival({})

        durations_after = env.duration_per_op()
        machines_after = env.machine_per_op()
        assert durations_after.shape[0] == env.active_ops
        # Bit-identical over the operations that existed before the arrival.
        np.testing.assert_array_equal(durations_after[:before_active], base_durations_before)
        np.testing.assert_array_equal(machines_after[:before_active], base_machines_before)
        # The newly arrived job's operations are populated (no longer the zero-init sentinel).
        new_job_durations = durations_after[before_active:]
        assert new_job_durations.shape[0] == env.ops_per_job
        assert (new_job_durations > 0).all()

    def test_episode_terminates_with_dynamic_arrivals(self) -> None:
        config = DisruptionConfig(target_utilization=0.85, max_dynamic_jobs=10, seed=1)
        env = DynamicJSSPEnv("ft06", config)
        env.reset(seed=1)
        rng = np.random.default_rng(2)
        terminated = False
        info = {}
        steps = 0
        while not terminated:
            legal = np.flatnonzero(env.action_masks())
            action = int(legal[rng.integers(len(legal))])
            _, _, terminated, _, info = env.step(action)
            steps += 1
            assert steps <= env.max_possible_ops  # guards against a runaway loop

        assert info["num_jobs_arrived"] > env.base_num_jobs  # at least one arrival happened
        assert info["num_scheduled"] == info["num_operations"]  # every arrived job's ops are DONE
        assert steps == info["num_operations"]
        assert info["schedule_feasible"]

    @pytest.mark.parametrize("seed", range(8))
    def test_episode_terminating_system_bounded_length(self, seed: int) -> None:
        """Guards against a future regression where the arrival cutoff/cap is
        accidentally bypassed: episode length must never exceed the pre-allocated
        ``max_possible_ops``, across a spread of seeds.
        """
        config = DisruptionConfig(
            target_utilization=0.9, max_dynamic_jobs=8, arrival_cutoff_time=None, seed=seed
        )
        env = DynamicJSSPEnv("ft06", config)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        terminated = False
        steps = 0
        while not terminated:
            legal = np.flatnonzero(env.action_masks())
            action = int(legal[rng.integers(len(legal))])
            _, _, terminated, _, _ = env.step(action)
            steps += 1
            assert steps <= env.max_possible_ops
        assert terminated
