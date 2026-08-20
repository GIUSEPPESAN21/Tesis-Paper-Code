"""Unit tests for :mod:`envs.jssp_env`.

Two things matter most for a scheduling environment: (1) it never produces an invalid
schedule, and (2) an optimal action sequence is actually reachable. :class:`TestConstraintSafety`
covers (1) by rolling out every dispatching-rule baseline (including uniformly random) on
several instances and checking the resulting schedule against
:func:`~envs.disjunctive_graph.is_schedule_feasible`. :class:`TestOptimalSequence` covers
(2) with a hand-derived toy optimum and a CP-SAT-proved optimum on ft06 (55, the literature
value -- see ``VALIDATION_WEEK1.md``).
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.cpsat import dispatch_order_from_starts, solve_cpsat
from baselines.dispatching_rules import RULES, run_rule
from envs.disjunctive_graph import is_schedule_feasible
from envs.instance import JSSPInstance
from envs.jssp_env import NUM_NODE_FEATURES, JSSPEnv
from envs.loader import load_by_name


@pytest.fixture
def toy() -> JSSPInstance:
    """Same hand-computable toy instance as ``tests/test_disjunctive_graph.py``."""
    return JSSPInstance(
        name="toy",
        machines=np.array([[0, 1], [1, 0]]),
        durations=np.array([[3, 2], [4, 1]]),
        num_machines=2,
    )


class TestResetAndSpaces:
    def test_reset_returns_consistent_shapes(self) -> None:
        env = JSSPEnv("ft06")
        obs, info = env.reset(seed=0)
        n = env.instance.num_operations
        assert obs["node_features"].shape == (n, NUM_NODE_FEATURES)
        assert obs["adj_precedence"].shape == (n, n)
        assert obs["adj_machine"].shape == (n, n)
        assert obs["action_mask"].shape == (n,)
        assert info["instance"] == "ft06"

    def test_duration_and_machine_per_op_delegate_to_instance(self) -> None:
        """T8.1: JSSPEnv.duration_per_op()/machine_per_op() are pure delegation to the
        base instance's arrays -- the static half of the arrival-safety fix, mirrored
        by DynamicJSSPEnv's growable override (tests/test_dynamic_env.py)."""
        env = JSSPEnv("ft06")
        np.testing.assert_array_equal(env.duration_per_op(), env.instance.flat_durations)
        np.testing.assert_array_equal(env.machine_per_op(), env.instance.flat_machines)

    def test_initial_mask_is_first_operation_of_every_job(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        obs, _ = env.reset()
        # op0 = job0/pos0, op2 = job1/pos0 are the two job frontiers.
        np.testing.assert_array_equal(obs["action_mask"], [1, 0, 1, 0])

    def test_observation_space_contains_reset_observation(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_reward_scale_defaults_to_inverse_lower_bound(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        assert env.reward_scale == pytest.approx(1.0 / env.static_lb)


class TestStepMechanics:
    def test_hand_dispatch_sequence_matches_manual_computation(self, toy: JSSPInstance) -> None:
        """Dispatch order [0, 2, 1, 3] is the by-hand optimal schedule (see
        ``test_disjunctive_graph.py``): start times [0, 4, 0, 4], makespan 6.
        """
        env = JSSPEnv(toy, reward_scale=1.0)
        env.reset()

        _, _, terminated, _, info = env.step(0)
        assert env.start_times[0] == 0 and env.end_times[0] == 3
        assert not terminated

        env.step(2)
        assert env.start_times[2] == 0 and env.end_times[2] == 4

        env.step(1)
        assert env.start_times[1] == 4 and env.end_times[1] == 6

        _, reward, terminated, truncated, info = env.step(3)
        assert env.start_times[3] == 4 and env.end_times[3] == 5
        assert terminated
        assert not truncated
        assert info["makespan"] == 6
        assert info["schedule_feasible"]
        assert env.static_lb == 6  # optimal: makespan equals the lower bound

    def test_illegal_action_not_ready_raises(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        env.reset()
        with pytest.raises(ValueError, match="not yet ready"):
            env.step(1)  # job0/pos1 before job0/pos0

    def test_illegal_action_already_scheduled_raises(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        env.reset()
        env.step(0)
        with pytest.raises(ValueError, match="already scheduled"):
            env.step(0)

    def test_out_of_range_action_raises(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        env.reset()
        with pytest.raises(ValueError, match="outside"):
            env.step(99)

    def test_reward_telescopes_to_h0_minus_makespan(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy, reward_scale=1.0)
        env.reset()
        h0 = env._h_estimate
        total_reward = 0.0
        for action in (0, 2, 1, 3):
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        assert total_reward == pytest.approx(h0 - env.makespan())

    def test_action_mask_matches_action_masks_method(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        obs, _ = env.reset()
        for action in (0, 2, 1, 3):
            np.testing.assert_array_equal(obs["action_mask"], env.action_masks().astype(np.int8))
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                np.testing.assert_array_equal(obs["action_mask"], env.action_masks().astype(np.int8))

    def test_include_machine_edges_false_gives_empty_adjacency(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy, include_machine_edges=False)
        obs, _ = env.reset()
        assert np.all(obs["adj_machine"] == 0)

    def test_reward_mode_rejects_unknown_value(self, toy: JSSPInstance) -> None:
        with pytest.raises(ValueError, match="reward_mode"):
            JSSPEnv(toy, reward_mode="dense_and_sparse")

    def test_sparse_reward_mode_is_zero_until_terminal(self, toy: JSSPInstance) -> None:
        """``AblationPlan.md`` Axis 2: sparse reward is 0 at every non-terminal step and
        ``-makespan * reward_scale`` at the terminal step only."""
        env = JSSPEnv(toy, reward_scale=1.0, reward_mode="sparse")
        env.reset()
        rewards = []
        terminated = False
        for action in (0, 2, 1, 3):
            _, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
        assert rewards[:-1] == [0.0, 0.0, 0.0]
        assert terminated
        assert rewards[-1] == pytest.approx(-env.makespan())

    def test_sparse_and_dense_reward_totals_differ_by_h0_constant(self, toy: JSSPInstance) -> None:
        """Both telescope to the same total up to the ``H(s_0)`` constant only dense adds."""
        dense_env = JSSPEnv(toy, reward_scale=1.0, reward_mode="dense")
        dense_env.reset()
        h0 = dense_env._h_estimate
        dense_total = sum(dense_env.step(a)[1] for a in (0, 2, 1, 3))

        sparse_env = JSSPEnv(toy, reward_scale=1.0, reward_mode="sparse")
        sparse_env.reset()
        sparse_total = sum(sparse_env.step(a)[1] for a in (0, 2, 1, 3))

        assert dense_total == pytest.approx(h0 - dense_env.makespan())
        assert sparse_total == pytest.approx(-sparse_env.makespan())
        assert dense_total - sparse_total == pytest.approx(h0)


class TestConstraintSafety:
    """Every dispatching rule, on every tested instance, must yield a feasible schedule."""

    def test_rules_registry_has_ten_unique_pdrs_plus_random(self) -> None:
        """Tripwire (Week 9, ``Baselines_TechSpec.md`` §7): nine PDRs + random. ``len(RULES)``
        is 11, not 10, because ``"mopnr"``/``"mor"`` are two keys for the same callable
        (see ``RULES``'s own docstring); ``len(set(RULES.values()))`` is the count that
        actually equals 10 and is what a silently-unregistered new rule would break.
        """
        assert len(set(RULES.values())) == 10
        assert len(RULES) == 11

    @pytest.mark.parametrize("rule_name", list(RULES.keys()))
    @pytest.mark.parametrize("instance_name", ["ft06", "la01"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_rule_rollout_is_always_feasible(
        self, rule_name: str, instance_name: str, seed: int
    ) -> None:
        env = JSSPEnv(instance_name)
        result = run_rule(env, rule_name, seed=seed)
        assert result["feasible"], f"{rule_name}@{instance_name} seed={seed}: {result['violations']}"
        assert result["makespan"] >= env.static_lb

    def test_random_policy_schedules_every_operation_exactly_once(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        result = run_rule(env, "random", seed=42)
        assert sorted(result["dispatch_order"]) == list(range(toy.num_operations))

    def test_independent_feasibility_check_on_final_start_times(self) -> None:
        env = JSSPEnv("ft06")
        run_rule(env, "spt", seed=0)
        feasible, violations = is_schedule_feasible(env.instance, env.start_times)
        assert feasible, violations


class TestOptimalSequence:
    def test_ft06_cpsat_optimal_sequence_reaches_known_makespan(self) -> None:
        """Solve ft06 to proven optimality with CP-SAT, replay the resulting schedule's
        dispatch order through the environment, and check the literature makespan (55)
        is reproduced -- the central correctness claim for Week 1 (see
        ``VALIDATION_WEEK1.md``).
        """
        instance = load_by_name("ft06")
        result = solve_cpsat(instance, time_limit_s=30.0)
        assert result.is_optimal
        assert result.makespan == 55

        order = dispatch_order_from_starts(instance, result.start_times)
        env = JSSPEnv(instance)
        env.reset()
        terminated = False
        info: dict = {}
        for action in order:
            _, _, terminated, _, info = env.step(action)
        assert terminated
        assert info["makespan"] == 55
        assert info["schedule_feasible"]

    def test_toy_optimal_sequence_reaches_lower_bound(self, toy: JSSPInstance) -> None:
        env = JSSPEnv(toy)
        env.reset()
        for action in (0, 2, 1, 3):
            _, _, terminated, _, info = env.step(action)
        assert info["makespan"] == env.static_lb
