"""Unit tests for the Brandimarte loader and the flexible job-shop environment.

Two things are being pinned here, and they are different in kind:

1. **The loader is correct.** A benchmark parser that is quietly wrong invalidates every
   downstream number, so the strongest available check is used: CP-SAT is run to
   optimality on the parsed instance and must reach the *literature* optimum (40 for
   ``mk01``).  That simultaneously validates the parse, the machine indexing, and the
   claim that ``envs.fjsp_loader``'s rectangularity padding is makespan-neutral.
2. **The environment's MDP is well-formed.** The masking, the reward telescoping, and the
   feasibility of every reachable schedule --- the same properties ``tests/test_env.py``
   pins for :class:`~envs.jssp_env.JSSPEnv`, restated for the flexible action space.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.interface import CPSATFJSPBaseline
from envs.fjsp_env import (
    FJSPEnv,
    eligibility_matrices,
    fjsp_static_lower_bound,
)
from envs.fjsp_loader import (
    FJSPParseError,
    fjsp_known_optimum,
    load_fjsp_by_name,
    parse_brandimarte,
    real_operation_mask,
)
from envs.jssp_env import NUM_MACHINE_FEATURES, NUM_NODE_FEATURES

MK01_OPTIMUM = 40


class TestBrandimarteLoader:
    def test_mk01_shape(self) -> None:
        instance = load_fjsp_by_name("mk01")
        assert instance.name == "mk01"
        assert instance.num_jobs == 10
        assert instance.num_machines == 6
        # Ragged in the source (jobs of 5 and 6 operations), padded to the maximum.
        assert instance.ops_per_job == 6
        assert instance.num_operations == 60
        assert int(real_operation_mask(instance).sum()) == 55  # the literature op count

    def test_padding_operations_are_zero_duration_and_single_machine(self) -> None:
        instance = load_fjsp_by_name("mk01")
        padded = ~real_operation_mask(instance)
        assert padded.sum() == 5
        for op in np.flatnonzero(padded):
            job, pos = instance.job_of(int(op)), instance.position_of(int(op))
            assert instance.durations[job][pos] == [0]
            assert len(instance.eligible_machines[job][pos]) == 1

    def test_every_machine_index_is_in_range(self) -> None:
        instance = load_fjsp_by_name("mk01")
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                for machine in instance.eligible_machines[job][pos]:
                    assert 0 <= machine < instance.num_machines

    def test_lower_bound_does_not_exceed_the_known_optimum(self) -> None:
        """The single cheapest sanity check on a benchmark parse."""
        instance = load_fjsp_by_name("mk01")
        assert fjsp_static_lower_bound(instance) <= fjsp_known_optimum("mk01")

    def test_known_optimum_table(self) -> None:
        assert fjsp_known_optimum("mk01") == MK01_OPTIMUM
        with pytest.raises(KeyError):
            fjsp_known_optimum("not-an-instance")

    @pytest.mark.slow
    def test_cpsat_reaches_the_literature_optimum_on_the_padded_instance(self) -> None:
        """End-to-end validation of both the parse and the padding's neutrality.

        If the parse were wrong the optimum would differ; if the padding consumed real
        machine capacity the optimum would be worse than 40.  Reaching exactly 40, with
        a *proven* ``OPTIMAL`` status, rules out both.
        """
        result = CPSATFJSPBaseline().solve(load_fjsp_by_name("mk01"), time_limit_s=180.0)
        assert result.status == "OPTIMAL"
        assert result.makespan == MK01_OPTIMUM
        assert result.feasible, result.violations

    def test_parser_rejects_a_truncated_file(self) -> None:
        with pytest.raises(FJSPParseError):
            parse_brandimarte("2 3\n2 1 0 5 1 1 4\n", "broken")

    def test_parser_normalises_one_indexed_machines(self) -> None:
        """1-indexed transcriptions are shifted down, as ``envs.loader`` does for Taillard."""
        text = "1 2\n2 1 1 7 1 2 9\n"  # one job, two ops, machines {1} and {2}
        instance = parse_brandimarte(text, "toy")
        assert instance.eligible_machines[0][0] == [0]
        assert instance.eligible_machines[0][1] == [1]


class TestFJSPEnvSpaces:
    def test_observation_shapes_and_membership(self) -> None:
        env = FJSPEnv("mk01")
        obs, _ = env.reset(seed=0)
        n_ops, n_mach = env.instance.num_operations, env.instance.num_machines
        assert obs["node_features"].shape == (n_ops, NUM_NODE_FEATURES)
        assert obs["machine_features"].shape == (n_mach, NUM_MACHINE_FEATURES)
        assert obs["adj_precedence"].shape == (n_ops, n_ops)
        assert obs["adj_machine"].shape == (n_ops, n_ops)
        assert obs["adj_eligibility"].shape == (n_ops, n_mach)
        assert obs["action_mask"].shape == (n_ops * n_mach,)
        assert env.observation_space.contains(obs)

    def test_action_space_is_the_flattened_pair_space(self) -> None:
        env = FJSPEnv("mk01")
        assert env.action_space.n == env.instance.num_operations * env.instance.num_machines

    def test_action_encoding_round_trips(self) -> None:
        env = FJSPEnv("mk01")
        for op in (0, 17, 59):
            for machine in range(env.instance.num_machines):
                assert env.decode_action(env.encode_action(op, machine)) == (op, machine)

    def test_eligibility_matrix_matches_the_instance(self) -> None:
        instance = load_fjsp_by_name("mk01")
        eligibility, durations = eligibility_matrices(instance)
        for job in range(instance.num_jobs):
            for pos in range(instance.ops_per_job):
                op = instance.op_id(job, pos)
                expected = instance.eligible_machines[job][pos]
                np.testing.assert_array_equal(
                    np.flatnonzero(eligibility[op]), sorted(expected)
                )
                for machine, duration in zip(expected, instance.durations[job][pos]):
                    assert durations[op, machine] == duration
                ineligible = set(range(instance.num_machines)) - set(expected)
                assert all(np.isinf(durations[op, m]) for m in ineligible)


class TestFJSPEnvMasking:
    def test_initial_mask_is_first_operation_of_each_job_times_its_eligible_set(self) -> None:
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        mask = env.action_masks().reshape(
            env.instance.num_operations, env.instance.num_machines
        )
        for job in range(env.instance.num_jobs):
            for pos in range(env.instance.ops_per_job):
                op = env.instance.op_id(job, pos)
                if pos == 0:
                    np.testing.assert_array_equal(
                        np.flatnonzero(mask[op]), sorted(env.instance.eligible_machines[job][0])
                    )
                else:
                    assert not mask[op].any()

    def test_mask_is_never_empty_before_termination(self) -> None:
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        terminated = False
        while not terminated:
            legal = np.flatnonzero(env.action_masks())
            assert legal.size > 0
            _, _, terminated, _, _ = env.step(int(rng.choice(legal)))

    def test_illegal_actions_raise(self) -> None:
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        with pytest.raises(ValueError, match="outside"):
            env.step(env.action_space.n)
        # Operation 1 is not yet ready (its job predecessor is unscheduled).
        with pytest.raises(ValueError, match="not yet ready"):
            env.step(env.encode_action(1, 0))
        # An ineligible machine for a ready operation.
        ineligible = [
            m
            for m in range(env.instance.num_machines)
            if m not in env.instance.eligible_machines[0][0]
        ]
        with pytest.raises(ValueError, match="not eligible"):
            env.step(env.encode_action(0, ineligible[0]))

    def test_observation_action_mask_matches_action_masks(self) -> None:
        env = FJSPEnv("mk01")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(1)
        np.testing.assert_array_equal(obs["action_mask"], env.action_masks().astype(np.int8))
        for _ in range(20):
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, _, _, _, _ = env.step(action)
            np.testing.assert_array_equal(obs["action_mask"], env.action_masks().astype(np.int8))


class TestFJSPEnvDynamics:
    def test_episode_length_equals_operation_count(self) -> None:
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        steps, terminated = 0, False
        while not terminated:
            _, _, terminated, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            steps += 1
        assert steps == env.instance.num_operations

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_every_random_rollout_is_feasible(self, seed: int) -> None:
        """Masking alone must guarantee a feasible schedule, whatever the policy does."""
        env = FJSPEnv("mk01")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        terminated, info = False, {}
        while not terminated:
            _, _, terminated, _, info = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
        assert info["schedule_feasible"], info["violations"]
        assert info["makespan"] >= env.static_lb

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reward_telescopes_to_h0_minus_makespan(self, seed: int) -> None:
        """``sum_t r_t = (H(s_0) - C_max) * reward_scale`` -- the property ``gamma=1`` needs."""
        env = FJSPEnv("mk01")
        _, info = env.reset(seed=seed)
        h0 = info["makespan_estimate"]
        rng = np.random.default_rng(seed)
        total_reward, terminated = 0.0, False
        while not terminated:
            _, reward, terminated, _, info = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            total_reward += reward
        expected = (h0 - info["makespan"]) * env.reward_scale
        assert total_reward == pytest.approx(expected, abs=1e-6)

    def test_rewards_are_non_positive(self) -> None:
        """``H`` is non-decreasing along an episode, so no step can be rewarded."""
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        terminated = False
        while not terminated:
            _, reward, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            assert reward <= 1e-9

    def test_machine_choice_changes_the_schedule(self) -> None:
        """The action's machine component is actually used, not silently ignored."""
        instance = load_fjsp_by_name("mk01")
        alternatives = instance.eligible_machines[0][0]
        assert len(alternatives) >= 2, "mk01's first operation should be flexible"

        ends = []
        for machine in alternatives[:2]:
            env = FJSPEnv(instance)
            env.reset(seed=0)
            env.step(env.encode_action(0, machine))
            ends.append((int(env.machine_of_op[0]), float(env.end_times[0])))
        assert ends[0][0] != ends[1][0]

    def test_zero_duration_padding_does_not_block_a_machine(self) -> None:
        """The neutrality claim in ``envs.fjsp_loader``'s docstring, checked directly."""
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        padded = np.flatnonzero(~real_operation_mask(env.instance))
        assert padded.size > 0
        # Drive one job to its padding operation and confirm machine 0 is untouched.
        job = env.instance.job_of(int(padded[0]))
        rng = np.random.default_rng(0)
        while env._job_next_pos[job] < env.instance.ops_per_job:
            legal = np.flatnonzero(env.action_masks())
            job_actions = [
                a for a in legal if env.instance.job_of(env.decode_action(int(a))[0]) == job
            ]
            free_before = env.machine_free_time.copy()
            op_id, _ = env.decode_action(int(job_actions[0]))
            is_padding = op_id in set(padded.tolist())
            env.step(int(rng.choice(job_actions)))
            if is_padding:
                np.testing.assert_array_equal(env.machine_free_time, free_before)

    def test_reset_clears_episode_state(self) -> None:
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        env.step(env.encode_action(0, env.instance.eligible_machines[0][0][0]))
        env.reset(seed=0)
        assert env.makespan() == 0.0
        assert not env.scheduled.any()
        assert (env.machine_of_op == -1).all()
        assert env.dispatch_order == []


class TestFJSPEnvContract:
    def test_scheduling_env_contract_methods(self) -> None:
        """The four methods ``baselines`` code is written against all work."""
        env = FJSPEnv("mk01")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        assert env.action_masks().dtype == bool
        assert env.makespan() > 0.0
        assert env.remaining_work_per_op().shape == (env.instance.num_operations,)
        records = env.schedule_dataframe_records()
        assert len(records) == 10
        assert set(records[0]) == {
            "op_id", "job", "position", "machine", "start", "end", "duration"
        }
        assert env.static_lb >= 1
        assert len(env.dispatch_order) == 10

    def test_lower_bound_is_valid_against_the_known_optimum(self) -> None:
        env = FJSPEnv("mk01")
        assert env.static_lb <= fjsp_known_optimum("mk01")
