"""Tests for the heterogeneous observation keys added to :class:`~envs.jssp_env.JSSPEnv`.

Kept separate from ``tests/test_env.py`` rather than merged into it: the Week 18 change
is purely additive (``machine_features`` and ``adj_eligibility`` on top of an untouched
observation), and ``tests/test_env.py`` passing *unmodified* is itself the regression
evidence that the addition changed nothing. Merging the new assertions in would destroy
that evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.jssp_env import (
    MACHINE_FEATURE_NAMES,
    NUM_MACHINE_FEATURES,
    NUM_NODE_FEATURES,
    JSSPEnv,
)


class TestConstants:
    def test_machine_feature_names_match_the_declared_width(self) -> None:
        assert len(MACHINE_FEATURE_NAMES) == NUM_MACHINE_FEATURES == 7

    def test_operation_feature_width_is_unchanged(self) -> None:
        """The additive contract: machine nodes did not touch the operation node width."""
        assert NUM_NODE_FEATURES == 8


class TestHeterogeneousObservationKeys:
    @pytest.mark.parametrize("instance", ["ft06", "la06", "ft10"])
    def test_shapes_and_space_membership(self, instance: str) -> None:
        env = JSSPEnv(instance)
        obs, _ = env.reset(seed=0)
        n_ops, n_mach = env.instance.num_operations, env.instance.num_machines
        assert obs["machine_features"].shape == (n_mach, NUM_MACHINE_FEATURES)
        assert obs["adj_eligibility"].shape == (n_ops, n_mach)
        assert env.observation_space.contains(obs)

    def test_original_keys_are_still_present_and_unchanged_in_shape(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        n = env.instance.num_operations
        assert obs["node_features"].shape == (n, NUM_NODE_FEATURES)
        assert obs["adj_precedence"].shape == (n, n)
        assert obs["adj_machine"].shape == (n, n)
        assert obs["action_mask"].shape == (n,)

    def test_eligibility_is_one_hot_and_matches_the_instance_routing(self) -> None:
        """Under fixed routing every operation has exactly one eligible machine."""
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        eligibility = obs["adj_eligibility"]
        np.testing.assert_array_equal(eligibility.sum(axis=1), np.ones(env.instance.num_operations))
        np.testing.assert_array_equal(
            eligibility.argmax(axis=1), env.instance.flat_machines
        )

    def test_eligibility_is_structural_and_never_changes(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        initial = obs["adj_eligibility"].copy()
        for _ in range(15):
            obs, _, _, _, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        np.testing.assert_array_equal(obs["adj_eligibility"], initial)


class TestMachineFeatures:
    def test_initial_state_every_machine_free_and_fully_loaded(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        features = obs["machine_features"]
        np.testing.assert_array_equal(features[:, 0], 1.0)  # is_free
        np.testing.assert_array_equal(features[:, 1], 0.0)  # is_busy
        np.testing.assert_array_equal(features[:, 3], 0.0)  # time_to_available
        np.testing.assert_array_equal(features[:, 5], 0.0)  # utilization_accum at t=0
        # assigned_pending_load: the whole instance is unscheduled, so each machine's
        # pending load is exactly its total load.
        expected = env.instance.machine_totals / env.instance.total_processing_time
        np.testing.assert_allclose(features[:, 4], expected, rtol=1e-5)

    def test_status_columns_are_a_one_hot(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs, _, _, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            status = obs["machine_features"][:, 0:3]
            np.testing.assert_allclose(status.sum(axis=1), 1.0, atol=1e-6)

    def test_static_shop_columns_are_constant(self) -> None:
        """``is_broken`` stays 0 and ``estimated_availability`` stays 1 with no breakdowns."""
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(1)
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            np.testing.assert_array_equal(obs["machine_features"][:, 2], 0.0)
            np.testing.assert_array_equal(obs["machine_features"][:, 6], 1.0)

    def test_pending_load_drains_to_zero_over_an_episode(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(2)
        previous = obs["machine_features"][:, 4].sum()
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            current = obs["machine_features"][:, 4].sum()
            assert current <= previous + 1e-6  # monotone non-increasing
            previous = current
        np.testing.assert_allclose(obs["machine_features"][:, 4], 0.0, atol=1e-6)

    def test_features_are_finite_and_non_negative_throughout(self) -> None:
        env = JSSPEnv("la06")
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(3)
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            features = obs["machine_features"]
            assert np.isfinite(features).all()
            assert (features >= 0.0).all()

    def test_utilization_never_exceeds_one(self) -> None:
        """``busy_so_far / t`` is a utilisation ratio; > 1 would mean a bookkeeping bug."""
        env = JSSPEnv("ft06")
        env.reset(seed=0)
        rng = np.random.default_rng(4)
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            assert (obs["machine_features"][:, 5] <= 1.0 + 1e-6).all()


class TestCurrentTime:
    def test_starts_at_zero(self) -> None:
        env = JSSPEnv("ft06")
        env.reset(seed=0)
        assert env.current_time() == 0.0

    def test_reports_the_makespan_once_nothing_is_left(self) -> None:
        env = JSSPEnv("ft06")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
        assert env.current_time() == env.makespan()

    def test_never_exceeds_the_realised_makespan_mid_episode(self) -> None:
        env = JSSPEnv("ft06")
        env.reset(seed=0)
        rng = np.random.default_rng(5)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(
                int(rng.choice(np.flatnonzero(env.action_masks())))
            )
            assert 0.0 <= env.current_time() <= env.makespan() + 1e-6
