"""Smoke tests for the GIN features extractor and the MaskablePPO wiring.

These are deliberately cheap (a handful of PPO update steps on the small ``ft06``
instance) -- the goal is catching shape/plumbing bugs fast in CI, not measuring
learning quality. The longer >=1000-timestep run lives in
``experiments/train_ft06.py`` and is reported in ``VALIDATION_WEEK1.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from envs.jssp_env import JSSPEnv
from models.policy import JSSPGraphFeaturesExtractor, build_maskable_ppo


class TestFeaturesExtractor:
    def test_forward_shape_matches_features_dim(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        extractor = JSSPGraphFeaturesExtractor(env.observation_space, hidden_dim=16, num_layers=2)

        batch = {
            key: torch.as_tensor(np.expand_dims(value, 0), dtype=torch.float32)
            for key, value in obs.items()
        }
        out = extractor(batch)
        assert out.shape == (1, extractor.features_dim)
        assert torch.isfinite(out).all()

    def test_handles_batch_dimension(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        extractor = JSSPGraphFeaturesExtractor(env.observation_space, hidden_dim=8, num_layers=2)

        batch = {
            key: torch.as_tensor(np.stack([value, value, value]), dtype=torch.float32)
            for key, value in obs.items()
        }
        out = extractor(batch)
        assert out.shape == (3, extractor.features_dim)


class TestMaskablePPOSmoke:
    def test_agent_constructs_and_predicts_legal_actions(self) -> None:
        env = JSSPEnv("ft06")
        model = build_maskable_ppo(
            env,
            hidden_dim=16,
            num_layers=2,
            n_steps=36,
            batch_size=36,
            verbose=0,
            seed=0,
        )
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        assert env.action_masks()[int(action)]

    def test_short_learn_call_runs_without_crashing(self) -> None:
        """A few PPO updates on ft06 -- exercises rollout, GAE, and the loss end to end."""
        env = JSSPEnv("ft06")
        model = build_maskable_ppo(
            env,
            hidden_dim=16,
            num_layers=2,
            n_steps=36,
            batch_size=36,
            n_epochs=1,
            verbose=0,
            seed=0,
        )
        model.learn(total_timesteps=72)  # 2 rollouts of one full ft06 episode each

    def test_full_episode_rollout_stays_feasible(self) -> None:
        """Greedy rollout under an (untrained) masked policy must still respect the
        env's own constraint checks -- masking guarantees legality regardless of the
        policy's learning progress.
        """
        env = JSSPEnv("ft06")
        model = build_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, verbose=0, seed=0
        )
        obs, _ = env.reset(seed=0)
        terminated = False
        info: dict = {}
        steps = 0
        while not terminated and steps < env.instance.num_operations:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated
        assert info["schedule_feasible"]
