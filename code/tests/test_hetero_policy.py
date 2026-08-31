"""Tests for the Week 18 heterogeneous encoder and size-agnostic actor/critic heads.

Covers the required-test list in ``GNN_PPO_Architecture_Spec_v2.md`` §6:

* ``test_hetero_encoder_output_shapes``
* ``test_per_node_actor_head_parameter_count_independent_of_n``
* ``test_weight_transfer_across_instance_sizes``
* ``test_action_masking_unchanged_semantics``
* ``test_rl_baseline_matches_protocol``

plus the FJSP per-pair head, which is this sprint's extension beyond that list.
``test_homogeneous_path_still_passes`` is not re-implemented here: it *is*
``tests/test_policy.py``, which must keep passing unmodified, and duplicating it would
create a second copy to keep in sync.

These are deliberately cheap (small ``hidden_dim``, a handful of PPO steps) --- the goal
is catching shape/plumbing bugs fast in CI, not measuring learning quality.  The long
runs live in ``experiments/train_hetero.py`` and are reported in
``TRAINING_REPORT_WEEK18.md``.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from baselines.interface import Baseline, TrainedRLBaseline
from envs.fjsp_env import FJSPEnv
from envs.fjsp_loader import load_fjsp_by_name
from envs.jssp_env import NUM_MACHINE_FEATURES, NUM_NODE_FEATURES, JSSPEnv
from envs.loader import load_by_name
from models.gin import HeteroGINEncoder, HeteroRelationalGATHybridLayer, HeteroRelationalGINLayer
from models.policy import (
    JSSPHeteroGraphFeaturesExtractor,
    PerNodeActorHead,
    PerPairActorHead,
    PerNodeMaskableActorCriticPolicy,
    PooledCriticHead,
    build_hetero_maskable_ppo,
)


def _batch(obs: dict[str, np.ndarray], repeat: int = 1) -> dict[str, torch.Tensor]:
    """Stack one observation into a batch of ``repeat`` identical tensors."""
    return {
        key: torch.as_tensor(np.stack([value] * repeat), dtype=torch.float32)
        for key, value in obs.items()
    }


class TestHeteroEncoder:
    @pytest.mark.parametrize("instance", ["ft06", "la06", "ft10"])
    def test_hetero_encoder_output_shapes(self, instance: str) -> None:
        """Spec Table 2.1 shapes, across several instance sizes (36 / 75 / 100 ops)."""
        env = JSSPEnv(instance)
        obs, _ = env.reset(seed=0)
        hidden_dim, batch = 16, 3
        encoder = HeteroGINEncoder(
            in_dim_ope=NUM_NODE_FEATURES,
            in_dim_mach=NUM_MACHINE_FEATURES,
            hidden_dim=hidden_dim,
            num_layers=3,
        )
        tensors = _batch(obs, batch)
        node_ope, node_mach, graph = encoder(
            tensors["node_features"],
            tensors["machine_features"],
            tensors["adj_precedence"],
            tensors["adj_machine"],
            tensors["adj_eligibility"],
        )
        n_ops = env.instance.num_operations
        n_mach = env.instance.num_machines
        assert node_ope.shape == (batch, n_ops, hidden_dim)
        assert node_mach.shape == (batch, n_mach, hidden_dim)
        assert graph.shape == (batch, 2 * hidden_dim)
        assert encoder.graph_dim == 2 * hidden_dim
        assert torch.isfinite(graph).all()

    def test_graph_embedding_is_concat_of_the_two_pools(self) -> None:
        """The critic's input really is ``concat(pool(ope), pool(mach))``, not one pool."""
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        encoder = HeteroGINEncoder(
            in_dim_ope=NUM_NODE_FEATURES, in_dim_mach=NUM_MACHINE_FEATURES, hidden_dim=8
        )
        tensors = _batch(obs)
        node_ope, node_mach, graph = encoder(
            tensors["node_features"],
            tensors["machine_features"],
            tensors["adj_precedence"],
            tensors["adj_machine"],
            tensors["adj_eligibility"],
        )
        expected = torch.cat([node_ope.mean(dim=1), node_mach.mean(dim=1)], dim=-1)
        torch.testing.assert_close(graph, expected)

    def test_eligibility_relation_actually_carries_signal(self) -> None:
        """Zeroing ``adj_eligibility`` must change the output.

        Directly guards the failure mode ``TrainingPlan_Week17.md`` §3.1 warns about: an
        all-zero or degenerate eligibility matrix silently turns the heterogeneous
        encoder into a homogeneous one with a dead branch, which trains *slower* than the
        homogeneous ablation while looking superficially fine.
        """
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        torch.manual_seed(0)
        encoder = HeteroGINEncoder(
            in_dim_ope=NUM_NODE_FEATURES, in_dim_mach=NUM_MACHINE_FEATURES, hidden_dim=8
        )
        tensors = _batch(obs)
        args = (
            tensors["node_features"],
            tensors["machine_features"],
            tensors["adj_precedence"],
            tensors["adj_machine"],
        )
        with_edges = encoder(*args, tensors["adj_eligibility"])[0]
        without_edges = encoder(*args, torch.zeros_like(tensors["adj_eligibility"]))[0]
        assert not torch.allclose(with_edges, without_edges)

    def test_layer_rejects_disabled_machine_edges_gracefully(self) -> None:
        """``use_machine_edges=False`` (the ablation) still runs and changes nothing else."""
        layer = HeteroRelationalGINLayer(
            in_dim_ope=NUM_NODE_FEATURES,
            in_dim_mach=NUM_MACHINE_FEATURES,
            out_dim_ope=8,
            out_dim_mach=8,
            use_machine_edges=False,
        )
        assert layer.lin_machine is None
        h_ope, h_mach = layer(
            torch.randn(2, 5, NUM_NODE_FEATURES),
            torch.randn(2, 3, NUM_MACHINE_FEATURES),
            torch.zeros(2, 5, 5),
            torch.zeros(2, 5, 5),
            torch.ones(2, 5, 3),
        )
        assert h_ope.shape == (2, 5, 8)
        assert h_mach.shape == (2, 3, 8)


class TestGATHybridEncoder:
    """``AblationPlan.md`` Axis 1b: attention on the machine-update term only."""

    @pytest.mark.parametrize("instance", ["ft06", "la06"])
    def test_gat_hybrid_encoder_output_shapes(self, instance: str) -> None:
        env = JSSPEnv(instance)
        obs, _ = env.reset(seed=0)
        hidden_dim, batch = 16, 3
        encoder = HeteroGINEncoder(
            in_dim_ope=NUM_NODE_FEATURES,
            in_dim_mach=NUM_MACHINE_FEATURES,
            hidden_dim=hidden_dim,
            num_layers=3,
            encoder_type="gat_hybrid",
        )
        tensors = _batch(obs, batch)
        node_ope, node_mach, graph = encoder(
            tensors["node_features"],
            tensors["machine_features"],
            tensors["adj_precedence"],
            tensors["adj_machine"],
            tensors["adj_eligibility"],
        )
        n_ops = env.instance.num_operations
        n_mach = env.instance.num_machines
        assert node_ope.shape == (batch, n_ops, hidden_dim)
        assert node_mach.shape == (batch, n_mach, hidden_dim)
        assert graph.shape == (batch, 2 * hidden_dim)
        assert torch.isfinite(graph).all()

    def test_gat_hybrid_rejects_unknown_encoder_type(self) -> None:
        with pytest.raises(ValueError, match="encoder_type"):
            HeteroGINEncoder(
                in_dim_ope=NUM_NODE_FEATURES,
                in_dim_mach=NUM_MACHINE_FEATURES,
                encoder_type="attention",
            )

    def test_gat_hybrid_attention_ignores_ineligible_operations(self) -> None:
        """Zeroing a machine's ineligible-operation rows must not change its output ---
        the softmax should already have assigned them ~zero weight via the -1e9 mask."""
        torch.manual_seed(0)
        layer = HeteroRelationalGATHybridLayer(
            in_dim_ope=NUM_NODE_FEATURES,
            in_dim_mach=NUM_MACHINE_FEATURES,
            out_dim_ope=8,
            out_dim_mach=8,
        )
        x_ope = torch.randn(1, 6, NUM_NODE_FEATURES)
        x_mach = torch.randn(1, 2, NUM_MACHINE_FEATURES)
        adj_eligibility = torch.zeros(1, 6, 2)
        adj_eligibility[0, :3, 0] = 1.0  # ops 0-2 eligible only for machine 0
        adj_eligibility[0, 3:, 1] = 1.0  # ops 3-5 eligible only for machine 1
        adj_precedence = torch.zeros(1, 6, 6)
        adj_machine = torch.zeros(1, 6, 6)

        _, h_mach_before = layer(x_ope, x_mach, adj_precedence, adj_machine, adj_eligibility)

        x_ope_perturbed = x_ope.clone()
        x_ope_perturbed[0, 3:] += 100.0  # perturb only ops ineligible for machine 0
        _, h_mach_after = layer(
            x_ope_perturbed, x_mach, adj_precedence, adj_machine, adj_eligibility
        )
        torch.testing.assert_close(h_mach_before[0, 0], h_mach_after[0, 0])
        assert not torch.allclose(h_mach_before[0, 1], h_mach_after[0, 1])

    def test_build_hetero_maskable_ppo_accepts_gat_hybrid_encoder(self) -> None:
        env = JSSPEnv("ft06")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=8, num_layers=2, encoder_type="gat_hybrid", n_steps=32, seed=0
        )
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        assert env.action_masks()[int(action)]


class TestActorCriticHeads:
    def test_per_node_actor_head_parameter_count_independent_of_n(self) -> None:
        """Spec §6: one head, two instance sizes, identical parameter count and valid output."""
        head = PerNodeActorHead(in_dim=16, hidden_dim=16, num_layers=2)
        before = sum(p.numel() for p in head.parameters())
        for n_ope in (36, 100):  # ft06, ft10
            logits = head(torch.randn(2, n_ope, 16))
            assert logits.shape == (2, n_ope)
        after = sum(p.numel() for p in head.parameters())
        assert before == after

    def test_per_pair_actor_head_parameter_count_independent_of_n_and_m(self) -> None:
        """The FJSP head is size-agnostic in *both* dimensions."""
        head = PerPairActorHead(ope_dim=16, mach_dim=16, hidden_dim=16, num_layers=2)
        before = sum(p.numel() for p in head.parameters())
        for n_ope, n_mach in ((60, 6), (120, 15)):
            logits = head(torch.randn(2, n_ope, 16), torch.randn(2, n_mach, 16))
            assert logits.shape == (2, n_ope * n_mach)
        assert sum(p.numel() for p in head.parameters()) == before

    def test_per_pair_head_ordering_matches_env_action_encoding(self) -> None:
        """Logit index ``op * num_machines + machine`` must be that pair's score.

        A transposed reshape here would silently pair every logit with the wrong action
        while keeping all shapes valid --- the exact class of masking bug
        ``TrainingPlan_Week17.md`` §3.1 says would corrupt gradients without crashing.
        """
        head = PerPairActorHead(ope_dim=4, mach_dim=4, hidden_dim=8, num_layers=2)
        n_ope, n_mach = 5, 3
        ope = torch.randn(1, n_ope, 4)
        mach = torch.randn(1, n_mach, 4)
        flat = head(ope, mach)[0]
        for op in range(n_ope):
            for machine in range(n_mach):
                pair_logit = head.mlp(
                    torch.cat([ope[0, op], mach[0, machine]], dim=-1)
                ).squeeze(-1)
                torch.testing.assert_close(flat[op * n_mach + machine], pair_logit)

    def test_pooled_critic_head_output_shape(self) -> None:
        head = PooledCriticHead(in_dim=32, hidden_dim=16, num_layers=2)
        assert head(torch.randn(7, 32)).shape == (7,)


class TestHeteroFeaturesExtractor:
    def test_returns_pooled_graph_and_stashes_node_embeddings(self) -> None:
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        extractor = JSSPHeteroGraphFeaturesExtractor(
            env.observation_space, hidden_dim=16, num_layers=2
        )
        out = extractor(_batch(obs, 4))
        assert out.shape == (4, extractor.features_dim) == (4, 32)
        assert extractor.last_node_embeddings_ope.shape == (4, 36, 16)
        assert extractor.last_node_embeddings_mach.shape == (4, 6, 16)
        assert torch.isfinite(out).all()

    def test_features_dim_is_independent_of_instance_size(self) -> None:
        """Unlike ``JSSPGraphFeaturesExtractor``'s ``(num_ops + 1) * hidden_dim``."""
        small = JSSPHeteroGraphFeaturesExtractor(JSSPEnv("ft06").observation_space, hidden_dim=16)
        large = JSSPHeteroGraphFeaturesExtractor(JSSPEnv("ft10").observation_space, hidden_dim=16)
        assert small.features_dim == large.features_dim

    def test_rejects_a_homogeneous_observation_space(self) -> None:
        from gymnasium import spaces  # noqa: PLC0415

        space = spaces.Dict({"node_features": spaces.Box(0, 1, shape=(4, NUM_NODE_FEATURES))})
        with pytest.raises(KeyError, match="missing heterogeneous key"):
            JSSPHeteroGraphFeaturesExtractor(space)


class TestHeteroPolicy:
    def test_constructs_and_predicts_legal_actions_jssp(self) -> None:
        env = JSSPEnv("ft06")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0, gamma=1.0
        )
        assert isinstance(model.policy, PerNodeMaskableActorCriticPolicy)
        assert isinstance(model.policy.action_net, PerNodeActorHead)
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        assert env.action_masks()[int(action)]

    def test_constructs_and_predicts_legal_actions_fjsp(self) -> None:
        env = FJSPEnv("mk01")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=60, batch_size=60, seed=0, gamma=1.0
        )
        assert isinstance(model.policy.action_net, PerPairActorHead)
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        assert env.action_masks()[int(action)]

    def test_rejects_an_action_space_matching_neither_variant(self) -> None:
        from gymnasium import spaces  # noqa: PLC0415

        env = JSSPEnv("ft06")
        with pytest.raises(ValueError, match="matches neither"):
            PerNodeMaskableActorCriticPolicy(
                env.observation_space, spaces.Discrete(7), lambda _: 3e-4
            )

    @pytest.mark.parametrize(
        ("factory", "instance", "n_steps"),
        [(JSSPEnv, "ft06", 36), (FJSPEnv, "mk01", 60)],
    )
    def test_short_learn_call_runs_without_crashing(self, factory, instance, n_steps) -> None:
        """A few PPO updates -- exercises rollout, GAE and the masked loss end to end."""
        env = factory(instance)
        model = build_hetero_maskable_ppo(
            env,
            hidden_dim=16,
            num_layers=2,
            n_steps=n_steps,
            batch_size=n_steps,
            n_epochs=1,
            seed=0,
            gamma=1.0,
        )
        model.learn(total_timesteps=2 * n_steps)

    @pytest.mark.parametrize(
        ("factory", "instance"), [(JSSPEnv, "ft06"), (FJSPEnv, "mk01")]
    )
    def test_full_episode_rollout_stays_feasible(self, factory, instance) -> None:
        """Masking guarantees legality regardless of the policy's learning progress."""
        env = factory(instance)
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=32, batch_size=32, seed=0, gamma=1.0
        )
        obs, _ = env.reset(seed=0)
        terminated, info, steps = False, {}, 0
        while not terminated and steps <= env.instance.num_operations:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, _, terminated, _, info = env.step(int(action))
            steps += 1
        assert terminated
        assert info["schedule_feasible"], info.get("violations")

    def test_weight_transfer_across_instance_sizes(self) -> None:
        """Spec §6: ``ft06`` weights load into a policy wrapping ``la06`` (N = 75)."""
        source = build_hetero_maskable_ppo(
            JSSPEnv("ft06"), hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        source.learn(total_timesteps=36)  # a few gradient steps: prove the mechanism
        target = build_hetero_maskable_ppo(
            JSSPEnv("la06"), hidden_dim=16, num_layers=2, n_steps=75, batch_size=75, seed=1
        )
        target.policy.load_state_dict(source.policy.state_dict(), strict=True)

        env = JSSPEnv("la06")
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        distribution = target.policy.get_distribution(_batch(obs), action_masks=mask)
        probs = distribution.distribution.probs[0].detach().numpy()

        assert probs.shape == (env.instance.num_operations,)
        assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)
        assert math.isclose(float(probs[mask].sum()), 1.0, rel_tol=1e-5)
        assert float(probs[~mask].sum()) == pytest.approx(0.0, abs=1e-7)

    def test_action_masking_unchanged_semantics(self) -> None:
        """Spec §6: the per-node head masks the same action set the flat head did.

        Both policies are built against the same env and seed; the assertion is on the
        *support* of the resulting distribution, which is what masking governs --- not on
        the probabilities, which two different architectures have no reason to share.
        """
        from models.policy import build_maskable_ppo  # noqa: PLC0415

        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        for _ in range(5):  # advance to a partially-scheduled state with a smaller frontier
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        obs = env._observation()
        mask = env.action_masks()

        flat = build_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        per_node = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        batch = _batch(obs)
        flat_probs = flat.policy.get_distribution(batch, action_masks=mask).distribution.probs
        node_probs = per_node.policy.get_distribution(batch, action_masks=mask).distribution.probs

        flat_support = (flat_probs[0] > 0).numpy()
        node_support = (node_probs[0] > 0).numpy()
        np.testing.assert_array_equal(flat_support, mask)
        np.testing.assert_array_equal(node_support, mask)

    def test_initial_policy_is_near_uniform_over_legal_actions(self) -> None:
        """Guards the entropy-collapse-at-init failure mode (plan §3.2 cause (a)).

        A fresh policy's entropy should sit just below ``ln(number of legal actions)``;
        an actor head initialised at a large logit scale would start far below it and
        the run would look like an instant collapse.
        """
        env = JSSPEnv("ft06")
        obs, _ = env.reset(seed=0)
        model = build_hetero_maskable_ppo(
            env, hidden_dim=32, num_layers=3, n_steps=36, batch_size=36, seed=0
        )
        mask = env.action_masks()
        distribution = model.policy.get_distribution(_batch(obs), action_masks=mask)
        entropy = float(distribution.entropy().detach())
        assert entropy == pytest.approx(math.log(int(mask.sum())), rel=0.05)

    def test_save_load_round_trip_preserves_actions(self, tmp_path) -> None:
        """``_get_constructor_parameters`` must carry the head kwargs, or load() breaks."""
        from sb3_contrib import MaskablePPO  # noqa: PLC0415

        env = JSSPEnv("ft06")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        path = tmp_path / "policy.zip"
        model.save(path)
        reloaded = MaskablePPO.load(path)

        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        before, _ = model.predict(obs, action_masks=mask, deterministic=True)
        after, _ = reloaded.predict(obs, action_masks=mask, deterministic=True)
        assert int(before) == int(after)


class TestTrainedRLBaseline:
    def test_rl_baseline_matches_protocol(self, tmp_path) -> None:
        """Spec §6/§8: the adapter satisfies ``Baseline`` around an untrained policy."""
        env = JSSPEnv("ft06")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        path = tmp_path / "policy.zip"
        model.save(path)

        baseline = TrainedRLBaseline(policy_path=str(path))
        # ``Baseline`` is a plain (non-runtime_checkable) Protocol, so conformance is
        # asserted structurally rather than with isinstance -- the same way every other
        # adapter is exercised in tests/test_baselines_interface.py.
        assert set(inspect.signature(Baseline.solve).parameters) == set(
            inspect.signature(baseline.solve).parameters
        ) | {"self"}
        assert baseline.name == "rl"

        instance = load_by_name("ft06")
        result = baseline.solve(instance, seed=0, time_limit_s=5.0)
        assert result.method == "rl"
        assert result.instance == "ft06"
        assert result.start_times.shape == (instance.num_operations,)
        assert result.feasible, result.violations
        assert result.status == "OK"
        assert not result.is_optimal
        assert result.makespan >= 55  # ft06's proven optimum: no schedule can beat it
        assert result.wall_clock_time > 0.0

    def test_rl_baseline_handles_fjsp_instances(self, tmp_path) -> None:
        env = FJSPEnv("mk01")
        model = build_hetero_maskable_ppo(
            env, hidden_dim=16, num_layers=2, n_steps=60, batch_size=60, seed=0
        )
        path = tmp_path / "policy_fjsp.zip"
        model.save(path)

        result = TrainedRLBaseline(policy_path=str(path)).solve(
            load_fjsp_by_name("mk01"), seed=0
        )
        assert result.feasible, result.violations
        assert result.makespan >= 40  # mk01's proven optimum
        assert "machine_of_op" in result.extra

    def test_rl_baseline_rejects_a_size_mismatched_policy(self, tmp_path) -> None:
        """A JSSP-trained policy must refuse an FJSP instance loudly, not crash on shapes."""
        model = build_hetero_maskable_ppo(
            JSSPEnv("ft06"), hidden_dim=16, num_layers=2, n_steps=36, batch_size=36, seed=0
        )
        path = tmp_path / "policy.zip"
        model.save(path)
        with pytest.raises(ValueError, match="actor head"):
            TrainedRLBaseline(policy_path=str(path)).solve(load_fjsp_by_name("mk01"), seed=0)
