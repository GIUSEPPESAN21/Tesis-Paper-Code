"""Wires :class:`~models.gin.GINEncoder` into an SB3-family actor-critic PPO policy.

Why ``sb3_contrib.MaskablePPO`` instead of plain ``stable_baselines3.PPO``
---------------------------------------------------------------------------
:class:`~envs.jssp_env.JSSPEnv` masks illegal actions (see ``action_masks()``): at most
one operation per job is ever dispatchable, so on a 6-job instance only ~6 of the
``Discrete(N)`` actions are legal at any step. Vanilla SB3 ``PPO`` has no first-class
mechanism to exclude actions from the categorical distribution before sampling; the
usual workarounds (large negative reward on illegal actions, or resampling) either
corrupt the on-policy gradient estimate or waste rollout steps. ``sb3_contrib``
(maintained by the same team as ``stable_baselines3``) adds exactly one thing on top of
``PPO``: a ``MaskableCategorical`` distribution that zeroes out illegal-action
probability *before* sampling and before computing the log-prob used in the PPO loss.
It auto-discovers masks by calling ``env.action_masks()`` -- which
:class:`~envs.jssp_env.JSSPEnv` already implements for exactly this purpose -- so no
extra wrapper is needed. This keeps us inside "reuse a reliable, tested library" while
still getting correct masked action selection, which is not optional for this MDP (an
unmasked policy would sample violations of job precedence constantly).

Why a custom features extractor instead of the default ``MultiInputPolicy`` CNN/MLP mix
------------------------------------------------------------------------------------------
SB3's built-in ``CombinedExtractor`` (used by ``MultiInputPolicy``) processes each key of
a ``Dict`` observation independently and concatenates the results. That is wrong here:
``node_features`` only becomes meaningful jointly with ``adj_precedence`` and
``adj_machine`` -- message passing is the entire point. :class:`JSSPGraphFeaturesExtractor`
instead receives the whole dict in one ``forward`` call and runs it through
:class:`~models.gin.GINEncoder`, exactly as a graph encoder should.

Fixed graph size (documented limitation)
-----------------------------------------
``features_dim`` is computed from ``observation_space["node_features"].shape[0]`` at
construction time, i.e. from one specific instance's operation count. This policy is
therefore tied to a single instance size, which is fine for the Week 1 goal (train and
smoke-test on ``ft06``) but will not transfer across instances of different sizes.
Instance-agnostic transfer needs a *per-node* action head (e.g. a shared linear scorer
applied to each node embedding, as in Zhang et al. 2020's L2D) instead of the flattened
node-embeddings-as-features approach used here, and is left for a later week once basic
correctness of the pipeline is established.

Week 18: the size-agnostic heterogeneous path (this file's second half)
------------------------------------------------------------------------
:class:`JSSPHeteroGraphFeaturesExtractor` + :class:`PerNodeMaskableActorCriticPolicy`
resolve exactly the limitation the paragraph above documents, per
``GNN_PPO_Architecture_Spec_v2.md`` §3.  Everything above them is unchanged and remains
the pipeline's homogeneous ablation configuration --- ``tests/test_policy.py`` passes
unmodified.

The one structural obstacle is that SB3's ``BaseFeaturesExtractor`` contract is "return
one flat ``(B, features_dim)`` tensor", while a per-node action head needs the *per-node*
embeddings, which are ``(B, N, hidden_dim)``.  The standard escape hatch --- and what
§3.3 of the spec specifies --- is for the extractor to return the pooled graph embedding
as its nominal features (satisfying the contract, and exactly what the critic wants
anyway) while stashing the per-node tensors as attributes for the policy's action head to
read.  ``MaskableCategorical`` never notices: it still receives one ``(B, n_actions)``
logits tensor and one ``(B, n_actions)`` mask, exactly as before.

Verified against the installed ``sb3_contrib==2.9.0`` / ``stable_baselines3==2.9.0``
(the versions ``requirements.txt`` pins), as the spec's §3.3 requires: in this release the
single place a policy turns latent features into action logits is
``MaskableActorCriticPolicy._get_action_dist_from_latent``, called from ``forward``,
``evaluate_actions`` and ``get_distribution`` --- all three of which call
``extract_features`` first, so the stashed tensors are always the ones belonging to the
observation being scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.gin import GINEncoder, HeteroGINEncoder, build_mlp


class JSSPGraphFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 features extractor wrapping :class:`~models.gin.GINEncoder`.

    Args:
        observation_space: The environment's ``Dict`` observation space; must have
            ``node_features``, ``adj_precedence`` and ``adj_machine`` keys shaped as
            produced by :class:`~envs.jssp_env.JSSPEnv`.
        hidden_dim: Width of every GIN layer; also the width of the pooled graph
            embedding appended to the flattened node embeddings.
        num_layers: Number of GIN message-passing rounds.
        mlp_layers: Depth of the MLP inside each GIN layer.
        use_machine_edges: Passed through to :class:`~models.gin.GINEncoder`.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_layers: int = 2,
        use_machine_edges: bool = True,
    ) -> None:
        num_ops, in_dim = observation_space["node_features"].shape
        # Flattened per-node embeddings + one pooled graph embedding.
        features_dim = (num_ops + 1) * hidden_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.num_ops = int(num_ops)
        self.hidden_dim = hidden_dim
        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            use_machine_edges=use_machine_edges,
            pool="mean",
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of observations into a flat feature vector.

        Args:
            observations: Dict of batched tensors as produced by SB3's observation
                preprocessing, with ``node_features`` shape ``(B, N, F)`` and
                ``adj_precedence``/``adj_machine`` shape ``(B, N, N)``.

        Returns:
            Tensor of shape ``(B, features_dim)``: flattened node embeddings
            concatenated with the pooled graph embedding.
        """
        node_h, graph_h = self.encoder(
            observations["node_features"],
            observations["adj_precedence"],
            observations["adj_machine"],
        )
        return torch.cat([node_h.reshape(node_h.shape[0], -1), graph_h], dim=-1)


def build_maskable_ppo(
    env: Any,
    hidden_dim: int = 64,
    num_layers: int = 3,
    mlp_layers: int = 2,
    net_arch: dict[str, list[int]] | None = None,
    seed: int | None = None,
    **ppo_kwargs: Any,
) -> MaskablePPO:
    """Construct a :class:`~sb3_contrib.MaskablePPO` agent with the GIN feature extractor.

    Args:
        env: A :class:`~envs.jssp_env.JSSPEnv` (or a ``VecEnv`` of them, all sharing the
            same instance size). Action masking is auto-discovered from ``env.action_masks()``.
        hidden_dim: GIN hidden width, forwarded to :class:`JSSPGraphFeaturesExtractor`.
        num_layers: GIN depth, forwarded to :class:`JSSPGraphFeaturesExtractor`.
        mlp_layers: Depth of each GIN layer's internal MLP.
        net_arch: Actor/critic MLP head sizes applied on top of the extracted graph
            features, e.g. ``dict(pi=[128, 64], vf=[128, 64])``. Defaults to a single
            128-unit hidden layer for both heads.
        seed: RNG seed forwarded to ``MaskablePPO`` for reproducibility.
        **ppo_kwargs: Extra keyword arguments forwarded to ``MaskablePPO`` (e.g.
            ``n_steps``, ``batch_size``, ``learning_rate``, ``verbose``).

    Returns:
        A ``MaskablePPO`` instance ready for ``.learn(...)``.
    """
    policy_kwargs: dict[str, Any] = dict(
        features_extractor_class=JSSPGraphFeaturesExtractor,
        features_extractor_kwargs=dict(
            hidden_dim=hidden_dim, num_layers=num_layers, mlp_layers=mlp_layers
        ),
        net_arch=net_arch if net_arch is not None else dict(pi=[128], vf=[128]),
    )
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        **ppo_kwargs,
    )


# ===========================================================================================
# Week 18: heterogeneous encoder + size-agnostic actor/critic heads
# ===========================================================================================


class PerNodeActorHead(nn.Module):
    """Score every *operation* node with one shared MLP: ``(B, N, D) -> (B, N)``.

    The JSSP action head.  The **same** module instance (the same weights) is applied to
    every operation node, so the parameter count is independent of ``N`` --- which is
    what makes a ``state_dict`` trained on ``ft06`` (``N = 36``) loadable into a policy
    wrapping ``la06`` (``N = 75``) without reshaping anything
    (``GNN_PPO_Architecture_Spec_v2.md`` §3.1/§3.4).

    Args:
        in_dim: Width of an operation-node embedding (the encoder's ``hidden_dim``).
        hidden_dim: Hidden width of the scoring MLP.
        num_layers: Depth of the scoring MLP.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim=1, num_layers=num_layers)

    def forward(
        self, node_embeddings_ope: torch.Tensor, node_embeddings_mach: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Produce one logit per operation node.

        Args:
            node_embeddings_ope: Shape ``(B, N_ope, in_dim)``.
            node_embeddings_mach: Accepted and ignored, so this head is call-compatible
                with :class:`PerPairActorHead` and the policy below can hold either
                behind one call signature.

        Returns:
            Logits of shape ``(B, N_ope)``.
        """
        del node_embeddings_mach
        return self.mlp(node_embeddings_ope).squeeze(-1)


class PerPairActorHead(nn.Module):
    """Score every ``(operation, machine)`` pair: ``(B, N, D), (B, M, D) -> (B, N * M)``.

    The FJSP action head --- the machine-selection component
    ``baselines/interface.py`` and ``GNN_PPO_Architecture_Spec_v2.md`` §0 both name as
    the missing piece that kept FJSP out of the Week 17 sprint.  A pair's logit is
    ``MLP([h_ope_i ; h_mach_k])``, with one shared MLP over all ``N * M`` pairs, so like
    :class:`PerNodeActorHead` the parameter count is independent of the instance size ---
    of ``N_ope`` *and* of ``N_mach``.

    Output ordering is ``pair_id = op_id * num_machines + machine``, matching
    :meth:`~envs.fjsp_env.FJSPEnv.encode_action` exactly, so
    ``MaskableCategorical``'s logits line up index-for-index with
    ``FJSPEnv.action_masks()``.

    Args:
        ope_dim: Width of an operation-node embedding.
        mach_dim: Width of a machine-node embedding.
        hidden_dim: Hidden width of the scoring MLP.
        num_layers: Depth of the scoring MLP.
    """

    def __init__(
        self,
        ope_dim: int = 64,
        mach_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = build_mlp(ope_dim + mach_dim, hidden_dim, out_dim=1, num_layers=num_layers)

    def forward(
        self, node_embeddings_ope: torch.Tensor, node_embeddings_mach: torch.Tensor
    ) -> torch.Tensor:
        """Produce one logit per ``(operation, machine)`` pair.

        Args:
            node_embeddings_ope: Shape ``(B, N_ope, ope_dim)``.
            node_embeddings_mach: Shape ``(B, N_mach, mach_dim)``.

        Returns:
            Logits of shape ``(B, N_ope * N_mach)``, flattened operation-major.
        """
        batch, n_ope, _ = node_embeddings_ope.shape
        n_mach = node_embeddings_mach.shape[1]
        ope = node_embeddings_ope.unsqueeze(2).expand(-1, -1, n_mach, -1)
        mach = node_embeddings_mach.unsqueeze(1).expand(-1, n_ope, -1, -1)
        pair = torch.cat([ope, mach], dim=-1)
        return self.mlp(pair).squeeze(-1).reshape(batch, n_ope * n_mach)


class PooledCriticHead(nn.Module):
    """State-value readout over the concatenated pooled embeddings: ``(B, 2H) -> (B,)``.

    ``in_dim`` is ``2 * hidden_dim``: the concatenation of the pooled operation and
    pooled machine embeddings :class:`~models.gin.HeteroGINEncoder` returns.  The critic
    therefore sees both a "how much work is left" signal and a "how loaded/available are
    the machines" signal; the Sprint-1 homogeneous critic could only see the first.

    Args:
        in_dim: Width of the graph embedding (``2 * hidden_dim``).
        hidden_dim: Hidden width of the value MLP.
        num_layers: Depth of the value MLP.
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim=1, num_layers=num_layers)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Estimate ``V(s)``.

        Args:
            graph_embedding: Shape ``(B, in_dim)``.

        Returns:
            Values of shape ``(B,)``.
        """
        return self.mlp(graph_embedding).squeeze(-1)


class _ValueNetAdapter(nn.Module):
    """Wrap :class:`PooledCriticHead` to return ``(B, 1)``, the shape SB3's ``value_net``
    contract implies (it is an ``nn.Linear(latent_dim_vf, 1)`` by default and every
    caller either flattens or indexes ``[:, 0]``).  Kept separate so
    :class:`PooledCriticHead` can expose the ``(B,)`` signature the architecture spec
    specifies without either the spec or SB3 having to bend.
    """

    def __init__(self, critic: PooledCriticHead) -> None:
        super().__init__()
        self.critic = critic

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Return ``V(s)`` with a trailing singleton dimension."""
        return self.critic(graph_embedding).unsqueeze(-1)


class JSSPHeteroGraphFeaturesExtractor(BaseFeaturesExtractor):
    """Heterogeneous graph encoder as an SB3 features extractor.

    Returns the pooled graph embedding as its nominal ``(B, features_dim)`` features ---
    which is exactly what :class:`PooledCriticHead` consumes --- and stashes the two
    per-node embedding tensors on :attr:`last_node_embeddings_ope` /
    :attr:`last_node_embeddings_mach` for :class:`PerNodeMaskableActorCriticPolicy`'s
    action head to read.  See this module's docstring for why the stash is necessary and
    why it is safe.

    ``features_dim`` is ``2 * hidden_dim`` and therefore **independent of the instance
    size**, unlike :class:`JSSPGraphFeaturesExtractor`'s ``(num_ops + 1) * hidden_dim``.

    Args:
        observation_space: The environment's ``Dict`` observation space; must have
            ``node_features``, ``machine_features``, ``adj_precedence``, ``adj_machine``
            and ``adj_eligibility`` keys, as produced by
            :class:`~envs.jssp_env.JSSPEnv` and :class:`~envs.fjsp_env.FJSPEnv`.
        hidden_dim: Width of every GIN layer, both node types.
        num_layers: Number of message-passing rounds.
        mlp_layers: Depth of the MLP inside each GIN layer.
        use_machine_edges: Passed through to :class:`~models.gin.HeteroGINEncoder`.
        encoder_type: ``"gin"`` (default) or ``"gat_hybrid"`` (``AblationPlan.md`` Axis
            1b), forwarded to :class:`~models.gin.HeteroGINEncoder`.

    Attributes:
        last_node_embeddings_ope: ``(B, N_ope, hidden_dim)`` from the most recent
            ``forward``.
        last_node_embeddings_mach: ``(B, N_mach, hidden_dim)`` from the most recent
            ``forward``.

    Raises:
        KeyError: If the observation space is missing a heterogeneous key --- i.e. if it
            came from an environment predating ``NUM_MACHINE_FEATURES``.
    """

    REQUIRED_KEYS = (
        "node_features",
        "machine_features",
        "adj_precedence",
        "adj_machine",
        "adj_eligibility",
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_layers: int = 2,
        use_machine_edges: bool = True,
        encoder_type: str = "gin",
    ) -> None:
        missing = [key for key in self.REQUIRED_KEYS if key not in observation_space.spaces]
        if missing:
            raise KeyError(
                f"observation space is missing heterogeneous key(s) {missing}. "
                "JSSPHeteroGraphFeaturesExtractor needs an environment that emits machine "
                "nodes and eligibility edges (envs.jssp_env.JSSPEnv or envs.fjsp_env.FJSPEnv)."
            )
        num_ops, in_dim_ope = observation_space["node_features"].shape
        num_machines, in_dim_mach = observation_space["machine_features"].shape
        super().__init__(observation_space, features_dim=2 * hidden_dim)

        self.num_ops = int(num_ops)
        self.num_machines = int(num_machines)
        self.hidden_dim = int(hidden_dim)
        self.encoder = HeteroGINEncoder(
            in_dim_ope=in_dim_ope,
            in_dim_mach=in_dim_mach,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            use_machine_edges=use_machine_edges,
            pool="mean",
            encoder_type=encoder_type,
        )
        self.last_node_embeddings_ope: torch.Tensor | None = None
        self.last_node_embeddings_mach: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of heterogeneous observations.

        Args:
            observations: Dict of batched tensors as produced by SB3's observation
                preprocessing.

        Returns:
            The pooled graph embedding, shape ``(B, 2 * hidden_dim)``.  The per-node
            tensors are left on :attr:`last_node_embeddings_ope` /
            :attr:`last_node_embeddings_mach`.
        """
        node_ope, node_mach, graph = self.encoder(
            observations["node_features"],
            observations["machine_features"],
            observations["adj_precedence"],
            observations["adj_machine"],
            observations["adj_eligibility"],
        )
        self.last_node_embeddings_ope = node_ope
        self.last_node_embeddings_mach = node_mach
        return graph


class PerNodeMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """``MaskableActorCriticPolicy`` whose action logits come from a per-node head.

    Replaces SB3's default ``action_net`` (one ``nn.Linear(features_dim, n_actions)``,
    whose parameter count grows with the instance size) and ``value_net`` with
    :class:`PerNodeActorHead`/:class:`PerPairActorHead` and :class:`PooledCriticHead`.
    Which actor head is used is inferred from the action space, so one policy class
    covers both problem variants:

    * ``action_space.n == N_ope``            -> :class:`PerNodeActorHead` (JSSP, fixed routing)
    * ``action_space.n == N_ope * N_mach``   -> :class:`PerPairActorHead` (FJSP, flexible routing)

    Action masking is untouched: ``sb3_contrib`` still auto-discovers
    ``env.action_masks()`` and ``MaskableCategorical`` still applies it to a
    ``(B, n_actions)`` logits tensor.  Only the tensor's *provenance* changed.

    Args:
        observation_space: The environment's ``Dict`` observation space.
        action_space: ``Discrete(N_ope)`` or ``Discrete(N_ope * N_mach)``.
        lr_schedule: SB3 learning-rate schedule.
        head_hidden_dim: Hidden width of the actor/critic head MLPs.
        head_num_layers: Depth of the actor/critic head MLPs.
        actor_logit_gain: Orthogonal-init gain applied to the actor head's final layer.
            Small by design (SB3 uses ``0.01`` for its own ``action_net``): a
            near-uniform initial policy is what keeps early entropy near
            ``ln(legal actions)`` instead of collapsing, the failure mode
            ``TrainingPlan_Week17.md`` §3.2 flags as most likely for a new head.
        **kwargs: Forwarded to :class:`~sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy`.

    Raises:
        ValueError: If ``action_space.n`` matches neither the operation count nor the
            operation-machine pair count implied by ``observation_space``.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Any,
        head_hidden_dim: int = 64,
        head_num_layers: int = 2,
        actor_logit_gain: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_num_layers = int(head_num_layers)
        self.actor_logit_gain = float(actor_logit_gain)

        num_ops = int(observation_space["node_features"].shape[0])
        num_machines = int(observation_space["machine_features"].shape[0])
        n_actions = int(action_space.n)
        if n_actions == num_ops:
            self._head_kind = "per_node"
        elif n_actions == num_ops * num_machines:
            self._head_kind = "per_pair"
        else:
            raise ValueError(
                f"action space Discrete({n_actions}) matches neither the operation count "
                f"({num_ops}, JSSP) nor the operation-machine pair count "
                f"({num_ops * num_machines}, FJSP) implied by the observation space"
            )

        # An empty net_arch makes SB3's MlpExtractor an identity, so latent_pi/latent_vf
        # *are* the extractor's pooled graph embedding -- which is what PooledCriticHead
        # wants, and the actor head ignores the latent entirely (see
        # _get_action_dist_from_latent).
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        """Build the default networks, then swap in the size-agnostic heads.

        Calling ``super()._build`` first (rather than reimplementing it) keeps this class
        correct across ``sb3_contrib`` patch releases: everything SB3 sets up --- the MLP
        extractor, the masked distribution, orthogonal init of the shared trunk --- still
        happens exactly as the library intends, and only the two output modules are
        replaced afterwards.  The optimizer is rebuilt last so it owns the new parameters
        and not the discarded ones.

        Args:
            lr_schedule: SB3 learning-rate schedule.
        """
        super()._build(lr_schedule)

        hidden_dim = self.features_extractor.hidden_dim
        if self._head_kind == "per_node":
            self.action_net: nn.Module = PerNodeActorHead(
                in_dim=hidden_dim,
                hidden_dim=self.head_hidden_dim,
                num_layers=self.head_num_layers,
            )
        else:
            self.action_net = PerPairActorHead(
                ope_dim=hidden_dim,
                mach_dim=hidden_dim,
                hidden_dim=self.head_hidden_dim,
                num_layers=self.head_num_layers,
            )
        self.value_net: nn.Module = _ValueNetAdapter(
            PooledCriticHead(
                in_dim=2 * hidden_dim,
                hidden_dim=self.head_hidden_dim,
                num_layers=self.head_num_layers,
            )
        )

        if self.ortho_init:
            self.action_net.apply(lambda m: self.init_weights(m, gain=np.sqrt(2)))
            self.value_net.apply(lambda m: self.init_weights(m, gain=1.0))
            # Small final-layer gain -> near-uniform initial policy; see the class
            # docstring and TrainingPlan_Week17.md Section 3.2.
            final_linear = [m for m in self.action_net.mlp if isinstance(m, nn.Linear)][-1]
            self.init_weights(final_linear, gain=self.actor_logit_gain)

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskableDistribution:
        """Score the stashed per-node embeddings instead of ``action_net(latent_pi)``.

        Args:
            latent_pi: SB3's actor latent.  Ignored: with ``net_arch=dict(pi=[])`` it is
                the pooled graph embedding, which is a *summary* of the graph and cannot
                by construction carry per-node information.  It is still accepted so the
                override matches the base class's signature exactly.

        Returns:
            The masked categorical distribution over actions.

        Raises:
            RuntimeError: If no embeddings have been stashed, i.e. this was called
                without a preceding ``extract_features`` --- which would silently score
                a stale graph.
        """
        del latent_pi
        node_ope = self.features_extractor.last_node_embeddings_ope
        node_mach = self.features_extractor.last_node_embeddings_mach
        if node_ope is None or node_mach is None:
            raise RuntimeError(
                "no node embeddings stashed: _get_action_dist_from_latent was reached "
                "without a preceding features-extractor forward pass"
            )
        action_logits = self.action_net(node_ope, node_mach)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Include this subclass's own kwargs so ``save``/``load`` round-trips."""
        data = super()._get_constructor_parameters()
        data.update(
            head_hidden_dim=self.head_hidden_dim,
            head_num_layers=self.head_num_layers,
            actor_logit_gain=self.actor_logit_gain,
        )
        return data


def build_hetero_maskable_ppo(
    env: Any,
    hidden_dim: int = 64,
    num_layers: int = 3,
    mlp_layers: int = 2,
    head_num_layers: int = 2,
    use_machine_edges: bool = True,
    encoder_type: str = "gin",
    seed: int | None = None,
    device: str = "cpu",
    **ppo_kwargs: Any,
) -> MaskablePPO:
    """Construct a :class:`~sb3_contrib.MaskablePPO` on the heterogeneous, size-agnostic policy.

    The heterogeneous counterpart of :func:`build_maskable_ppo`, which is left unchanged
    as the homogeneous ablation configuration.

    Args:
        env: A :class:`~envs.jssp_env.JSSPEnv`, an :class:`~envs.fjsp_env.FJSPEnv`, or a
            ``VecEnv`` of either (all sharing one instance size). Action masking is
            auto-discovered from ``env.action_masks()``.
        hidden_dim: Width of every heterogeneous GIN layer, both node types.
        num_layers: Number of message-passing rounds.
        mlp_layers: Depth of each GIN layer's internal MLP.
        head_num_layers: Depth of the actor and critic head MLPs.
        use_machine_edges: Include the machine-sharing relation; ``False`` is the
            "no disjunctive edges" ablation.
        encoder_type: ``"gin"`` (default) or ``"gat_hybrid"`` (``AblationPlan.md`` Axis
            1b), forwarded to :class:`JSSPHeteroGraphFeaturesExtractor`.
        seed: RNG seed forwarded to ``MaskablePPO`` for reproducibility.
        device: Pinned explicitly to ``"cpu"`` by default rather than left at SB3's own
            ``"auto"`` default (``E7_E8_FinalPlan.md`` §2.1 precondition #2 /
            ``E0_CLOSEOUT_WEEK24.md``'s root-cause: on a machine whose GPU driver
            reports ``torch.cuda.is_available() -> True`` while actually being
            unusable, ``device="auto"`` lets SB3 split the model across ``cuda``/``cpu``
            silently). This matches what ``baselines.interface``'s evaluation path
            already does safely. Pass ``device="cuda"`` explicitly to override on a
            machine with a confirmed-working GPU.
        **ppo_kwargs: Extra keyword arguments forwarded to ``MaskablePPO`` (e.g.
            ``n_steps``, ``batch_size``, ``learning_rate``, ``gamma``,
            ``tensorboard_log``).

    Returns:
        A ``MaskablePPO`` instance ready for ``.learn(...)``.
    """
    policy_kwargs: dict[str, Any] = dict(
        features_extractor_class=JSSPHeteroGraphFeaturesExtractor,
        features_extractor_kwargs=dict(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            use_machine_edges=use_machine_edges,
            encoder_type=encoder_type,
        ),
        head_hidden_dim=hidden_dim,
        head_num_layers=head_num_layers,
    )
    return MaskablePPO(
        PerNodeMaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        **ppo_kwargs,
    )
