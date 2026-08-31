# Heterogeneous GNN Encoder + PPO Policy — Architecture Spec v1.0
### Layer-by-layer target architecture for `models/gin.py` / `models/policy.py`, next week's coding sprint

**Audience:** a coding agent implementing this on top of the already-validated Sprint-1 pipeline (`models/gin.py::RelationalGINLayer`/`GINEncoder`, `models/policy.py::JSSPGraphFeaturesExtractor`/`build_maskable_ppo`, wired to `sb3_contrib.MaskablePPO`; `tests/test_policy.py`, 5/5 passing per `VALIDATION_WEEK1.md`).

**Prime directive: EXTEND, DO NOT REWRITE.** `RelationalGINLayer`, `GINEncoder`, `JSSPGraphFeaturesExtractor`, `build_maskable_ppo` keep their exact current signatures and behavior — `tests/test_policy.py` must keep passing unmodified. Everything below is new, additively-named classes in the same two files (or new sibling modules if that keeps the diff cleaner), not a replacement of the homogeneous JSSP path, which remains the training pipeline's fallback/ablation configuration (`use_machine_edges=False`-equivalent already exists as a documented ablation in `RelationalGINLayer`; the homogeneous path is the natural "further ablation" of the heterogeneous one, not dead code to delete).

**Source-of-truth cross-reference:** formalizes `../../archive/capitulos/Section4.4.md` (already written, companion document) — Part B5 (heterogeneous GNN, WL-expressiveness), Part B2.2 (disjunctive graph), Part D2 (node/machine feature vectors), Part B4 (PPO). This spec resolves `models/policy.py`'s own documented limitation ("Instance-agnostic transfer needs a per-node action head... left for a later week") and is the stated prerequisite for wiring `DynamicJSSPEnv` into training (`VALIDATION_WEEK6.md`, "Next steps").

---

## 0. Scope

**In scope:** the heterogeneous encoder (operation + machine nodes, four relation types), the size-agnostic per-node actor head and pooled critic head, the `MaskableActorCriticPolicy` subclass needed to wire a per-node head into `sb3_contrib` (SB3's default policy ties the action head to a fixed `features_dim`, which is exactly the limitation being removed), and the PPO hyperparameter starting point.

**Out of scope:** wiring this policy into `DynamicJSSPEnv` (needs the machine `broken` state and the growable-array observation contract, §6.1 of `../docs/especificaciones/DynamicEnv_TechSpec.md` — a real next step this architecture is built to support, but not this sprint's deliverable); the FJSP dual action head (operation, machine) — this sprint's action space stays `Discrete(num_operations)`, fixed routing, per `../../archive/capitulos/Section4.4.md` §4.4.3's explicit scope note.

---

## 1. Node feature dimensions

### 1.1 Operation nodes — unchanged, 8-dim (`envs.jssp_env.NUM_NODE_FEATURES`)

No change to `envs/jssp_env.py`'s existing `NODE_FEATURE_NAMES`: `processing_time`, `is_scheduled`, `is_ready`, `is_job_complete`, `earliest_start_estimate`, `remaining_job_work`, `job_progress`, `machine_remaining_load`. `in_dim_ope = 8`.

### 1.2 Machine nodes — new, 7-dim, per Part D2

```python
#: New constant, envs/jssp_env.py (additive — does not touch NUM_NODE_FEATURES/NODE_FEATURE_NAMES)
NUM_MACHINE_FEATURES = 7
MACHINE_FEATURE_NAMES: tuple[str, ...] = (
    "is_free",              # one-hot component 1/3
    "is_busy",               # one-hot component 2/3
    "is_broken",              # one-hot component 3/3 -- always 0.0 under JSSPEnv (static);
                               # populated under DynamicJSSPEnv once wired (later sprint)
    "time_to_available",       # 0 if free; else (free_time - t)/horizon; normalized like op features
    "assigned_pending_load",    # sum of durations of ready-or-later ops needing this machine / W_max
    "utilization_accum",         # busy time so far / t  (0 at t=0, by convention)
    "estimated_availability",     # running empirical A_k estimate; 1.0 under JSSPEnv (no breakdowns)
)
```

`in_dim_mach = 7`. Every machine node exists for the full episode (cardinality `instance.num_machines`, fixed, unlike the growable operation-node set) — no masking needed for machine nodes themselves, only for the action space over operation nodes (§4).

**Where this is computed:** as a new `_machine_features()` method on `JSSPEnv`, added the same way `node_features` is already computed in `step()`/`_observe()` — a pure additive extension to `observation_space`'s `Dict`, adding a `"machine_features"` key (shape `(num_machines, 7)`) and a `"adj_eligibility"` key (shape `(num_operations, num_machines)`, one-hot per operation row — its required machine) alongside the three existing keys. **Do not remove or rename `node_features`/`adj_precedence`/`adj_machine`/`action_mask`** — this is the same additive-Dict-key discipline already used for `../docs/especificaciones/DynamicEnv_TechSpec.md` §6.3's machine-state block, applied here to the static env first.

### 1.3 Eligibility edges carry no separate edge feature

Unlike Song et al.'s `GATedge` (which needs an explicit edge scalar — processing time — because attention coefficients are computed per-edge), this spec's sum-then-MLP aggregation (§2) does not need one: an operation's own `processing_time` feature (§1.1) is already part of what gets summed into the machine node's incoming message, and the machine's own `assigned_pending_load` already aggregates duration information from a different angle. Adding a redundant edge feature would duplicate information already on the node without a clear expressiveness benefit, so `adj_eligibility` is a plain 0/1 adjacency, not a weighted one. Revisit only if ablation results show the encoder underusing per-operation duration in the machine update.

---

## 2. GNN architecture — heterogeneous, GIN aggregation

### 2.1 Aggregation type: GIN (sum + MLP), not GAT — recommendation and justification

**Recommendation: extend the existing sum-aggregation GIN design uniformly across all four relations** (precedence-predecessor, precedence-successor, machine-sharing, and the new operation↔machine eligibility relation), rather than introducing attention.

**Justification, verified against the two closest works in the literature (not asserted from memory):**

- **L2D** (Zhang et al., NeurIPS 2020) — the closest precedent in objective (PPO + makespan-shaped reward + masked action space) — uses a homogeneous GIN with **sum** neighbor pooling and **mean** graph pooling throughout, confirmed directly from the official implementation's hyperparameter file (`zcaicaros/L2D`, `Params.py`: `neighbor_pooling_type='sum'`, `graph_pool_type='average'`, `num_layers=3`, `hidden_dim=64`, `num_mlp_layers_feature_extract=2`). No attention anywhere.
- **Song et al.** (IEEE TII 2023) — the closest precedent in *graph heterogeneity* (operation + machine nodes) — uses a **hybrid**, confirmed directly from the official implementation (`songwenas12/fjsp-drl`, `graph/hgnn.py`): a `GATedge` module (multi-head attention, `num_heads=[1, 1]` in their actual experiments — i.e. effectively single-head despite the parameterized capability) is used **only** for the operation→machine update (aggregating a machine's candidate operations into its embedding, using processing time as an edge feature inside the attention coefficient), while the operation↔operation relations (precedence, machine-sharing) use `MLPsim` — a plain sum-then-MLP aggregator, architecturally identical in spirit to GIN, **not** attention.

Neither closest precedent uses attention uniformly. Combined with (a) the WL-expressiveness argument already adopted in `models/gin.py`'s own docstring and Part B5 (Xu et al. 2019, Theorem 3: sum aggregation is as discriminative as 1-WL; mean/max — and, by the same argument, softmax-normalized attention weights, which behave more like a weighted mean than a sum — cannot distinguish a machine clique of 5 identical operations from one of 10), and (b) the lower implementation risk of extending the already-validated `RelationalGINLayer` rather than introducing a new attention mechanism from scratch, this spec's default is **uniform sum+MLP**, including for the new eligibility relation. Song et al.'s hybrid (attention specifically on the machine-update relation) is documented here as a **precisely specified future ablation**, not a vague alternative:

```python
# Documented future ablation, NOT this sprint's default:
# swap RelationalHeteroGINLayer's machine-update term for a GATedge-style attention
# aggregator (single head, matching Song et al.'s actual num_heads=1 configuration),
# keeping every operation-side relation on sum+MLP unchanged.
```

### 2.2 Layer-by-layer specification

```python
# models/gin.py -- new classes, additive to the existing RelationalGINLayer/GINEncoder

class HeteroRelationalGINLayer(nn.Module):
    """One heterogeneous GIN update: operation nodes (4 relations) + machine nodes (1 relation)."""

    def __init__(
        self,
        in_dim_ope: int, in_dim_mach: int,
        out_dim_ope: int, out_dim_mach: int,
        hidden_dim: int = 64, mlp_layers: int = 2, learn_eps: bool = True,
    ) -> None:
        # Operation-side: reuses the exact same 4 weight matrices as RelationalGINLayer
        # (lin_pred, lin_succ, lin_machine) PLUS one new lin_elig (operation <- machine).
        # Machine-side: one new lin_cand (machine <- candidate operations).
        ...

    def forward(self, x_ope, x_mach, adj_precedence, adj_machine, adj_eligibility):
        # x_ope: (B, N_ope, in_dim_ope); x_mach: (B, N_mach, in_dim_mach)
        # adj_eligibility: (B, N_ope, N_mach), one nonzero entry per operation row.
        ope_msg = (
            self.lin_pred(bmm(adj_precedence.transpose(1, 2), x_ope))
            + self.lin_succ(bmm(adj_precedence, x_ope))
            + self.lin_machine(bmm(adj_machine, x_ope))
            + self.lin_elig(bmm(adj_eligibility, x_mach))          # NEW: machine -> operation
        )
        mach_msg = self.lin_cand(bmm(adj_eligibility.transpose(1, 2), x_ope))  # NEW: operation -> machine
        h_ope = self.mlp_ope((1 + eps_ope) * x_ope + ope_msg)
        h_mach = self.mlp_mach((1 + eps_mach) * x_mach + mach_msg)
        return h_ope, h_mach


class HeteroGINEncoder(nn.Module):
    """Stack of HeteroRelationalGINLayer; mirrors GINEncoder's constructor exactly."""

    def __init__(
        self,
        in_dim_ope: int = 8, in_dim_mach: int = 7,
        hidden_dim: int = 64, num_layers: int = 3, mlp_layers: int = 2,
        pool: str = "mean",
    ) -> None:
        ...

    def forward(self, x_ope, x_mach, adj_precedence, adj_machine, adj_eligibility):
        # Returns (node_embeddings_ope, node_embeddings_mach, graph_embedding), where
        # graph_embedding = concat(pool(node_embeddings_ope), pool(node_embeddings_mach))
        ...
```

**Table 2.1.** Layer-by-layer dimensions, this spec's recommended defaults.

| Component | Value | Source |
|---|---|---|
| `in_dim_ope` | 8 | `envs.jssp_env.NUM_NODE_FEATURES`, unchanged |
| `in_dim_mach` | 7 | §1.2, new |
| `hidden_dim` (every layer, both node types) | 64 | Matches the current validated code (`GINEncoder` default) and L2D's `hidden_dim=64` exactly |
| `num_layers` (message-passing rounds) | 3 | Matches the current validated code (`GINEncoder` default `num_layers=3`) and L2D's `num_layers=3` exactly |
| `mlp_layers` (depth of each layer's internal MLP) | 2 | Matches the current validated code and L2D's `num_mlp_layers_feature_extract=2` exactly |
| Graph pooling | mean, concatenated across the two node types | Extends the current code's `pool="mean"` rationale (critic must compare across instance sizes) to the second node type; L2D's `graph_pool_type='average'` agrees for the (homogeneous) operation side |

**Rationale for keeping `hidden_dim=64`/`num_layers=3` rather than Song et al.'s wider `hidden_size_ope=128`:** the current, already-tested pipeline is calibrated around 64, and two independent JSSP-scale precedents (this thesis's own Sprint 1, and L2D on the same instance families) validate 64 as sufficient; Song et al.'s wider setting reflects their larger FJSP action space (routing + sequencing) rather than a JSSP-specific requirement. Widening to 128 is a documented, cheap thing to try if the heterogeneous encoder underperforms the homogeneous one on the validated instance tiers, not a default to adopt pre-emptively.

---

## 3. Actor and critic heads

### 3.1 Actor: per-node scorer (size-agnostic)

```python
class PerNodeActorHead(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim=1, num_layers=num_layers)  # reuses models/gin.py's helper

    def forward(self, node_embeddings_ope: torch.Tensor) -> torch.Tensor:
        # node_embeddings_ope: (B, N_ope, hidden_dim) -> logits: (B, N_ope)
        return self.mlp(node_embeddings_ope).squeeze(-1)
```

The **same** `PerNodeActorHead` instance (same weights) is applied to every operation node — this is what makes the parameter count independent of `N_ope`, per `../../archive/capitulos/Section4.4.md` §4.4.3. 2 layers, `hidden_dim=64`, matching the encoder's width for implementation simplicity (one fewer hyperparameter to track relative to L2D's separate `hidden_dim_actor=32`).

### 3.2 Critic: pooled, two-node-type readout

```python
class PooledCriticHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 64, num_layers: int = 2) -> None:
        # in_dim = 2 * hidden_dim: concat(pooled operation embedding, pooled machine embedding)
        super().__init__()
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim=1, num_layers=num_layers)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(graph_embedding).squeeze(-1)  # (B,)
```

2 layers, `hidden_dim=64`. `in_dim=128` because `HeteroGINEncoder`'s graph embedding concatenates the pooled operation and pooled machine embeddings (§2.2) — the critic sees both a "how much work is left" signal (from operations) and a "how loaded/available are the machines" signal (from machines), which the homogeneous Sprint-1 critic could not see directly.

### 3.3 Wiring into `sb3_contrib.MaskablePPO`: why a custom policy class is required, not just a custom extractor

`JSSPGraphFeaturesExtractor` (Sprint 1) works as a **features extractor** because SB3's default `MaskableActorCriticPolicy` applies one `nn.Linear(features_dim, action_space.n)` on top of whatever the extractor returns — that final `Linear` is exactly the size-tied component `../../archive/capitulos/Section4.4.md` §4.4.3 removes. A per-node scorer cannot be expressed as a features extractor alone; it must replace that final `Linear` itself. This spec adds:

```python
# models/policy.py -- new, additive

class JSSPHeteroGraphFeaturesExtractor(BaseFeaturesExtractor):
    """Returns BOTH the per-node operation embeddings (for the actor) and the pooled
    graph embedding (for the critic), instead of one flattened vector. features_dim is
    declared as hidden_dim (64), not (num_ops+1)*hidden_dim -- SB3's BaseFeaturesExtractor
    contract expects a flat (B, features_dim) tensor for its bookkeeping, so this class
    returns the POOLED graph embedding as the nominal "features" (satisfying SB3's shape
    contract) while stashing the per-node operation embeddings as an extra attribute
    (self.last_node_embeddings) consumed by the custom policy's action head below --
    the standard SB3 escape hatch for a features extractor that needs to expose more than
    one tensor (used, e.g., by SB3's own recurrent-policy extractors)."""
    ...

class PerNodeMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Overrides _get_action_dist_from_latent (or the equivalent forward hook, matching
    the installed sb3_contrib version's exact API) to score self.features_extractor
    .last_node_embeddings through PerNodeActorHead instead of self.action_net(latent_pi).
    action_masks() integration is UNCHANGED -- MaskableCategorical still receives the
    per-node logits (now variable-length, exactly N_ope wide) and the existing action
    mask (Section 4.3.2), exactly as today."""
    ...
```

**Verify the exact override point against the installed `sb3_contrib` version before implementing** — the internal method name/signature for "where the policy turns latent features into action logits" has changed across `sb3_contrib` releases; pin the version in `requirements.txt`/`pyproject.toml` if not already pinned, and add a version-guard comment at the override site.

### 3.4 Size generalization: what "independent of `N`" actually buys, operationally

`Discrete(N)` is still declared per `JSSPEnv` instance (`N = instance.num_operations`) — SB3 requires a fixed `action_space` per env, and this spec does not change that. What the per-node head buys is that **the learned weights** (`HeteroGINEncoder`, `PerNodeActorHead`, `PooledCriticHead`) have a parameter count independent of `N`, so a `state_dict` trained on `ft06` (`N=36`) can be loaded into a freshly constructed `PerNodeMaskableActorCriticPolicy` wrapping `ft10` (`N=100`) — zero-shot size transfer happens at weight-loading time, not by using one dynamic action space across sizes. This is the operational mechanism behind H2 (`../docs/planificacion/Step 4 Thesis.md` D7), and it should be stated explicitly in the Chapter 6 evaluation script (load `ft06`-trained weights into a policy object constructed against each test-size env, do not attempt to retrain the wrapper). A second mechanism — padding every env to `max_possible_ops` and masking, exactly `../docs/especificaciones/DynamicEnv_TechSpec.md` §6.1's already-implemented growable-array pattern — is the natural fit once this policy is wired into `DynamicJSSPEnv` (out of scope this sprint, §0) or if a future sprint wants genuine within-VecEnv size-mixing during training rather than weight transfer after training; flag both mechanisms in the thesis text as the two valid readings of "size-agnostic," since they are not the same claim.

---

## 4. Action masking integration

Unchanged mechanism, updated data path: `env.action_masks()` (Section 4.3.2, unmodified) still returns a boolean vector of length `N_ope`; `sb3_contrib`'s auto-discovery still calls it the same way. The only change is that the logits `MaskableCategorical` masks are now `PerNodeActorHead`'s per-node output (§3.1) instead of a single `Linear(features_dim, N)`'s output — from `MaskableCategorical`'s point of view, nothing changes, since it only ever sees a `(B, N)` logits tensor and a `(B, N)` mask tensor, regardless of how the logits were produced.

---

## 5. PPO hyperparameters — starting point

**Table 5.1.** Recommended starting configuration, with confidence level and source for each value.

| Hyperparameter | Recommended | Confidence | Source |
|---|---:|---|---|
| `gamma` | 1.0 | **High** — structural, not tunable | Matches this thesis's own reward-telescoping design (Part B4: $\sum_t r_t = H_0 - C_{\max}$ requires undiscounted return); **also** the exact value used by both L2D (`Params.py`: `gamma=1`) and Song et al. (`config.json`: `"gamma": 1.0`) |
| `clip_range` ($\epsilon$) | 0.2 | **High** | Exact value in both L2D (`eps_clip=0.2`) and Song et al. (`"eps_clip": 0.2`); also the canonical Schulman et al. (2017) default |
| `ent_coef` | 0.01 | **High** | Exact value in both L2D (`entloss_coef=0.01`) and Song et al. (`"entropy_coeff": 0.01`) |
| `learning_rate` | 2e-4 (Adam) | **Medium** | Song et al.'s exact value (`"lr": 0.0002`) — the closer architectural precedent (heterogeneous graph); L2D uses a 10x smaller `lr=2e-5`. Recommend starting at Song et al.'s value since this architecture is heterogeneous like theirs, not homogeneous like L2D's; **if training is unstable (reward variance not decreasing, value loss diverging), fall back toward L2D's 2e-5** rather than assuming the discrepancy is a bug — a 10x lr gap between two published, working configurations on a related problem class is itself evidence this hyperparameter needs empirical confirmation on this thesis's own instances, not a value to trust blindly from either source. |
| `gae_lambda` | 0.95 | **Low-Medium** | Neither L2D's nor Song et al.'s published configuration files (as fetched from their official repositories) report a GAE $\lambda$ value explicitly — L2D's `k_epochs=1` (a single PPO update epoch per rollout) suggests they may not rely on GAE's bias/variance tradeoff the way a multi-epoch PPO run does. This spec falls back to the canonical Schulman et al. (2016, GAE paper, arXiv:1506.02438) default of 0.95, already the value assumed in this thesis's own Part B4 formula — **do not attribute 0.95 to L2D or Song et al. in the thesis text; attribute it to the GAE paper directly**, since that is the actually-verified source. |
| `n_epochs` (PPO update epochs per rollout) | 3–4 | **Medium** | Song et al. uses `K_epochs=3`; L2D uses the unusually low `k_epochs=1`. Recommend Song et al.'s value (closer to the canonical SB3 default of 10, and this thesis's episodes are short, so a very low epoch count like L2D's risks under-using each rollout) as the starting point; do not default to SB3's generic 10 without checking for overfitting on the small per-episode rollout buffer this problem produces. |
| `vf_coef` | 0.5 | **Medium** | Song et al.'s exact value (`"vf_coeff": 0.5`, alongside `"A_coeff": 1.0` for the policy term) — matches the canonical SB3 default ratio (policy weight 1 : value weight 0.5) exactly. L2D instead uses an inverted, asymmetric weighting (`ploss_coef=2`, `vloss_coef=1`, i.e. policy overweighted 2:1 relative to value) with no stated justification recoverable from the config file alone; **do not silently copy L2D's ratio** — start from Song et al.'s canonical-matching ratio, and only investigate L2D's inverted weighting as a documented ablation if the critic is diagnosed as the bottleneck (e.g., value loss staying high while policy loss drops). |
| `n_steps` (per-env rollout length) | 4–8 × `num_operations` of the training instance | **This spec's own choice** | Neither precedent's config reports `n_steps` in a directly transferable form (both structure training around complete-episode batches rather than SB3's fixed-length truncated rollout window); since a JSSP episode is exactly `num_operations` steps long (36 for `ft06`, up to 100 for `ft10`/`ft20`), this spec recommends sizing the rollout buffer as a small multiple of episode length so each PPO update sees several complete episodes, not a truncated slice of one — e.g. `n_steps=6*36=216` when training on `ft06`. |
| `n_envs` (parallel environments) | 4–8 | **Medium** | L2D uses `num_envs=4`; Song et al. uses `parallel_iter=20`. This spec recommends L2D's more modest range as the single-GPU/CPU student-budget-appropriate starting point (`../docs/planificacion/Step 4 Thesis.md` §1.5) — Song et al.'s 20 parallel environments likely assumes more compute than this thesis's budget affords; revisit upward only if the single-GPU budget is confirmed to have headroom. |
| `batch_size` (SB3 minibatch size) | 64 | **Medium** | Canonical SB3 default; Song et al.'s `minibatch_size=512` assumes their larger `n_steps × parallel_iter` product — recompute this thesis's own `batch_size` as a divisor of `n_steps × n_envs` once those are fixed empirically, rather than importing 512 directly into a much smaller rollout buffer. |

**Do not adopt any single source's full configuration wholesale.** Table 5.1 is a synthesis, not a copy of either paper — each row states its own confidence and source so the thesis text (`../../archive/capitulos/Section4.4.md` §4.4.4, already written) can cite precisely which choices are literature-grounded and which are this document's own reasoned starting point, consistent with the flagging culture already established across every other technical spec in this project (`../docs/especificaciones/DynamicEnv_TechSpec.md` §10, `../docs/especificaciones/Baselines_TechSpec.md` §9, `../docs/especificaciones/GA_TechSpec.md` §7).

---

## 6. Required tests (additive to `tests/test_policy.py`)

- `test_hetero_encoder_output_shapes` — `HeteroGINEncoder.forward` returns `(node_embeddings_ope, node_embeddings_mach, graph_embedding)` with the exact shapes implied by Table 2.1, for a batch of mixed instance sizes (padded or looped, not assumed uniform).
- `test_per_node_actor_head_parameter_count_independent_of_n` — construct `PerNodeActorHead` once, run it on tensors of two different `N_ope` (e.g. `ft06`'s 36 and `ft10`'s 100), assert the parameter count (`sum(p.numel() for p in head.parameters())`) is identical — the direct test of the size-agnostic claim (§3.4), not just an assertion in a docstring.
- `test_weight_transfer_across_instance_sizes` — train (or randomly initialize, if a full training run is too slow for a unit test — a few gradient steps suffice to prove the mechanism, not to prove convergence) a policy on `ft06`, save its `state_dict`, load it into a freshly constructed policy wrapping `la06` (`N=75`), and confirm the forward pass runs without shape errors and produces a valid probability distribution over `la06`'s legal actions — the direct test of §3.4's weight-transfer claim.
- `test_homogeneous_path_still_passes` — re-run the existing `tests/test_policy.py` suite unmodified after this sprint's changes; **zero regressions**, exactly the same discipline as `../docs/especificaciones/DynamicEnv_TechSpec.md`'s `test_default_config_matches_static_env_exactly` and `../docs/especificaciones/Baselines_TechSpec.md`'s `test_existing_rules_byte_identical`.
- `test_action_masking_unchanged_semantics` — for a fixed seed/instance, confirm the set of legal actions reported by `action_masks()` and consumed by `MaskableCategorical` is identical whether reached through the old flattened head or the new per-node head (only the logit-producing mechanism should differ, not the masking contract).

---

## 7. Sources

- Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.* NeurIPS 2020. Hyperparameters verified directly from the official implementation: [zcaicaros/L2D, `Params.py`](https://github.com/zcaicaros/L2D/blob/master/Params.py).
- Song, W., Chen, X., Li, Q., & Cao, Z. (2023). *Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning.* IEEE Transactions on Industrial Informatics, 19(2), 1600–1610. Hyperparameters and architecture verified directly from the official implementation: [songwenas12/fjsp-drl, `config.json`](https://github.com/songwenas12/fjsp-drl/blob/main/config.json) and [`graph/hgnn.py`](https://github.com/songwenas12/fjsp-drl/blob/main/graph/hgnn.py).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. Canonical `clip_range`/general PPO defaults.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. Source of the `gae_lambda=0.95` fallback default (§5, not independently confirmed in either JSSP-specific precedent's published config).
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR 2019, arXiv:1810.00826. Theorem 3, the expressiveness argument for uniform sum-aggregation (§2.1), already cited in Part B5 and `models/gin.py`.
