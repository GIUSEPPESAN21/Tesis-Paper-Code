# Heterogeneous GNN Encoder + PPO Policy — Architecture Spec v2.0
### Layer-by-layer target architecture for `models/gin.py` / `models/policy.py`, Week 17 coding sprint

**Audience:** a coding agent implementing this on top of the Week 15-refactored pipeline. Two things changed since v1.0 (Week 11) that this revision reconciles the spec against:

1. **`envs/base_env.py::SchedulingEnv`** now exists. `JSSPEnv` and `DynamicJSSPEnv` are siblings under it, not parent/child. Nothing in v1.0's architecture assumed the old (misleading) inheritance, so no §-by-§ content changes were forced by this — but every cross-reference to "`JSSPEnv`'s `__init__`" below now means "`JSSPEnv.__init__`, which calls `SchedulingEnv._init_common`," and the four-method contract (`action_masks`, `makespan`, `remaining_work_per_op`, `schedule_dataframe_records`) this spec's encoder consumes is now formally declared on `SchedulingEnv`, not just implied by convention. See `REFACTOR_NOTES_WEEK15.md` STEP1a.
2. **`run_evaluation.py`, the unified evaluation harness, is real, not planned.** `baselines/interface.py::RLPlaceholderBaseline` already occupies the `"rl"` slot in every CSV this harness produces, with `status="NOT_IMPLEMENTED"`. §8 (new in this revision) specifies exactly how this sprint's trained policy replaces that placeholder with zero downstream CSV/CLI changes — the harness was built to make that swap a non-event, and this spec should not let that guarantee go unused.

Everything else — node/machine feature dimensions, the heterogeneous GIN layer, actor/critic heads, PPO hyperparameters, action masking — is unchanged in substance from v1.0 and is reproduced here so this file is a complete, standalone reference for the sprint. Where a claim was re-verified against the current codebase (not just carried over), that is noted inline.

**Prime directive, unchanged: EXTEND, DO NOT REWRITE.** `RelationalGINLayer`, `GINEncoder`, `JSSPGraphFeaturesExtractor`, `build_maskable_ppo` (`models/gin.py` / `models/policy.py`) keep their exact current signatures and behavior — `tests/test_policy.py` must keep passing unmodified. Everything below is new, additively-named classes in the same two files (or new sibling modules if that keeps the diff cleaner). The homogeneous JSSP path remains the training pipeline's fallback/ablation configuration, not dead code to delete.

**Source-of-truth cross-reference:** formalizes `../../archive/capitulos/Section4.4.md` (Part B5, heterogeneous GNN / WL-expressiveness; Part B2.2, disjunctive graph; Part D2, node/machine feature vectors; Part B4, PPO). Resolves `models/policy.py`'s own documented limitation ("Instance-agnostic transfer needs a per-node action head... left for a later week") and is the stated prerequisite for wiring `DynamicJSSPEnv` into training (`VALIDATION_WEEK6.md`, "Next steps" — still explicitly deferred, see §0).

---

## 0. Scope

**In scope:** the heterogeneous encoder (operation + machine nodes, four relation types), the size-agnostic per-node actor head and pooled critic head, the `MaskableActorCriticPolicy` subclass needed to wire a per-node head into `sb3_contrib`, the PPO hyperparameter starting point, and (new in v2.0) the integration point with `run_evaluation.py`.

**Out of scope, unchanged from v1.0:**

- Wiring this policy into `DynamicJSSPEnv` — needs the machine `broken` state and the growable-array observation contract (`../docs/especificaciones/DynamicEnv_TechSpec.md` §6.1). `VALIDATION_WEEK6.md`'s "Next steps" and `REFACTOR_NOTES_WEEK15.md`'s scope-boundary list both independently confirm this is still not done as of Week 15 — a real next step this architecture is built to support, not this sprint's deliverable.
- The FJSP dual action head (operation, machine). This sprint's action space stays `Discrete(num_operations)`, fixed routing, per `../../archive/capitulos/Section4.4.md` §4.4.3. Note: Week 15 completed FJSP support for the *baseline* methods (`CPSATFJSPBaseline`, `GABaseline`) and explicitly left `PDRBaseline` rejecting FJSP instances by design — the RL policy trained this sprint has no FJSP counterpart yet either, for the same underlying reason (a machine-selection component the current architecture doesn't have).

---

## 1. Node feature dimensions

### 1.1 Operation nodes — unchanged, 8-dim (`envs.jssp_env.NUM_NODE_FEATURES`)

Re-verified directly against `envs/jssp_env.py` as it stands after Week 15: `NUM_NODE_FEATURES = 8`, `NODE_FEATURE_NAMES = ("processing_time", "is_scheduled", "is_ready", "is_job_complete", "earliest_start_estimate", "remaining_job_work", "job_progress", "machine_remaining_load")`. Week 15 touched `base_env.py`, `fjsp_instance.py`, `baselines/`, and `run_evaluation.py` only — its own "what was explicitly left alone" section confirms no changes to `envs/jssp_env.py`'s observation/reward design. `in_dim_ope = 8`, unchanged.

### 1.2 Machine nodes — new, 7-dim, per Part D2

```python
#: New constant, envs/jssp_env.py (additive — does not touch NUM_NODE_FEATURES/NODE_FEATURE_NAMES)
NUM_MACHINE_FEATURES = 7
MACHINE_FEATURE_NAMES: tuple[str, ...] = (
    "is_free",                 # one-hot component 1/3
    "is_busy",                 # one-hot component 2/3
    "is_broken",                # one-hot component 3/3 -- always 0.0 under JSSPEnv (static);
                                  # populated under DynamicJSSPEnv once wired (later sprint,
                                  # still out of scope per §0)
    "time_to_available",        # 0 if free; else (free_time - t)/horizon; normalized like op features
    "assigned_pending_load",     # sum of durations of ready-or-later ops needing this machine / W_max
    "utilization_accum",         # busy time so far / t  (0 at t=0, by convention)
    "estimated_availability",    # running empirical A_k estimate; 1.0 under JSSPEnv (no breakdowns)
)
```

`in_dim_mach = 7`. Every machine node exists for the full episode (cardinality `instance.num_machines`, fixed) — no masking needed for machine nodes themselves, only for the action space over operation nodes (§4).

**Where this is computed:** as a new `_machine_features()` method on `JSSPEnv`, added the same way `node_features` is already computed in `_observation()` (verified: `JSSPEnv._observation()` currently builds exactly `node_features` / `adj_precedence` / `adj_machine` / `action_mask` and nothing else — this is a pure additive extension of that method and of `observation_space`'s `Dict`). Add a `"machine_features"` key (shape `(num_machines, 7)`) and an `"adj_eligibility"` key (shape `(num_operations, num_machines)`, one-hot per operation row — its required machine, derivable from `self._flat_machines` which `JSSPEnv` already stores). **Do not remove or rename `node_features`/`adj_precedence`/`adj_machine`/`action_mask`** — every `Baseline` adapter and PDR relies on the current `SchedulingEnv` contract (`action_masks()`, `makespan()`, `remaining_work_per_op()`, `schedule_dataframe_records()`) staying intact regardless of what the observation `Dict` gains, and `run_evaluation.py` never touches `observation_space` directly, so this addition is invisible to every non-RL code path — only `models/policy.py`'s new extractor (§3.3) reads the new keys.

### 1.3 Eligibility edges carry no separate edge feature

Unchanged from v1.0: unlike Song et al.'s `GATedge` (needs an explicit edge scalar for its attention coefficients), this spec's sum-then-MLP aggregation does not need one — an operation's own `processing_time` feature is already summed into the machine node's incoming message, and the machine's own `assigned_pending_load` aggregates duration information from the other direction. `adj_eligibility` is a plain 0/1 adjacency. Revisit only if ablation results show the encoder underusing per-operation duration in the machine update.

---

## 2. GNN architecture — heterogeneous, GIN aggregation

### 2.1 Aggregation type: GIN (sum + MLP), not GAT — recommendation and justification

**Recommendation, unchanged: extend the existing sum-aggregation GIN design uniformly across all four relations** (precedence-predecessor, precedence-successor, machine-sharing, and the new operation↔machine eligibility relation), rather than introducing attention.

**Justification, verified against the two closest works in the literature:**

- **L2D** (Zhang et al., NeurIPS 2020) — closest precedent in objective (PPO + makespan-shaped reward + masked action space) — uses a homogeneous GIN with **sum** neighbor pooling and **mean** graph pooling throughout, confirmed from the official implementation's hyperparameter file (`zcaicaros/L2D`, `Params.py`: `neighbor_pooling_type='sum'`, `graph_pool_type='average'`, `num_layers=3`, `hidden_dim=64`, `num_mlp_layers_feature_extract=2`). No attention anywhere.
- **Song et al.** (IEEE TII 2023) — closest precedent in *graph heterogeneity* — uses a **hybrid**, confirmed from the official implementation (`songwenas12/fjsp-drl`, `graph/hgnn.py`): a `GATedge` module (`num_heads=[1, 1]` in their actual experiments) only for the operation→machine update, while operation↔operation relations use `MLPsim`, a plain sum-then-MLP aggregator architecturally identical in spirit to GIN.

Neither closest precedent uses attention uniformly. Combined with (a) the WL-expressiveness argument already adopted in `models/gin.py`'s own docstring and Part B5 (Xu et al. 2019, Theorem 3: sum aggregation is as discriminative as 1-WL; mean/max/softmax-attention cannot distinguish a machine clique of 5 identical operations from one of 10), and (b) lower implementation risk relative to introducing attention from scratch, this spec's default remains **uniform sum+MLP**, including for the eligibility relation. Song et al.'s hybrid is a **documented future ablation**, not this sprint's default:

```python
# Documented future ablation, NOT this sprint's default:
# swap HeteroRelationalGINLayer's machine-update term for a GATedge-style attention
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
        # Operation-side: reuses the exact same weight-matrix pattern as RelationalGINLayer
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

**Table 2.1.** Layer-by-layer dimensions, recommended defaults (unchanged from v1.0).

| Component | Value | Source |
|---|---|---|
| `in_dim_ope` | 8 | `envs.jssp_env.NUM_NODE_FEATURES`, unchanged, re-verified against current code |
| `in_dim_mach` | 7 | §1.2, new |
| `hidden_dim` (every layer, both node types) | 64 | Matches the current validated `GINEncoder` default and L2D's `hidden_dim=64` exactly |
| `num_layers` (message-passing rounds) | 3 | Matches the current validated `GINEncoder` default (`num_layers=3`) and L2D's `num_layers=3` exactly |
| `mlp_layers` (depth of each layer's internal MLP) | 2 | Matches the current validated code and L2D's `num_mlp_layers_feature_extract=2` exactly |
| Graph pooling | mean, concatenated across the two node types | Extends the current `pool="mean"` rationale (critic must compare across instance sizes) to the second node type |

**Rationale for keeping `hidden_dim=64`/`num_layers=3` rather than Song et al.'s wider `hidden_size_ope=128`:** the current, already-tested pipeline is calibrated around 64; two independent JSSP-scale precedents (this thesis's own Sprint 1, and L2D on the same instance families) validate 64 as sufficient. Song et al.'s wider setting reflects their larger FJSP action space, not a JSSP-specific requirement. Widening to 128 is a documented, cheap thing to try if the heterogeneous encoder underperforms the homogeneous one on the validated instance tiers, not a pre-emptive default.

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

The **same** `PerNodeActorHead` instance (same weights) is applied to every operation node — parameter count independent of `N_ope`, per `../../archive/capitulos/Section4.4.md` §4.4.3. 2 layers, `hidden_dim=64`, matching the encoder's width.

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

2 layers, `hidden_dim=64`, `in_dim=128` (concat of pooled operation and pooled machine embeddings). The critic sees both a "how much work is left" signal and a "how loaded/available are the machines" signal — the homogeneous Sprint-1 critic could not see the latter directly.

### 3.3 Wiring into `sb3_contrib.MaskablePPO`: why a custom policy class is required

`JSSPGraphFeaturesExtractor` (Sprint 1, unchanged since — re-verified against `models/policy.py` as it stands after Week 15) works as a **features extractor** because SB3's default `MaskableActorCriticPolicy` applies one `nn.Linear(features_dim, action_space.n)` on top of whatever the extractor returns. A per-node scorer must replace that final `Linear` itself, not just feed it. This spec adds:

```python
# models/policy.py -- new, additive

class JSSPHeteroGraphFeaturesExtractor(BaseFeaturesExtractor):
    """Returns BOTH the per-node operation embeddings (for the actor) and the pooled
    graph embedding (for the critic), instead of one flattened vector. features_dim is
    declared as hidden_dim (64), not (num_ops+1)*hidden_dim -- SB3's BaseFeaturesExtractor
    contract expects a flat (B, features_dim) tensor, so this class returns the POOLED
    graph embedding as the nominal "features" (satisfying SB3's shape contract) while
    stashing the per-node operation embeddings as an extra attribute
    (self.last_node_embeddings) consumed by the custom policy's action head below --
    the standard SB3 escape hatch for a features extractor that needs to expose more than
    one tensor."""
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

**Verify the exact override point against the installed `sb3_contrib` version before implementing** (`.venv` currently pins `sb3_contrib==2.9.0`, `stable_baselines3==2.9.0` — confirm the override point against that exact release, not a newer one, and pin it explicitly if `requirements.txt`/`pyproject.toml` doesn't already; the internal method name for "where the policy turns latent features into action logits" has changed across `sb3_contrib` releases).

### 3.4 Size generalization: what "independent of `N`" actually buys, operationally

Unchanged from v1.0. `Discrete(N)` is still declared per `JSSPEnv` instance (`N = instance.num_operations`) — SB3 requires a fixed `action_space` per env. What the per-node head buys is that the learned weights (`HeteroGINEncoder`, `PerNodeActorHead`, `PooledCriticHead`) have a parameter count independent of `N`, so a `state_dict` trained on `ft06` (`N=36`) can be loaded into a freshly constructed `PerNodeMaskableActorCriticPolicy` wrapping `ft10` (`N=100`) or any `la*` instance — zero-shot size transfer at weight-loading time, not one dynamic action space across sizes. State this explicitly in the Chapter 6 evaluation script (load `ft06`-trained weights into a policy object constructed against each test-size env; do not attempt to retrain the wrapper). A second mechanism — padding every env to `max_possible_ops` and masking, `../docs/especificaciones/DynamicEnv_TechSpec.md` §6.1's already-implemented growable-array pattern — is the natural fit once this policy is wired into `DynamicJSSPEnv` (still out of scope, §0) or if a future sprint wants genuine within-`VecEnv` size-mixing during training. Flag both mechanisms in the thesis text as the two valid readings of "size-agnostic," since they are not the same claim.

---

## 4. Action masking integration

Unchanged mechanism, updated data path: `env.action_masks()` (declared on `SchedulingEnv`, implemented per-subclass — `JSSPEnv.action_masks()` unmodified by Week 15) still returns a boolean vector of length `N_ope`; `sb3_contrib`'s auto-discovery still calls it the same way. The only change is that the logits `MaskableCategorical` masks are now `PerNodeActorHead`'s per-node output (§3.1) instead of a single `Linear(features_dim, N)`'s output — from `MaskableCategorical`'s point of view, nothing changes, since it only ever sees a `(B, N)` logits tensor and a `(B, N)` mask tensor.

---

## 5. PPO hyperparameters — starting point

See `../docs/logs/TrainingPlan_Week17.md` §2 for the full table with confidence levels and sources — reproduced there rather than duplicated here so the training plan is self-contained for whoever runs the sprint day-to-day. Structural takeaway repeated here because it governs the architecture, not just the run: `gamma=1.0` is not a tunable choice — it is required by this thesis's own reward-telescoping design (Part B4: $\sum_t r_t = H_0 - C_{\max}$ needs undiscounted return), and both L2D and Song et al. independently use the same value for the same structural reason.

---

## 6. Required tests (additive to `tests/test_policy.py`)

- `test_hetero_encoder_output_shapes` — `HeteroGINEncoder.forward` returns `(node_embeddings_ope, node_embeddings_mach, graph_embedding)` with the exact shapes implied by Table 2.1, for a batch of mixed instance sizes (padded or looped, not assumed uniform).
- `test_per_node_actor_head_parameter_count_independent_of_n` — construct `PerNodeActorHead` once, run it on tensors of two different `N_ope` (e.g. `ft06`'s 36 and `ft10`'s 100), assert the parameter count is identical.
- `test_weight_transfer_across_instance_sizes` — train (or randomly initialize — a few gradient steps suffice to prove the mechanism, not convergence) a policy on `ft06`, save its `state_dict`, load it into a freshly constructed policy wrapping `la06` (`N=75`), confirm the forward pass runs without shape errors and produces a valid probability distribution over `la06`'s legal actions.
- `test_homogeneous_path_still_passes` — re-run the existing `tests/test_policy.py` suite unmodified after this sprint's changes; **zero regressions**, the same discipline `REFACTOR_NOTES_WEEK15.md` documents for `tests/test_env.py`/`tests/test_dynamic_env.py` (run continuously during the sprint, not just at the end).
- `test_action_masking_unchanged_semantics` — for a fixed seed/instance, confirm the set of legal actions reported by `action_masks()` and consumed by `MaskableCategorical` is identical whether reached through the old flattened head or the new per-node head.
- **New in v2.0** — `test_rl_baseline_matches_protocol` — construct the `Baseline`-protocol wrapper from §8 around a freshly initialized (not necessarily trained) policy and assert it satisfies `baselines.interface.Baseline`'s contract: `.name == "rl"`, `.solve(instance, seed=..., time_limit_s=...)` returns a `BaselineResult` with `start_times` of the right shape and `feasible` computed via the same `is_schedule_feasible` every other adapter uses. This is the direct regression guard for the swap described in §8 — if it silently breaks the protocol, `run_evaluation.py --methods rl` fails the same way `RLPlaceholderBaseline`'s removal would, just later and less clearly.

---

## 7. Sources

- Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.* NeurIPS 2020. [zcaicaros/L2D, `Params.py`](https://github.com/zcaicaros/L2D/blob/master/Params.py).
- Song, W., Chen, X., Li, Q., & Cao, Z. (2023). *Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning.* IEEE Transactions on Industrial Informatics, 19(2), 1600–1610. [songwenas12/fjsp-drl, `config.json`](https://github.com/songwenas12/fjsp-drl/blob/main/config.json) and [`graph/hgnn.py`](https://github.com/songwenas12/fjsp-drl/blob/main/graph/hgnn.py).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR 2019, arXiv:1810.00826. Theorem 3.

---

## 8. Integration with the unified evaluation harness (new in v2.0)

`REFACTOR_NOTES_WEEK15.md` built `run_evaluation.py` and `baselines/interface.py::RLPlaceholderBaseline` specifically so that landing this sprint's policy "changes zero downstream code: not the CLI, not the CSV columns, not whatever reads the CSV for Wilcoxon/Friedman/Nemenyi." This section makes that swap concrete so the sprint actually cashes in that guarantee instead of inventing a parallel evaluation path.

**8.1 What has to be built:** a `TrainedRLBaseline` class in `baselines/interface.py`, satisfying the same `Baseline` protocol as every other adapter (`name: str`, `solve(instance, seed=None, time_limit_s=None) -> BaselineResult`):

```python
# baselines/interface.py -- new, additive; replaces RLPlaceholderBaseline in resolve_methods()

@dataclass
class TrainedRLBaseline:
    """Wraps a trained PerNodeMaskableActorCriticPolicy as a Baseline adapter.

    Loads once (state_dict path passed at construction), runs one greedy (or
    sampled, per seed) rollout per .solve() call against a fresh JSSPEnv --
    the same "construct env, roll out to termination, read off start_times"
    shape as run_pdr_online() in run_evaluation.py, since a trained policy is
    also fundamentally an online (env) -> op_id policy, not an offline solver.
    """
    name: str = "rl"
    policy_path: str = ...

    def solve(self, instance, seed=None, time_limit_s=None) -> BaselineResult:
        env = JSSPEnv(instance)
        obs, _ = env.reset(seed=seed)
        terminated = False
        while not terminated:
            action, _ = self._policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, terminated, _, info = env.step(int(action))
        feasible, violations = is_schedule_feasible(instance, env.start_times)
        return BaselineResult(
            instance=instance.name, method=self.name, start_times=env.start_times.copy(),
            makespan=info["makespan"], wall_clock_time=..., feasible=feasible,
            violations=violations, status="OK", is_optimal=False,
            extra={"dispatch_order": env.dispatch_order},
        )
```

**8.2 What changes in `run_evaluation.py`:** exactly one line, in `resolve_methods()` — the `elif name == "rl": baselines.append(RLPlaceholderBaseline())` branch becomes `baselines.append(TrainedRLBaseline(policy_path=...))`. Nothing else: `evaluate_one()` already dispatches non-PDR, non-`"rl"`-named baselines correctly once `TrainedRLBaseline.name == "rl"` and `disruption_name == "none"` route through `baseline.solve()` directly (the `if baseline.name == "rl" or disruption_name == "none"` branch in `evaluate_one()` already special-cases `"rl"` — re-read that condition when implementing this, since it currently exists *because* the placeholder must never be routed through `run_predictive_reactive()`, and a trained policy inherits that same routing for a different, better reason: see 8.3).

**8.3 Flagged decision: the trained policy is evaluable online, unlike GA/CP-SAT — this is a documented future improvement, not this sprint's scope.** `run_predictive_reactive()` exists because GA/CP-SAT are offline solvers with no notion of "solve while a machine might break." A trained RL policy is not offline in that sense — like a PDR, it is already an `(env, rng) -> op_id` mapping, so evaluating it under disruption is naturally a **direct rollout against `DynamicJSSPEnv`**, exactly like `run_pdr_online()`, not predictive-reactive replay. Wiring that is blocked on the same out-of-scope item as everywhere else in this document (§0): the policy this sprint trains is built against `JSSPEnv`'s static observation contract, and `DynamicJSSPEnv` wiring needs the growable-array head. Until that lands, `TrainedRLBaseline` only supports `--disruption none`, and should raise a clear `SystemExit`-style error (matching `validate_methods_for_disruption`'s existing pattern) if invoked otherwise — do not silently attempt to roll a static-graph policy out against a growable dynamic environment and produce a shape-error crash mid-sweep instead of a clear message.

**8.4 Why this section exists in the architecture spec and not only the training plan:** `PerNodeMaskableActorCriticPolicy.predict()`'s call signature (§8.1) is exactly what §3.4's weight-transfer claim has to support — the same loaded `state_dict`, wrapping envs of different `instance.num_operations`, is what lets one `TrainedRLBaseline` instance be re-used across every instance `run_evaluation.py --instances` iterates over, without reconstructing the encoder per instance. If §3's per-node head did not deliver size-agnostic weights, `TrainedRLBaseline` would need a policy-per-instance-size, which the current `Baseline` protocol (one `.solve(instance, ...)` call, any instance) does not anticipate and `run_evaluation.py`'s loop does not support without changes. This is the concrete, harness-visible payoff of §3.4's architectural choice, not just a training-time nicety.
