# Training Plan — Week 17 Coding Sprint

**Goal of this sprint:** implement the heterogeneous GNN + size-agnostic PPO policy specified in `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md`, and produce the first evidence — on the training instance and, zero-shot, on larger ones — that it learns the right thing, not just that it runs. This is the same bar `VALIDATION_WEEK1.md` set for the homogeneous pipeline ("prove the full pipeline runs... produce the reward-curve evidence") applied to a materially bigger architecture, at a materially bigger instance size, for the first time.

**Not this sprint's goal:** full convergence at every instance size, `DynamicJSSPEnv` wiring, or FJSP support. See `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §0 for the explicit scope boundary.

---

## 1. Curriculum: which sizes, and why

`../docs/planificacion/Step 4 Thesis.md` §1.5 fixes the compute budget explicitly: **one student, one GPU, 9 months** ("Alcance, delimitaciones y supuestos... GPU única"), and its own execution-order note (§"Secuencia de ejecución experimental," item 5) states the resulting policy directly: *"Respetar el presupuesto de cómputo (GPU única): entrenar en instancias pequeñas (6×6, 10×10) y explotar la generalización en tamaño para evaluar en grandes."* That sentence is this curriculum, made concrete against the actual bundled instance set (`data/instances/`, 40 files, sizes verified directly rather than assumed):

| Stage | Instances | Size (jobs × machines) | `num_operations` | Role |
|---|---|---|---|---|
| **A — primary training** | `ft06` | 6×6 | 36 | Debug/validate the new heterogeneous encoder and per-node head at the cheapest possible size. Reuses `experiments/train_ft06.py`'s existing harness (`Monitor`-wrapped `JSSPEnv`, `EpisodeStatsCallback`) — extend, don't replace, per the architecture spec's prime directive. |
| **B — scale-up training** | `ft10`, `la16`–`la20` | 10×10 | 100 | Second and last *training* tier this sprint, matching `../docs/planificacion/Step 4 Thesis.md`'s explicit "6×6, 10×10" instruction. `la16`–`la20` are five distinct 10×10 instances (not five seeds of one) — training across all five (round-robin instance swap via `env.reset(options={"instance": ...})`, already supported by `JSSPEnv.reset`) is what actually exercises size-agnostic weights *during* training, not just at zero-shot transfer time in Stage C. |
| **C — zero-shot generalization eval, not training** | `la01`–`la05` (10×5), `la06`–`la10` (15×5), `la11`–`la15` (20×5), `la21`–`la30` (15×10 / 20×10), `la31`–`la35` (30×10), `la36`–`la40` (15×15) | 40–225 ops | Load Stage B's `state_dict` into a freshly constructed `PerNodeMaskableActorCriticPolicy` wrapping each size, **no further gradient updates** — the direct test of §3.4's weight-transfer claim, and the "Generalización 2–3×" row of `../docs/planificacion/Step 4 Thesis.md`'s results matrix (`la31`–`la35` at 300 ops is ~3× `ft10`'s 100; `la36`–`la40` at 225 ops with 15×15 density is the structurally hardest size in the bundled set, not just the largest by op count). |

**Why not train directly on the largest instances:** `n_steps` scales with `num_operations` (§2 below), so a `la36`-sized rollout costs roughly 6× a `ft06`-sized one per environment step, before accounting for the heterogeneous encoder's larger `adj_eligibility` matrix (`num_operations × num_machines`, dense). Training small and transferring is not just a fallback for a constrained budget — it is the specific claim H2 of this thesis (`../docs/planificacion/Step 4 Thesis.md` D7), so Stage C is not merely an evaluation convenience, it is a required experiment either way.

**Hardware note:** no specific GPU model is pinned in the thesis planning documents beyond "single GPU" / student budget. This plan assumes that constraint holds (a single consumer-class GPU, or CPU fallback for Stage A given how small `ft06`'s graph is — 36 operation nodes + 6 machine nodes is tractable on CPU in minutes per rollout). If the actual available hardware differs materially (e.g., a shared cluster GPU with a larger memory/time budget), Stage B's five-instance round-robin and Stage C's full 30-instance sweep are the first things to expand — flag this explicitly rather than silently assuming more or less compute than is actually available.

---

## 2. Hyperparameter starting point

Table reproduced in full here (rather than only in `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md`, which cross-references this file) so the sprint can run entirely from this document. **Do not adopt any single source's full configuration wholesale** — each row states its own confidence and source, consistent with the flagging culture already established in `../docs/especificaciones/DynamicEnv_TechSpec.md` §10, `../docs/especificaciones/Baselines_TechSpec.md` §9, and `../docs/especificaciones/GA_TechSpec.md` §7.

| Hyperparameter | Recommended | Confidence | Source |
|---|---:|---|---|
| `gamma` | 1.0 | **High** — structural, not tunable | Required by this thesis's reward-telescoping design (Part B4: $\sum_t r_t = H_0 - C_{\max}$ needs undiscounted return); also the exact value in both L2D (`Params.py`: `gamma=1`) and Song et al. (`config.json`: `"gamma": 1.0`) |
| `clip_range` ($\epsilon$) | 0.2 | **High** | Exact value in both L2D (`eps_clip=0.2`) and Song et al. (`"eps_clip": 0.2`); canonical Schulman et al. (2017) default |
| `ent_coef` | 0.01 | **High** | Exact value in both L2D (`entloss_coef=0.01`) and Song et al. (`"entropy_coeff": 0.01`) |
| `learning_rate` | 2e-4 (Adam) | **Medium** | Song et al.'s exact value (`"lr": 0.0002`) — the closer architectural precedent (heterogeneous graph). L2D uses `lr=2e-5`, 10× smaller. Start at Song et al.'s value; **if training is unstable (reward variance not decreasing, value loss diverging), fall back toward L2D's 2e-5** rather than assuming a bug — a 10× gap between two published, working configurations is itself evidence this needs empirical confirmation on this thesis's own instances |
| `gae_lambda` | 0.95 | **Low–Medium** | Neither L2D's nor Song et al.'s published config reports GAE $\lambda$ explicitly (L2D's `k_epochs=1` suggests they may not rely on GAE's bias/variance tradeoff the way a multi-epoch run does). Falls back to the canonical Schulman et al. (2016, GAE paper) default of 0.95 — **attribute 0.95 to the GAE paper in the thesis text, not to L2D or Song et al.** |
| `n_epochs` | 3–4 | **Medium** | Song et al. uses `K_epochs=3`; L2D uses `k_epochs=1`. Recommend Song et al.'s value (closer to SB3's generic default of 10, and this problem's episodes are short, so a very low epoch count risks under-using each rollout); do not default to 10 without checking for overfitting on the small per-episode buffer |
| `vf_coef` | 0.5 | **Medium** | Song et al.'s exact value (`"vf_coeff": 0.5`), matching the canonical SB3 default ratio (policy 1 : value 0.5). L2D uses an inverted `ploss_coef=2, vloss_coef=1` with no stated justification recoverable from its config — **do not silently copy L2D's ratio**; only investigate it as a documented ablation if the critic is diagnosed as the bottleneck |
| `n_steps` (per-env rollout length) | Stage A (`ft06`, N=36): **216** (6×N). Stage B (10×10, N=100): **600** (6×N) | **This spec's own choice** | Neither precedent's config reports `n_steps` in a directly transferable form. Since a JSSP episode is exactly `num_operations` steps (deterministic, per `envs/jssp_env.py`'s own docstring), size the rollout buffer as a small multiple of episode length so each PPO update sees several complete episodes, not a truncated slice of one |
| `n_envs` | 4 | **Medium** | L2D uses `num_envs=4`; Song et al. uses `parallel_iter=20`. This plan recommends L2D's more modest range as the single-GPU/CPU student-budget-appropriate starting point (`../docs/planificacion/Step 4 Thesis.md` §1.5) — Song et al.'s 20 parallel environments likely assumes more compute than this budget affords; revisit upward only if headroom is confirmed |
| `batch_size` | 64 | **Medium** | Canonical SB3 default. Song et al.'s `minibatch_size=512` assumes their much larger `n_steps × parallel_iter` product (600 × 20 = 12,000 vs. this plan's 216×4=864 or 600×4=2,400) — 64 is a divisor of both this plan's Stage A and Stage B buffer sizes |
| `num_layers` (GIN depth) | 3 | **High** | Matches the current validated `GINEncoder` default and L2D's `num_layers=3` exactly (see `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` Table 2.1) |
| `hidden_dim` | 64 | **High** | Matches the current validated code and L2D's `hidden_dim=64` exactly |

**Stage-specific `total_timesteps` target (this plan's own choice, not literature-sourced):** Stage A, **300,000–500,000 timesteps** — at `n_steps=216 × n_envs=4 = 864` per update, this is roughly 350–580 PPO updates, i.e. several thousand complete `ft06` episodes, which is the right order of magnitude to see a clear reward/entropy/makespan trend on a 36-node graph within a single sprint day of wall-clock time. Stage B, **300,000–400,000 timesteps** as a stretch target within the sprint — the larger graph (100 vs. 36 nodes, plus the new machine/eligibility relation) costs more per step, so if wall-clock runs out, a partial Stage B run that shows the *right-shaped* curve (§3) is an acceptable sprint deliverable; full Stage B convergence is not required this week, matching the "not this sprint's goal" framing above.

---

## 3. Success criteria: learning correctly vs. broken

The single most useful thing this document can specify precisely is what a **healthy** curve looks like versus a **broken-pipeline** curve, because the two are easy to conflate at a glance ("reward isn't increasing much" could mean either "still learning, be patient" or "gradients aren't flowing at all"). Criteria below are stated per metric, with the concrete numeric anchors this codebase already has on record (`BASELINES_RESULTS_WEEK12.md`) rather than only qualitative language.

### 3.1 Episode return (reward curve)

- **Structural fact, not a target:** because `r_t = (H(s_t) - H(s_{t+1})) \cdot \text{reward\_scale}` telescopes, the undiscounted episode return always equals exactly $(H_0 - C_{\max}) \cdot \text{reward\_scale}$, where $\text{reward\_scale} = 1/\text{static\_lb}$. Since $C_{\max} \geq \text{static\_lb}$ always, return is bounded above by roughly $(H_0 - \text{static\_lb})/\text{static\_lb}$ (small, near zero) and unbounded below by a bad policy's makespan.
- **Healthy:** return starts negative (early policy makespans well above the lower bound) and trends **upward, toward 0**, with the trend visible within the first ~20–30% of Stage A's timestep budget and continuing to tighten after. This is exactly the "reward should trend toward 0 as it approaches the $-C_{\max}$ optimum" behavior — restated precisely here as "return $\to (H_0 - \text{static\_lb})/\text{static\_lb}$, i.e. toward its structural ceiling, not literally to 0" (the two coincide only when $H_0 \approx \text{static\_lb}$, which is typically close but not exact).
- **Concrete anchor for `ft06`:** `static_lb = 55` (proven optimum, from `BASELINES_RESULTS_WEEK12.md`). The best PDR (`fifo`) reaches makespan 67 (21.82% RPD); GA/CP-SAT reach 55 (0% RPD). A healthy Stage A run should see best-episode makespan cross below 67 (beat the best fixed rule) well before the timestep budget is exhausted, and ideally approach the 55–60 range by the end — this is the RL-beats-best-PDR criterion `../docs/planificacion/Step 4 Thesis.md`'s own results-matrix row ("Estático JSSP: mejor que la mejor PDR; por encima de CP-SAT") sets as the target shape, not just a training nicety.
- **Broken:** return flat at its initial (very negative) value for the entire run — no learning signal is reaching the policy at all. Check, in order: is `action_masks()` actually being consumed by `MaskableCategorical` (a masking bug silently falling back to an unmasked distribution would make most sampled actions raise `ValueError` inside `JSSPEnv.step()`, so this failure mode is more likely to crash than silently plateau — but a masking bug on the *machine-eligibility* side, new in the heterogeneous encoder, could still corrupt gradients without crashing); is `gamma=1.0` actually wired through (a `gamma<1` bug would still show learning, but with a systematically different — usually worse — final makespan than the reward-telescoping argument predicts); are the new `adj_eligibility`/`machine_features` tensors reaching the encoder as the right dtype/shape (an all-zero or all-one `adj_eligibility` would make the eligibility relation a no-op, degrading the heterogeneous encoder to something worse than useless — a homogeneous encoder with a dead branch and slower training, not equivalent to `use_machine_edges=False`).

### 3.2 Entropy

- **Healthy:** entropy starts near $\ln(\text{avg legal actions})$ (a near-uniform distribution over the frontier operations, which for `ft06` is at most 6 legal actions early on) and **decays gradually** over the course of training, tracking the return curve's improvement — the policy becomes more confident as it gets better, not before.
- **Broken — collapse:** entropy drops to near-zero within the first few PPO updates (tens of thousands of timesteps or less), while return is still near its bad initial value. This is the single clearest "something is structurally wrong, not just slow" signal, and the most likely causes given this sprint's specific changes are: (a) a `PerNodeActorHead` weight-initialization or scale issue producing extreme logits before any training has had a chance to shape them (check the MLP's final-layer initialization — `build_mlp`'s default `nn.Linear` init should not by itself cause this, so an extreme-logit collapse points at something size-dependent, e.g. logits not being properly masked to `-inf` before softmax, so illegal-action mass concentrates onto one path); (b) `adj_eligibility` degenerate to one-hot-everywhere-identical, making every machine's message identical and collapsing the encoder's useful signal early. Investigate with `PerNodeActorHead`'s raw logit magnitudes (log them, §4), not just the resulting entropy number.
- **Broken — no decay:** entropy stays flat near its maximum for the entire run even as return improves. Less severe than collapse, but indicates the policy is improving through the critic/advantage signal alone without ever committing to better actions — check `ent_coef` isn't set too high, and that `clip_range` isn't preventing the policy distribution from actually moving (persistently high clip fraction, §3.4).

### 3.3 Value loss / explained variance

- **Healthy:** value loss decreases over training and stabilizes at a low, non-zero plateau; `explained_variance` (SB3 logs this natively) trends toward 1.0.
- **Broken:** value loss diverges (increasing, or oscillating with growing amplitude) — classic sign the `learning_rate` is too high for this architecture's larger parameter count (heterogeneous encoder has roughly double the linear layers of the homogeneous one); per the hyperparameter table's flagged fallback, try L2D's `2e-5` before assuming a code bug. `explained_variance` stuck near 0 or negative for the whole run indicates the critic's `PooledCriticHead` input (the 128-dim concatenated pooled embedding) isn't actually informative — check that `adj_eligibility`/`machine_features` are reaching the *critic* path too, not just the actor path (an easy mistake given the extractor now returns two different things, per `JSSPHeteroGraphFeaturesExtractor`'s docstring in the architecture spec §3.3).

### 3.4 PPO-specific diagnostics (log even though not requested explicitly, because they disambiguate the above)

- **`approx_kl`:** should stay small and bounded, not spike — a spike coincides with (and helps diagnose) a `learning_rate` that's too aggressive for the new, larger network.
- **`clip_fraction`:** persistently near 1.0 (nearly every update is being clipped) suggests steps are too large; persistently near 0 suggests the policy has stopped moving (consistent with the "no decay" entropy failure mode above).

### 3.5 Makespan curve (the human-readable summary of all of the above)

Plot alongside `known_optimum` exactly as `experiments/train_ft06.py::plot_curves` already does — this sprint's Stage A run should produce the same two-panel (return, makespan) figure, extended with the entropy/value-loss/KL panels from TensorBoard (§4) for anything that needs deeper diagnosis than the makespan plot alone can show.

---

## 4. Monitoring and logging plan

**Tooling choice: TensorBoard, not Weights & Biases, for this sprint.** `stable_baselines3`/`sb3_contrib` have first-class built-in TensorBoard support (`tensorboard_log=...` on `MaskablePPO`'s constructor, zero extra code) that already logs the PPO-specific diagnostics in §3.4 (`approx_kl`, `clip_fraction`, `explained_variance`, value/policy loss) out of the box. W&B would need an extra `WandbCallback` integration and, more importantly, an internet-connected run under an account — an unnecessary dependency for a single-GPU, single-experimenter sprint where TensorBoard's local `logdir` is sufficient and matches `experiments/train_ft06.py`'s existing "headless, reproducible on a reviewer's machine" design principle (`matplotlib.use("Agg")`, no external service calls anywhere in the current training script). Revisit W&B once experiments need to be compared across machines or shared with the committee live — flagged as a future decision, not made here by default.

### 4.1 What to log, and where it comes from

| Metric | Source | Panel |
|---|---|---|
| `rollout/ep_rew_mean`, `rollout/ep_len_mean` | SB3 built-in, from the `Monitor` wrapper (already used in `train_ft06.py`) | `rollout/` |
| `train/entropy_loss`, `train/policy_gradient_loss`, `train/value_loss`, `train/approx_kl`, `train/clip_fraction`, `train/explained_variance` | SB3 built-in (`MaskablePPO.train()` logs these every update automatically once `tensorboard_log` is set) | `train/` |
| Per-episode makespan vs. `known_optimum` | **New**, extend `EpisodeStatsCallback` (`experiments/train_ft06.py`) to also call `self.logger.record("custom/makespan_rpd_pct", ...)` each time an episode ends, reusing the exact RPD formula `run_evaluation.py::build_row` already uses so the number is comparable to the baselines CSV without a second formula to keep in sync | `custom/` |
| Raw per-node actor logit magnitude (mean/max abs value, pre-mask) | **New**, a lightweight hook on `PerNodeActorHead.forward` (log every N steps, not every step, to avoid slowing training) — this is the direct diagnostic for the entropy-collapse failure mode in §3.2 | `custom/` |
| Machine-node embedding norm (mean over the batch) | **New**, same mechanism, on `HeteroGINEncoder`'s machine-branch output — a norm collapsing to ~0 or exploding indicates the new eligibility relation (§2 of the architecture spec) is unstable before the actor/critic heads even see it | `custom/` |
| Stage C (generalization) results: RPD vs. optimum per instance, per size tier | **New**, a small script (`experiments/eval_generalization.py`, mirroring `run_evaluation.py`'s CSV-row shape so the same downstream Wilcoxon/Friedman tooling can consume it later) run once at the end of Stage B, not logged to TensorBoard (it's a one-shot table, not a training curve) — write it as a CSV with the same columns `run_evaluation.py` produces, `method="rl_zeroshot"`, so it drops into the existing results pipeline without a new schema |

### 4.2 Practical logging setup

```python
# experiments/train_ft06.py (Stage A) / a new experiments/train_hetero.py (Stage B) --
# extends build_maskable_ppo's caller, does not modify build_maskable_ppo itself.
model = build_maskable_ppo(
    env,
    hidden_dim=64, num_layers=3,
    n_steps=216,          # Stage A; 600 for Stage B
    batch_size=64,
    n_epochs=3,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    learning_rate=2e-4,   # fall back to 2e-5 per the hyperparameter table's flagged note if unstable
    gamma=1.0,
    seed=args.seed,
    tensorboard_log=str(RESULTS_DIR / "tb_logs"),
    verbose=1,
)
model.learn(total_timesteps=args.timesteps, callback=[EpisodeStatsCallback(), CustomDiagnosticsCallback()])
```

Run TensorBoard locally against `experiments/results/tb_logs/` (`tensorboard --logdir experiments/results/tb_logs`) — no code beyond the `tensorboard_log` argument and the two additive callbacks above is required to get the SB3-native metrics; only the `custom/*` panel needs new callback code.

### 4.3 Checkpointing

Save `model.save(...)` at minimum: (a) after Stage A completes, (b) every ~50,000 timesteps during Stage B (so a mid-run crash doesn't lose the whole day's compute — Stage B is the more expensive, more failure-prone stage given the new architecture), and (c) at the end of Stage B, which is the checkpoint Stage C's zero-shot transfer and `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §8's `TrainedRLBaseline` both load from. Name checkpoints with the instance/stage/timestep in the filename (`ppo_hetero_stageB_ft10_la1620_step300000.zip`) — `experiments/train_ft06.py`'s current single fixed filename per instance (`ppo_gin_{instance}.zip`) is fine for a single-instance smoke test but would silently overwrite across Stage B's five-instance round-robin if copied verbatim.

---

## 5. Sprint schedule (5 working days)

| Day | Work | Exit criterion |
|---|---|---|
| 1 | `envs/jssp_env.py`: add `_machine_features()`, `NUM_MACHINE_FEATURES`, `MACHINE_FEATURE_NAMES`, `adj_eligibility`, additive `observation_space` keys (architecture spec §1.2). `models/gin.py`: `HeteroRelationalGINLayer`, `HeteroGINEncoder` (§2.2). Unit tests: `test_hetero_encoder_output_shapes`, homogeneous-path regression (§6 of the spec). | `tests/test_env.py`, `tests/test_policy.py`, and the new hetero-encoder shape test all pass; `git diff` touches only additive code per the prime directive. |
| 2 | `models/policy.py`: `PerNodeActorHead`, `PooledCriticHead`, `JSSPHeteroGraphFeaturesExtractor`, `PerNodeMaskableActorCriticPolicy` (§3.1–3.3), pinned against the installed `sb3_contrib==2.9.0`/`stable_baselines3==2.9.0`. Unit tests: parameter-count-independent-of-N, weight-transfer, action-masking-unchanged-semantics. | All new `tests/test_policy.py` additions pass; a random-weight forward pass on `ft06` and `la06` both succeed through the same policy object. |
| 3 | Stage A training run (`ft06`, hyperparameters from §2, TensorBoard wired per §4). Watch the §3 diagnostics live; if entropy collapses or value loss diverges within the first ~50k timesteps, stop and debug rather than letting the run finish uninterpretably. | Reward curve trends toward its structural ceiling; best-episode makespan crosses below 67 (best PDR, `fifo`) at least once; no collapse/divergence per §3. |
| 4 | Stage B training run (`ft10` + `la16`–`la20`, round-robin via `env.reset(options={"instance": ...})`), warm-started from Stage A's checkpoint (size-agnostic weights make this a meaningful warm start, not just a convenience — this is worth reporting as an empirical curriculum-learning result in its own right, not only a compute-saving trick). | Same §3 criteria, evaluated per-instance; checkpoint saved per §4.3. |
| 5 | Stage C zero-shot generalization sweep (`experiments/eval_generalization.py`, all `la*` instances not used in Stage B training), wire the `TrainedRLBaseline` stub from `../docs/especificaciones/GNN_PPO_Architecture_Spec_v2.md` §8 against the Stage B checkpoint, write up results (RPD vs. optimum per size tier, entropy/reward curve figures). | CSV of Stage C results in the `run_evaluation.py`-compatible schema; `TrainedRLBaseline` passes `test_rl_baseline_matches_protocol`; short write-up ready for committee/advisor review. |

---

## 6. Open risks to flag going into the sprint (not resolved here, on purpose)

- **`sb3_contrib` version-pinned override point (architecture spec §3.3):** the exact method to override for a per-node action head is version-specific. Confirm against the installed `2.9.0` before Day 2 starts, not during it.
- **10×10 wall-clock cost is genuinely unknown until Day 4 is underway** — Stage B's `total_timesteps` target (§2) is this plan's own estimate, not measured. If Day 4's actual throughput is far below expectation, the correct response is to reduce Stage B's timestep target and say so explicitly in the write-up, not to silently let Stage C run on an under-trained checkpoint and present it as more converged than it is.
- **Stage C's zero-shot numbers may look worse than Stage A/B's in-distribution numbers, and that is expected, not a bug** — `../docs/planificacion/Step 4 Thesis.md`'s own results matrix predicts "degradación moderada, aún ≤ PDR" (moderate degradation, still ≤ PDR) for the 2–3× generalization case, not parity. Don't over-interpret a first-sprint generalization gap as an architecture failure; it's a baseline measurement this sprint exists to produce, to be improved on in later sprints.
