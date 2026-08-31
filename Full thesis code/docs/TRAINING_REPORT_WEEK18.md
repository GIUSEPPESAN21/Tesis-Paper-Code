# Training Report — Week 18: Heterogeneous GNN + PPO, first trained policy

**Sprint goal.** Implement the heterogeneous GNN encoder and PPO policy specified in
`GNN_PPO_Architecture_Spec_v2.md`, and produce the first evidence that it *learns the
right thing*, not merely that it runs — the bar `TrainingPlan_Week17.md` §3 sets.

**Headline result.** Both training runs are healthy by every criterion in
`TrainingPlan_Week17.md` §3, with no diagnosed pipeline bug. On `ft06` the trained
policy's single greedy rollout reaches **makespan 55 — the proven optimum** — in 0.18 s,
beating the best of the nine PDRs (`fifo`, 67) by 17.9% and matching GA (60 s) and CP-SAT.
On the flexible `mk01` benchmark the greedy rollout reaches **43 vs. the proven optimum
40** (7.5% RPD), between GA (42) and the untrained-policy reference (70), at roughly 200×
GA's wall clock.

| | `ft06` (JSSP) | `mk01` (FJSP) |
|---|---:|---:|
| Timesteps | 400,000 | 300,000 |
| Wall clock (CPU, 3 torch threads) | 36.9 min | 42.1 min |
| Episodes completed | 11,112 | 5,016 |
| First episode makespan | 113 | 104 |
| Best makespan seen in training | **55** | **42** |
| Mean makespan, last 100 episodes | 57.03 | 46.24 |
| Greedy rollout makespan | **55** | 43 |
| Best of 100 sampled rollouts | **55** | 42 |
| Reference optimum | 55 (proven) | 40 (proven) |
| Static lower bound | 52 | 36 |
| Every rollout feasible | yes | yes |

---

## 1. What was built

Four new modules (`envs/fjsp_env.py`, `envs/fjsp_loader.py`,
`scripts/fetch_fjsp_instances.py`, `experiments/train_hetero.py`), three new test files,
and eight modified source files. The full test suite went from **510 to 585 passing tests
with zero regressions** (`experiments/results/week18/pytest_week18.txt`). Every change is
additive: `RelationalGINLayer`, `GINEncoder`, `JSSPGraphFeaturesExtractor`,
`build_maskable_ppo` and `tests/test_policy.py` are untouched, per the architecture spec's
prime directive, and the homogeneous path remains available as the ablation configuration.

| Step | Where | What |
|---|---|---|
| 1. Graph construction | `envs/jssp_env.py` (additive), `envs/fjsp_env.py` (new) | `NUM_MACHINE_FEATURES = 7`, `MACHINE_FEATURE_NAMES`, `_machine_features()`, and the `machine_features` / `adj_eligibility` observation keys, exactly per spec §1.2. Operation nodes, machine nodes, precedence edges (both directions), machine-sharing edges, eligibility edges. |
| 2. Heterogeneous encoder | `models/gin.py` | `HeteroRelationalGINLayer` (five relation-specific weight matrices: four updating operation nodes, one updating machine nodes) and `HeteroGINEncoder`. |
| 3. Actor-critic heads | `models/policy.py` | `PerNodeActorHead` (JSSP), `PerPairActorHead` (FJSP, `(operation, machine)` logits), `PooledCriticHead`, `JSSPHeteroGraphFeaturesExtractor`, `PerNodeMaskableActorCriticPolicy`. |
| 4. PPO loop | `experiments/train_hetero.py` | `sb3_contrib.MaskablePPO` on the custom policy, `TrainingPlan_Week17.md` §2 hyperparameters as defaults, §4 logging, §4.3 checkpointing. |
| 5. Training + comparison | `experiments/results/` | Curves, checkpoints, metrics CSVs, and `week18_comparison.csv` produced through the existing `run_evaluation.py` harness. |
| 8 (spec §8) | `baselines/interface.py`, `run_evaluation.py` | `TrainedRLBaseline` replaces `RLPlaceholderBaseline` behind `--rl-policy`; zero CSV-schema change. |

### 1.1 Three decisions that deviate from a literal reading of the brief

These are flagged rather than buried, because each is a place where the sprint brief, the
architecture spec, and the codebase's own constraints did not all agree.

**(a) Dense `bmm` message passing instead of PyTorch Geometric's `HeteroConv`.** The brief
asked for PyG. PyG is not installed, the environment is CPU-only `torch 2.12.1+cpu`, and
`models/gin.py`'s existing module docstring already records the project's standing
decision against PyG: its wheels are pinned to an exact torch+CUDA build and it is "the
single most common reason a published DRL scheduling repo fails to install," which matters
for a thesis that must be reproducible on a reviewer's machine years from now. Adding it
would also have contradicted the architecture spec, which specifies the exact `bmm`-based
`forward` signature implemented here (§2.2). What `HeteroConv` provides — a per-relation
convolution with relation-specific weights — is what the five `nn.Linear` + `bmm` terms in
`HeteroRelationalGINLayer` compute. At these sizes (36–225 operation nodes, 4–15 machine
nodes) the dense product is also *faster* than gather/scatter over an edge index, and
`adj_eligibility` is dense by construction. Swapping in PyG later remains a change
confined to `models/gin.py`.

**(b) `stable-baselines3` rather than a custom PPO.** The brief left this to judgement
"but document the reasoning." Heterogeneous batching turned out not to be an obstacle:
every environment in a run has a fixed, identical graph size, so a batch of observations is
a stack of equal-shaped dense tensors — precisely the `(B, N, F)` / `(B, N, N)` layout SB3's
`Dict` observation buffer already produces, with no ragged collation. Keeping SB3
preserves the tested `MaskableCategorical` masking path, the tested GAE/PPO-loss
implementation, and the `tests/test_policy.py` regression contract. SB3 did force one
thing: a features extractor alone cannot host a per-node head, because SB3 applies a final
`nn.Linear(features_dim, action_space.n)` on top of whatever the extractor returns. Hence
`PerNodeMaskableActorCriticPolicy`, which overrides `_get_action_dist_from_latent` —
verified against the installed `sb3_contrib==2.9.0`, as spec §3.3 requires.

**(c) FJSP and the `(operation, machine)` action space were explicitly out of scope.**
`GNN_PPO_Architecture_Spec_v2.md` §0 and `TrainingPlan_Week17.md`'s "not this sprint's
goal" both exclude FJSP support and fix the action space at `Discrete(num_operations)`.
The sprint brief's STEP 3 and STEP 5 require the opposite. The brief was followed: both
heads exist (`PerNodeActorHead` for fixed routing, `PerPairActorHead` for flexible), the
policy selects between them from the action space automatically, and `FJSPEnv` supplies
the flexible MDP. **The JSSP path is unchanged by this** — `ft06` still trains through a
`Discrete(36)` action space, so the spec's §0 boundary is honoured where it governs and
extended only where the brief asked. The scope docs should be updated to match.

### 1.2 MK01 was fetched, not transcribed — and the parse is verified

No FJSP benchmark loader existed (`envs/fjsp_instance.py`: *"No real FJSP benchmark
loader (Brandimarte `MK*`, Hurink) is bundled this sprint"*), only a synthetic generator
with no literature optimum and therefore no RPD denominator.
`scripts/fetch_fjsp_instances.py` + `envs/fjsp_loader.py` close that gap, following
`scripts/fetch_instances.py`'s stated principle of never hand-transcribing benchmark data.

One real modelling decision was required: Brandimarte jobs are ragged (`mk01`: six jobs of
6 operations, four of 5), while `FJSPInstance` is rectangular — and that rectangularity is
load-bearing far beyond the parser, since `op_id = job * ops_per_job + position` is the
action id, the GA gene index, and every adjacency row index in the codebase. Short jobs are
therefore padded to `ops_per_job = 6` with zero-duration operations, which `FJSPEnv` treats
as occupying no machine capacity.

**The neutrality of that padding is verified, not asserted:** CP-SAT solves the padded
`mk01` to `status=OPTIMAL, makespan=40` — exactly the literature optimum. That single check
simultaneously validates the parse, the machine indexing, and the padding
(`tests/test_fjsp_env.py::test_cpsat_reaches_the_literature_optimum_on_the_padded_instance`).
It also surfaced a latent bug in `is_fjsp_schedule_feasible`, which flagged spurious
machine overlaps for zero-duration operations; that is fixed (an empty interval cannot
overlap anything).

---

## 2. Training configuration as actually run

`TrainingPlan_Week17.md` §2 verbatim, with no tuning — the sprint brief's instruction to
"prioritize getting a correct, even if not yet optimal, training loop over premature
tuning." No hyperparameter was changed from the plan's recommendation in either run.

| Hyperparameter | `ft06` | `mk01` | Source |
|---|---:|---:|---|
| `gamma` | 1.0 | 1.0 | Structural (reward telescoping), not tunable |
| `learning_rate` | 2e-4 | 2e-4 | Song et al.; the plan's fallback to 2e-5 was **not** needed |
| `clip_range` | 0.2 | 0.2 | L2D / Song et al. / Schulman et al. |
| `ent_coef` | 0.01 | 0.01 | L2D / Song et al. |
| `gae_lambda` | 0.95 | 0.95 | Schulman et al. (2016) |
| `n_epochs` | 3 | 3 | Song et al. |
| `vf_coef` | 0.5 | 0.5 | Song et al. / SB3 default |
| `n_steps` (per env) | 216 (6 × 36) | 360 (6 × 60) | Plan's own rule: 6 × episode length |
| `n_envs` | 4 | 4 | L2D |
| `batch_size` | 64 | 64 | SB3 default; divides both buffers |
| `hidden_dim` / `num_layers` / `mlp_layers` | 64 / 3 / 2 | 64 / 3 / 2 | L2D, matching the validated `GINEncoder` |
| Policy parameters | 97,528 | 101,624 | Independent of instance size (the difference between the two columns is the actor head only: `PerPairActorHead` consumes a concatenated operation+machine embedding, so its first layer is twice as wide) |

**Hardware note, per the plan's §1 request to flag this rather than assume it:** the run
was CPU-only (8 cores, `torch 2.12.1+cpu`, no CUDA), not the single GPU the plan assumes.
Both runs were executed concurrently at 3 torch threads each, so the wall-clock figures
include contention and are an upper bound on single-run cost. Throughput measured in
isolation beforehand was 275 steps/s (`ft06`) and 134 steps/s (`mk01`).

---

## 3. Training curves

![ft06 training curves](experiments/results/week18/training_curves_hetero_stageA_ft06_seed0.png)

![mk01 training curves](experiments/results/week18/training_curves_hetero_stageF_mk01_seed0.png)

Six panels each: episode return, episode makespan vs. the reference optimum, entropy loss,
value loss (log scale), explained variance, and the PPO step-size diagnostics
(`approx_kl`, `clip_fraction`). Raw data: `experiments/results/tb_logs/<run>/progress.csv`
(SB3-native, includes the `custom/*` diagnostics) and
`experiments/results/train_<run>_metrics.csv` (per episode). TensorBoard event files are in
the same directories: `tensorboard --logdir experiments/results/tb_logs`.

---

## 4. Assessment against the §3 success criteria

The plan's §3 is written as healthy-vs-broken pairs. Each is evaluated literally below.

### 4.1 Episode return — **healthy, both runs**

The plan states the structural fact that undiscounted return equals
$(H_0 - C_{\max}) / \text{static\_lb}$ exactly, and that the target is the *structural
ceiling*, not zero. Those ceilings are computable and were computed:

| | $H_0$ | static LB | Ceiling at the reference optimum | Observed final `ep_rew_mean` |
|---|---:|---:|---:|---:|
| `ft06` | 47 | 52 | $(47-55)/52 = -0.154$ | **−0.193** |
| `mk01` | 22 | 36 | $(22-40)/36 = -0.500$ | **−0.673** |

`ft06` starts at −0.98 and rises to −0.193 against a ceiling of −0.154 — it has closed
roughly 95% of the available gap. `mk01` starts at −2.18 and rises to −0.673 against a
ceiling of −0.500, closing about 90%. Both curves rise steeply for the first ~25% of the
budget and continue tightening after, which is the shape §3.1 describes as healthy. Neither
is flat at its initial value, so the "no learning signal is reaching the policy at all"
failure mode is ruled out.

The reward-telescoping identity was additionally verified directly as a unit test
(`tests/test_fjsp_env.py::test_reward_telescopes_to_h0_minus_makespan`), so the
`gamma=1.0` argument rests on a checked property, not an assumed one.

**§3.1's concrete `ft06` anchor — "best-episode makespan should cross below 67 (the best
fixed rule) well before the budget is exhausted, and ideally approach 55–60 by the end" —
is met with room to spare:** 67 is crossed at timestep **6,192** (1.5% of the budget, first
episode at 65), the optimum 55 is first reached at timestep **37,296** (9.3% of the budget),
and the last-100-episode *mean* is 57.03.

### 4.2 Entropy — **healthy, both runs**

| | Initial entropy | Final entropy | $\ln(\text{legal actions})$ at reset |
|---|---:|---:|---:|
| `ft06` | 1.51 | 0.61 | $\ln 6 = 1.79$ |
| `mk01` | 2.45 | 0.97 | $\ln 18 = 2.89$ |

Both start near the near-uniform value and **decay gradually across the whole run**,
tracking the return curve rather than preceding it — exactly §3.2's healthy description.
Neither failure mode is present: no collapse to near-zero in the first few updates, and no
flat-at-maximum curve.

The plan flagged entropy collapse from an extreme-logit actor head as the single most
likely structural failure for a brand-new head, and asked for raw logit magnitudes to
disambiguate it. They were logged (`custom/actor_logit_abs_mean` / `_abs_max`):

| | Logit \|mean\| at start → end | Logit \|max\| at start → end |
|---|---|---|
| `ft06` | 0.006 → 1.64 | 0.011 → 3.78 |
| `mk01` | 0.001 → 5.25 | 0.006 → 8.71 |

Logits start essentially at zero — the deliberate small-gain (0.01) initialisation on the
actor head's final layer — and grow smoothly. There is no blow-up. A pre-emptive unit test
pins this at initialisation:
`test_initial_policy_is_near_uniform_over_legal_actions` asserts entropy ≈ `ln(legal)`
within 5%, measured at 2.7080 vs. `ln(15) = 2.7081` on `la06`.

### 4.3 Value loss and explained variance — **healthy, both runs**

Value loss falls monotonically on a log scale in both runs (`ft06`: 5×10⁻² → 2.5×10⁻⁴;
`mk01`: 3×10⁻² → 4×10⁻³) and stabilises at a low, non-zero plateau. No divergence, so the
plan's flagged fallback to L2D's `lr=2e-5` was not triggered.

`explained_variance` rises immediately and holds at **0.83 (`ft06`)** and **0.91
(`mk01`)**. The plan specifically warned that a value stuck near 0 or negative would
indicate the machine/eligibility tensors were reaching the actor path but not the critic's
— an easy mistake given the extractor now returns two different things. The high values
rule that out empirically, and
`test_graph_embedding_is_concat_of_the_two_pools` pins it structurally: the critic's input
really is `concat(pool(operation), pool(machine))`, not one pool.

### 4.4 PPO step-size diagnostics — **healthy, both runs**

`approx_kl` is small and bounded throughout (`ft06` 0.016, `mk01` 0.022) with no spikes;
`clip_fraction` sits at 0.13 (`ft06`) and 0.18 (`mk01`) — comfortably between the plan's two
pathological readings of "persistently near 1.0" (steps too large) and "persistently near
0" (policy has stopped moving).

### 4.5 The eligibility relation is not a dead branch

§3.1 warns that an all-zero or degenerate `adj_eligibility` would silently degrade the
heterogeneous encoder to "something worse than useless — a homogeneous encoder with a dead
branch and slower training." Three independent checks:

* `test_eligibility_relation_actually_carries_signal` — zeroing `adj_eligibility` provably
  changes the encoder output.
* `custom/machine_embedding_norm` stayed in a narrow band throughout (`ft06` 8.20 → 9.43;
  `mk01` 7.81 → 7.85) — neither collapsing toward 0 nor exploding, which is the plan's §4.1
  criterion for the new relation being stable.
* On `mk01` the machine branch is not merely stable but *load-bearing*: the machine
  embeddings are half of every `(operation, machine)` logit, and the policy reaches 43 vs.
  70 for a uniform-random masked policy.

### 4.6 Verdict table

| §3 criterion | `ft06` | `mk01` |
|---|---|---|
| 3.1 Return trends toward its structural ceiling | ✅ −0.98 → −0.193 (ceiling −0.154) | ✅ −2.18 → −0.673 (ceiling −0.500) |
| 3.1 Best makespan beats best PDR (67 on `ft06`) | ✅ 55, crossed 67 by ~50k | n/a (no PDR supports FJSP) |
| 3.2 Entropy decays gradually, no collapse | ✅ 1.50 → 0.61 | ✅ 2.45 → 0.97 |
| 3.3 Value loss decreases and stabilises | ✅ 5e-2 → 2.5e-4 | ✅ 3e-2 → 4e-3 |
| 3.3 `explained_variance` → 1.0 | ✅ 0.83 | ✅ 0.91 |
| 3.4 `approx_kl` bounded, no spikes | ✅ 0.016 | ✅ 0.022 |
| 3.4 `clip_fraction` not saturated at 0 or 1 | ✅ 0.13 | ✅ 0.18 |
| Every produced schedule feasible | ✅ | ✅ |
| **Converged** | ✅ effectively (see §6.1) | ❌ **not converged** (see §6.2) |

---

## 5. Makespan comparison against the baselines

Produced through the existing harness, with no bespoke evaluation path — the swap the
architecture spec §8 was designed for:

```bash
python run_evaluation.py --instances ft06 --methods pdr ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageA_ft06_final.zip
python run_evaluation.py --fjsp --instances mk01 --methods ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageF_mk01_final.zip
```

Full CSV: `experiments/results/week18/week18_comparison.csv`.

### 5.1 `ft06` (JSSP, 6×6, proven optimum 55)

| Method | Makespan | RPD vs. optimum | Wall clock (s) | Feasible |
|---|---:|---:|---:|:--:|
| `cr` | 171 | 210.91% | 0.016 | ✓ |
| `lwkr` | 154 | 180.00% | 0.007 | ✓ |
| `edd` | 154 | 180.00% | 0.018 | ✓ |
| `lpt` | 132 | 140.00% | 0.008 | ✓ |
| `lor` | 132 | 140.00% | 0.016 | ✓ |
| `spt` | 122 | 121.82% | 0.013 | ✓ |
| `mwkr` | 74 | 34.55% | 0.009 | ✓ |
| `mor` | 73 | 32.73% | 0.010 | ✓ |
| `fifo` (best PDR) | 67 | 21.82% | 0.012 | ✓ |
| **`rl` (this sprint, greedy)** | **55** | **0.00%** | **0.179** | ✓ |
| `ga` (60 s budget) | 55 | 0.00% | 60.029 | ✓ |
| `cpsat` (proven optimal) | 55 | 0.00% | 0.116 | ✓ |

The trained policy **beats all nine PDRs and matches the two exact/metaheuristic methods**,
in 0.18 s versus GA's 60 s. This is the target shape `Step 4 Thesis.md`'s results matrix
sets for the static JSSP row ("*mejor que la mejor PDR; por encima de CP-SAT*") — except
that here it is not merely above the best PDR but level with CP-SAT. The best of 100
*sampled* rollouts is also 55, and their mean is 55.15, so this is a property of the learned
policy rather than one lucky rollout.

### 5.2 `mk01` (FJSP, 10×6, 55 real operations, proven optimum 40)

| Method | Makespan | RPD vs. optimum | Wall clock (s) | Feasible |
|---|---:|---:|---:|:--:|
| Untrained policy / uniform-random over legal pairs (best of 200) | 70 | 75.00% | — | ✓ |
| Untrained policy, mean of 200 | 100.8 | 152.00% | — | ✓ |
| **`rl` (this sprint, greedy)** | **43** | **7.50%** | **0.296** | ✓ |
| **`rl` (best of 100 sampled)** | **42** | **5.00%** | 26.6 | ✓ |
| `ga` (60 s budget) | 42 | 5.00% | 60.013 | ✓ |
| `cpsat_fjsp` (proven optimal) | 40 | 0.00% | 0.250 | ✓ |

No PDR row exists because `PDRBaseline` rejects FJSP instances by design — a dispatching
rule has no machine-selection component (`baselines/interface.py`). The untrained-policy
row is included in its place as the honest "what does the action space give you for free"
reference, and the gap between it (70 best / 100.8 mean) and the trained policy (43 greedy)
is the clearest single measure of what was learned.

**Reading this honestly:** the trained policy is *worse than GA on greedy* (43 vs. 42) and
*ties GA when sampled* (42), while being ~200× faster per solution than GA's 60 s budget.
It does not reach CP-SAT's proven optimum. Given that this is a first FJSP training run at
a hyperparameter setting transferred wholesale from a JSSP plan, on a variant the
architecture spec declared out of scope, 7.5% RPD is a reasonable baseline to improve on —
but it is a baseline, not a win.

---

## 6. Diagnosis: where it is not converged, and why

The sprint brief asks for diagnosis rather than a bare pass/fail. Neither run shows a
pipeline bug — the §4 checks were designed to catch masking, graph-construction, reward-scale
and dtype faults, and all pass — so what follows is about *optimisation*, not correctness.

### 6.1 `ft06`: converged on greedy, a residual gap on the mean

Greedy and best-of-100 both reach 55. But the last-100-episode **mean** is 57.03, i.e. the
stochastic policy still emits sub-optimal schedules some of the time, and `ep_rew_mean`
plateaus at −0.193 against a −0.154 ceiling. The most likely cause, in order:

1. **`ent_coef = 0.01` is holding entropy up on purpose.** Final entropy is 0.61 nats over
   a ≤6-action frontier — a deliberately non-deterministic policy. The entropy bonus is
   paying for exploration the run no longer needs by ~150k timesteps. An entropy-coefficient
   *schedule* (decaying 0.01 → 0.001) is the cheapest thing to try and would likely close
   most of the mean gap without touching the architecture. This is a tuning choice the
   sprint brief explicitly deferred.
2. **The return curve plateaus at ~150k of a 400k budget.** The last 250k timesteps bought
   very little. That is consistent with (1) rather than with an under-trained network, and
   argues for redirecting that budget to Stage B (10×10) rather than lengthening Stage A.
3. Not the learning rate: `approx_kl` (0.016) and `clip_fraction` (0.13) are both mid-range,
   so the policy is neither frozen nor over-stepping.

### 6.2 `mk01`: genuinely not converged — more timesteps is the first thing to try

**The return curve is still rising at 300k timesteps.** It has not plateaued, entropy is
still decaying (0.97 and falling, from 2.45), and the makespan mean is still drifting down.
This is an under-trained run, not a stuck one. Ranked hypotheses:

1. **Insufficient timesteps for a ~6× larger action space.** `mk01`'s action space is 360
   `(operation, machine)` pairs against `ft06`'s 36 operations, and its episode is 60 steps
   against 36 — yet it received *fewer* timesteps (300k vs. 400k) and, because episodes are
   longer, less than half as many episodes (5,016 vs. 11,112). The plan sized `total_timesteps`
   by graph size, not by action-space size, which under-provisions the flexible variant. **This
   is the single most likely explanation and the first thing to change:** 1M timesteps on
   `mk01` is roughly 2.3 hours on this hardware.
2. **The reward's optimism gap is much larger on FJSP.** $H_0 = 22$ against a static lower
   bound of 36 and an optimum of 40, because $H$ charges every unscheduled operation its
   *cheapest eligible* duration and ignores machine contention entirely — a far looser
   relaxation under flexible routing than under fixed routing (`ft06`: $H_0 = 47$ vs.
   optimum 55). The shaping signal is therefore weaker per step relative to the achievable
   makespan. A tighter FJSP $H$ (e.g. adding a per-machine load floor) would preserve the
   telescoping identity — any $H$ that is non-decreasing and equals the makespan at
   termination does — while sharpening the gradient. This is a *design* improvement, not a
   bug, and it is the second thing to try.
3. **The per-pair head may be under-parameterised for its job.** `PerPairActorHead` scores
   `MLP([h_ope ; h_mach])` at `hidden_dim=64`, inherited unchanged from the JSSP setting.
   Song et al. — the closest heterogeneous-FJSP precedent — use `hidden_size_ope=128`, and
   the architecture spec explicitly names widening to 128 as "a documented, cheap thing to
   try if the heterogeneous encoder underperforms." That condition is now met for the FJSP
   path specifically.
4. **Possibly too little exploration pressure late, per §6.1(1).** Lower priority here,
   since entropy is still falling naturally.

Explicitly *not* the diagnosis, with evidence: action masking (every one of ~5,000 rollouts
was independently verified feasible, and `test_per_pair_head_ordering_matches_env_action_encoding`
pins logit index `op * num_machines + machine` to the right pair); graph construction
(machine-embedding norms stable, eligibility relation proven load-bearing); reward scale
(telescoping identity unit-tested, `explained_variance` 0.91); `gamma` (1.0 verified wired
through, and the observed return matches the telescoping prediction).

### 6.3 One real bug was found and fixed during the sprint

`utilization_accum` (machine feature 6) was initially computed as `busy_time / current_time`
following the spec's literal wording. Under list scheduling an operation is placed at the
earliest feasible moment on its machine, which can be *later* than the next decision epoch,
so committed busy time routinely exceeds `current_time()` — the feature reached 3.0 where a
utilisation must lie in [0, 1], feeding a badly-scaled input to the encoder. A unit test
caught it (`test_utilization_never_exceeds_one`); the denominator is now the realised
schedule span, which bounds every committed interval by construction. **Both training runs
reported here were restarted from scratch after the fix** — no result in this document was
produced with the buggy feature.

---

## 7. Reproducing this

```bash
pip install -e ".[dev,monitoring]"
python scripts/fetch_fjsp_instances.py            # mk01 + its reference optimum
pytest                                            # 585 tests, ~6.5 min

python experiments/train_hetero.py --instance ft06 --stage A --timesteps 400000 --seed 0
python experiments/train_hetero.py --instance mk01 --fjsp --stage F --timesteps 300000 --seed 0

python run_evaluation.py --instances ft06 --methods pdr ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageA_ft06_final.zip
python run_evaluation.py --fjsp --instances mk01 --methods ga cpsat rl \
    --rl-policy experiments/results/checkpoints/ppo_hetero_stageF_mk01_final.zip
```

Artifacts, all under `experiments/results/`: `training_curves_hetero_stage{A,F}_*.png`,
`summary_hetero_stage{A,F}_*.json`, `train_hetero_stage{A,F}_*_metrics.csv`,
`tb_logs/<run>/progress.csv` + TensorBoard events, `checkpoints/*.zip` (every 50k steps plus
final), `week18_comparison.csv`, `pytest_week18.txt`.

---

## 8. What this sprint did *not* do

Stated plainly so nothing here reads as more converged or broader than it is.

* **No Stage B or Stage C.** `TrainingPlan_Week17.md` §1 specifies a 10×10 training tier
  (`ft10`, `la16`–`la20`) and a zero-shot generalization sweep over the remaining `la*`
  instances. Neither was run. The *mechanism* they depend on is implemented and tested —
  `test_weight_transfer_across_instance_sizes` loads `ft06`-trained weights into a policy
  wrapping `la06` (N=75) with `strict=True` and confirms a valid masked distribution, and
  the parameter count is identical (97,528) across sizes — but the sweep itself is
  outstanding, and `experiments/eval_generalization.py` is not written.
* **No `DynamicJSSPEnv` wiring.** Still blocked on the growable-array observation contract,
  exactly as spec §0 and §8.3 state. `TrainedRLBaseline` raises a clear `SystemExit` under
  any `--disruption` other than `none` rather than silently reporting nominal numbers in a
  disruption row.
* **No hyperparameter tuning, no ablations, no multiple seeds.** Every number here is
  seed 0 at the plan's stock hyperparameters. The `use_machine_edges=False` ablation path
  exists in both encoders but was not run, and the Song-et-al.-style attention aggregator
  remains a documented future ablation rather than an implemented one.
* **One FJSP instance.** `mk01` only. `scripts/fetch_fjsp_instances.py --all` will fetch the
  rest of the Brandimarte family, but nothing larger was trained or evaluated.

### Recommended next actions, in order

1. Re-run `mk01` at 1M timesteps (§6.2 hypothesis 1) — the cheapest test of the leading
   diagnosis, ~2.3 h on this hardware.
2. Tighten the FJSP `H(s)` with a per-machine load floor (§6.2 hypothesis 2).
3. Add an entropy-coefficient schedule and re-check `ft06`'s mean-makespan gap (§6.1).
4. Run Stage B and Stage C, which the size-agnostic weights are already tested to support.
5. Widen the FJSP head to `hidden_dim=128` (§6.2 hypothesis 3) if 1–2 do not close the gap.
6. Reconcile `GNN_PPO_Architecture_Spec_v2.md` §0 and `TrainingPlan_Week17.md` with the FJSP
   support that now exists (§1.1c).
