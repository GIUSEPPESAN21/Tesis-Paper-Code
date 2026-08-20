# HPO and Ablation Sprint — Week 21 Report

**Status: IN PROGRESS.** This report is being filled in as background runs complete; see
§0 for what is done, running, and still queued as of this writing. Per this sprint's
brief, any part not completed at full scale is flagged explicitly rather than silently
reduced or omitted.

---

## 0. Honest status and sequencing

Two things changed the literal sprint brief before any HPO trial ran, both flagged to
and confirmed by the user up front:

1. **`HPO_TechSpec.md` gates the search on `TrainingDiagnosis.md` §4 steps 1–2** (a
   multi-seed `ft06` sanity check and a `mk01` re-run at 1,000,000 timesteps), neither of
   which had been run before this sprint (only `seed0` runs existed). Those gate runs
   were launched first, in the background, before any Optuna trial.
2. **Compute reality**: this machine is CPU-only (`torch==2.12.1+cpu`, no CUDA, 8 cores).
   The tech spec's own estimates put Phase 1 HPO at 1–2 days and Phase 2 confirmation
   (dominated by `mk01`, ~20h/config × 3 configs) at 60+ hours — before ablations or the
   full curriculum retrain even start. The full protocol as literally scoped is a
   multi-day-to-week continuous-compute job, not something completable inside one
   sitting. Approach taken: build and correctness-test every step at tiny scale first,
   then launch the real full-budget runs as long-running background jobs.

**Completed so far:**
- Gate run 1 (`ft06`, seeds 1–4, 400k timesteps each) — launched, [STATUS: see §1].
- Gate run 2 (`mk01`, 1,000,000 timesteps) — launched, [STATUS: see §1].
- `experiments/hpo_hetero.py` implemented per `HPO_TechSpec.md` §6 (two-phase Optuna
  search, entropy-schedule callback, chunked-training pruning). Both implementation
  risks the spec flagged were resolved by direct verification against the installed
  `sb3_contrib==2.9.0`/`stable_baselines3==2.9.0`, not assumption (§2 below).
- Smoke-tested at tiny scale on both `ft06` (per-node head) and `mk01` (per-pair head) —
  [STATUS: see §2].
- Ablation code added: `HeteroRelationalGATHybridLayer` (`models/gin.py`), `reward_mode`
  on `JSSPEnv` (`envs/jssp_env.py`), wired through `models/policy.py` and covered by new
  tests in `tests/test_hetero_policy.py` / `tests/test_env.py` — full suite [STATUS: see
  §2].
- `experiments/run_ablations.py` (Runs 0/1a/1b/2) and `experiments/train_curriculum.py`
  (la01–la20 → mk01–mk05, with weight transfer and periodic zero-shot generalization
  eval on held-out larger instances) implemented and smoke-tested.
- Fetched `mk02`–`mk07` (only `mk01` was previously bundled) — needed for the curriculum
  step and for FJSP-side generalization held-out instances.

**Blocked / not yet run at full scale (depends on the above completing):**
- Phase 1 Optuna search (40 trials × 2 studies) — queued behind the gate runs.
- Phase 2 full-budget confirmation — queued behind Phase 1.
- Ablation study at matched full budget — queued behind Phase 2 (needs the winning
  config).
- Full curriculum retrain — queued behind Phase 2 (needs the winning config).

---

## 1. Gate: `TrainingDiagnosis.md` §4 steps 1–2

### 1.1 `ft06` multi-seed sanity check (seeds 1–4, 400k timesteps, existing config) — DONE

`TrainingDiagnosis.md` §3 rank-5 concern (every Week 18 headline number rests on a
single seed) is resolved for `ft06` specifically: the result replicates.

| Seed | Greedy makespan | Greedy RPD | Best training makespan | Sampled-mean makespan (100 rollouts) |
|---|---|---|---|---|
| 0 (original) | 55.0 | 0.00% | 55.0 | 55.15 |
| 1 | 55.0 | 0.00% | 55.0 | 55.69 |
| 2 | 55.0 | 0.00% | 55.0 | 55.88 |
| 3 | 55.0 | 0.00% | 55.0 | 59.22 |
| 4 | 55.0 | 0.00% | 55.0 | 55.89 |

**Every one of the 5 seeds' greedy (deterministic) rollout reaches makespan 55** — the
reference optimum, matching CP-SAT, 0% RPD — with zero variance on the metric that
matters most (the one `TrainingDiagnosis.md` §2.3 warned not to over-read as an H1
result, but which is at least now known to be a property of the policy/instance at this
config, not of seed 0 specifically). The only spread visible is in the stochastic
best-of-100-sampled mean (55.15–59.22), which is expected policy-entropy noise, not a
sign of an unstable optimum. **This is the strongest possible outcome for the gate
check**: the HPO search's `ft06` objective (mean greedy RPD across seeds, per
`HPO_TechSpec.md` §2) is being built on a genuinely stable signal, not single-seed luck.

### 1.2 `mk01` re-run at 1,000,000 timesteps — DONE

`TrainingDiagnosis.md` §4 step 2, rank-1 hypothesis (insufficient timesteps for the
larger `(operation, machine)` action space), confirmed as at least partially correct:

| | 300k (seed0, original) | 1,000,000 (this gate run) |
|---|---|---|
| Greedy makespan | 43.0 | **42.0** |
| Greedy RPD vs. optimum (40) | 7.50% | **5.00%** |
| Best training makespan | 42.0 | 41.0 |
| Best-of-100-sampled | 42.0 | 42.0 |
| Last-100-episode mean makespan | 46.24 | 43.05 |
| Wall clock | 2528.9 s (42.1 min) | 8712.9 s (145.2 min) |

Extending the budget alone (no hyperparameter change) closes the gap from 7.5% to 5.0%
RPD and now **matches GA's greedy result (42)** for the first time — previously RL
(43) was strictly worse than GA (42) at 300k. It still doesn't reach CP-SAT's proven
optimum (40). This is direct evidence that hypothesis 1 (undertraining) explains a real
part of the original gap, though not all of it: RL still sits at the same makespan GA
finds, so hypotheses 3 (loose FJSP shaping) and 4 (`PerPairActorHead` width) remain live
candidates for closing the remaining 2 points, per `TrainingDiagnosis.md`'s own
conditional sequencing (steps 4/6 apply only if steps 2 and 5 together don't close the
gap). The checkpoint from this run is `experiments/results/checkpoints/ppo_hetero_stageF1M_mk01_final.zip`
(kept distinct from the original 300k `ppo_hetero_stageF_mk01_final.zip`, not overwritten,
per this project's flagging discipline).

---

## 2. Implementation notes and correctness checks

### 2.1 Entropy-schedule risk (`HPO_TechSpec.md` §6, risk 1)

Verified directly against the installed `sb3_contrib.MaskablePPO.__init__` signature:
`ent_coef: float = 0.0`, no `Callable[[float], float]` union (unlike `learning_rate`/
`clip_range`, which do accept one). `PPO.train()` reads `self.ent_coef` fresh on every
call. Resolution: `EntropyScheduleCallback` mutates `model.ent_coef` at the start of
every rollout — no SB3 internals needed patching.

### 2.2 Chunked-`learn()` risk (`HPO_TechSpec.md` §6, risk 2)

Verified directly against `BaseAlgorithm._setup_learn`: with `reset_num_timesteps=False`,
the current call's `total_timesteps` is added on top of `self.num_timesteps`, so
`progress_remaining`-based schedules stay correct across chunks. Regression check
(`experiments/hpo_hetero.py --verify-chunking`): one uninterrupted `learn()` vs. four
chunked calls produce **byte-identical final weights** once the chunk size is aligned to
the rollout buffer size (`n_steps * n_envs`). A first attempt with a chunk size *not*
aligned to the buffer failed this check — not a bug, but confirmation that SB3's
`total_timesteps` is a lower bound, not an exact target (it always finishes the rollout
in progress). Consequence for the real runs: `checkpoint_every=50_000` will not evenly
divide typical buffer sizes, so every trial's realized timestep count overshoots
slightly past each nominal checkpoint — expected and harmless, noted here so it isn't
mistaken for a bug if checkpoint step counts don't land exactly on round numbers.

### 2.3 Test suite

Full suite (`pytest`) run after all code changes: [STATUS: see run log
`experiments/results/gate_logs/pytest_week21.txt`].

---

## 3. HPO results (Phase 1 + Phase 2)

[TO FILL IN once Phase 1/2 complete: best configuration per problem (`ft06`, `mk01`),
justification, parameter-importance plot, Phase 2 confirmation table.]

---

## 4. Ablation results

[TO FILL IN: Run 0 (reference) / 1a (homogeneous GIN) / 1b (GAT-hybrid) / 2 (sparse
reward) comparison table, at matched compute budget under the winning HPO config.]

---

## 5. Full curriculum retrain and generalization

[TO FILL IN: `la01`–`la20` → `mk01`–`mk05` stage-by-stage results, zero-shot
generalization curves on the held-out `la21`–`la40` / `mk06`–`mk07` tiers, final
checkpoint path.]

---

## 6. What was prioritized and why (compute/time flagging)

[TO FILL IN at sprint close: final honest accounting of what ran at full budget vs. what
was reduced/deferred, matching this report's §0 running log.]
