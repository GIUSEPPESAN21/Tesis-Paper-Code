# Hyperparameter Search — Technical Specification (Optuna)

**Companion to `TrainingDiagnosis.md`. Do not start this search before `TrainingDiagnosis.md` §4 steps 1–2 (the multi-seed `ft06` sanity check and the extended `mk01` re-run) are complete** — §2 explains why searching against an unverified single-seed objective wastes the search budget on noise.

---

## 1. Scope and what this search is *for*

Five axes, per the sprint brief: **learning rate, GNN layers/width, PPO clip epsilon, entropy coefficient, discount factor.** `n_steps`, `n_envs`, `batch_size`, `n_epochs`, `vf_coef`, `gae_lambda` stay fixed at `TrainingPlan_Week17.md` §2's values — they are out of the brief's explicit scope, and `TRAINING_REPORT_WEEK18.md` §4 gives no evidence any of them is a problem (`approx_kl`/`clip_fraction` both mid-range on both runs, per that report's Table 4.4).

**This is a search for a better *training recipe*, not a replacement for the diagnostic fixes in `TrainingDiagnosis.md` §4.** Concretely: hypothesis 1 in that document (insufficient `mk01` timesteps) is a budget problem no hyperparameter search fixes, and hypothesis 3 (loose FJSP shaping potential) is a reward-design problem no hyperparameter search fixes either. This search targets the *interaction* between the architecture/optimizer settings and the training dynamics `TRAINING_REPORT_WEEK18.md` §6 already diagnosed — chiefly the entropy-floor/plateau pattern flagged in `TrainingDiagnosis.md` §2.1, where a schedule or a different `ent_coef` might close a gap no amount of extra `mk01` budget alone would.

---

## 2. Objective function: what a "trial" measures, and why single-seed is not good enough here

**Primary metric (minimize): mean of greedy-rollout RPD across $k\geq3$ seeds**, not a single seed's RPD. `TrainingDiagnosis.md` §3 rank 5 elevates the single-seed problem to a first-tier validity concern for the Week 18 report itself; building a hyperparameter search on top of an objective that is *itself* only evaluated once per trial would import the same problem into every trial's ranking — a config that looks best might just be the config whose seed 0 got lucky. **Every trial trains $k=3$ seeds at the trial's sampled configuration and reports the mean greedy RPD across them** as the value Optuna optimizes; the per-seed spread is logged (not optimized) as a secondary diagnostic, since a config with a lower mean but wildly higher variance across seeds is a worse thesis result even at equal mean RPD (harder to reproduce, weaker for the pre-registered Wilcoxon tests in `../../thesis/PreRegistration.md`, which assume the policy's behavior is stable enough to be meaningfully paired against a PDR instance-by-instance).

**Two independent studies, not one shared study across problem types.** `ft06` (JSSP) and `mk01` (FJSP) are diagnosed with different symptoms in `TrainingDiagnosis.md` §3 (an entropy floor vs. genuine undertraining/looser shaping) and have different action spaces (36 vs. 360), episode lengths (36 vs. 60 steps), and current headline results (already at the reference optimum vs. 7.5% RPD). A single combined objective (e.g., averaging RPD across both) would let a config that helps one problem and hurts the other look mediocre-but-acceptable on both, which is the opposite of what a diagnostic search is for. Run `study_ft06` and `study_mk01` as separate Optuna studies, each with its own search space instance (same axes, same ranges — the *space* is shared, the *study* is not).

**Pruning metric (intermediate, per seed within a trial): `custom/makespan_rpd_pct` at fixed checkpoints**, already logged by `HeteroEpisodeStatsCallback` (`TrainingPlan_Week17.md` §4.1, confirmed implemented in `experiments/train_hetero.py` per `TRAINING_REPORT_WEEK18.md` §1). Report the smoothed (last-100-episode mean) RPD to `trial.report(value, step)` at every checkpoint interval (every 50,000 timesteps, matching the existing `CheckpointCallback` cadence already in `experiments/train_hetero.py`), and call `trial.should_prune()` after each report — this reuses logging the codebase already has, rather than adding a second metric-collection path.

---

## 3. Search space

| Hyperparameter | Type | Range / choices | Sampling | Rationale |
|---|---|---|---|---|
| `learning_rate` | float | $[1\times10^{-5},\ 5\times10^{-4}]$ | log-uniform | Brackets both literature anchors already in `TrainingPlan_Week17.md`'s hyperparameter table — L2D's $2\times10^{-5}$ and Song et al.'s $2\times10^{-4}$ — with headroom on both sides rather than only interpolating between them, since Week 18 ran at $2\times10^{-4}$ without needing the $2\times10^{-5}$ fallback (report §2), so the search should be free to explore *above* $2\times10^{-4}$ too, not just toward the fallback. |
| `num_layers` (GNN message-passing depth) | categorical | $\{2, 3, 4\}$ | — | Centered on the validated default (3, matching `GINEncoder`/L2D). 2 tests whether the current depth is already more than needed at these small graph sizes (36–360 nodes); 4 tests the over-smoothing risk `models/gin.py`'s own docstring already names as the reason more layers "mainly cause over-smoothing" at this problem's low graph diameter — worth checking empirically rather than only citing the concern. |
| `hidden_dim` (GNN width, both node types) | categorical | $\{32, 64, 96, 128\}$ | — | 64 is the validated default. 128 is `TrainingDiagnosis.md` rank-4 hypothesis (`PerPairActorHead` under-parameterized for `mk01`), included here so the search can test it systematically instead of as a single manual follow-up run. 32 and 96 fill in the range so the search isn't limited to only the two endpoints anyone would guess by hand. |
| `clip_range` ($\epsilon$) | categorical | $\{0.1, 0.15, 0.2, 0.25, 0.3\}$ | — | Centered on 0.2 (L2D / Song et al. / Schulman et al.'s shared canonical value, confirmed working in Week 18: `clip_fraction` mid-range on both runs). Discrete rather than continuous — PPO's clip range is not usually sensitive at fine granularity, and a discrete grid keeps the search space's effective size smaller for a fixed trial budget. |
| `ent_coef` **schedule**, not a single value | two floats + one fraction | `ent_coef_start` $\in[5\times10^{-3},\ 5\times10^{-2}]$ (log-uniform); `ent_coef_end` $\in[1\times10^{-4},\ \text{ent\_coef\_start}]$ (log-uniform, constrained below the start); `decay_fraction` $\in[0.2, 0.8]$ (uniform) — the fraction of total training timesteps over which `ent_coef` decays linearly from start to end, then holds at `ent_coef_end` | see above | Directly operationalizes `TrainingDiagnosis.md` §2.1/§4 action item 3: a fixed `ent_coef=0.01` is implicated in `ft06`'s entropy-plateau/return-plateau coincidence. A single scalar `ent_coef` is included as a **special case check**: any trial where the sampled `ent_coef_start` and `ent_coef_end` land close together with a `decay_fraction` near 0 effectively degenerates to the current fixed-value behavior, so the search naturally nests the current configuration as one reachable point rather than assuming a schedule is strictly better without testing the null. |
| `gamma` | **fixed at 1.0, not searched** | — | — | See §3.1 below — this is a deliberate deviation from the brief's literal five-axis list, flagged rather than silently applied. |

### 3.1 Why `gamma` is not actually searched, despite being named in the brief

`TrainingPlan_Week17.md` §2 and `../../thesis/PreRegistration.md`'s underlying framework (`Step 4 Thesis.md` Part B4/D3) both state `gamma=1.0` is **structural**, not tunable: the dense reward's telescoping identity ($\sum_t r_t = H_0 - C_{\max}$) requires undiscounted return, and `TRAINING_REPORT_WEEK18.md` §4.1 confirms this was verified as a unit-tested property (`test_reward_telescopes_to_h0_minus_makespan`) this sprint, not merely assumed. Searching `gamma$ away from 1.0 would optimize a metric that is no longer guaranteed to equal $-C_{\max}$ at all — the search could find a `gamma<1` configuration that scores well on discounted return while making the actual makespan worse, which would be optimizing the wrong thing and calling it a hyperparameter win.

**This document honors the brief's explicit request without silently dropping the axis: `gamma` is available as an optional, separate, narrow diagnostic arm, not part of either primary study.** If it is run at all: `study_gamma_diagnostic`, $\{0.99, 0.995, 1.0\}$ only (a 3-point grid, not a continuous search — a discount that close to 1 is a robustness check, not a real search over a meaningfully different regime), on `ft06` only (the cheaper, already-converged problem, so any deviation is easy to attribute to `gamma` specifically rather than confounded with `mk01`'s known undertraining), reported **separately** in the eventual sprint write-up as "does departing from the structurally-required $\gamma=1$ measurably hurt, as the reward-telescoping argument predicts" rather than folded into the primary `learning_rate`/`hidden_dim`/etc. results table. Recommended priority: **skip this arm unless primary-study budget is exhausted with time remaining** — it is confirmatory of an already-strong theoretical argument, not exploratory of an open question the way the other four axes are.

---

## 4. Sampler and pruner

**Sampler: `optuna.samplers.TPESampler`** (default in modern Optuna, `multivariate=True` recommended given the `ent_coef_start`/`ent_coef_end`/`decay_fraction` group is not independent — a joint sampler avoids wasting trials on `ent_coef_end > ent_coef_start` regions the constraint already excludes, but TPE with `multivariate=True` learns the joint structure faster than independent per-parameter sampling would). Not a grid search: `num_layers` × `hidden_dim` × `clip_range` alone is $3\times4\times5=60$ discrete combinations before the two continuous axes and the three-parameter entropy schedule are even considered — a full grid is not affordable at this problem's per-trial cost (§5).

**Pruner: `optuna.pruners.HyperbandPruner`** (or `SuccessiveHalvingPruner` if Hyperband's own bracket-management overhead is judged not worth it for a study this small — both use the same `trial.report`/`should_prune()` calls, so switching between them later is a one-line change, not a rework). Hyperband is preferred over the simpler `MedianPruner` specifically because early `mk01` trials are expected to look uniformly bad for the first several checkpoints (report §6.2: the return curve is *still rising* at 300k on the current default config, so almost every config will look "not yet good" at the earliest checkpoints) — a median-based pruner risks killing viable trials just because nothing has separated from the pack yet, whereas Hyperband's successive-halving structure allocates more budget to surviving brackets over time rather than making an early comparative judgment.

---

## 5. Budget: low-fidelity search, high-fidelity confirmation

Per-trial cost at full budget (`ft06`: 400k timesteps × 3 seeds $\approx$ 37 min × 3 $\approx$ 1.85h; `mk01` at the diagnostic-corrected 1M timesteps × 3 seeds, extrapolating report §2's throughput figures, $\approx$ 6–7h) makes a full-budget search infeasible for a one-week sprint on the single-GPU/CPU-only budget `TrainingPlan_Week17.md` §1 and the report's own hardware note (§2, CPU-only, 8 cores) both establish. **Two-phase protocol:**

**Phase 1 — low-fidelity search.** Each trial trains at **25% of the confirmed full budget** (`ft06`: 100k timesteps; `mk01`: 250k, i.e. a quarter of the diagnosis-corrected 1M target), $k=2$ seeds per trial (not 3 — Phase 1's job is to eliminate clearly-bad regions of the search space cheaply, not to produce a publication-grade mean; §2's $k\geq3$ requirement applies to Phase 2), pruned aggressively via Hyperband. Target: **40 trials** per study (`ft06`, `mk01` separately) — at $\sim$9 min/seed × 2 seeds × 40 trials for `ft06$ ($\approx$12h, poolable across the 8 available cores by running 3–4 trials concurrently at reduced `n_envs` if needed) and proportionally more for `mk01`, this is a one- to two-day background job, not the whole sprint.

**Phase 2 — high-fidelity confirmation.** Take the **top 3 configurations** from Phase 1 (by pruned/completed value, not just the single best — Phase 1's short budget makes the ranking noisy, and the point of keeping 3 is to not bet the whole sprint on a Phase-1 winner that doesn't hold up), re-run each at the **full budget** (`ft06`: 400k; `mk01`: 1,000,000, per `TrainingDiagnosis.md` step 2) with the full $k=3$ seeds this document's §2 objective requires. The configuration that wins Phase 2 — not Phase 1 — is the one that becomes the new default in `TrainingPlan_Week17.md`'s hyperparameter table, and that update should itself be a dated, flagged edit (matching the flagging discipline used throughout this project's other tech specs), not a silent overwrite of the values `../../thesis/PreRegistration.md` cites as already-tested.

**Total added compute, rough order of magnitude:** Phase 1 $\approx$1–2 days background compute across both studies; Phase 2 $\approx$ 3 configs × (1.85h `ft06` + $\sim$20h `mk01` at full 1M-timestep/3-seed budget) — **`mk01`'s Phase 2 is the dominant cost and should be sequenced after `TrainingDiagnosis.md` step 2's own single-seed 1M-timestep confirmation run**, so Phase 2 isn't the first time the 1M-timestep budget is spent at all. If the sprint's compute budget cannot absorb full `mk01` Phase 2 for all 3 candidates, reduce to the single Phase-1 winner for `mk01` specifically and flag the reduced confirmation explicitly, rather than silently running fewer seeds without noting it.

---

## 6. Implementation sketch

Builds on `experiments/train_hetero.py`'s existing `build_hetero_maskable_ppo(env, hidden_dim, num_layers, mlp_layers, head_num_layers, use_machine_edges, seed, **ppo_kwargs)` (signature confirmed directly from `models/policy.py`) and `HeteroEpisodeStatsCallback`. New file: `experiments/hpo_hetero.py`, additive, following the same "extend the Week 1/18 harness, don't replace it" discipline the codebase has used at every prior sprint.

```python
# experiments/hpo_hetero.py (new, additive)
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from envs.jssp_env import JSSPEnv
from envs.fjsp_env import FJSPEnv
from envs.loader import known_optimum
from envs.fjsp_loader import fjsp_known_optimum
from models.policy import build_hetero_maskable_ppo
from experiments.train_hetero import HeteroEpisodeStatsCallback  # reused, not reimplemented


def suggest_config(trial: optuna.Trial) -> dict:
    ent_start = trial.suggest_float("ent_coef_start", 5e-3, 5e-2, log=True)
    ent_end = trial.suggest_float("ent_coef_end", 1e-4, ent_start, log=True)  # constrained below start
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        num_layers=trial.suggest_categorical("num_layers", [2, 3, 4]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 96, 128]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.25, 0.3]),
        ent_coef_start=ent_start,
        ent_coef_end=ent_end,
        decay_fraction=trial.suggest_float("decay_fraction", 0.2, 0.8),
    )
    # gamma is NOT sampled here -- fixed at 1.0, per §3.1. A separate
    # study_gamma_diagnostic script (not this one) handles the {0.99,0.995,1.0} arm.


def entropy_schedule(start: float, end: float, decay_fraction: float):
    """Returns an SB3-compatible callable: progress_remaining (1->0) -> ent_coef."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0 at start of training, 1 at end
        if progress >= decay_fraction:
            return end
        return start + (end - start) * (progress / decay_fraction)
    return schedule
    # NOTE: MaskablePPO's ent_coef accepts a float, not a schedule, in sb3_contrib==2.9.0
    # (confirmed against the installed version per GNN_PPO_Architecture_Spec_v2.md §3.3's
    # own version-pinning discipline) -- implementing this requires either (a) a small
    # custom callback that overwrites model.ent_coef every n_steps rollout using this
    # function evaluated at the current progress, or (b) confirming whether 2.9.0's
    # `ent_coef` argument accepts a Schedule callable directly (SB3's `learning_rate`
    # argument does; `ent_coef` may not -- VERIFY THIS FIRST, before assuming the
    # schedule wires through the same constructor path as learning_rate).


def run_trial(trial: optuna.Trial, *, instance: str, fjsp: bool, low_fidelity: bool) -> float:
    config = suggest_config(trial)
    timesteps = (100_000 if instance == "ft06" else 250_000) if low_fidelity else \
                (400_000 if instance == "ft06" else 1_000_000)
    n_seeds = 2 if low_fidelity else 3
    optimum = fjsp_known_optimum(instance) if fjsp else known_optimum(instance)

    seed_rpds = []
    for seed in range(n_seeds):
        env = FJSPEnv(instance) if fjsp else JSSPEnv(instance)
        model = build_hetero_maskable_ppo(
            env,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            seed=seed,
            learning_rate=config["learning_rate"],
            clip_range=config["clip_range"],
            ent_coef=entropy_schedule(  # see the callback-wiring note above
                config["ent_coef_start"], config["ent_coef_end"], config["decay_fraction"]
            ),
            gamma=1.0,  # fixed, not from config -- see §3.1
            gae_lambda=0.95, n_epochs=3, vf_coef=0.5,  # unchanged, out of this search's scope
            n_steps=216 if instance == "ft06" else 360,
            batch_size=64,
        )
        callback = HeteroEpisodeStatsCallback(optimum=optimum)
        # Prune on the *smoothed* RPD, reported every 50k timesteps, matching the
        # existing CheckpointCallback cadence (TrainingPlan_Week17.md §4.3).
        for checkpoint_step in range(50_000, timesteps + 1, 50_000):
            model.learn(total_timesteps=50_000, callback=callback, reset_num_timesteps=False)
            recent_rpd = callback.recent_makespan_rpd_pct(window=100)  # new, small helper method
            trial.report(recent_rpd, step=checkpoint_step + seed * timesteps)  # offset per seed
            if trial.should_prune():
                raise optuna.TrialPruned()
        seed_rpds.append(callback.greedy_rollout_rpd(env, optimum))  # existing eval path, reused

    trial.set_user_attr("seed_rpds", seed_rpds)  # log spread, don't optimize it (§2)
    return sum(seed_rpds) / len(seed_rpds)


def main():
    for instance, fjsp in [("ft06", False), ("mk01", True)]:
        study = optuna.create_study(
            study_name=f"hpo_{instance}",
            direction="minimize",
            sampler=TPESampler(multivariate=True, seed=0),
            pruner=HyperbandPruner(),
            storage=f"sqlite:///experiments/results/optuna_{instance}.db",  # resumable
            load_if_exists=True,
        )
        study.optimize(
            lambda t: run_trial(t, instance=instance, fjsp=fjsp, low_fidelity=True),
            n_trials=40,
        )
        # Phase 2: re-run top 3 at full fidelity, outside the Optuna loop (§5) --
        # a small separate script reading study.trials_dataframe(), not shown here.
```

**Two implementation risks flagged explicitly, matching this codebase's existing flagging discipline, rather than assumed away:**

1. **Whether `sb3_contrib==2.9.0`'s `MaskablePPO(ent_coef=...)` accepts a schedule callable the same way `learning_rate` does is not yet verified in this document** — SB3's convention is that `learning_rate` accepts either a float or a `Callable[[float], float]` (progress-remaining based), but not every hyperparameter follows that convention uniformly across `stable_baselines3`/`sb3_contrib` releases. **Verify against the installed version before implementing the schedule as a native constructor argument; if unsupported, implement it as a small `BaseCallback` that sets `model.ent_coef` directly each rollout** (the fallback path, noted in the code sketch's comment) — either way, add a regression test analogous to `GNN_PPO_Architecture_Spec_v2.md` §3.3's own version-pinning discipline for `_get_action_dist_from_latent`.
2. **`reset_num_timesteps=False` across repeated `model.learn()` calls within one trial must be checked against SB3's own semantics for `n_steps`/rollout-buffer continuity** — chunking training into 50k-timestep calls for pruning purposes should not silently reset the optimizer's internal step counter or corrupt the learning-rate schedule if one is added later; a one-off unit test comparing a single uninterrupted `learn(total_timesteps=200_000)` call against four chunked `learn(total_timesteps=50_000)` calls, checked for byte-identical final weights at a fixed seed, is the cheap way to confirm this before trusting Phase 1's results.

---

## 7. How a winning configuration feeds back into the rest of the project

- **`TrainingPlan_Week17.md` §2's hyperparameter table gets a dated, flagged update** to the Phase-2 winning configuration per problem type — not a silent overwrite, since `../../thesis/PreRegistration.md` and `AblationPlan.md` both currently reference "the reference configuration" as a specific, fixed thing.
- **`AblationPlan.md`'s reference configuration (§1 of that document) is *not* automatically re-baselined to the HPO winner.** Ablations compare against a known, already-multi-seed-verified reference; swapping the reference mid-plan without re-running the ablations already scheduled against the old one would silently invalidate any ablation work done in between. Recommend: finish this HPO search and Phase 2 confirmation *before* Month 7's ablation runs begin, so `AblationPlan.md`'s reference configuration is set once, after HPO, not twice.
- **`../../thesis/PreRegistration.md` is not reopened by this search.** H1/H2/H5's frozen decision rules (§§2–3, 6 of that document) apply to whatever the final trained policy turns out to be, HPO-tuned or not — the pre-registration was frozen against the *hypotheses*, not against a specific hyperparameter configuration, so a better-tuned policy simply produces the numbers that document's already-fixed tests are applied to. No amendment to that document is triggered by running this search.
