"""Optuna hyperparameter search over the heterogeneous GNN + PPO training recipe.

Companion to ``HPO_TechSpec.md``. Extends ``experiments/train_hetero.py``'s harness
rather than reimplementing it: same env factories, same ``HeteroEpisodeStatsCallback``,
same ``build_hetero_maskable_ppo`` constructor, same greedy-rollout evaluation path.

Two implementation risks the tech spec (§6) flagged, resolved here:

1. ``sb3_contrib==2.9.0``'s ``MaskablePPO(ent_coef=...)`` is a plain ``float`` in the
   constructor signature (verified directly: unlike ``learning_rate``/``clip_range``, no
   ``Callable[[float], float]`` union). ``PPO.train()`` reads ``self.ent_coef`` fresh on
   every call, so :class:`EntropyScheduleCallback` mutating ``model.ent_coef`` between
   rollouts is sufficient -- no SB3 internals need patching.
2. ``reset_num_timesteps=False`` across chunked ``model.learn()`` calls: verified against
   ``BaseAlgorithm._setup_learn`` directly -- when ``reset_num_timesteps=False``, the
   ``total_timesteps`` argument of the *current* call is added on top of
   ``self.num_timesteps``, so ``self._total_timesteps`` (and therefore
   ``progress_remaining``-based schedules) stays correct across chunks. ``--verify-chunking``
   runs the cheap regression check the tech spec asks for (tiny-scale, not full-budget).

Usage
-----
    python experiments/hpo_hetero.py --instance ft06 --n-trials 40 --low-fidelity
    python experiments/hpo_hetero.py --instance mk01 --fjsp --n-trials 40 --low-fidelity
    python experiments/hpo_hetero.py --instance ft06 --smoke-test        # mechanics check
    python experiments/hpo_hetero.py --verify-chunking                  # risk-2 regression check
    python experiments/hpo_hetero.py --instance ft06 --phase2-top-n 3   # full-budget confirmation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import optuna
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.train_hetero import (
    HeteroEpisodeStatsCallback,
    evaluate_policy_rollouts,
    make_env_factory,
    reference_optimum,
)
from models.policy import build_hetero_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

FULL_BUDGET = {"ft06": 400_000, "mk01": 1_000_000}
LOW_FIDELITY_BUDGET = {"ft06": 100_000, "mk01": 250_000}


# ------------------------------------------------------------------------- search space


def suggest_config(trial: optuna.Trial) -> dict[str, Any]:
    """§3 of ``HPO_TechSpec.md``. ``gamma`` is fixed at 1.0, not sampled (§3.1)."""
    ent_start = trial.suggest_float("ent_coef_start", 5e-3, 5e-2, log=True)
    ent_end = trial.suggest_float("ent_coef_end", 1e-4, ent_start, log=True)
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        num_layers=trial.suggest_categorical("num_layers", [2, 3, 4]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 96, 128]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.25, 0.3]),
        ent_coef_start=ent_start,
        ent_coef_end=ent_end,
        decay_fraction=trial.suggest_float("decay_fraction", 0.2, 0.8),
    )


def entropy_schedule(start: float, end: float, decay_fraction: float) -> Callable[[float], float]:
    """``progress`` (0 at training start, 1 at end) -> ``ent_coef``. Linear decay then hold."""

    def schedule(progress: float) -> float:
        if progress >= decay_fraction:
            return end
        return start + (end - start) * (progress / decay_fraction)

    return schedule


class EntropyScheduleCallback(BaseCallback):
    """Mutates ``model.ent_coef`` at the start of every rollout. See module docstring risk 1."""

    def __init__(self, schedule_fn: Callable[[float], float], total_timesteps: int) -> None:
        super().__init__()
        self.schedule_fn = schedule_fn
        self.total_timesteps = total_timesteps

    def _on_rollout_start(self) -> None:
        progress = min(1.0, self.model.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self.schedule_fn(progress)

    def _on_step(self) -> bool:
        return True


# --------------------------------------------------------------------------- trial loop


def train_one_seed(
    *,
    config: dict[str, Any],
    instance: str,
    fjsp: bool,
    seed: int,
    timesteps: int,
    n_envs: int,
    checkpoint_every: int,
    n_sampled_eval: int,
    trial: optuna.Trial | None,
    seed_offset: int,
) -> tuple[float, HeteroEpisodeStatsCallback]:
    """Train one seed at ``config``, optionally reporting to ``trial`` for pruning.

    Returns (greedy_rpd_pct, stats_callback). If ``trial`` is given and the trial is
    pruned mid-training, raises ``optuna.TrialPruned`` (Phase 1 use). If ``trial`` is
    None (Phase 2 confirmation), trains the full budget uninterrupted.
    """
    optimum = reference_optimum(instance, fjsp)
    factory = make_env_factory(instance, fjsp)
    probe = factory()
    episode_length = probe.unwrapped.instance.num_operations
    n_steps = 6 * episode_length
    probe.close()

    venv = DummyVecEnv([factory for _ in range(n_envs)])
    model = build_hetero_maskable_ppo(
        venv,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        seed=seed,
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef_start"],
        gamma=1.0,
        gae_lambda=0.95,
        n_epochs=3,
        vf_coef=0.5,
        n_steps=n_steps,
        batch_size=64,
    )
    stats_callback = HeteroEpisodeStatsCallback(optimum=optimum)
    schedule_fn = entropy_schedule(
        config["ent_coef_start"], config["ent_coef_end"], config["decay_fraction"]
    )
    entropy_callback = EntropyScheduleCallback(schedule_fn, total_timesteps=timesteps)

    try:
        checkpoints = list(range(checkpoint_every, timesteps + 1, checkpoint_every))
        if not checkpoints or checkpoints[-1] != timesteps:
            checkpoints.append(timesteps)
        done = 0
        for checkpoint_step in checkpoints:
            chunk = checkpoint_step - done
            if chunk <= 0:
                continue
            model.learn(
                total_timesteps=chunk,
                callback=[stats_callback, entropy_callback],
                reset_num_timesteps=False,
            )
            done = checkpoint_step
            if trial is not None:
                recent_rpd = stats_callback.recent_makespan_rpd_pct(window=100)
                trial.report(recent_rpd, step=seed_offset + checkpoint_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        evaluation = evaluate_policy_rollouts(model, instance, fjsp, n_sampled_eval, seed)
        greedy_rpd = 100.0 * (evaluation["greedy_makespan"] - optimum) / optimum
        return greedy_rpd, stats_callback
    finally:
        venv.close()


def run_trial(
    trial: optuna.Trial,
    *,
    instance: str,
    fjsp: bool,
    low_fidelity: bool,
    n_envs: int,
    checkpoint_every: int,
    timesteps_override: int | None,
    n_seeds_override: int | None,
) -> float:
    config = suggest_config(trial)
    if timesteps_override is not None:
        timesteps = timesteps_override
    else:
        timesteps = (LOW_FIDELITY_BUDGET if low_fidelity else FULL_BUDGET)[instance]
    n_seeds = n_seeds_override if n_seeds_override is not None else (2 if low_fidelity else 3)
    n_sampled_eval = 20 if low_fidelity else 100

    seed_rpds = []
    for seed in range(n_seeds):
        rpd, _ = train_one_seed(
            config=config,
            instance=instance,
            fjsp=fjsp,
            seed=seed,
            timesteps=timesteps,
            n_envs=n_envs,
            checkpoint_every=checkpoint_every,
            n_sampled_eval=n_sampled_eval,
            trial=trial,
            seed_offset=seed * timesteps,
        )
        seed_rpds.append(rpd)

    trial.set_user_attr("seed_rpds", seed_rpds)
    return sum(seed_rpds) / len(seed_rpds)


# --------------------------------------------------------------------------- phase 2


def confirm_config(
    *,
    config: dict[str, Any],
    instance: str,
    fjsp: bool,
    timesteps: int,
    n_seeds: int,
    n_envs: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    """Phase 2: re-run one config at full budget / full seed count, no pruning."""
    seed_rpds = []
    for seed in range(n_seeds):
        rpd, _ = train_one_seed(
            config=config,
            instance=instance,
            fjsp=fjsp,
            seed=seed,
            timesteps=timesteps,
            n_envs=n_envs,
            checkpoint_every=checkpoint_every,
            n_sampled_eval=100,
            trial=None,
            seed_offset=0,
        )
        seed_rpds.append(rpd)
    return {
        "config": config,
        "seed_rpds": seed_rpds,
        "mean_rpd": sum(seed_rpds) / len(seed_rpds),
        "timesteps": timesteps,
        "n_seeds": n_seeds,
    }


def top_n_configs(study: optuna.Study, n: int) -> list[dict[str, Any]]:
    """Top ``n`` completed (non-pruned) trials by objective value."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    return [t.params for t in completed[:n]]


# --------------------------------------------------------------------------- verify chunking


def verify_chunking(instance: str = "ft06") -> bool:
    """Risk 2 regression check: one uninterrupted ``learn()`` vs. four chunked calls
    must produce byte-identical final weights at a fixed seed. Tiny scale, n_envs=2,
    ft06's ``n_steps=216`` -> rollout buffer = 432 -- chunk size is deliberately set to
    exactly one buffer's worth (432) so both runs execute the same number of
    rollout-collection/PPO-update cycles. SB3's ``total_timesteps`` is a lower bound,
    not an exact target: it always completes the rollout in progress, so a chunk size
    that does *not* divide the buffer size evenly (e.g. checkpoint_every=50_000 against
    a ~a few-hundred-timestep buffer) makes each chunk overshoot to the next rollout
    boundary -- expected, not a bug, but real trials will therefore run slightly past
    each nominal checkpoint. This check isolates whether ``reset_num_timesteps=False``
    itself (progress/optimizer continuity) is sound, independent of that overshoot.
    """
    config = dict(
        learning_rate=2e-4,
        num_layers=3,
        hidden_dim=32,
        clip_range=0.2,
        ent_coef_start=0.01,
        ent_coef_end=0.01,
        decay_fraction=0.5,
    )
    n_envs = 2
    n_steps = 6 * 36  # ft06 episode length
    buffer_size = n_steps * n_envs
    total = buffer_size * 4

    factory = make_env_factory(instance, fjsp=False)
    venv_a = DummyVecEnv([factory for _ in range(n_envs)])
    model_a = build_model_for_config_direct(venv_a, config, seed=0)
    model_a.learn(total_timesteps=total, reset_num_timesteps=True)
    venv_a.close()

    venv_b = DummyVecEnv([factory for _ in range(n_envs)])
    model_b = build_model_for_config_direct(venv_b, config, seed=0)
    for _ in range(4):
        model_b.learn(total_timesteps=buffer_size, reset_num_timesteps=False)
    venv_b.close()

    params_a = model_a.policy.state_dict()
    params_b = model_b.policy.state_dict()
    identical = all(
        torch.equal(params_a[k], params_b[k]) for k in params_a
    )
    print(f"[verify-chunking] byte-identical final weights: {identical}")
    return identical


def build_model_for_config_direct(venv, config: dict[str, Any], seed: int):
    factory_probe = venv.envs[0]
    episode_length = factory_probe.unwrapped.instance.num_operations
    return build_hetero_maskable_ppo(
        venv,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        seed=seed,
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef_start"],
        gamma=1.0,
        gae_lambda=0.95,
        n_epochs=3,
        vf_coef=0.5,
        n_steps=6 * episode_length,
        batch_size=32,
    )


# ---------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="ft06")
    parser.add_argument("--fjsp", action="store_true")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--low-fidelity", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--timesteps-override", type=int, default=None)
    parser.add_argument("--n-seeds-override", type=int, default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="tiny-scale run (2 trials, 2000 timesteps, 1 seed) to validate mechanics",
    )
    parser.add_argument("--verify-chunking", action="store_true")
    parser.add_argument(
        "--phase2-top-n",
        type=int,
        default=0,
        help="if >0, skip the Optuna loop and run Phase-2 full-budget confirmation on "
        "the top-N completed trials from --storage's existing study instead",
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_chunking:
        ok = verify_chunking(args.instance)
        return 0 if ok else 1

    if args.smoke_test:
        args.n_trials = 2
        args.timesteps_override = 2000
        args.n_seeds_override = 1
        args.checkpoint_every = 1000
        args.n_envs = 2

    suffix = "_smoke" if args.smoke_test else ""
    study_name = args.study_name or f"hpo_{args.instance}{suffix}"
    storage = args.storage or f"sqlite:///{(RESULTS_DIR / f'optuna_{args.instance}{suffix}.db').as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=0),
        pruner=HyperbandPruner(),
        storage=storage,
        load_if_exists=True,
    )

    if args.phase2_top_n > 0:
        configs = top_n_configs(study, args.phase2_top_n)
        if not configs:
            print("[phase2] no completed trials found in study -- run Phase 1 first")
            return 1
        results = []
        full_timesteps = args.timesteps_override or FULL_BUDGET[args.instance]
        for i, config in enumerate(configs):
            print(f"[phase2] confirming config {i + 1}/{len(configs)}: {config}")
            result = confirm_config(
                config=config,
                instance=args.instance,
                fjsp=args.fjsp,
                timesteps=full_timesteps,
                n_seeds=args.n_seeds_override or 3,
                n_envs=args.n_envs,
                checkpoint_every=args.checkpoint_every,
            )
            results.append(result)
            print(f"[phase2] config {i + 1} mean RPD={result['mean_rpd']:.2f}% seeds={result['seed_rpds']}")
        results.sort(key=lambda r: r["mean_rpd"])
        out_path = RESULTS_DIR / f"hpo_phase2_{args.instance}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[phase2] winner mean RPD={results[0]['mean_rpd']:.2f}% params={results[0]['config']}")
        print(f"[phase2] all results -> {out_path}")
        return 0

    study.optimize(
        lambda t: run_trial(
            t,
            instance=args.instance,
            fjsp=args.fjsp,
            low_fidelity=args.low_fidelity,
            n_envs=args.n_envs,
            checkpoint_every=args.checkpoint_every,
            timesteps_override=args.timesteps_override,
            n_seeds_override=args.n_seeds_override,
        ),
        n_trials=args.n_trials,
    )

    print(f"[done] {len(study.trials)} trials, best value={study.best_value:.3f}")
    print(f"[done] best params={study.best_params}")

    best_path = RESULTS_DIR / f"hpo_best_{args.instance}{suffix}.json"
    best_path.write_text(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "n_pruned": sum(
                    1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
                ),
            },
            indent=2,
        )
    )
    print(f"[done] best config -> {best_path}")

    try:
        from optuna.visualization.matplotlib import plot_param_importances

        ax = plot_param_importances(study)
        ax.figure.tight_layout()
        importance_path = RESULTS_DIR / f"hpo_param_importance_{args.instance}{suffix}.png"
        ax.figure.savefig(importance_path, dpi=150)
        print(f"[done] param importance plot -> {importance_path}")
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"[warn] could not produce param-importance plot: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
