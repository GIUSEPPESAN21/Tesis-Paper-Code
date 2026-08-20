"""E7 -- disruption-aware reward (kappa*Delta_inest) training entrypoint.

``E7_E8_FinalPlan.md`` Section 2.1 precondition #3: no training entrypoint against
``DynamicJSSPEnv`` existed before this script (``experiments/train_hetero.py`` and
``train_curriculum.py`` both import and train only ``JSSPEnv``/``FJSPEnv``). This script
fine-tunes an existing static-JSSP checkpoint directly against ``DynamicJSSPEnv`` so the
resulting policy natively learns the dynamic environment's completion-gated action
legality (see ``EXPERIMENT_LOG_WEEK24.md``'s E4-E6 RL-arm wiring-fix finding for why this
matters: the adapter-only path used for H3 carries a quantified, non-fixable-without-
retraining legality mismatch), rather than only ever seeing the dynamic contract through
an evaluation-time adapter.

Design decisions (dated 2026-07-31, recorded here and cross-referenced from
``AblationPlan.md`` Section 2.3 per that document's own instruction to write this down
*before* training starts):

1. **Domain-randomized disruption regime during training** (``E7_E8_FinalPlan.md``
   Section 2.2, "Option B"): every episode samples one of the same 9 frozen regimes
   ``PreRegistration_FINAL.md`` Section 4 / E4-E6 evaluate on, uniformly --
   ``DomainRandomizedRegimeWrapper`` below. This is what makes H4's "majority of 9
   regimes" decision rule a meaningful test of general disruption-awareness rather than
   regime-specific overfitting.
2. **Two-stage curriculum, not domain-randomized instance size.** SB3's ``DummyVecEnv``
   requires every parallel env to share one observation/action space, and
   ``DynamicJSSPEnv``'s padded space size depends on the base instance
   (``max_possible_ops = base_num_ops + max_dynamic_jobs * ops_per_job``) -- so unlike
   the disruption regime, the *instance* cannot be domain-randomized within a single
   ``model.learn()`` call. Instead: stage D1 fine-tunes on ``ft06`` (warm-started from
   the existing static curriculum checkpoint), stage D2 continues fine-tuning that
   checkpoint on ``la01`` -- exactly the weight-transfer mechanism
   ``experiments/train_curriculum.py::transfer_weights`` already uses between stages,
   reused here rather than reinvented. The final (post-D2) checkpoint is what E7's
   evaluation grid scores zero-shot on both instances, the same way the existing
   ``curriculum_stage19_la20_final`` checkpoint (last fine-tuned on la20) is already
   evaluated on ft06 and la01 today for H1-H3.
3. **``max_dynamic_jobs=20`` for every regime**, matching the CLI default Week 24's
   actual E4-E6 execution used (``run_e4e5e6.sh`` / ``run_e4e5e6_rl.sh`` never override
   ``--max-dynamic-jobs``), not the stricter "set explicitly per instance" language in
   ``PreRegistration_FINAL.md`` Section 4 that was never actually implemented -- chosen
   for exact train/eval regime consistency with the already-accepted-as-final E4-E6
   results, not as a new methodological decision.

Usage
-----
    # Stage D1: warm-start from the static curriculum checkpoint, train on ft06.
    python experiments/train_dynamic.py --instance ft06 --kappa 0.5 \\
        --init-checkpoint experiments/results/curriculum_full/checkpoints/curriculum_stage19_la20_final.zip \\
        --stage D1 --timesteps 100000

    # Stage D2: continue from stage D1's checkpoint, train on la01.
    python experiments/train_dynamic.py --instance la01 --kappa 0.5 \\
        --init-checkpoint experiments/results/dynamic/checkpoints/dynamic_kappa0.5_D1_ft06_final.zip \\
        --stage D2 --timesteps 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: this script may run without a display
import numpy as np
import torch
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.interface import _DynamicNodeFeatureAdapter
from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from envs.loader import known_optimum
from experiments.train_hetero import (
    HeteroDiagnosticsCallback,
    HeteroEpisodeStatsCallback,
    _read_progress_csv,
    plot_training_curves,
    write_metrics_csv,
)
from models.policy import build_hetero_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "dynamic"

#: max_dynamic_jobs=20 everywhere, per this module's docstring point 3.
_MAX_DYNAMIC_JOBS = 20


def frozen_regimes() -> list[DisruptionConfig]:
    """The 9 regimes ``PreRegistration_FINAL.md`` Section 4 / E4-E6 evaluate on."""
    regimes = [DisruptionConfig(target_availability=a) for a in (0.95, 0.90, 0.80)]
    regimes += [DisruptionConfig(processing_time_sigma=s) for s in (0.1, 0.25, 0.5)]
    regimes += [
        DisruptionConfig(target_utilization=u, max_dynamic_jobs=_MAX_DYNAMIC_JOBS)
        for u in (0.7, 0.85, 0.95)
    ]
    return regimes


class DomainRandomizedRegimeWrapper(gym.Wrapper):
    """Samples one of the 9 frozen regimes uniformly at every ``reset()``.

    Swaps ``DynamicJSSPEnv.config`` directly rather than reconstructing the environment:
    ``_allocate_episode_state`` (called by both ``__init__`` and ``reset``) rebuilds
    every disruption generator fresh from ``self.config`` each reset, and everything
    genuinely fixed at construction time (``max_possible_ops``, ``max_jobs_total``, the
    pre-allocated padded arrays) is derived only from ``max_dynamic_jobs``, which every
    regime in :func:`frozen_regimes` shares. So this is a safe, cheap swap -- it does not
    require reallocating any array per episode, unlike constructing a fresh env would.

    Args:
        env: A raw (unwrapped) :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`.
        regimes: The candidate regimes, sampled uniformly.
        seed: Seeds this wrapper's own regime-selection RNG. Independent of whatever
            seed ``reset(seed=...)`` is called with, so repeated resets (as SB3's
            auto-reset-on-episode-end does, with no explicit seed) still vary the
            regime and the underlying disruption realization.
    """

    def __init__(
        self, env: DynamicJSSPEnv, regimes: list[DisruptionConfig], seed: int | None = None
    ) -> None:
        super().__init__(env)
        self._regimes = regimes
        self._rng = np.random.default_rng(seed)
        self.last_regime_index: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        regime_index = int(self._rng.integers(len(self._regimes)))
        self.last_regime_index = regime_index
        episode_seed = int(self._rng.integers(0, 2**31 - 1))
        self.env.unwrapped.config = self._regimes[regime_index]
        return self.env.reset(seed=seed if seed is not None else episode_seed, options=options)


def _display_path(path: Path) -> str:
    """``path`` relative to :data:`REPO_ROOT` when possible, else the absolute path.

    ``--output-dir`` accepts any location (the smoke test in ``README.md`` deliberately
    points it outside the repo, e.g. a scratch/temp directory, to avoid littering the
    working tree) -- ``Path.relative_to`` raises ``ValueError`` in that case, which
    previously crashed this script's own final summary/print step *after* training and
    the checkpoint save had already succeeded, discarding a real result over a cosmetic
    path-formatting choice. This is display-only; it never affects where anything is
    actually written.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def make_dynamic_env_factory(instance: str, kappa: float, seed: int | None):
    """Zero-argument factory building one ``Monitor``-wrapped, adapted, domain-
    randomized :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv`.

    Wrapper order (innermost to outermost): raw env -> regime randomization (needs the
    raw env's mutable ``.config``) -> node-feature adapter (drops ``is_preempted`` so the
    observation matches the frozen checkpoint's 8-column contract) -> ``Monitor`` (SB3
    episode-stats bookkeeping, outermost so it sees the final wrapped observation).
    """

    def _factory():
        base_config = DisruptionConfig(max_dynamic_jobs=_MAX_DYNAMIC_JOBS)
        env = DynamicJSSPEnv(instance, base_config, kappa=kappa)
        env = DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=seed)
        env = _DynamicNodeFeatureAdapter(env)
        return Monitor(env)

    return _factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default="ft06", help="base JSSP instance (fixed-routing only)")
    parser.add_argument("--kappa", type=float, required=True, help="disruption-aware reward weight")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="warm-start weights (strict=True transfer, matching "
        "experiments/train_curriculum.py::transfer_weights); omit to train from scratch",
    )
    parser.add_argument("--stage", default="D1", help="label used in checkpoint/run-name filenames")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=None, help="default: 6 x base instance operation count")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    # 96/4, not build_hetero_maskable_ppo's own 64/3 defaults: curriculum_stage19_la20_final
    # (this script's only intended --init-checkpoint) was itself trained with
    # experiments/results/hpo_winner.json's winning config (hidden_dim=96, num_layers=4).
    # A fresh model built at 64/3 fails MaskablePPO.load(...).policy.load_state_dict(...,
    # strict=True) with a shape mismatch on every encoder/actor/critic layer -- caught by
    # this script's first-ever smoke test (2026-08-01) before any real training budget was
    # spent on it.
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    run_name = args.run_name or f"dynamic_kappa{args.kappa}_{args.stage}_{args.instance}_seed{args.seed}"
    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "tb_logs" / run_name
    checkpoint_dir = output_dir / "checkpoints"
    for directory in (output_dir, log_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    factory = make_dynamic_env_factory(args.instance, args.kappa, seed=args.seed)
    probe = factory()
    base_ops = probe.unwrapped.base_num_ops
    n_steps = args.n_steps if args.n_steps is not None else 6 * base_ops
    optimum = known_optimum(args.instance)
    static_lb = probe.unwrapped.static_lb
    probe.close()

    venv = DummyVecEnv(
        [make_dynamic_env_factory(args.instance, args.kappa, seed=args.seed + i) for i in range(args.n_envs)]
    )
    model = build_hetero_maskable_ppo(
        venv,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mlp_layers=args.mlp_layers,
        n_steps=n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
        verbose=1,
    )

    transfer_info = {"transferred_keys": 0, "skipped_keys": 0, "source": None}
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {args.init_checkpoint}")
        source_model = MaskablePPO.load(str(args.init_checkpoint), device="cpu")
        source_state = source_model.policy.state_dict()
        # strict=True: the architecture is size-agnostic (per-node/per-machine MLPs with
        # no hardcoded operation/machine count -- see models/policy.py), so a checkpoint
        # trained on a different instance size (static la20, 100 ops) transfers cleanly
        # onto this instance's differently-padded DynamicJSSPEnv space, exactly the same
        # mechanism experiments/train_curriculum.py::transfer_weights already relies on
        # for its own same-head-kind (JSSP->JSSP) stage transitions.
        model.policy.load_state_dict(source_state, strict=True)
        transfer_info = {
            "transferred_keys": len(source_state),
            "skipped_keys": 0,
            "source": str(args.init_checkpoint),
        }
        del source_model

    formats = ["stdout", "csv", "tensorboard"]
    try:
        model.set_logger(configure(str(log_dir), formats))
    except AssertionError:
        print("[warn]  tensorboard not installed; logging to CSV + stdout only")
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

    print(
        f"[setup] {args.instance} (DynamicJSSPEnv, kappa={args.kappa}, stage={args.stage}): "
        f"{base_ops} base operations, action space {venv.action_space.n}, "
        f"static LB {static_lb}, reference optimum {optimum}"
    )
    print(
        f"[setup] n_envs={args.n_envs} n_steps={n_steps} (buffer {n_steps * args.n_envs}) "
        f"warm_start={transfer_info['source']} transferred_keys={transfer_info['transferred_keys']}"
    )

    stats_callback = HeteroEpisodeStatsCallback(optimum=optimum)
    callbacks = [
        stats_callback,
        HeteroDiagnosticsCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix=f"dynamic_kappa{args.kappa}_{args.stage}_{args.instance}",
            verbose=0,
        ),
    ]

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=True)
    train_seconds = time.perf_counter() - t0

    final_checkpoint = checkpoint_dir / f"dynamic_kappa{args.kappa}_{args.stage}_{args.instance}_final.zip"
    model.save(final_checkpoint)

    progress = _read_progress_csv(log_dir / "progress.csv")
    plot_training_curves(
        stats_callback, progress, args.instance, optimum, output_dir / f"training_curves_{run_name}.png"
    )
    write_metrics_csv(stats_callback, optimum, output_dir / f"train_{run_name}_metrics.csv")

    summary = {
        "run_name": run_name,
        "instance": args.instance,
        "kappa": args.kappa,
        "stage": args.stage,
        "warm_start_checkpoint": transfer_info["source"],
        "transferred_keys": transfer_info["transferred_keys"],
        "timesteps": args.timesteps,
        "episodes": len(stats_callback.episode_makespans),
        "train_wall_clock_s": round(train_seconds, 1),
        "static_lower_bound": static_lb,
        "reference_optimum": optimum,
        "best_training_makespan": stats_callback.best_makespan,
        "last_100_mean_makespan": (
            float(np.mean(stats_callback.episode_makespans[-100:]))
            if stats_callback.episode_makespans
            else None
        ),
        "checkpoint": _display_path(final_checkpoint),
        "hyperparameters": {
            "n_envs": args.n_envs,
            "n_steps": n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seed": args.seed,
        },
    }
    summary_path = output_dir / f"summary_{run_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n[done] {args.timesteps} timesteps in {train_seconds / 60:.1f} min, "
        f"{summary['episodes']} episodes"
    )
    print(
        f"[done] best training makespan {stats_callback.best_makespan} "
        f"(reference optimum {optimum}, static LB {static_lb})"
    )
    print(f"[done] checkpoint -> {_display_path(final_checkpoint)}")
    print(f"[done] summary -> {_display_path(summary_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
