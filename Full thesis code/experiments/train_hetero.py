"""Week 18 training driver for the heterogeneous GNN + PPO policy.

Extends ``experiments/train_ft06.py``'s harness (``Monitor``-wrapped env,
``EpisodeStatsCallback``, headless ``Agg`` plotting, per-episode metrics CSV) rather
than replacing it, per ``GNN_PPO_Architecture_Spec_v2.md``'s prime directive.  What is
new here and not in the Week 1 script:

* the heterogeneous, size-agnostic policy (:func:`~models.policy.build_hetero_maskable_ppo`);
* **both** problem variants --- ``JSSPEnv`` (``--instance ft06``) and ``FJSPEnv``
  (``--instance mk01 --fjsp``), which differ only in which environment class is
  constructed and which reference-optimum table is consulted;
* the full ``TrainingPlan_Week17.md`` §2 hyperparameter set as defaults, so a bare
  invocation reproduces the planned configuration;
* the §4.1 logging plan: SB3's native ``rollout/*`` and ``train/*`` panels plus the
  three ``custom/*`` diagnostics that plan asks for (per-episode makespan RPD, actor
  logit magnitude, machine-embedding norm), written to TensorBoard **and** to a CSV so
  the curves can be re-plotted without a TensorBoard install;
* §4.3 checkpointing, with the stage/instance/timestep in every filename.

Why ``stable-baselines3`` rather than a hand-written PPO
---------------------------------------------------------
The architecture spec's §3.3 anticipated that heterogeneous-graph *batching* might make
SB3 impractical and left the choice open.  It does not, for one specific reason: every
environment in a given run has a fixed, identical graph size, so a batch of observations
is a stack of equal-shaped dense tensors --- exactly the ``(B, N, F)`` / ``(B, N, N)``
layout SB3's ``Dict`` observation buffer already produces, with no ragged batching, no
``Batch.from_data_list``, and no custom collation.  The heterogeneous part lives entirely
inside the features extractor.  Keeping SB3 therefore preserves the tested masking path
(``MaskableCategorical``), the tested GAE/PPO-loss implementation, and the
``tests/test_policy.py`` regression contract, and it is what
``GNN_PPO_Architecture_Spec_v2.md`` §8's ``TrainedRLBaseline`` assumes it can
``.load(...)``.  A custom PPO would have re-implemented all of that to gain nothing.
The one thing SB3 forced was a policy-class override rather than a features-extractor
alone --- see ``models/policy.py``'s module docstring.

Usage
-----
    python experiments/train_hetero.py --instance ft06 --timesteps 400000
    python experiments/train_hetero.py --instance mk01 --fjsp --timesteps 300000
    tensorboard --logdir experiments/results/tb_logs
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fjsp_env import FJSPEnv
from envs.fjsp_loader import fjsp_known_optimum
from envs.jssp_env import JSSPEnv
from envs.loader import known_optimum
from models.policy import build_hetero_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results"


# --------------------------------------------------------------------------- callbacks


class HeteroEpisodeStatsCallback(BaseCallback):
    """Per-episode return / makespan / RPD, to memory and to the SB3 logger.

    The Week 1 :class:`~experiments.train_ft06.EpisodeStatsCallback` recorded the first
    two in memory only.  This version also pushes ``custom/makespan`` and
    ``custom/makespan_rpd_pct`` into the logger (so they appear in TensorBoard next to
    the ``rollout/*`` panels) using the same RPD formula ``run_evaluation.py::build_row``
    uses --- one formula, two consumers, per ``TrainingPlan_Week17.md`` §4.1.

    Args:
        optimum: Reference makespan for the instance being trained on, used as the RPD
            denominator.
    """

    def __init__(self, optimum: float) -> None:
        super().__init__()
        self.optimum = float(optimum)
        self.episode_returns: list[float] = []
        self.episode_makespans: list[float] = []
        self.episode_timesteps: list[int] = []
        self.best_makespan: float = float("inf")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info or "makespan" not in info:
                continue
            makespan = float(info["makespan"])
            self.episode_returns.append(float(info["episode"]["r"]))
            self.episode_makespans.append(makespan)
            self.episode_timesteps.append(int(self.num_timesteps))
            self.best_makespan = min(self.best_makespan, makespan)

            self.logger.record("custom/makespan", makespan)
            self.logger.record(
                "custom/makespan_rpd_pct", 100.0 * (makespan - self.optimum) / self.optimum
            )
            self.logger.record("custom/best_makespan", self.best_makespan)
        return True

    def recent_makespan_rpd_pct(self, window: int = 100) -> float:
        """Mean RPD (%) over the last ``window`` episodes, for Optuna pruning reports.

        Returns ``inf`` if no episodes have completed yet, so an unstarted trial reports
        as unambiguously bad rather than raising.
        """
        if not self.episode_makespans:
            return float("inf")
        recent = self.episode_makespans[-window:]
        return 100.0 * (float(np.mean(recent)) - self.optimum) / self.optimum


class HeteroDiagnosticsCallback(BaseCallback):
    """Log the two architecture-specific diagnostics ``TrainingPlan_Week17.md`` §4.1 asks for.

    * ``custom/actor_logit_abs_mean`` / ``custom/actor_logit_abs_max`` --- raw, *pre-mask*
      per-node logit magnitudes.  §3.2 names an extreme-logit blow-up as the most likely
      cause of an entropy collapse in a brand-new action head, and entropy alone cannot
      distinguish that from ordinary sharpening.
    * ``custom/machine_embedding_norm`` (and its operation-node counterpart) --- the mean
      L2 norm of the encoder's machine-branch output.  A norm collapsing toward 0 or
      exploding says the new eligibility relation is unstable *before* the heads see it.

    Implemented as forward hooks so the cost is one ``.item()`` per forward pass rather
    than a second forward pass per rollout.  The values reported are the most recent
    forward's, which is what a "is this diverging right now" diagnostic wants.
    """

    def _init_callback(self) -> None:
        self._stats: dict[str, float] = {}
        policy = self.model.policy
        policy.action_net.register_forward_hook(self._actor_hook)
        policy.features_extractor.register_forward_hook(self._encoder_hook)

    def _actor_hook(self, module: torch.nn.Module, inputs, output: torch.Tensor) -> None:
        with torch.no_grad():
            self._stats["actor_logit_abs_mean"] = float(output.abs().mean())
            self._stats["actor_logit_abs_max"] = float(output.abs().max())

    def _encoder_hook(self, module: torch.nn.Module, inputs, output: torch.Tensor) -> None:
        with torch.no_grad():
            if module.last_node_embeddings_mach is not None:
                self._stats["machine_embedding_norm"] = float(
                    module.last_node_embeddings_mach.norm(dim=-1).mean()
                )
            if module.last_node_embeddings_ope is not None:
                self._stats["operation_embedding_norm"] = float(
                    module.last_node_embeddings_ope.norm(dim=-1).mean()
                )

    def _on_rollout_end(self) -> None:
        for key, value in self._stats.items():
            self.logger.record(f"custom/{key}", value)

    def _on_step(self) -> bool:
        return True


# ------------------------------------------------------------------------- env factory


def _display_path(path: Path) -> str:
    """``path`` relative to :data:`REPO_ROOT` when possible, else the absolute path.

    ``--output-dir`` accepts any location; ``Path.relative_to`` raises ``ValueError``
    when it points outside ``REPO_ROOT``, which previously crashed this script's final
    summary/print step *after* training and the checkpoint save had already succeeded
    (caught by the reproducibility smoke test in ``README.md``, which deliberately uses
    an out-of-repo scratch directory). Display-only; never affects where anything is
    actually written.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def make_env_factory(instance: str, fjsp: bool):
    """Return a zero-argument factory building one ``Monitor``-wrapped environment.

    Args:
        instance: Instance name, e.g. ``"ft06"`` or ``"mk01"``.
        fjsp: Build an :class:`~envs.fjsp_env.FJSPEnv` instead of a
            :class:`~envs.jssp_env.JSSPEnv`.

    Returns:
        A callable suitable for ``DummyVecEnv``.
    """

    def _factory():
        env = FJSPEnv(instance) if fjsp else JSSPEnv(instance)
        return Monitor(env)

    return _factory


def reference_optimum(instance: str, fjsp: bool) -> int:
    """Literature reference makespan for ``instance``.

    Args:
        instance: Instance name.
        fjsp: Consult the FJSP table (``data/fjsp_known_optima.csv``) instead of the
            JSSP one.

    Returns:
        The reference makespan.
    """
    return fjsp_known_optimum(instance) if fjsp else known_optimum(instance)


# ------------------------------------------------------------------------- evaluation


def evaluate_policy_rollouts(
    model, instance: str, fjsp: bool, n_sampled: int, seed: int
) -> dict[str, float]:
    """Roll the trained policy out and report greedy and best-of-``n_sampled`` makespans.

    Both numbers matter and they answer different questions: the greedy rollout is what
    :class:`~baselines.interface.TrainedRLBaseline` will report under
    ``deterministic=True`` (one rollout, no search), while best-of-N is the fair
    comparison against a *stochastic* baseline given a comparable wall-clock budget.

    Args:
        model: A trained ``MaskablePPO``.
        instance: Instance name.
        fjsp: Whether to build an FJSP environment.
        n_sampled: Number of stochastic rollouts.
        seed: Seed for the evaluation environment.

    Returns:
        Dict with ``greedy_makespan``, ``sampled_best_makespan``,
        ``sampled_mean_makespan``, ``feasible`` and timing fields.
    """
    env = FJSPEnv(instance) if fjsp else JSSPEnv(instance)

    def rollout(deterministic: bool, rollout_seed: int) -> tuple[float, bool]:
        obs, _ = env.reset(seed=rollout_seed)
        terminated, info = False, {}
        while not terminated:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic
            )
            obs, _, terminated, _, info = env.step(int(action))
        return float(info["makespan"]), bool(info["schedule_feasible"])

    t0 = time.perf_counter()
    greedy_makespan, greedy_feasible = rollout(True, seed)
    greedy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sampled = [rollout(False, seed + 1 + i) for i in range(n_sampled)]
    sampled_time = time.perf_counter() - t0

    makespans = [m for m, _ in sampled] or [float("nan")]
    return {
        "greedy_makespan": greedy_makespan,
        "greedy_feasible": greedy_feasible,
        "greedy_wall_clock_s": greedy_time,
        "sampled_best_makespan": float(np.min(makespans)),
        "sampled_mean_makespan": float(np.mean(makespans)),
        "sampled_rollouts": n_sampled,
        "sampled_wall_clock_s": sampled_time,
        "all_feasible": bool(greedy_feasible and all(f for _, f in sampled)),
    }


# ---------------------------------------------------------------------------- plotting


def _read_progress_csv(path: Path) -> dict[str, list[float]]:
    """Read SB3's ``progress.csv`` into column lists, dropping blank cells."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    columns: dict[str, list[float]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(float(value) if value not in ("", None) else float("nan"))
    return columns


def plot_training_curves(
    stats: HeteroEpisodeStatsCallback,
    progress: dict[str, list[float]],
    instance: str,
    optimum: float,
    out_path: Path,
) -> None:
    """Save the six-panel training figure ``TrainingPlan_Week17.md`` §3/§4 calls for.

    Panels: episode return, episode makespan vs. the reference optimum, entropy loss,
    value loss, explained variance, and ``approx_kl`` / ``clip_fraction`` --- i.e. the
    metrics §3.1-§3.4 specify healthy-vs-broken criteria for, in one figure.

    Args:
        stats: The episode-statistics callback from the finished run.
        progress: Columns parsed from SB3's ``progress.csv``.
        instance: Instance name, for titles.
        optimum: Reference makespan, drawn as a horizontal line on the makespan panel.
        out_path: Where to write the PNG.
    """
    steps = progress.get("time/total_timesteps", [])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax_return, ax_makespan, ax_entropy = axes[0]
    ax_value, ax_explained, ax_ppo = axes[1]

    ax_return.plot(stats.episode_timesteps, stats.episode_returns, lw=0.6, alpha=0.4)
    if len(stats.episode_returns) >= 20:
        window = max(len(stats.episode_returns) // 50, 5)
        smoothed = np.convolve(
            stats.episode_returns, np.ones(window) / window, mode="valid"
        )
        ax_return.plot(stats.episode_timesteps[window - 1 :], smoothed, lw=2, color="tab:blue")
    ax_return.set_xlabel("timestep")
    ax_return.set_ylabel("episode return (scaled, undiscounted)")
    ax_return.set_title(f"{instance}: episode return")

    ax_makespan.plot(
        stats.episode_timesteps, stats.episode_makespans, lw=0.6, alpha=0.35, color="tab:orange"
    )
    if len(stats.episode_makespans) >= 20:
        window = max(len(stats.episode_makespans) // 50, 5)
        smoothed = np.convolve(
            stats.episode_makespans, np.ones(window) / window, mode="valid"
        )
        ax_makespan.plot(
            stats.episode_timesteps[window - 1 :], smoothed, lw=2, color="tab:orange"
        )
    ax_makespan.axhline(optimum, color="tab:green", ls="--", label=f"reference optimum ({optimum})")
    ax_makespan.set_xlabel("timestep")
    ax_makespan.set_ylabel("episode makespan")
    ax_makespan.set_title(f"{instance}: episode makespan")
    ax_makespan.legend()

    def line(ax, key: str, label: str, color: str) -> None:
        values = progress.get(key)
        if values and steps:
            ax.plot(steps[: len(values)], values, lw=1.4, color=color, label=label)

    line(ax_entropy, "train/entropy_loss", "entropy loss", "tab:red")
    ax_entropy.set_title(f"{instance}: entropy loss (negative entropy)")
    ax_entropy.set_xlabel("timestep")
    ax_entropy.legend()

    line(ax_value, "train/value_loss", "value loss", "tab:purple")
    ax_value.set_yscale("log")
    ax_value.set_title(f"{instance}: value loss (log scale)")
    ax_value.set_xlabel("timestep")
    ax_value.legend()

    line(ax_explained, "train/explained_variance", "explained variance", "tab:brown")
    ax_explained.axhline(1.0, color="tab:green", ls="--", lw=1)
    ax_explained.axhline(0.0, color="tab:gray", ls=":", lw=1)
    ax_explained.set_title(f"{instance}: explained variance")
    ax_explained.set_xlabel("timestep")
    ax_explained.legend()

    line(ax_ppo, "train/approx_kl", "approx_kl", "tab:cyan")
    line(ax_ppo, "train/clip_fraction", "clip_fraction", "tab:olive")
    ax_ppo.set_title(f"{instance}: PPO step-size diagnostics")
    ax_ppo.set_xlabel("timestep")
    ax_ppo.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_metrics_csv(stats: HeteroEpisodeStatsCallback, optimum: float, out_path: Path) -> None:
    """Persist per-episode metrics for provenance / later re-plotting."""
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "episode_return", "episode_makespan", "rpd_pct"])
        for step, ret, makespan in zip(
            stats.episode_timesteps, stats.episode_returns, stats.episode_makespans
        ):
            writer.writerow([step, ret, makespan, 100.0 * (makespan - optimum) / optimum])


# --------------------------------------------------------------------------------- main


def run_one_seed(args: argparse.Namespace, seed: int) -> dict:
    """Train one seed end-to-end and return the summary dict.

    T2.1 (Recomendacion del tutor: semillas multiples de entrenamiento): extracted,
    unchanged line-for-line, from the pre-T2.1 body of ``main()`` -- every ``args.seed``
    reference below is now the ``seed`` parameter, so a single-seed call (``--seeds``
    omitted) is byte-for-bit identical to the original ``main()``.
    """
    torch.set_num_threads(args.torch_threads)
    run_name = args.run_name or f"hetero_stage{args.stage}_{args.instance}_seed{seed}"
    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "tb_logs" / run_name
    checkpoint_dir = output_dir / "checkpoints"
    for directory in (output_dir, log_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    factory = make_env_factory(args.instance, args.fjsp)
    probe = factory()
    episode_length = probe.unwrapped.instance.num_operations
    n_steps = args.n_steps if args.n_steps is not None else 6 * episode_length
    optimum = reference_optimum(args.instance, args.fjsp)
    static_lb = probe.unwrapped.static_lb
    probe.close()

    venv = DummyVecEnv([factory for _ in range(args.n_envs)])
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
        seed=seed,
        verbose=1,
    )
    # CSV alongside TensorBoard so the curves survive without a TensorBoard install --
    # the same "reproducible on a reviewer's machine" principle train_ft06.py follows.
    # TensorBoard is the plan's chosen monitoring tool (TrainingPlan_Week17.md Section 4)
    # but is an optional extra, so its absence degrades to CSV rather than failing a run.
    formats = ["stdout", "csv", "tensorboard"]
    try:
        model.set_logger(configure(str(log_dir), formats))
    except AssertionError:
        print("[warn]  tensorboard not installed; logging to CSV + stdout only")
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

    print(
        f"[setup] {args.instance} ({'FJSP' if args.fjsp else 'JSSP'}): "
        f"{episode_length} operations/episode, action space {venv.action_space.n}, "
        f"static LB {static_lb}, reference optimum {optimum}"
    )
    print(
        f"[setup] n_envs={args.n_envs} n_steps={n_steps} "
        f"(buffer {n_steps * args.n_envs}) lr={args.learning_rate} gamma={args.gamma}"
    )

    stats_callback = HeteroEpisodeStatsCallback(optimum=optimum)
    callbacks = [
        stats_callback,
        HeteroDiagnosticsCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix=f"ppo_hetero_stage{args.stage}_{args.instance}",
            verbose=0,
        ),
    ]

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    train_seconds = time.perf_counter() - t0

    final_checkpoint = checkpoint_dir / f"ppo_hetero_stage{args.stage}_{args.instance}_final.zip"
    model.save(final_checkpoint)

    evaluation = evaluate_policy_rollouts(
        model, args.instance, args.fjsp, args.eval_rollouts, seed
    )
    progress = _read_progress_csv(log_dir / "progress.csv")

    plot_training_curves(
        stats_callback,
        progress,
        args.instance,
        optimum,
        output_dir / f"training_curves_{run_name}.png",
    )
    write_metrics_csv(stats_callback, optimum, output_dir / f"train_{run_name}_metrics.csv")

    summary = {
        "run_name": run_name,
        "instance": args.instance,
        "problem": "FJSP" if args.fjsp else "JSSP",
        "timesteps": args.timesteps,
        "episodes": len(stats_callback.episode_makespans),
        "train_wall_clock_s": round(train_seconds, 1),
        "static_lower_bound": static_lb,
        "reference_optimum": optimum,
        "best_training_makespan": stats_callback.best_makespan,
        "first_episode_makespan": (
            stats_callback.episode_makespans[0] if stats_callback.episode_makespans else None
        ),
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
            "seed": seed,
        },
        **evaluation,
    }
    summary_path = output_dir / f"summary_{run_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[done] {args.timesteps} timesteps in {train_seconds / 60:.1f} min, "
          f"{summary['episodes']} episodes")
    print(f"[done] best training makespan {stats_callback.best_makespan} "
          f"(reference optimum {optimum}, static LB {static_lb})")
    print(f"[done] greedy rollout makespan {evaluation['greedy_makespan']}, "
          f"best of {args.eval_rollouts} sampled {evaluation['sampled_best_makespan']}")
    print(f"[done] summary -> {_display_path(summary_path)}")
    return summary


def _run_seeds_in_subprocesses(argv: list[str], seeds: list[int], n_workers: int) -> int:
    """T2.1's ``--n-workers`` path: one subprocess per seed (up to ``n_workers`` at a
    time). See ``experiments/train_dynamic.py``'s identical helper for the full
    rationale (SB3/torch per-process state isn't pool-friendly)."""
    import subprocess

    base_argv: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("--seeds", "--n-workers"):
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        base_argv.append(token)
        i += 1

    running: list[subprocess.Popen] = []
    exit_codes: list[int] = []
    remaining = list(seeds)
    while remaining or running:
        while remaining and len(running) < n_workers:
            seed = remaining.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), *base_argv, "--seeds", str(seed)]
            print(f"[train_hetero] launching seed {seed}: {' '.join(cmd)}")
            running.append(subprocess.Popen(cmd))
        for proc in list(running):
            ret = proc.poll()
            if ret is not None:
                exit_codes.append(ret)
                running.remove(proc)
        if running:
            time.sleep(2.0)

    if any(code != 0 for code in exit_codes):
        print(f"[train_hetero] {sum(1 for c in exit_codes if c != 0)}/{len(exit_codes)} seed(s) failed", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="ft06")
    parser.add_argument("--fjsp", action="store_true", help="use FJSPEnv (flexible routing)")
    parser.add_argument("--stage", default="A", help="label used in checkpoint filenames")
    parser.add_argument("--timesteps", type=int, default=400_000)
    # TrainingPlan_Week17.md Section 2 defaults.
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=None, help="default: 6 x episode length")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="single-seed default; superseded by --seeds if given")
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="T2.1: train one run per seed (e.g. --seeds 0 1 2 3 4), each writing its "
        "own summary_..._seed{N}.json. Omit to keep the pre-T2.1 single-seed behavior.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="T2.1: with --seeds and --n-workers > 1, run seeds concurrently as "
        "separate subprocesses. --n-workers 1 (default) runs sequentially in-process.",
    )
    parser.add_argument("--eval-rollouts", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    seeds = args.seeds if args.seeds else [args.seed]

    if len(seeds) > 1 and args.n_workers > 1:
        return _run_seeds_in_subprocesses(argv or sys.argv[1:], seeds, args.n_workers)

    summaries = []
    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n[train_hetero] === seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) ===")
        summaries.append(run_one_seed(args, seed))

    if len(seeds) > 1:
        wall_clocks = [s["train_wall_clock_s"] for s in summaries]
        print(
            f"\n[done] {len(seeds)} seed(s) complete. train_wall_clock_s: "
            f"mean={np.mean(wall_clocks):.1f}s std={np.std(wall_clocks, ddof=1) if len(wall_clocks) > 1 else 0.0:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
