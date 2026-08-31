"""Week 1 smoke test: train the GIN+MaskablePPO pipeline on ft06 end to end.

This is deliberately not a convergence run (see ``models/policy.py`` and
``VALIDATION_WEEK1.md`` for why). The goal is to prove the full pipeline --
environment -> action masking -> GIN encoder -> PPO update -- runs for a sustained
number of timesteps without crashing, and to produce the reward-curve evidence
``VALIDATION_WEEK1.md`` reports against.

Usage
-----
    python experiments/train_ft06.py                       # defaults, ~5k timesteps
    python experiments/train_ft06.py --timesteps 20000 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: this script may run without a display
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.jssp_env import JSSPEnv
from envs.loader import known_optimum
from models.policy import build_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results"


class EpisodeStatsCallback(BaseCallback):
    """Record per-episode return and makespan as training progresses.

    ``Monitor`` (wrapped around the env below) attaches an ``"episode"`` dict with the
    undiscounted return to ``info`` on the timestep that ends an episode; this callback
    also pulls ``info["makespan"]``, which :meth:`~envs.jssp_env.JSSPEnv.step` attaches
    on termination, so both curves come from the single source of truth the env itself
    reports.
    """

    def __init__(self) -> None:
        super().__init__()
        self.episode_returns: list[float] = []
        self.episode_makespans: list[float] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done and "episode" in info:
                self.episode_returns.append(float(info["episode"]["r"]))
                self.episode_makespans.append(float(info["makespan"]))
                self.episode_timesteps.append(int(self.num_timesteps))
        return True


def plot_curves(callback: EpisodeStatsCallback, instance_name: str, out_path: Path) -> None:
    """Save a two-panel (return, makespan) training curve figure."""
    optimum = known_optimum(instance_name)

    fig, (ax_return, ax_makespan) = plt.subplots(1, 2, figsize=(11, 4))

    ax_return.plot(callback.episode_timesteps, callback.episode_returns, marker=".", lw=1)
    ax_return.set_xlabel("timestep")
    ax_return.set_ylabel("episode return (scaled, undiscounted)")
    ax_return.set_title(f"{instance_name}: episode return")

    ax_makespan.plot(
        callback.episode_timesteps, callback.episode_makespans, marker=".", lw=1, color="tab:orange"
    )
    ax_makespan.axhline(optimum, color="tab:green", ls="--", label=f"known optimum ({optimum})")
    ax_makespan.set_xlabel("timestep")
    ax_makespan.set_ylabel("episode makespan")
    ax_makespan.set_title(f"{instance_name}: episode makespan")
    ax_makespan.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_metrics_csv(callback: EpisodeStatsCallback, out_path: Path) -> None:
    """Persist per-episode metrics for provenance / later re-plotting."""
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "episode_return", "episode_makespan"])
        writer.writerows(
            zip(callback.episode_timesteps, callback.episode_returns, callback.episode_makespans)
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="ft06")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=360, help="rollout length (10 ft06 episodes)")
    parser.add_argument("--batch-size", type=int, default=180)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(JSSPEnv(args.instance))
    model = build_maskable_ppo(
        env,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=1,
    )

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    model.save(args.output_dir / f"ppo_gin_{args.instance}.zip")
    plot_path = args.output_dir / f"reward_curve_{args.instance}.png"
    plot_curves(callback, args.instance, plot_path)
    write_metrics_csv(callback, args.output_dir / f"train_{args.instance}_metrics.csv")

    n_episodes = len(callback.episode_makespans)
    best_makespan = min(callback.episode_makespans) if callback.episode_makespans else float("nan")
    optimum = known_optimum(args.instance)
    print(f"\n[done] {args.timesteps} timesteps, {n_episodes} episodes completed")
    print(f"[done] best makespan seen: {best_makespan} (known optimum: {optimum})")
    try:
        plot_display = plot_path.relative_to(REPO_ROOT)
    except ValueError:
        plot_display = plot_path
    print(f"[done] plot -> {plot_display}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
