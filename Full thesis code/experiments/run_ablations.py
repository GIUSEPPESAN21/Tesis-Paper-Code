"""``AblationPlan.md`` Runs 0/1a/1b/2 -- the axes unblocked without ``DynamicJSSPEnv``
wiring (Axis 3/H4 and Axis 4's disruption-aware term remain blocked, per that plan §2.3/§4).

* Run 0 (reference): heterogeneous GIN encoder, dense reward, single-objective.
* Run 1a (Axis 1, homogeneous): the Sprint-1 :func:`~models.policy.build_maskable_ppo` path.
* Run 1b (Axis 1, GAT-hybrid): :func:`~models.policy.build_hetero_maskable_ppo` with
  ``encoder_type="gat_hybrid"``.
* Run 2 (Axis 2, sparse reward): ``JSSPEnv(reward_mode="sparse")``, everything else
  matched to Run 0.

All variants train under an *identical* fixed timestep budget on the same instance(s), so
the comparison is architecture/reward-shape only. Axis 1 additionally runs Friedman +
Nemenyi across instances (``AblationPlan.md`` §2.1); Axis 2 additionally reports the
"timesteps to cross the best-PDR milestone" comparison (§2.2) since that -- not just
final RPD -- is what the credit-assignment argument predicts.

Usage
-----
    python experiments/run_ablations.py --instances ft06 --seeds 1 --timesteps 400000
    python experiments/run_ablations.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.jssp_env import JSSPEnv
from experiments.hpo_hetero import EntropyScheduleCallback, entropy_schedule
from experiments.train_hetero import HeteroEpisodeStatsCallback, evaluate_policy_rollouts, reference_optimum
from models.policy import build_hetero_maskable_ppo, build_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

VARIANTS = ("reference", "homogeneous", "gat_hybrid", "sparse_reward")

#: The best-PDR RPD milestone each instance's dense-reward reference run crosses,
#: per ``TrainingDiagnosis.md`` §5.2 (ft06: fifo=67 -> 21.82% RPD vs. optimum 55).
#: Used only to report Axis 2's "timesteps to milestone" comparison for instances this
#: table has an entry for; instances without one just skip that one column.
BEST_PDR_RPD_MILESTONE = {"ft06": 21.82}


def make_env_and_config(variant: str, instance: str, config: dict):
    """Return a zero-arg env factory and the ``build_*`` kwargs for ``variant``."""
    if variant == "homogeneous":
        factory = lambda: JSSPEnv(instance)  # noqa: E731
        build_fn = build_maskable_ppo
        extra = dict(mlp_layers=2)
    elif variant == "gat_hybrid":
        factory = lambda: JSSPEnv(instance)  # noqa: E731
        build_fn = build_hetero_maskable_ppo
        extra = dict(encoder_type="gat_hybrid")
    elif variant == "sparse_reward":
        factory = lambda: JSSPEnv(instance, reward_mode="sparse")  # noqa: E731
        build_fn = build_hetero_maskable_ppo
        extra = dict(encoder_type="gin")
    elif variant == "reference":
        factory = lambda: JSSPEnv(instance)  # noqa: E731
        build_fn = build_hetero_maskable_ppo
        extra = dict(encoder_type="gin")
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return factory, build_fn, extra


def train_one_variant(
    *, variant: str, instance: str, seed: int, timesteps: int, n_envs: int, config: dict
) -> dict:
    from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415

    factory, build_fn, extra = make_env_and_config(variant, instance, config)
    env_factory = lambda: Monitor(factory())  # noqa: E731
    probe = env_factory()
    episode_length = probe.unwrapped.instance.num_operations
    n_steps = 6 * episode_length
    probe.close()

    venv = DummyVecEnv([env_factory for _ in range(n_envs)])
    optimum = reference_optimum(instance, fjsp=False)
    model = build_fn(
        venv,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        seed=seed,
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        gamma=1.0,
        gae_lambda=0.95,
        n_epochs=3,
        vf_coef=0.5,
        n_steps=n_steps,
        batch_size=64,
        verbose=1,
        **extra,
    )
    stats_callback = HeteroEpisodeStatsCallback(optimum=optimum)
    callbacks = [stats_callback]
    if "ent_coef_start" in config and "ent_coef_end" in config:
        schedule_fn = entropy_schedule(
            config["ent_coef_start"], config["ent_coef_end"],
            config.get("decay_fraction", 0.25),
        )
        callbacks.append(EntropyScheduleCallback(schedule_fn, total_timesteps=timesteps))
    model.learn(total_timesteps=timesteps, callback=callbacks)

    evaluation = evaluate_policy_rollouts(model, instance, fjsp=False, n_sampled=20, seed=seed)
    greedy_rpd = 100.0 * (evaluation["greedy_makespan"] - optimum) / optimum

    milestone_rpd = BEST_PDR_RPD_MILESTONE.get(instance)
    milestone_step = None
    if milestone_rpd is not None:
        for step, makespan in zip(stats_callback.episode_timesteps, stats_callback.episode_makespans):
            rpd = 100.0 * (makespan - optimum) / optimum
            if rpd <= milestone_rpd:
                milestone_step = step
                break

    venv.close()
    return {
        "variant": variant,
        "instance": instance,
        "seed": seed,
        "timesteps": timesteps,
        "reference_optimum": optimum,
        "greedy_makespan": evaluation["greedy_makespan"],
        "greedy_rpd_pct": greedy_rpd,
        "best_training_makespan": stats_callback.best_makespan,
        "last_100_mean_makespan": (
            float(np.mean(stats_callback.episode_makespans[-100:]))
            if stats_callback.episode_makespans
            else None
        ),
        "milestone_rpd_pct": milestone_rpd,
        "milestone_reached_at_step": milestone_step,
        "milestone_reached_frac_of_budget": (
            milestone_step / timesteps if milestone_step is not None else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["ft06"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds per variant")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--config-json", type=Path, default=None,
                         help="JSON file with the winning HPO config, overrides the flags above")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "ablations")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)

    config = dict(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    if args.config_json is not None:
        loaded = json.loads(args.config_json.read_text())
        config.update(loaded.get("best_params", loaded))

    instances = args.instances
    variants = args.variants
    seeds = args.seeds
    timesteps = args.timesteps
    n_envs = args.n_envs
    output_dir = args.output_dir

    if args.smoke_test:
        instances = ["ft06"]
        variants = list(VARIANTS)
        seeds = 1
        timesteps = 2000
        n_envs = 2
        output_dir = RESULTS_DIR / "ablations_smoke"

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for instance in instances:
        for variant in variants:
            for seed in range(seeds):
                print(f"[ablation] {variant} / {instance} / seed {seed}")
                row = train_one_variant(
                    variant=variant, instance=instance, seed=seed,
                    timesteps=timesteps, n_envs=n_envs, config=config,
                )
                results.append(row)
                print(f"[ablation] {variant}/{instance}/seed{seed}: "
                      f"greedy RPD={row['greedy_rpd_pct']:.2f}%")

    csv_path = output_dir / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"[done] {len(results)} runs -> {csv_path}")

    json_path = output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
