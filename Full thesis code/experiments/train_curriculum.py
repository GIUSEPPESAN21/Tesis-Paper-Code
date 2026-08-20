"""Sequential curriculum training over ``la01``-``la20`` (JSSP) then ``mk01``-``mk05`` (FJSP).

Week 21 sprint, step 3. Extends ``experiments/train_hetero.py``'s single-instance harness
into a sequence: train on instance ``i``, transfer weights into a freshly-built model
sized for instance ``i+1`` (:func:`~models.policy.build_hetero_maskable_ppo`'s size-agnostic
heads are exactly what makes this possible -- ``tests/test_hetero_policy.py
::test_weight_transfer_across_instance_sizes``), and periodically evaluate the
in-progress policy zero-shot on a fixed held-out set of *larger* instances so
generalization behavior is tracked from the start of the curriculum, not only at the end.

Why weight transfer is only ``strict=True`` within one problem type
--------------------------------------------------------------------
:class:`~models.policy.PerNodeMaskableActorCriticPolicy` picks its action head
(:class:`~models.policy.PerNodeActorHead` vs :class:`~models.policy.PerPairActorHead`)
from the action-space shape at construction time (``models/policy.py``'s
``_head_kind`` inference). Moving from a JSSP instance to an FJSP instance changes which
head class the target model builds, so a full ``state_dict`` load across that boundary
would either fail (mismatched parameter shapes) or silently mismatch. At the JSSP->FJSP
boundary (``la20`` -> ``mk01``) this script therefore transfers **only the encoder**
weights (``features_extractor.*``, the shared ``HeteroGINEncoder``/``HeteroGATHybridEncoder``)
and lets the new action/critic heads initialize fresh -- everywhere else (JSSP->JSSP,
FJSP->FJSP) the full policy transfers, per the existing tested mechanism.

Usage
-----
    python experiments/train_curriculum.py --timesteps-per-stage 300000
    python experiments/train_curriculum.py --smoke-test   # tiny-scale mechanics check
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

import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fjsp_env import FJSPEnv
from envs.jssp_env import JSSPEnv
from experiments.hpo_hetero import EntropyScheduleCallback, entropy_schedule
from experiments.train_hetero import (
    HeteroDiagnosticsCallback,
    HeteroEpisodeStatsCallback,
    _display_path,
    evaluate_policy_rollouts,
    make_env_factory,
    reference_optimum,
)
from models.policy import build_hetero_maskable_ppo

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

#: Small-to-medium curriculum, per Week 21 sprint step 3. JSSP first (shared "per_node"
#: head across every stage), then FJSP (shared "per_pair" head across its stages).
JSSP_CURRICULUM = [f"la{i:02d}" for i in range(1, 21)]
FJSP_CURRICULUM = [f"mk{i:02d}" for i in range(1, 6)]
CURRICULUM = [(name, False) for name in JSSP_CURRICULUM] + [(name, True) for name in FJSP_CURRICULUM]

#: Held-out, never-trained-on, strictly-larger instances for zero-shot generalization
#: tracking. JSSP: la21-la40 tier (150-300 ops vs. the curriculum's 50-100). FJSP: mk06/07
#: (larger than any of mk01-05), fetched via ``scripts/fetch_fjsp_instances.py``.
JSSP_HELDOUT = ["la21", "la25", "la30", "la35", "la40"]
FJSP_HELDOUT = ["mk06", "mk07"]


def transfer_weights(source_model, target_model, same_head_kind: bool) -> dict[str, int]:
    """Transfer weights from ``source_model`` into ``target_model``.

    Args:
        source_model: The just-finished stage's ``MaskablePPO``.
        target_model: The next stage's freshly-built ``MaskablePPO``.
        same_head_kind: ``True`` for JSSP->JSSP or FJSP->FJSP transitions (full,
            ``strict=True`` transfer). ``False`` at the JSSP->FJSP boundary (encoder
            weights only, ``strict=False``; the action/critic heads initialize fresh).

    Returns:
        ``{"transferred_keys": n, "skipped_keys": n}`` for the run log.
    """
    source_state = source_model.policy.state_dict()
    if same_head_kind:
        target_model.policy.load_state_dict(source_state, strict=True)
        return {"transferred_keys": len(source_state), "skipped_keys": 0}

    encoder_state = {k: v for k, v in source_state.items() if k.startswith("features_extractor.")}
    result = target_model.policy.load_state_dict(encoder_state, strict=False)
    return {
        "transferred_keys": len(encoder_state),
        "skipped_keys": len(result.missing_keys) + len(result.unexpected_keys),
    }


def build_stage_model(instance: str, fjsp: bool, n_envs: int, seed: int, config: dict, tb_log_dir: Path):
    factory = make_env_factory(instance, fjsp)
    probe = factory()
    episode_length = probe.unwrapped.instance.num_operations
    n_steps = 6 * episode_length
    static_lb = probe.unwrapped.static_lb
    probe.close()

    venv = DummyVecEnv([factory for _ in range(n_envs)])
    model = build_hetero_maskable_ppo(
        venv,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        encoder_type=config.get("encoder_type", "gin"),
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
    )
    try:
        model.set_logger(configure(str(tb_log_dir), ["stdout", "csv", "tensorboard"]))
    except AssertionError:
        model.set_logger(configure(str(tb_log_dir), ["stdout", "csv"]))
    return model, venv, static_lb


def zero_shot_generalization_eval(
    model, heldout: list[tuple[str, bool]], n_sampled: int, seed: int, config: dict
) -> list[dict]:
    """Zero-shot-evaluate the current policy on each held-out (larger) instance.

    ``model.predict()`` validates the observation against the ``observation_space`` it
    was *constructed* with (fixed to the training instance's size), so a held-out
    instance of a different size cannot be evaluated through ``model`` directly even
    though the per-node/per-pair heads are architecturally size-agnostic. The tested
    mechanism (``tests/test_hetero_policy.py::test_weight_transfer_across_instance_sizes``)
    is instead: build a fresh model sized for the held-out instance and
    ``load_state_dict(..., strict=True)`` the trained weights into it, then evaluate
    *that* model. Only instances whose problem type matches the current head kind are
    passed in by the caller (JSSP-stage models cannot score FJSP action spaces).
    """
    rows = []
    source_state = model.policy.state_dict()
    for name, fjsp in heldout:
        eval_env = FJSPEnv(name) if fjsp else JSSPEnv(name)
        eval_model = build_hetero_maskable_ppo(
            eval_env,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            encoder_type=config.get("encoder_type", "gin"),
            seed=seed,
        )
        eval_model.policy.load_state_dict(source_state, strict=True)

        optimum = reference_optimum(name, fjsp)
        evaluation = evaluate_policy_rollouts(eval_model, name, fjsp, n_sampled, seed)
        rpd = 100.0 * (evaluation["greedy_makespan"] - optimum) / optimum
        rows.append({"instance": name, "greedy_makespan": evaluation["greedy_makespan"],
                      "reference_optimum": optimum, "rpd_pct": rpd})
    return rows


def run_curriculum(
    *,
    config: dict,
    timesteps_per_stage: int,
    n_envs: int,
    checkpoint_every: int,
    n_sampled_eval: int,
    curriculum: list[tuple[str, bool]],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stage_log: list[dict] = []
    generalization_log: list[dict] = []
    prev_model = None
    prev_head_kind = None

    for stage_idx, (instance, fjsp) in enumerate(curriculum):
        head_kind = "per_pair" if fjsp else "per_node"
        tb_log_dir = output_dir / "tb_logs" / f"curriculum_stage{stage_idx:02d}_{instance}"
        model, venv, static_lb = build_stage_model(
            instance, fjsp, n_envs, seed=stage_idx, config=config, tb_log_dir=tb_log_dir
        )

        transfer_info = {"transferred_keys": 0, "skipped_keys": 0}
        if prev_model is not None:
            transfer_info = transfer_weights(
                prev_model, model, same_head_kind=(prev_head_kind == head_kind)
            )
            prev_model = None  # release the previous stage's venv/model

        optimum = reference_optimum(instance, fjsp)
        stats_callback = HeteroEpisodeStatsCallback(optimum=optimum)
        callbacks = [
            stats_callback,
            HeteroDiagnosticsCallback(),
            CheckpointCallback(
                save_freq=max(checkpoint_every // n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix=f"curriculum_stage{stage_idx:02d}_{instance}",
                verbose=0,
            ),
        ]
        if "ent_coef_start" in config and "ent_coef_end" in config:
            schedule_fn = entropy_schedule(
                config["ent_coef_start"], config["ent_coef_end"],
                config.get("decay_fraction", 0.25),
            )
            callbacks.append(EntropyScheduleCallback(schedule_fn, total_timesteps=timesteps_per_stage))

        print(f"[stage {stage_idx:02d}] {instance} ({'FJSP' if fjsp else 'JSSP'}), "
              f"transfer={transfer_info}")
        t0 = time.perf_counter()
        model.learn(total_timesteps=timesteps_per_stage, callback=callbacks)
        train_seconds = time.perf_counter() - t0

        evaluation = evaluate_policy_rollouts(model, instance, fjsp, n_sampled_eval, seed=0)
        rpd = 100.0 * (evaluation["greedy_makespan"] - optimum) / optimum
        stage_row = {
            "stage": stage_idx,
            "instance": instance,
            "problem": "FJSP" if fjsp else "JSSP",
            "static_lower_bound": static_lb,
            "reference_optimum": optimum,
            "greedy_makespan": evaluation["greedy_makespan"],
            "rpd_pct": rpd,
            "train_wall_clock_s": round(train_seconds, 1),
            "transfer": transfer_info,
        }
        stage_log.append(stage_row)
        print(f"[stage {stage_idx:02d}] done: greedy RPD={rpd:.2f}% ({train_seconds/60:.1f} min)")

        heldout = JSSP_HELDOUT if not fjsp else FJSP_HELDOUT
        heldout_pairs = [(name, fjsp) for name in heldout]
        gen_rows = zero_shot_generalization_eval(
            model, heldout_pairs, n_sampled=min(20, n_sampled_eval), seed=0, config=config
        )
        for row in gen_rows:
            row.update(stage=stage_idx, trained_on=instance)
        generalization_log.extend(gen_rows)
        print(f"[stage {stage_idx:02d}] generalization: "
              + ", ".join(f"{r['instance']}={r['rpd_pct']:.1f}%" for r in gen_rows))

        final_ckpt = checkpoint_dir / f"curriculum_stage{stage_idx:02d}_{instance}_final.zip"
        model.save(final_ckpt)
        venv.close()
        prev_model = model
        prev_head_kind = head_kind

    stage_csv = output_dir / "curriculum_stages.csv"
    with stage_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "instance", "problem", "static_lower_bound", "reference_optimum",
                        "greedy_makespan", "rpd_pct", "train_wall_clock_s", "transfer"],
        )
        writer.writeheader()
        for row in stage_log:
            writer.writerow(row)

    gen_csv = output_dir / "curriculum_generalization.csv"
    with gen_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "trained_on", "instance", "greedy_makespan", "reference_optimum", "rpd_pct"],
        )
        writer.writeheader()
        for row in generalization_log:
            writer.writerow(row)

    final_checkpoint = checkpoint_dir / f"curriculum_stage{len(curriculum)-1:02d}_{curriculum[-1][0]}_final.zip"
    summary = {
        "config": config,
        "n_stages": len(curriculum),
        "timesteps_per_stage": timesteps_per_stage,
        "final_checkpoint": _display_path(final_checkpoint),
        "stage_log": stage_log,
        "generalization_log": generalization_log,
    }
    summary_path = output_dir / "curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] curriculum summary -> {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps-per-stage", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--eval-rollouts", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--encoder-type", default="gin", choices=["gin", "gat_hybrid"])
    parser.add_argument("--config-json", type=Path, default=None,
                         help="path to a JSON file with the winning HPO config "
                              "(overrides --learning-rate/--clip-range/--ent-coef/--hidden-dim/--num-layers)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "curriculum")
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny-scale run (2 stages, 2000 timesteps) to validate mechanics")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.torch_threads)

    config = dict(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder_type=args.encoder_type,
    )
    if args.config_json is not None:
        loaded = json.loads(args.config_json.read_text())
        config.update(loaded.get("best_params", loaded))

    curriculum = CURRICULUM
    timesteps_per_stage = args.timesteps_per_stage
    n_envs = args.n_envs
    checkpoint_every = args.checkpoint_every
    eval_rollouts = args.eval_rollouts
    output_dir = args.output_dir

    if args.smoke_test:
        curriculum = [("la01", False), ("la02", False), ("mk01", True)]
        timesteps_per_stage = 2000
        n_envs = 2
        checkpoint_every = 1000
        eval_rollouts = 5
        output_dir = RESULTS_DIR / "curriculum_smoke"

    run_curriculum(
        config=config,
        timesteps_per_stage=timesteps_per_stage,
        n_envs=n_envs,
        checkpoint_every=checkpoint_every,
        n_sampled_eval=eval_rollouts,
        curriculum=curriculum,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
