"""Tests for experiments/run_2x2_ablation.py (Fase 3, T3.1). Pure command-building
logic only -- no training is invoked."""

from __future__ import annotations

from experiments.run_2x2_ablation import build_cells, evaluation_command, training_command


class TestBuildCells:
    def test_four_cells_with_correct_env_reward_combinations(self) -> None:
        cells = build_cells(
            static_checkpoint="ckpt_static.zip", kappa_robust=0.5, kappa_d_checkpoint="ckpt_d.zip",
            instance="ft06", nominal_episode_fraction=0.2,
        )
        by_label = {c.label: c for c in cells}
        assert by_label["A"].env_type == "estatico" and by_label["A"].reward_type == "nominal"
        assert by_label["B"].env_type == "estatico" and by_label["B"].reward_type == "robusta"
        assert by_label["C"].env_type == "dinamico" and by_label["C"].reward_type == "nominal"
        assert by_label["D"].env_type == "dinamico" and by_label["D"].reward_type == "robusta"

    def test_cell_a_and_d_reuse_checkpoints_b_and_c_do_not(self) -> None:
        cells = build_cells(
            static_checkpoint="ckpt_static.zip", kappa_robust=0.5, kappa_d_checkpoint="ckpt_d.zip",
            instance="ft06", nominal_episode_fraction=0.2,
        )
        by_label = {c.label: c for c in cells}
        assert by_label["A"].reuse_checkpoint == "ckpt_static.zip"
        assert by_label["D"].reuse_checkpoint == "ckpt_d.zip"
        assert by_label["B"].reuse_checkpoint is None
        assert by_label["C"].reuse_checkpoint is None

    def test_cell_c_is_the_critical_missing_cell_kappa_zero_dynamic(self) -> None:
        cells = build_cells(
            static_checkpoint="ckpt.zip", kappa_robust=0.5, kappa_d_checkpoint=None,
            instance="ft06", nominal_episode_fraction=0.2,
        )
        cell_c = next(c for c in cells if c.label == "C")
        assert cell_c.kappa == 0.0
        assert cell_c.training_regime == "domain_randomized"


class TestTrainingCommand:
    def test_none_for_reused_checkpoint_cell(self) -> None:
        cells = build_cells(static_checkpoint="ckpt.zip", kappa_robust=0.5, kappa_d_checkpoint="ckpt_d.zip", instance="ft06", nominal_episode_fraction=0.2)
        cell_a = next(c for c in cells if c.label == "A")
        cmd = training_command(cell_a, instance="ft06", init_checkpoint="ckpt.zip", timesteps=100_000, seeds=[0], nominal_episode_fraction=0.2)
        assert cmd is None

    def test_cell_b_uses_training_regime_none(self) -> None:
        cells = build_cells(static_checkpoint="ckpt.zip", kappa_robust=0.5, kappa_d_checkpoint=None, instance="ft06", nominal_episode_fraction=0.2)
        cell_b = next(c for c in cells if c.label == "B")
        cmd = training_command(cell_b, instance="ft06", init_checkpoint="ckpt.zip", timesteps=100_000, seeds=[0, 1], nominal_episode_fraction=0.2)
        assert cmd is not None
        assert "--training-regime" in cmd
        assert cmd[cmd.index("--training-regime") + 1] == "none"
        assert "--nominal-episode-fraction" not in cmd  # only relevant under domain_randomized

    def test_cell_c_passes_nominal_episode_fraction(self) -> None:
        cells = build_cells(static_checkpoint="ckpt.zip", kappa_robust=0.5, kappa_d_checkpoint=None, instance="ft06", nominal_episode_fraction=0.3)
        cell_c = next(c for c in cells if c.label == "C")
        cmd = training_command(cell_c, instance="ft06", init_checkpoint="ckpt.zip", timesteps=100_000, seeds=[0], nominal_episode_fraction=0.3)
        assert "--nominal-episode-fraction" in cmd
        assert cmd[cmd.index("--nominal-episode-fraction") + 1] == "0.3"
        assert "--kappa" in cmd
        assert cmd[cmd.index("--kappa") + 1] == "0.0"


class TestEvaluationCommand:
    def test_builds_rl_only_evaluation(self) -> None:
        cells = build_cells(static_checkpoint="ckpt.zip", kappa_robust=0.5, kappa_d_checkpoint=None, instance="ft06", nominal_episode_fraction=0.2)
        cell_a = cells[0]
        cmd = evaluation_command(cell_a, checkpoint="ckpt.zip", instances=["ft06", "la01"], disruption="breakdowns")
        assert "--methods" in cmd and cmd[cmd.index("--methods") + 1] == "rl"
        assert "--disruption" in cmd and cmd[cmd.index("--disruption") + 1] == "breakdowns"
        assert "--rl-policy" in cmd and cmd[cmd.index("--rl-policy") + 1] == "ckpt.zip"
