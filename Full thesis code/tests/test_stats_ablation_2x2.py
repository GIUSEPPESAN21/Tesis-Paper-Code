"""Tests for experiments/stats/ablation_2x2.py (Fase 3, T3.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.stats.ablation_2x2 import (
    compute_factorial_effects,
    friedman_four_cells,
    tabla_ablacion_2x2_markdown,
)


class TestComputeFactorialEffects:
    def test_no_effects_when_all_cells_equal(self) -> None:
        effects = compute_factorial_effects({"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0})
        assert effects.env_effect == pytest.approx(0.0)
        assert effects.reward_effect == pytest.approx(0.0)
        assert effects.interaction == pytest.approx(0.0)

    def test_pure_env_effect(self) -> None:
        # dinamico (C,D) uniformly +5 over estatico (A,B), no reward effect, no interaction.
        effects = compute_factorial_effects({"A": 10.0, "B": 10.0, "C": 15.0, "D": 15.0})
        assert effects.env_effect == pytest.approx(5.0)
        assert effects.reward_effect == pytest.approx(0.0)
        assert effects.interaction == pytest.approx(0.0)

    def test_pure_reward_effect(self) -> None:
        effects = compute_factorial_effects({"A": 10.0, "B": 14.0, "C": 10.0, "D": 14.0})
        assert effects.env_effect == pytest.approx(0.0)
        assert effects.reward_effect == pytest.approx(4.0)
        assert effects.interaction == pytest.approx(0.0)

    def test_pure_interaction(self) -> None:
        # reward effect is +2 in estatico (B-A) but +8 in dinamico (D-C) -> interaction = 6.
        effects = compute_factorial_effects({"A": 10.0, "B": 12.0, "C": 10.0, "D": 18.0})
        assert effects.interaction == pytest.approx(6.0)

    def test_raises_on_missing_cell(self) -> None:
        with pytest.raises(KeyError):
            compute_factorial_effects({"A": 1.0, "B": 2.0, "C": 3.0})


class TestFriedmanFourCells:
    def test_runs_over_pivoted_blocks(self) -> None:
        long_df = pd.DataFrame(
            {
                "block": ["ft06", "ft06", "ft06", "ft06", "la01", "la01", "la01", "la01"],
                "cell": ["A", "B", "C", "D", "A", "B", "C", "D"],
                "rpd": [20.0, 22.0, 18.0, 25.0, 30.0, 28.0, 26.0, 35.0],
            }
        )
        result = friedman_four_cells(long_df, value_col="rpd")
        assert result.k_methods == 4
        assert result.n_blocks == 2
        assert set(result.average_ranks) == {"A", "B", "C", "D"}


class TestTablaAblacion2x2Markdown:
    def test_pendiente_corrida_when_no_data(self) -> None:
        md = tabla_ablacion_2x2_markdown(None, None)
        assert md.count("PENDIENTE-CORRIDA") >= 3

    def test_real_effects_when_cell_means_present(self) -> None:
        md = tabla_ablacion_2x2_markdown({"A": 10.0, "B": 12.0, "C": 10.0, "D": 18.0}, None)
        assert "PENDIENTE-CORRIDA" not in md.split("## Prueba de Friedman")[0]
        assert "+6.0000" in md  # the interaction from test_pure_interaction's example
